"""The demo's model: one loaded policy, one rollout at a time, one switch.

The session is what the window talks to, and it is deliberately the only thing
that talks to the robot. It reuses the evaluation stack unchanged —
:class:`~policy_abstraction.evaluation.backends.EvalBackends` for the servers
and the loaded policy,
:class:`~policy_abstraction.evaluation.episode_runner.EpisodeRunner` for the
control loop — and adds three things:

* **The guidance gate.** :attr:`DemoSession.guided` is a flag the runner polls
  at every replan (``guidance_gate``). Flipping it never rebuilds anything:
  the retrieved demo, the perception tracker and the alignment keep running
  either way, so the next chunk is guided or not, and RTC stitches it onto the
  tail of the last one exactly as it would any other replan.
* **The live snapshot.** The runner's ``on_step`` observer publishes an
  immutable :class:`LiveView` per control step — the exterior frame, the
  wrist frame, and whatever the abstractor produced this step (the instance
  map, the pairs, the distances, the perception overlay, the retrieved demo
  frame). The window polls it on a timer; nothing here touches a widget.
* **The trend history.** Every step with an alignment appends a
  :class:`TrendSample`, so the trend plot can draw the live distances against
  the retrieved demo's, and shade the stretches where guidance was on.

Threading, once: the rollout runs on a worker thread that owns the env
client; the GUI thread reads snapshots under a lock and sets flags. The only
robot command the GUI thread issues directly is the preview ``get_obs``
between rollouts, and it takes the client's lock without blocking to do it.
"""

from __future__ import annotations

import dataclasses
import enum
import pathlib
import threading
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

from policy_abstraction.data.recorder import BaseRecorder, RobodmRecorder
from policy_abstraction.evaluation.backends import EvalBackends
from policy_abstraction.visualization.drawing import pair_colors


class DemoState(enum.Enum):
    IDLE = "idle"
    """No rollout: the arm is parked and the view streams a preview."""
    RUNNING = "running"
    STOPPING = "stopping"
    """Stop requested; the rollout ends at its next step boundary and homes."""


class OverlayMode(enum.Enum):
    """What the exterior view draws on top of the camera frame."""

    AUTO = "auto"
    """Masks and pairs while guidance is on; the bare frame while it is off —
    so the picture itself says which mode the arm is in."""
    ON = "on"
    """Always draw them (perception runs either way, so they are always there)."""
    OFF = "off"

    def shows(self, guided: bool) -> bool:
        if self is OverlayMode.AUTO:
            return guided
        return self is OverlayMode.ON

    def next(self) -> "OverlayMode":
        order = list(OverlayMode)
        return order[(order.index(self) + 1) % len(order)]


@dataclasses.dataclass(frozen=True)
class ReferenceTrends:
    """The retrieved demo's distance trends — fixed for the rollout."""

    ref_traj: np.ndarray
    """``[T_ref, n_pairs]`` reference distance per pair over demo time."""
    ref_std: Optional[np.ndarray]
    """``[T_ref, n_pairs]`` multi-demo spread, or ``None`` for one demo."""
    pair_labels: tuple[str, ...]
    pair_subtraj: tuple[Optional[int], ...]
    """Which sub-trajectory each pair belongs to (``None`` = all)."""
    subtrajectories: tuple[tuple[int, int], ...]
    """``(start, end)`` reference index ranges, end exclusive."""
    colors: dict[str, tuple[int, int, int]]
    """RGB per pair label — the palette every view shares."""
    task: str

    @property
    def n_pairs(self) -> int:
        return len(self.pair_labels)


@dataclasses.dataclass(frozen=True)
class TrendSample:
    """One control step's alignment, for the trend plot."""

    step: int
    live: np.ndarray
    """``[n_pairs]`` live distance, NaN where a pair was not measurable."""
    aligned_t: int
    aligned_ts: np.ndarray
    """``[n_pairs]`` per-pair aligned reference index."""
    current_subtraj_idx: Optional[int]
    guidance_weights: np.ndarray
    relevances: dict[str, float]
    guided: bool
    """Whether the chunk executing at this step was guided."""
    replan: bool
    """This step queried the policy (a new chunk starts here)."""
    chunk_size: int
    guidance_factor: float = 0.0
    """The guidance factor applied to the chunk executing at this step: the
    configured factor scaled by the objects' relevance, and 0 while the
    switch withheld guidance or before the pipeline was up."""
    guidance_scale: float = 1.0
    """What a full-strength factor is (``--guidance.factor``), so
    ``guidance_factor / guidance_scale`` is the fraction of guidance applied."""


@dataclasses.dataclass(frozen=True)
class LiveView:
    """What the window paints: the newest frame and the newest analysis.

    Immutable and replaced wholesale, so a reader never sees a half-updated
    snapshot. Fields from the abstractor are ``None`` until it has run — before
    retrieval, in an unguided launch, or when perception found nothing.
    """

    image: Optional[np.ndarray] = None
    """Exterior (primary) camera frame, upright RGB."""
    wrist: Optional[np.ndarray] = None
    step: int = 0
    timestamp: float = 0.0
    guided: bool = False
    """The chunk executing now came from a guided query."""
    replan: bool = False
    intrinsic: Optional[np.ndarray] = None
    extrinsic: Optional[np.ndarray] = None
    abstraction: Any = None
    """The step's :class:`~policy_abstraction.evaluation.abstraction.AbstractionResult`."""
    seg: Optional[np.ndarray] = None
    """``[H, W]`` int32 instance map of the exterior view (0 = background)."""
    names: dict[int, str] = dataclasses.field(default_factory=dict)
    """Instance id → object name, for the instances in :attr:`seg`."""
    ee_pos: Optional[np.ndarray] = None
    perception_overlay: Optional[np.ndarray] = None
    """The detector's own rendering of its masks on the exterior view."""
    ref_frame: Optional[np.ndarray] = None
    """The retrieved demo's frame at the aligned time, pairs drawn on it."""
    guidance_info: Optional[dict] = None
    colors: dict[str, tuple[int, int, int]] = dataclasses.field(default_factory=dict)

    @property
    def analysed(self) -> bool:
        return self.abstraction is not None


class NullRecorder(BaseRecorder):
    """A recorder that keeps nothing.

    The runner runs retrieval and the demo analysis only when it has a
    recorder — they write their artefacts and metadata into the episode. A
    demo that does not record still needs the analysis, so it hands the
    runner this: every write is dropped, and nothing is spent inside the
    control loop encoding camera streams.
    """

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    def initialize(self, episode_data_path: str, episode_metadata: dict) -> None:
        self.metadata = dict(episode_metadata)

    def record(self, **kwargs: Any) -> None:
        return None

    def save(self) -> None:
        return None

    def update_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value


def frame_from_obs(obs: dict, camera) -> Optional[np.ndarray]:
    """The upright frame of *camera* in *obs*, or ``None``."""
    if camera is None:
        return None
    raw = obs.get(camera.image_key)
    if raw is None:
        return None
    return np.ascontiguousarray(camera.apply_image_transform(np.asarray(raw)))


class DemoSession:
    """One loaded policy, rollouts on demand, guidance switchable under them.

    Args:
        backends: Live backends from
            :func:`~policy_abstraction.evaluation.backends.start_backends`,
            launched in the *guided* configuration (retrieval, demo analysis
            and real perception on) — that is what gives the gate something to
            open. Launched unguided, the demo still runs and streams, and
            :attr:`guidance_available` stays False.
        task: The prompt handed to the policy; the env's own when ``None``.
        max_steps: Per-rollout cap; the env's task horizon when ``0``.
        guided: The switch's position at the first rollout.
        overlay_mode: What the exterior view draws; see :class:`OverlayMode`.
        guided_warmup: Also compile the guided path during set-up, so the
            first flip of the switch does not stall the arm.
        save_video: Write the replay video after each rollout (slow; the
            episode itself is always recorded when the run saves episodes).
        output_dir: Where episodes go; the run's episode dir when ``None``.
        on_state_change: Called (from any thread) whenever :attr:`state`, the
            switch or the overlay mode changes.
    """

    def __init__(
        self,
        backends: EvalBackends,
        *,
        task_id: int = 0,
        task: Optional[str] = None,
        max_steps: int = 0,
        guided: bool = False,
        overlay_mode: OverlayMode = OverlayMode.AUTO,
        guided_warmup: bool = True,
        save_video: bool = False,
        output_dir: Optional[pathlib.Path] = None,
        on_state_change: Optional[Callable[[], None]] = None,
        training_heatmap: Any = None,
        show_heatmap: bool = False,
    ) -> None:
        self.backends = backends
        self.training_heatmap = training_heatmap
        """A :class:`~policy_abstraction.demo.heatmap.TrainingHeatmap` of the
        demos' grasp positions, or ``None`` when no positions file was given."""
        self.show_heatmap = bool(show_heatmap) and training_heatmap is not None
        self.show_readouts = True
        """Whether the pair lines on the exterior view carry their distance
        numbers. Off, the lines and masks stay and the picture is quieter."""
        self.task_id = task_id
        self._task = task
        self._max_steps = int(max_steps)
        self._guided = threading.Event()
        if guided:
            self._guided.set()
        self.overlay_mode = overlay_mode
        self._guided_warmup = guided_warmup
        self._save_video = save_video
        self._on_state_change = on_state_change

        args = backends.args
        self.output_dir = pathlib.Path(
            output_dir
            if output_dir is not None
            else (
                getattr(args.output, "episode_data_path", None) or "data/demo/episodes"
            )
        )

        # How many actions a chunk contributes before the next replan: the
        # width of the reference snippet the trend plot overlays per chunk.
        policy_args = getattr(args, "policy", None)
        self._replan_steps = int(getattr(policy_args, "replan_steps", 25) or 25)

        self.state = DemoState.IDLE
        self.last_error: Optional[str] = None
        self.rollouts = 0
        """Rollouts finished in this session."""
        self.episode_paths: list[str] = []
        self.warming_up = False
        self.saving = False
        """True while the previous rollout's episode is being written to
        disk. The arm is already parked and a new rollout may start."""
        self.warmup_seconds: Optional[float] = None
        self.guided_warmup_seconds: Optional[float] = None

        self._thread: Optional[threading.Thread] = None
        self._save_threads: list[threading.Thread] = []
        self._stop_requested = threading.Event()
        self._paused = threading.Event()
        self._runner = None
        self._lock = threading.Lock()
        self._view = LiveView()
        self._trends: Optional[ReferenceTrends] = None
        self._history: list[TrendSample] = []
        self._colors: dict[str, tuple[int, int, int]] = {}
        self._last_ref_frame: Optional[np.ndarray] = None
        self._last_perception_overlay: Optional[np.ndarray] = None
        self._camera_params: dict = {}
        self._live_preview: Optional[bool] = None
        self._homes_with: Optional[str] = None
        self._live_step = 0
        self._step_limit = 0
        self._started_at = 0.0
        # The factor of the chunk on the arm, carried between replans.
        self._last_factor = 0.0
        self._last_scale = 1.0

        spec = backends.env.env_spec
        self.camera = spec.primary_camera if spec is not None else None
        """The exterior view — the primary camera of the env spec."""
        self.wrist_camera = spec.wrist_camera if spec is not None else None

        # One visualiser for the whole session rather than one per rollout,
        # so the viewer layout the operator picked survives the next Start.
        # None when --viz.no-enable.
        self.visualizer: Any = None
        try:
            from policy_abstraction.evaluation.backends import (
                build_runner_components,
            )

            self.visualizer = build_runner_components(args, backends.env.env_spec)[
                "visualizer"
            ]
        except Exception as exc:  # a fake backends in tests, or no viz
            logger.debug(f"no visualizer for the demo session ({exc})")
        # Open the viewer sink now rather than at the first rollout, so a
        # served web viewer has a URL by the time the window is built.
        start = getattr(self.visualizer, "start", None)
        if callable(start):
            try:
                start()
            except Exception as exc:
                logger.warning(f"Could not start the Rerun visualiser: {exc}")

    # ------------------------------------------------------------------
    # Set-up
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        if self._task:
            return self._task
        env_task = getattr(self.backends.env, "language_instruction", None)
        if env_task:
            return str(env_task)
        return str(self.backends.task_languages[self.task_id])

    def set_task(self, text: str) -> bool:
        """Change the instruction handed to the policy.

        The runner reads its task string at every policy query, so while a
        rollout runs the new prompt reaches the policy at the next replan;
        retrieval and the demo analysis keep the demo they chose for the
        original prompt. Between rollouts it is simply what the next one
        starts with. Blank text is ignored. Returns whether anything changed.
        """
        text = " ".join(str(text).split())
        if not text or text == self.task:
            return False
        self._task = text
        runner = self._runner
        if runner is not None and self.state is DemoState.RUNNING:
            try:
                runner.task_description = text
                logger.info(f"Instruction changed under the running policy: {text!r}")
            except Exception as exc:
                logger.warning(
                    f"Could not hand the running policy the new prompt: {exc}"
                )
        else:
            logger.info(f"Instruction for the next rollout: {text!r}")
        self._notify()
        return True

    def prepare_env(self) -> None:
        """Build the env, grab a preview frame, warm the policy up.

        Talks to the robot (the env's own reset may home it), so the caller
        decides when — after the window is up, so a slow move is watched
        rather than mistaken for a hang.
        """
        args = self.backends.args
        self.backends.env.make_env(
            task_id=self.task_id,
            camera_heights=args.camera.height,
            camera_widths=args.camera.width,
            seed=args.seed,
        )
        self._camera_params = self._fetch_camera_params()
        self.refresh_preview()
        self.warm_up_policy()

    def _fetch_camera_params(self) -> dict:
        if self.camera is None:
            return {}
        try:
            params = self.backends.env.get_camera_params(
                camera_names=[self.camera.name],
                height=self.backends.args.camera.height,
                width=self.backends.args.camera.width,
            )
        except TypeError:
            try:
                params = self.backends.env.get_camera_params()
            except Exception as exc:
                logger.debug(f"no camera params ({exc})")
                return {}
        except Exception as exc:
            logger.debug(f"no camera params ({exc})")
            return {}
        entry = (params or {}).get(self.camera.name) or {}
        out: dict = {}
        if entry.get("intrinsic") is not None:
            out["intrinsic"] = np.asarray(entry["intrinsic"], dtype=np.float64)
        if entry.get("extrinsic") is not None:
            out["extrinsic"] = np.asarray(entry["extrinsic"], dtype=np.float64)
        return out

    def warm_up_policy(self) -> None:
        """Pay both first-inference costs now: the plain path, then the
        guided one, so neither the first Start nor the first flip of the
        switch stalls the arm. Never raises."""
        self.warming_up = True
        self._notify()
        try:
            with logger.contextualize(
                startup_phase="WARMING UP THE POLICY",
                startup_detail=(
                    "One throw-away inference so the checkpoint compiles now "
                    "rather than during the first rollout. The arm is not "
                    "commanded."
                ),
            ):
                logger.info("Warming up the policy (one inference, actions dropped) …")
                self.warmup_seconds = self.backends.warm_up_policy(self.task)
                if self.warmup_seconds is not None:
                    logger.info(f"Policy warm-up took {self.warmup_seconds:.1f}s")
            if self._guided_warmup and hasattr(self.backends, "warm_up_guided_policy"):
                with logger.contextualize(
                    startup_phase="WARMING UP THE GUIDED PATH",
                    startup_detail=(
                        "One guided inference against a synthetic target, so "
                        "the guided graph compiles now rather than the first "
                        "time the switch is flipped. The arm is not commanded."
                    ),
                ):
                    logger.info("Warming up the guided path (actions dropped) …")
                    self.guided_warmup_seconds = self.backends.warm_up_guided_policy(
                        self.task
                    )
                    if self.guided_warmup_seconds is not None:
                        logger.info(
                            f"Guided warm-up took {self.guided_warmup_seconds:.1f}s"
                        )
        except Exception as exc:
            logger.warning(f"Policy warm-up raised ({exc}); the first rollout pays it.")
        finally:
            self.warming_up = False
            self._notify()

    # ------------------------------------------------------------------
    # The switch
    # ------------------------------------------------------------------

    @property
    def guided(self) -> bool:
        """The switch: whether the *next* replan is asked for guidance."""
        return self._guided.is_set()

    @guided.setter
    def guided(self, value: bool) -> None:
        if bool(value) == self._guided.is_set():
            return
        if value:
            self._guided.set()
        else:
            self._guided.clear()
        logger.info(f"Guidance switched {'ON' if value else 'OFF'}")
        self._notify()

    def toggle_guidance(self) -> bool:
        self.guided = not self.guided
        return self.guided

    def toggle_readouts(self) -> bool:
        """Show or hide the distance numbers on the pair overlay."""
        self.show_readouts = not self.show_readouts
        self._notify()
        return self.show_readouts

    def toggle_heatmap(self) -> bool:
        """Show or hide the training heat map. Stays off without positions."""
        if self.training_heatmap is None:
            self.show_heatmap = False
        else:
            self.show_heatmap = not self.show_heatmap
        self._notify()
        return self.show_heatmap

    def cycle_overlay_mode(self) -> OverlayMode:
        self.overlay_mode = self.overlay_mode.next()
        self._notify()
        return self.overlay_mode

    @property
    def guidance_available(self) -> bool:
        """Whether flipping the switch changes anything right now.

        False before the first rollout has retrieved and analysed its demo, in
        a run launched unguided, and when the abstractor could not be set up
        on this scene (nothing detected, VLM found no pairs).
        """
        runner = self._runner
        if runner is None:
            return False
        return bool(getattr(runner, "guidance_available", False))

    @property
    def guidance_configured(self) -> bool:
        """Whether the run was launched with the guided pipeline at all."""
        args = self.backends.args
        return bool(
            getattr(args, "guidance", None) is not None
            and args.guidance.enable
            and args.retrieval.enable
            and args.demo_analysis.enable
        )

    @property
    def abstraction_failed(self) -> bool:
        runner = self._runner
        return bool(
            runner is not None and getattr(runner, "_abstractor_setup_failed", False)
        )

    # ------------------------------------------------------------------
    # Snapshots for the window
    # ------------------------------------------------------------------

    def live_view(self) -> LiveView:
        with self._lock:
            return self._view

    @property
    def trends(self) -> Optional[ReferenceTrends]:
        with self._lock:
            return self._trends

    def trend_history(self, since: int = 0) -> list[TrendSample]:
        """Samples appended since index *since* (a copy, safe to iterate)."""
        with self._lock:
            return list(self._history[since:])

    @property
    def trend_count(self) -> int:
        with self._lock:
            return len(self._history)

    @property
    def live_step(self) -> int:
        return self._live_step

    @property
    def step_limit(self) -> int:
        return self._step_limit

    @property
    def elapsed(self) -> float:
        if self.state is DemoState.IDLE or not self._started_at:
            return 0.0
        return time.time() - self._started_at

    @property
    def is_running(self) -> bool:
        return self.state in (DemoState.RUNNING, DemoState.STOPPING)

    def refresh_preview(self) -> bool:
        """Pull a frame from the env while nothing is running.

        Probed once: a simulator that serves no ``get_obs`` says no on the wire,
        and after that the last frame simply stands. Never blocks on the
        client's lock — a skipped tick costs one frame, a blocked GUI thread
        costs the operator the picture of a moving arm.
        """
        if self.is_running or self._live_preview is False:
            return False
        lock = getattr(self.backends.env, "rpc_lock", None)
        if lock is not None and not lock.acquire(blocking=False):
            return False
        try:
            obs = self.backends.env.get_obs()
        except Exception as exc:
            if self._live_preview is None:
                logger.info(
                    f"This env serves no on-demand observations ({exc}); the view "
                    "holds the last rollout's frame between rollouts."
                )
            self._live_preview = False
            return False
        finally:
            if lock is not None:
                lock.release()
        self._live_preview = True
        image = frame_from_obs(obs, self.camera)
        if image is None:
            return False
        with self._lock:
            self._view = dataclasses.replace(
                self._view,
                image=image,
                wrist=frame_from_obs(obs, self.wrist_camera),
                step=0,
                timestamp=time.time(),
                replan=False,
                intrinsic=self._camera_params.get("intrinsic"),
                extrinsic=self._camera_params.get("extrinsic"),
            )
        return True

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def start_rollout(self) -> bool:
        """Run the policy on a worker thread. False if one is already running."""
        if self.state is not DemoState.IDLE:
            return False
        self._stop_requested.clear()
        self._paused.clear()
        self.last_error = None
        self.state = DemoState.RUNNING
        self._started_at = time.time()
        self._live_step = 0
        self._last_factor = 0.0
        with self._lock:
            self._history = []
            self._trends = None
            self._colors = {}
            self._last_ref_frame = None
            self._last_perception_overlay = None
            self._view = dataclasses.replace(
                self._view,
                abstraction=None,
                seg=None,
                names={},
                perception_overlay=None,
                ref_frame=None,
                guidance_info=None,
                colors={},
                guided=self.guided,
            )
        if self._thread is not None and self._thread.is_alive():
            # The previous rollout is still writing its episode; it no longer
            # touches the robot, so it may finish alongside the new one.
            self._save_threads.append(self._thread)
        self._thread = threading.Thread(
            target=self._run_rollout, name=f"demo-rollout-{self.rollouts}", daemon=True
        )
        self._thread.start()
        self._notify()
        return True

    @property
    def paused(self) -> bool:
        """The pause switch. Only meaningful while a rollout runs; the runner
        holds between steps while it is set and the arm keeps its last
        target."""
        return self._paused.is_set()

    def toggle_pause(self) -> bool:
        """Pause a running rollout, or resume a paused one."""
        if self.state is not DemoState.RUNNING:
            return False
        if self._paused.is_set():
            self._paused.clear()
            logger.info("Rollout resumed by the operator")
        else:
            self._paused.set()
            logger.info("Rollout paused by the operator")
        self._notify()
        return self.paused

    def request_stop(self) -> bool:
        """End the rollout at its next step boundary; the arm then homes."""
        if self.state is not DemoState.RUNNING:
            return False
        self.state = DemoState.STOPPING
        self._stop_requested.set()
        self._paused.clear()
        self._notify()
        return True

    def _should_stop(self) -> bool:
        return self._stop_requested.is_set()

    def _run_rollout(self) -> None:
        """Worker-thread body. Must never raise past this frame."""
        args = self.backends.args
        runner = None
        episode_path = None
        try:
            max_steps = self._max_steps or int(
                self.backends.task_max_steps[self.task_id]
            )
            self._step_limit = max_steps
            recorder: BaseRecorder
            if getattr(args.output, "save_episodes", True):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                episode_path = self.output_dir / f"demo_{self.rollouts:03d}_{stamp}.vla"
                recorder = RobodmRecorder()
            else:
                # Retrieval and the demo analysis run only with a recorder;
                # this one costs nothing per step.
                recorder = NullRecorder()
            overrides: dict[str, Any] = dict(
                stop_signal=self._should_stop,
                guidance_gate=self._guided.is_set,
                on_step=self._on_step,
                pause_signal=self._paused.is_set,
                render_overlays=True,
            )
            if self.visualizer is not None:
                overrides["visualizer"] = self.visualizer
            runner = self.backends.build_runner(
                task_description=self.task,
                initial_state=self._initial_state(),
                max_steps=max_steps,
                episode_data_path=str(episode_path) if episode_path else None,
                episode_idx=self.rollouts,
                recorder=recorder,
                **overrides,
            )
            self._runner = runner
            runner.run()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"Demo rollout raised: {exc}")
            logger.debug(traceback.format_exc())
        finally:
            self._finish(runner, episode_path)

    def _initial_state(self) -> Any:
        try:
            states = self.backends.env.get_task_init_states(task_id=self.task_id)
        except Exception as exc:
            logger.debug(f"env published no init states ({exc}); using None")
            return None
        if states is None or not len(states):
            return None
        return states[0]

    def _finish(self, runner, episode_path) -> None:
        """Park the arm first, then write the episode.

        Encoding a long rollout's video streams into the ``.vla`` takes a
        minute or more, and an operator who pressed Stop is watching an arm
        that has stopped moving but not gone home. So the arm is homed and
        the session is IDLE — the next rollout can start — before the save
        begins; the save then runs on this same thread with :attr:`saving`
        set, which the window shows.
        """
        self.rollouts += 1
        try:
            self._home()
        except Exception as exc:
            logger.error(f"Could not return the robot home: {exc}")
            self.last_error = f"homing failed: {exc}"
        self.state = DemoState.IDLE
        self.refresh_preview()
        self._notify()

        if runner is None:
            return
        self.saving = True
        self._notify()
        try:
            runner.save_episode(success=None)
            saved = getattr(runner, "episode_data_path", None) or episode_path
            if saved:
                self.episode_paths.append(str(saved))
        except Exception as exc:
            logger.warning(f"Failed to save the demo episode: {exc}")
        if self._save_video and episode_path is not None:
            try:
                runner.save_video(
                    str(self.output_dir.parent / "videos"), self.rollouts - 1, False
                )
            except Exception as exc:
                logger.debug(f"No replay video written: {exc}")
        self.saving = False
        self._notify()

    def _home(self) -> None:
        """Park the arm: ``home`` on the real cell, ``reset`` on a simulator."""
        env = self.backends.env
        if self._homes_with is None:
            try:
                env.home()
                self._homes_with = "home"
                return
            except Exception as exc:
                logger.debug(f"env has no home() ({exc}); using reset() to park")
                self._homes_with = "reset"
        if self._homes_with == "home":
            env.home()
        else:
            env.reset()

    # ------------------------------------------------------------------
    # The runner's observer (rollout thread)
    # ------------------------------------------------------------------

    def _on_step(
        self,
        *,
        images: dict | None = None,
        t: int = 0,
        abstraction_result=None,
        frame_bundle=None,
        guidance_info: dict | None = None,
        replan: bool = False,
        guided: bool = False,
        **_ignored,
    ) -> None:
        self._live_step = int(t)
        paused = bool(_ignored.get("paused", False))
        if replan:
            # A new chunk: what guidance it carries holds until the next one.
            info = guidance_info or {}
            self._last_factor = float(info.get("guidance_factor", 0.0) or 0.0)
            scale = info.get("hl_factor") or info.get("requested_factor")
            if scale:
                self._last_scale = max(float(scale), 1e-6)
        runner = self._runner
        abstractor = getattr(runner, "abstractor", None) if runner is not None else None

        image = wrist = None
        if images:
            if self.camera is not None and self.camera.name in images:
                image = np.ascontiguousarray(images[self.camera.name])
            if self.wrist_camera is not None and self.wrist_camera.name in images:
                wrist = np.ascontiguousarray(images[self.wrist_camera.name])

        seg = None
        names: dict[int, str] = {}
        ee_pos = None
        if abstraction_result is not None and abstractor is not None:
            seg = getattr(abstractor, "_latest_agentview_predicted_seg", None)
            names = dict(getattr(abstractor, "_int_name_map", {}) or {})
            ec = abstraction_result.entity_centroids or {}
            ee = ec.get("ee")
            ee_pos = np.asarray(ee, dtype=np.float64) if ee is not None else None
            if not self._colors and abstraction_result.pairs:
                self._colors = pair_colors(abstraction_result.pairs)
            overlays = abstraction_result.perception_overlays
            if overlays:
                self._last_perception_overlay = overlays[0]
        if frame_bundle is not None and frame_bundle.ref_image_annotated is not None:
            self._last_ref_frame = frame_bundle.ref_image_annotated

        camera_params = {}
        handler = getattr(runner, "_viz_handler", None)
        if handler is not None and self.camera is not None:
            camera_params = (handler.camera_params or {}).get(self.camera.name, {})
        intrinsic = camera_params.get("intrinsic", self._camera_params.get("intrinsic"))
        extrinsic = camera_params.get("extrinsic", self._camera_params.get("extrinsic"))

        trends = None
        sample = None
        if abstraction_result is not None and abstractor is not None:
            trends = self._reference_trends(abstractor)
            if trends is not None:
                sample = self._trend_sample(
                    trends,
                    abstraction_result,
                    t,
                    replan=replan,
                    guided=guided,
                    guidance_factor=self._last_factor if guided else 0.0,
                    guidance_scale=self._last_scale,
                )

        with self._lock:
            if trends is not None and self._trends is None:
                self._trends = trends
            if sample is not None:
                self._history.append(sample)
            if paused:
                # A preview frame while holding: keep the last analysis so
                # the overlays and panels stay put, only the picture moves.
                self._view = dataclasses.replace(
                    self._view,
                    image=image if image is not None else self._view.image,
                    wrist=wrist if wrist is not None else self._view.wrist,
                    timestamp=time.time(),
                    replan=False,
                )
                return
            self._view = LiveView(
                image=image if image is not None else self._view.image,
                wrist=wrist if wrist is not None else self._view.wrist,
                step=int(t),
                timestamp=time.time(),
                guided=bool(guided),
                replan=bool(replan),
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                abstraction=abstraction_result,
                seg=seg,
                names=names,
                ee_pos=ee_pos,
                perception_overlay=self._last_perception_overlay,
                ref_frame=self._last_ref_frame,
                guidance_info=guidance_info,
                colors=dict(self._colors),
            )

    def _reference_trends(self, abstractor) -> Optional[ReferenceTrends]:
        if self._trends is not None:
            return self._trends
        ref_traj = getattr(abstractor, "ref_traj", None)
        if ref_traj is None:
            return None
        pairs = list(getattr(abstractor, "_translated_pairs", []) or [])
        labels = tuple(abstractor.pair_labels)
        subtraj_of = {p.label: p.subtraj_idx for p in pairs}
        colors = self._colors or pair_colors(pairs)
        self._colors = colors
        ref_std = getattr(abstractor, "ref_std", None)
        return ReferenceTrends(
            ref_traj=np.asarray(ref_traj, dtype=np.float32),
            ref_std=None if ref_std is None else np.asarray(ref_std, dtype=np.float32),
            pair_labels=labels,
            pair_subtraj=tuple(subtraj_of.get(lbl) for lbl in labels),
            subtrajectories=tuple(
                (int(a), int(b))
                for a, b in (getattr(abstractor, "_subtrajectories", []) or [])
            ),
            colors=dict(colors),
            task=self.task,
        )

    def _trend_sample(
        self,
        trends: ReferenceTrends,
        result,
        t: int,
        *,
        replan: bool,
        guided: bool,
        guidance_factor: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> TrendSample:
        live = np.full(trends.n_pairs, np.nan, dtype=np.float32)
        for i, lbl in enumerate(trends.pair_labels):
            v = result.live_distances.get(lbl)
            if v is not None:
                live[i] = float(v)
        aligned_ts = np.asarray(result.aligned_ts, dtype=np.int32)
        if aligned_ts.shape != (trends.n_pairs,):
            aligned_ts = np.full(trends.n_pairs, int(result.aligned_t), dtype=np.int32)
        weights = np.asarray(result.guidance_weights, dtype=np.float32)
        if weights.shape != (trends.n_pairs,):
            weights = np.ones(trends.n_pairs, dtype=np.float32)
        return TrendSample(
            step=int(t),
            live=live,
            aligned_t=int(result.aligned_t),
            aligned_ts=aligned_ts,
            current_subtraj_idx=result.current_subtraj_idx,
            guidance_weights=weights,
            relevances=dict(result.relevances or {}),
            guided=bool(guided),
            replan=bool(replan),
            chunk_size=self._replan_steps,
            guidance_factor=float(guidance_factor),
            guidance_scale=float(guidance_scale),
        )

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def viewer_url(self) -> Optional[str]:
        """The Rerun web viewer's URL, for the window's embedded 3-D panel.

        Set when the visualiser serves its recording from this process
        (``--viz.serve``); ``None`` without a visualiser or when it streams to
        an external viewer instead.
        """
        return (
            getattr(self.visualizer, "web_viewer_url", None)
            if self.visualizer
            else None
        )

    @property
    def viewer_layout(self) -> Optional[str]:
        """``"full"`` / ``"scene"``, or ``None`` without a Rerun visualiser."""
        return getattr(self.visualizer, "layout", None) if self.visualizer else None

    def set_viewer_layout(self, layout: str) -> bool:
        """Switch the Rerun viewer between the 3-D scene alone and the full
        analysis layout. False when there is no viewer to switch."""
        if self.visualizer is None or not hasattr(self.visualizer, "set_layout"):
            return False
        try:
            self.visualizer.set_layout(layout)
        except Exception as exc:
            logger.warning(f"Could not switch the viewer layout: {exc}")
            return False
        self._notify()
        return True

    # ------------------------------------------------------------------

    def _notify(self) -> None:
        if self._on_state_change is not None:
            try:
                self._on_state_change()
            except Exception as exc:
                logger.debug(f"state-change callback raised: {exc}")

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop any rollout, wait for it, park the arm, drop the backends."""
        if self.is_running:
            logger.info("Stopping the in-flight rollout before shutting down …")
            self.request_stop()
            if self._thread is not None:
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    logger.error(
                        f"Rollout thread did not stop within {timeout:.0f}s; "
                        "shutting the servers down anyway. The env server's own "
                        "disconnect handler parks the arm."
                    )
        try:
            self._home()
        except Exception as exc:
            logger.warning(f"Could not park the arm on shutdown: {exc}")
        # Episodes still being written finish before the process goes.
        for thread in [*self._save_threads, self._thread]:
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)
        if self.visualizer is not None:
            try:
                self.visualizer.close()
            except Exception:
                pass
        self.backends.shutdown()


__all__ = [
    "DemoSession",
    "DemoState",
    "LiveView",
    "OverlayMode",
    "ReferenceTrends",
    "TrendSample",
    "frame_from_obs",
]

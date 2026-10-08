"""Episode execution logic for LIBERO evaluations."""

from __future__ import annotations

import collections
import math
import pathlib
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

import imageio
import numpy as np
from loguru import logger
from openpi_client import image_tools
from tqdm import tqdm

from policy_abstraction.data.recorder import BaseRecorder, RobodmRecorder
from policy_abstraction.dynamics.base import BaseDynamicsModel
from policy_abstraction.dynamics.identity import IdentityDynamicsModel
from policy_abstraction.envs.spec import CameraSpec, EnvSpec
from policy_abstraction.evaluation.abstraction import PerceptionMode, RolloutAbstractor
from policy_abstraction.evaluation.constants import VIDEO_FPS
from policy_abstraction.evaluation.guidance import (
    guidance_request_flags,
    pack_guidance,
    pack_guidance_params,
)
from policy_abstraction.evaluation.guidance_builder import (
    build_distance_guidance_params,
    merge_guidance_params,
)
from policy_abstraction.evaluation.obs_encoder import BaseObsEncoder, IdentityObsEncoder
from policy_abstraction.evaluation.retrieval import run_retrieval_and_analysis
from policy_abstraction.utils.transforms import quat2axisangle
from policy_abstraction.visualization.alignment_plot import AlignmentPlotter
from policy_abstraction.visualization.base import BaseVisualizer
from policy_abstraction.visualization.frame_bundle import FrameBundle
from policy_abstraction.visualization.handler import VisualizationHandler

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@contextmanager
def _profile(profiling_dict: dict[str, list[float]], key: str):
    """Append elapsed seconds to *profiling_dict[key]* on exit."""
    t0 = time.perf_counter()
    yield
    profiling_dict[key].append(time.perf_counter() - t0)


@contextmanager
def _timed():
    """Yield a lightweight object whose ``.elapsed`` is set on exit."""

    class _Timer:
        elapsed: float = 0.0

    timer = _Timer()
    t0 = time.perf_counter()
    yield timer
    timer.elapsed = time.perf_counter() - t0


def build_policy_input(
    env_spec,
    obs: dict,
    images: dict[str, np.ndarray],
    *,
    task_description: str,
    rng_seed: int,
) -> dict:
    """The policy server's input dict, built from the env spec.

    ``images`` is keyed by camera name; the spec's
    ``policy_input_keys.images`` decides which cameras feed which policy keys.
    The state vector uses ``policy_input_keys.state_layout`` when the spec
    provides one and falls back to the robot's internal state layout
    otherwise. ``__rng_seed__`` makes per-inference seeding deterministic.

    Module-level rather than a method because the warm-up query
    (:meth:`EvalBackends.warm_up_policy`) has to build the *identical* element
    before any episode exists: a shape the compiler has not seen is a shape it
    recompiles for, which would move the cost right back into the first
    rollout.
    """
    io_keys = env_spec.policy_input_keys
    element: dict = {}
    for cam_name, policy_key in io_keys.images:
        if cam_name not in images:
            raise ValueError(
                f"policy_input_keys references camera {cam_name!r} which "
                f"isn't currently available; have {list(images)}"
            )
        element[policy_key] = images[cam_name]

    state_layout = io_keys.state_layout or env_spec.robot.state_layout
    element[io_keys.state_key] = state_layout.build(obs, env_spec.robot.gripper)
    element[io_keys.prompt_key] = str(task_description)
    element["__rng_seed__"] = np.array(rng_seed, dtype=np.int64)
    return element


def policy_preprocess_imgs(
    images: dict[str, np.ndarray], resize_size: int
) -> dict[str, np.ndarray]:
    """Resize and normalise every camera image to the policy resolution.

    Returns a new dict keyed by the same camera names with each image as
    uint8 ``[resize_size, resize_size, 3]``.
    """
    return {
        name: image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, resize_size, resize_size)
        )
        for name, img in images.items()
    }


class EpisodeRunner:
    """Manages execution of a single evaluation episode.

    The action loop is composed of four pluggable components that can be
    supplied at construction time.  Each has a sensible default:

    +-----------------+---------------------+---------------------------------------+
    | Component       | Default             | Responsibility                        |
    +=================+=====================+=======================================+
    | ``obs_encoder`` | IdentityObsEncoder  | Compute features from raw obs         |
    +-----------------+---------------------+---------------------------------------+
    | ``dynamics_model`` | IdentityDynamicsModel | Predict future states from actions |
    +-----------------+---------------------+---------------------------------------+
    | ``visualizer``  | ``None`` (disabled) | Display obs, actions, robot state     |
    +-----------------+---------------------+---------------------------------------+
    | ``recorder``    | ``None`` (disabled) | Persist trajectory to disk            |
    +-----------------+---------------------+---------------------------------------+

    A ``RobodmRecorder`` is created automatically when *only*
    ``episode_data_path`` is provided and no explicit ``recorder`` is passed,
    preserving backward-compatibility with callers that used the old API.
    """

    def __init__(
        self,
        env,
        client,
        task_description,
        initial_state,
        args,
        max_steps: int,
        episode_data_path: str | None = None,
        *,
        env_spec: EnvSpec | None = None,
        seed: int = 0,
        episode_idx: int = 0,
        obs_encoder: BaseObsEncoder | None = None,
        dynamics_model: BaseDynamicsModel | None = None,
        visualizer: BaseVisualizer | None = None,
        recorder: BaseRecorder | None = None,
        retriever=None,
        analyzer=None,
        interaction_retriever=None,
        abstractor_kwargs: dict | None = None,
        robot_model_name: str | None = None,
        guidance_fn=None,
        use_distance_guidance: bool = False,
        guidance_factor: float = 1.0,
        debug_guidance: bool = False,
        aligned_t_offset: int = 0,
        normalize_guidance: bool = True,
        use_line_search: bool = True,
        guidance_max_alpha: float = 1.0,
        multi_alpha: bool = False,
        multi_alpha_combine: str = "sequential",
        relation_filter: frozenset[str] | None = None,
        high_level_base_factor: float | None = None,
        high_level_decay_k: float = 25.0,
        low_level_base_factor: float = 1.0,
        low_level_decay_k: float = 1.0,
        control_dt: float = 1.0 / 50,
        depth_stride: int = 0,
        demo_segmenter=None,
        stop_signal: Callable[[], bool] | None = None,
        guidance_gate: Callable[[], bool] | None = None,
        on_step: Callable[..., None] | None = None,
        pause_signal: Callable[[], bool] | None = None,
        render_overlays: bool = False,
    ) -> None:
        """
        Initialise the episode runner.

        Args:
            env:               LIBERO environment instance.
            client:            Policy client used for inference.
            task_description:  Natural-language task string.
            initial_state:     Initial state passed to ``env.set_init_state``.
            args:              Evaluation arguments namespace (``camera_height``,
                               ``camera_width``, ``resize_size``, ``replan_steps``, …).
            max_steps:         Maximum environment steps for this episode.
            episode_data_path: Optional path for saving episode data.  When
                               provided without an explicit *recorder*, a
                               :class:`RobodmRecorder` is created automatically.
            obs_encoder:       :class:`BaseObsEncoder` instance.  Defaults to
                               :class:`IdentityObsEncoder`.
            dynamics_model:    :class:`BaseDynamicsModel` instance.  Defaults to
                               :class:`IdentityDynamicsModel`.
            visualizer:        :class:`BaseVisualizer` instance.  Disabled when
                               ``None`` (default).
            recorder:          :class:`BaseRecorder` instance.  When ``None``
                               *and* ``episode_data_path`` is given, a
                               :class:`RobodmRecorder` is created automatically.
            demo_segmenter:    Supplies segmentation for a retrieved demo that
                               carries none of its own (any real recording).
                               See ``analysis.demo_segmenter``.
            depth_stride:      Record metric depth every N steps (0 = never).
                               Depth is stored uncompressed, so it dominates
                               episode size next to the h264 colour streams --
                               hence a stride rather than every step.
            stop_signal:       Optional zero-argument predicate polled once per
                               step. When it returns True the episode ends
                               cleanly at the next step boundary — the recorder
                               is flushed and ``run`` returns normally, with
                               ``stopped_early`` set. This is how an operator's
                               stop button, and the wall-clock timeout built on
                               it, end a rollout without killing the process or
                               the loaded policy. It is checked *between*
                               steps, so a stop lands within one control period
                               rather than mid-``env.step``.
            guidance_gate:     Optional zero-argument predicate polled at every
                               replan. While it returns False the policy is
                               queried *without* the ``__gp_*__`` guidance
                               params — the abstractor, perception and the
                               alignment keep stepping, so distances and
                               overlays stay live and the next replan that
                               finds the gate open is guided from an alignment
                               that never stopped tracking. This is how an
                               operator switches guidance on and off under a
                               running policy: nothing is rebuilt, and RTC
                               blends the switch into the unexecuted tail of
                               the previous chunk like any other replan. The
                               gate only has something to close when guidance
                               is configured (``use_distance_guidance`` and a
                               retrieved demo); it never turns guidance on in
                               a run launched without it.
            pause_signal:      Optional zero-argument predicate polled between
                               steps. While it returns True the loop holds:
                               no action is sent, so the arm stays where the
                               last target left it, and the plan, the
                               alignment and the RTC state are untouched, so
                               resuming continues the chunk exactly where it
                               stopped. While holding, the observer (below)
                               keeps receiving the env's current frames when
                               the env serves them on demand, so a display
                               stays live. A stop request ends the hold.
            render_overlays:   Render the annotated frame bundle (chunk
                               overlays, the retrieved demo's frame with pairs
                               drawn on it) even with no visualiser and no
                               recording. A live display that reads them
                               through ``on_step`` needs this; the batch
                               evaluator never does.
            on_step:           Optional observer called once per executed
                               step, after the recorder, with the same keyword
                               arguments as ``_record_timestep`` plus
                               ``replan`` (this step queried the policy) and
                               ``guided`` (that query carried guidance
                               params). Runs on the rollout thread; anything
                               it raises is logged and dropped, so a display
                               bug can never end an episode.
            abstractor_kwargs: Optional kwargs forwarded to
                               :class:`~policy_abstraction.evaluation.abstraction.RolloutAbstractor`
                               (e.g. ``vlm_model``, ``feature``, ``mta_window``).
                               When provided and retrieval+analysis succeeds, a
                               :class:`~policy_abstraction.evaluation.abstraction.RolloutAbstractor`
                               is constructed automatically and used for MTA at
                               each replanning step.  ``None`` disables abstraction.
        """
        self.env = env
        self.client = client
        self.task_description = task_description
        self.initial_state = initial_state
        self.args = args
        self.max_steps = max_steps
        self.episode_data_path = episode_data_path
        self._depth_stride = max(0, int(depth_stride))
        self._demo_segmenter = demo_segmenter

        # EnvSpec drives every camera/robot lookup that used to be hardcoded
        # to the LIBERO + Panda layout. Falls back to the cached spec on the
        # WebsocketEnvClient (populated by setup_suite) when not passed
        # explicitly so callers don't have to thread it through.
        spec = env_spec if env_spec is not None else getattr(env, "env_spec", None)
        if spec is None:
            raise ValueError(
                "EpisodeRunner requires an EnvSpec — either pass env_spec=… or "
                "use a WebsocketEnvClient that has called setup_suite() first."
            )
        self.env_spec: EnvSpec = spec
        self._cameras: tuple[CameraSpec, ...] = spec.cameras
        self._primary_camera: CameraSpec = spec.primary_camera
        self._wrist_camera: CameraSpec | None = spec.wrist_camera

        _layout = spec.robot.action_layout
        _dummy = np.zeros(_layout.action_dim, dtype=np.float32)
        if _layout.gripper_dim is not None:
            _dummy[_layout.gripper_dim] = -1.0
        self._dummy_action: np.ndarray = _dummy

        # Scoped RNG for deterministic per-inference policy seeds.
        # Each inference call draws a unique seed from this generator,
        # making the episode fully reproducible given (seed, episode_idx).
        self._episode_rng = np.random.default_rng([seed, episode_idx])

        # --- Pluggable components ---
        self.obs_encoder: BaseObsEncoder = obs_encoder or IdentityObsEncoder()
        self.dynamics_model: BaseDynamicsModel = (
            dynamics_model or IdentityDynamicsModel()
        )
        self.visualizer: BaseVisualizer | None = visualizer
        self.retriever = retriever
        self.analyzer = analyzer
        self.interaction_retriever = interaction_retriever
        self._abstractor_kwargs = abstractor_kwargs
        self.abstractor = None
        self.interaction_abstractor = None
        self._guidance_fn = guidance_fn
        self._use_distance_guidance = use_distance_guidance
        self._debug_guidance = debug_guidance
        self._normalize_guidance = normalize_guidance
        self._use_line_search = use_line_search
        self._guidance_max_alpha = float(guidance_max_alpha)
        self._multi_alpha = bool(multi_alpha)
        self._multi_alpha_combine = str(multi_alpha_combine)
        self._relation_filter = relation_filter
        self._aligned_t_offset = aligned_t_offset
        # high_level_base_factor defaults to guidance_factor for backward compatibility
        self._hl_base_factor: float = (
            high_level_base_factor
            if high_level_base_factor is not None
            else guidance_factor
        )
        self._guidance_factor = self._hl_base_factor  # kept for external callers
        self._hl_decay_k = high_level_decay_k
        self._ll_base_factor = low_level_base_factor
        self._ll_decay_k = low_level_decay_k
        self._control_dt = control_dt
        self._action_shape: tuple[int, int, int] | None = (
            None  # cached after first inference
        )
        self._alignment_plotter: AlignmentPlotter | None = None
        self._constraint_keys: list[str] = []

        # Backwards-compatible auto-create of RobodmRecorder
        if recorder is not None:
            self.recorder: BaseRecorder | None = recorder
        elif episode_data_path:
            self.recorder = RobodmRecorder()
        else:
            self.recorder = None

        # --- Profiling stats ---
        self.profiling: dict[str, list[float]] = collections.defaultdict(list)

        # --- Internal bookkeeping ---
        self.action_plan: collections.deque = collections.deque()
        self.replay_images: list[np.ndarray] = []
        self.num_inferences: int = 0
        self.steps_executed: int = 0
        """Policy steps actually executed, excluding the settle phase.
        Counted at the ``env.step`` itself rather than at the bottom of
        the loop, so an episode that ends on ``done`` — which breaks
        before the bottom — is not under-reported by one. Read after
        :meth:`run` to record how far a stopped rollout got."""
        self.stopped_early: bool = False
        """True when ``stop_signal`` ended the episode rather than ``done`` or
        the step budget."""
        self._stop_signal = stop_signal
        self._guidance_gate = guidance_gate
        self._on_step = on_step
        self._pause_signal = pause_signal
        self.paused: bool = False
        """True while the loop is holding on ``pause_signal``."""
        self.guidance_applied: bool = False
        """Whether the most recent policy query carried guidance params.
        Read by an operator display to say which mode the arm is in."""
        self._retrieval_done: bool = False
        self._abstractor_setup_failed: bool = False
        """True when the abstractor could not be set up on the first frame
        (object not detected, VLM returned nothing, ...). The episode then runs
        unguided; the flag is also written to the episode metadata."""
        _viz_enabled = (
            visualizer is not None or bool(episode_data_path) or bool(render_overlays)
        )
        self._viz_handler = VisualizationHandler(
            args.camera.height,
            args.camera.width,
            alpha=0.65,
            env_spec=self.env_spec,
            enabled=_viz_enabled,
        )

        # --- Set up recorder -------------------------------------------------
        if self.recorder is not None and episode_data_path:
            # Flat keys preserved for stable per-episode .vla metadata schema.
            # Record a short backend label (``"groot"``, ``"openpi"``, ...)
            # so replays can display which policy produced these actions.
            _policy_cls = type(args.policy).__name__
            policy_backend_label = (
                _policy_cls.removesuffix("PolicyArgs").removesuffix("Args").lower()
                or _policy_cls.lower()
            )
            episode_metadata = {
                "task_description": str(task_description),
                "max_steps": max_steps,
                "resize_size": args.camera.resize_size,
                "camera_height": args.camera.height,
                "camera_width": args.camera.width,
                "replan_steps": args.policy.replan_steps,
                "num_steps_wait": args.env.num_steps_wait,
                "seed": getattr(args, "seed", None),
                "policy_backend": policy_backend_label,
                # Full camera/robot/policy layout so replays can reconstruct
                # the same EnvSpec the live runner used (multi-env support).
                "env_spec": self.env_spec.to_dict(),
            }
            self.recorder.initialize(episode_data_path, episode_metadata)

        # --- Robot base (used by visualiser and recorder) --------------------
        if hasattr(env, "get_robot_base"):
            _rb = env.get_robot_base()
            self._robot_base_pos = np.asarray(_rb["base_pos"], dtype=np.float64)
            self._robot_base_ori = np.asarray(_rb["base_ori"], dtype=np.float64)
        else:
            self._robot_base_pos = np.asarray(env.robots[0].base_pos, dtype=np.float64)
            self._robot_base_ori = np.asarray(env.robots[0].base_ori, dtype=np.float64)

        if self.recorder is not None:
            self.recorder.update_metadata(
                "robot_base_pos", self._robot_base_pos.tolist()
            )
            self.recorder.update_metadata(
                "robot_base_ori", self._robot_base_ori.tolist()
            )
            if robot_model_name:
                self.recorder.update_metadata("robot_model_name", robot_model_name)

        # --- Set up visualiser -----------------------------------------------
        if self.visualizer is not None:
            if hasattr(self.visualizer, "set_env_spec"):
                self.visualizer.set_env_spec(self.env_spec)
            # Derive a short policy label from the policy-args class name so
            # the viewer can show e.g. "groot" / "openpi" alongside the task.
            # Loose coupling: any class named ``XxxPolicyArgs`` is mapped to
            # ``xxx``; unknown shapes fall back to the lowercased class name.
            policy_cls = type(self.args.policy).__name__
            policy_label = (
                policy_cls.removesuffix("PolicyArgs").removesuffix("Args").lower()
                or policy_cls.lower()
            )
            self.visualizer.setup(
                task_description=str(task_description),
                policy_label=policy_label,
            )
            self.visualizer.set_robot_base(self._robot_base_pos, self._robot_base_ori)

        # --- Camera parameters for image projection --------------------------
        self._viz_handler.fetch_camera_params(self.env)

    # ------------------------------------------------------------------
    # Main episode loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[bool, int]:
        """
        Execute the episode.

        Returns:
            ``(success, num_inferences)`` tuple.
        """
        # Drop per-episode policy state before the first inference. Only
        # backends that opt in via ``resets_per_episode`` are called: every
        # client in the repo happens to define ``reset``, but most are no-ops
        # and one (GR00T) sends an endpoint its server may not implement, so
        # duck-typing this would risk a per-episode stall for no benefit. The
        # lerobot client opts in because its RTC buffer would otherwise blend
        # the previous episode's action tail into this episode's first chunk.
        if getattr(self.client, "resets_per_episode", False):
            self.client.reset()

        self.env.reset()
        obs = self.env.set_init_state(self.initial_state)

        t = 0
        done = False

        logger.info(f"Starting episode for task: {self.task_description}")

        pbar = tqdm(
            total=self.max_steps,
            desc=str(self.task_description),
            unit="step",
            dynamic_ncols=True,
        )
        _prof_lines: list[str] = []

        while t < self.max_steps + self.args.env.num_steps_wait:
            try:
                # ---- Operator / timeout stop ---------------------------------
                # Checked before any work so a stop requested during the
                # previous step's inference takes effect immediately rather
                # than paying for one more policy query.
                if self._stop_signal is not None and self._stop_signal():
                    self.stopped_early = True
                    logger.info(f"Episode stopped early at step {t}")
                    break

                # ---- Operator pause ------------------------------------------
                if self._pause_signal is not None and self._pause_signal():
                    self._hold_while_paused(t)
                    if self._stop_signal is not None and self._stop_signal():
                        self.stopped_early = True
                        logger.info(f"Episode stopped early at step {t} (while paused)")
                        break

                # ---- Wait phase: let objects settle --------------------------
                if t < self.args.env.num_steps_wait:
                    obs, reward, done, info = self.env.step(self._dummy_action)
                    t += 1
                    continue

                # Wrist-camera extrinsic moves with the EEF, so it must be
                # refreshed before any consumer (retrieval setup, abstraction,
                # render) reads `self._viz_handler.camera_params` this step.
                self._viz_handler.fetch_camera_params(self.env)

                # ---- Retrieval (once, on the first real frame) ---------------
                if not self._retrieval_done:
                    self._retrieval_done = True
                    self._run_retrieval_setup(obs)

                # ---- Encode observation --------------------------------------
                robot_state = self.env_spec.robot.state_layout.build(
                    obs, self.env_spec.robot.gripper, dtype=np.float32
                )

                encoded_obs = self.obs_encoder.encode(obs, robot_state)

                # ---- Preprocess images for policy and video ------------------
                # One entry per camera in the env spec, keyed by name. Wrist
                # is optional — single-camera envs flow through unchanged.
                images = {
                    cam.name: self._extract_camera_image(obs, cam)
                    for cam in self._cameras
                }

                # ---- Abstraction step (MTA alignment) ------------------------
                abstraction_result = None
                interaction_result = None
                _alignment_plot_img = None
                frame_bundle = FrameBundle(
                    images=dict(images),
                    primary_name=self._primary_camera.name,
                )

                with _timed() as t_abs:
                    if self.abstractor is not None:
                        if (
                            self.abstractor._perception_mode
                            == PerceptionMode.GT_POINTCLOUD
                        ):
                            obs = self._enrich_obs_with_gt_pointclouds(obs)
                        abstraction_result = self.abstractor.step(
                            obs, self._viz_handler.camera_params or {}
                        )

                with _timed() as t_align:
                    if (
                        self._alignment_plotter is not None
                        and abstraction_result is not None
                    ):
                        is_replan = len(self.action_plan) == 0
                        _alignment_plot_img = self._alignment_plotter.update(
                            timestep=t,
                            live_distances=abstraction_result.live_distances,
                            aligned_t=abstraction_result.aligned_t,
                            current_subtraj_idx=abstraction_result.current_subtraj_idx,
                            chunk_size=self.args.policy.replan_steps
                            if is_replan
                            else None,
                            aligned_ts=abstraction_result.aligned_ts
                            if is_replan
                            else None,
                            is_replan=is_replan,
                            guidance_weights=abstraction_result.guidance_weights,
                            relevances=abstraction_result.relevances,
                        )

                # ---- Query policy if action plan is empty --------------------
                action_chunks = None
                guidance_info: dict | None = None
                if not self.action_plan:
                    dynamic_guidance, guidance_info, interaction_result = (
                        self._build_dynamic_guidance(obs, abstraction_result)
                    )
                    # The operator's switch. Everything above still ran, so
                    # the recorded factor says what was *applied*: zero, with
                    # the value the gate withheld kept alongside it.
                    gate_open = self._guidance_gate is None or bool(
                        self._guidance_gate()
                    )
                    if not gate_open and guidance_info is not None:
                        guidance_info = {
                            **guidance_info,
                            "requested_factor": guidance_info["guidance_factor"],
                            "guidance_factor": 0.0,
                            "gated": True,
                        }

                    with _timed() as t_policy:
                        policy_images = self._policy_preprocess_imgs(images)
                        action_chunks = self._query_policy(
                            obs,
                            policy_images,
                            guidance_fn=dynamic_guidance or self._guidance_fn,
                            use_guidance=gate_open,
                        )

                    # ---- Dynamics prediction ---------------------------------
                    predicted_action_chunk_states = self.dynamics_model.predict_states(
                        np.array(list(action_chunks)), robot_state, dt=self._control_dt
                    )
                    predicted_states = (
                        self.dynamics_model.predict_states(
                            np.array(list(self.action_plan)),
                            robot_state,
                            dt=self._control_dt,
                        )
                        if self.action_plan
                        else np.empty((0, len(robot_state)))
                    )

                    # ---- Render frame bundle ---------------------------------
                    with _timed() as t_render:
                        self._viz_handler.update_projections(
                            predicted_action_chunk_states
                        )
                        frame_bundle = self._viz_handler.render(
                            images,
                            obs=obs,
                            camera_params=self._viz_handler.camera_params,
                            abstraction_result=abstraction_result,
                            abstractor=self.abstractor,
                        )
                        if _alignment_plot_img is not None:
                            frame_bundle.alignment_plot = _alignment_plot_img
                            frame_bundle.combined_replay = (
                                self._viz_handler._compose_replay(frame_bundle)
                            )
                        if frame_bundle.combined_replay is not None:
                            self.replay_images.append(frame_bundle.combined_replay)

                    # ---- Visualise -------------------------------------------
                    with _timed() as t_viz:
                        if self.visualizer is not None:
                            self.visualizer.update(
                                timestep=t,
                                obs=obs,
                                images=images,
                                encoded_obs=encoded_obs,
                                robot_state=robot_state,
                                robot_joints=self._build_robot_joints(obs),
                                action_chunks=action_chunks,
                                predicted_states=predicted_states,
                                action_chunk_states=predicted_action_chunk_states,
                                camera_params=self._viz_handler.camera_params or None,
                                abstraction_result=abstraction_result,
                                frame_bundle=frame_bundle,
                                guidance_info=guidance_info,
                                replan_steps=self.args.policy.replan_steps,
                            )

                    # guidance_build was already appended by _build_dynamic_guidance;
                    # pop it so _log_replan_profiling can re-append it alongside
                    # the other per-replan timings.
                    _t_guidance = (
                        self.profiling["guidance_build"].pop()
                        if self.profiling["guidance_build"]
                        else 0.0
                    )
                    _t_perception = (
                        abstraction_result.perception_latency
                        if abstraction_result is not None
                        else 0.0
                    )
                    _prof_lines = self._log_replan_profiling(
                        {
                            "perception": _t_perception,
                            "abstraction": max(t_abs.elapsed - _t_perception, 0.0),
                            "alignment_plot": t_align.elapsed,
                            "guidance_build": _t_guidance,
                            "policy_infer": t_policy.elapsed,
                            "frame_bundle": t_render.elapsed,
                            "visualizer": t_viz.elapsed,
                        },
                        pbar,
                    )

                # ---- Execute action ------------------------------------------
                robot_joints = self._build_robot_joints(obs)
                action = self.action_plan.popleft()
                obs, reward, done, info = self.env.step(action.tolist())
                self.steps_executed += 1

                # ---- Visualise executed action & reward ----------------------
                if self.visualizer is not None:
                    self.visualizer.update(
                        timestep=t,
                        obs=obs,
                        images=images,
                        encoded_obs=encoded_obs,
                        robot_state=robot_state,
                        robot_joints=robot_joints,
                        action=action,
                        reward=float(reward),
                        done=bool(done),
                        camera_params=self._viz_handler.camera_params or None,
                        frame_bundle=None,
                    )

                # ---- Record timestep -----------------------------------------
                depths = None
                if self._depth_stride and t % self._depth_stride == 0:
                    depths = {
                        cam.name: d
                        for cam in self._cameras
                        if (d := self._extract_camera_depth(obs, cam)) is not None
                    } or None

                step_kwargs: dict[str, Any] = dict(
                    images=images,
                    depths=depths,
                    robot_state=robot_state,
                    robot_joints=robot_joints,
                    action=action,
                    t=t,
                    reward=reward,
                    done=done,
                    action_chunks=action_chunks,
                    abstraction_result=abstraction_result,
                    interaction_result=interaction_result,
                    frame_bundle=frame_bundle,
                    guidance_info=guidance_info,
                )
                self._record_timestep(**step_kwargs)
                if self._on_step is not None:
                    try:
                        self._on_step(
                            **step_kwargs,
                            replan=action_chunks is not None,
                            guided=self.guidance_applied,
                        )
                    except Exception as exc:
                        logger.warning(f"on_step observer raised, ignoring: {exc}")

                if done:
                    break

                t += 1
                if t > self.args.env.num_steps_wait:
                    pbar.update(1)

            except Exception as exc:
                # ``exc_info`` is a stdlib-logging kwarg; loguru treats any
                # kwarg as a str.format field and then formats the message,
                # so a traceback containing ``{a[:7]!r}`` raised KeyError here
                # and masked the real error. Pass the exception as an argument
                # so its text is never parsed.
                logger.opt(exception=True).error(
                    "Error during episode execution: {}", exc
                )
                raise

        pbar.close()
        # Print final profiling summary below the completed progress bar.
        if self.profiling:
            _fp = sys.stderr
            for line in _prof_lines:
                _fp.write(f"{line}\n")
            _fp.flush()
        return done, self.num_inferences

    def _hold_while_paused(self, t: int) -> None:
        """Wait out a pause, streaming the env's frames to the observer.

        Nothing is commanded, so the arm holds its last target. ``get_obs``
        is a UR5 server RPC; a simulator that lacks it simply pauses blind.
        """
        self.paused = True
        logger.info(f"Episode paused at step {t}")
        can_observe = self._on_step is not None and hasattr(self.env, "get_obs")
        try:
            while self._pause_signal is not None and self._pause_signal():
                if self._stop_signal is not None and self._stop_signal():
                    return
                if can_observe:
                    try:
                        obs = self.env.get_obs()
                        images = {
                            cam.name: self._extract_camera_image(obs, cam)
                            for cam in self._cameras
                            if cam.image_key in obs
                        }
                        self._on_step(  # type: ignore[misc]
                            images=images,
                            t=t,
                            replan=False,
                            guided=self.guidance_applied,
                            paused=True,
                        )
                    except Exception as exc:
                        logger.debug(f"paused preview unavailable ({exc})")
                        can_observe = False
                time.sleep(0.1)
        finally:
            self.paused = False
            logger.info(f"Episode resumed at step {t}")

    # ------------------------------------------------------------------
    # Retrieval & abstractor setup (runs once on first real frame)
    # ------------------------------------------------------------------

    def _run_retrieval_setup(self, obs: dict) -> None:
        """Run demo retrieval, build abstractors, and send visualiser blueprint."""
        with _profile(self.profiling, "retrieval_setup"):
            if self.retriever is None or self.recorder is None:
                self._send_blueprint()
                return

            first_frame = self._extract_camera_image(obs, self._primary_camera)

            with _profile(self.profiling, "setup/retrieval_and_analysis"):
                top_result, analysis, seg_name_map, retrieval_viz_data, top_episode = (
                    run_retrieval_and_analysis(
                        self.recorder,
                        self.retriever,
                        self.analyzer,
                        self.task_description,
                        first_frame,
                        k=1,
                        segmenter=self._demo_segmenter,
                    )
                )

            with _profile(self.profiling, "setup/viz_logging"):
                if self.visualizer is not None and retrieval_viz_data:
                    self.visualizer.log_retrieval_info(**retrieval_viz_data)

            _mds = (retrieval_viz_data or {}).get("multi_demo_stats")
            if _mds is not None and self.recorder is not None:
                self.recorder.update_metadata(
                    "multi_demo_stats",
                    {
                        "ref_traj": _mds["ref_traj"].tolist(),
                        "mean": _mds["mean"].tolist(),
                        "std": _mds["std"].tolist(),
                        "min": _mds["min"].tolist(),
                        "max": _mds["max"].tolist(),
                        "pair_labels": list(_mds["pair_labels"]),
                        "n_demos": int(_mds["n_demos"]),
                    },
                )

            if (
                self._abstractor_kwargs is not None
                and top_result is not None
                and analysis is not None
            ):
                self._setup_abstractor(
                    obs,
                    top_result,
                    analysis,
                    seg_name_map,
                    retrieval_viz_data,
                    top_episode,
                )
                self._setup_interaction_abstractor(first_frame)

        self._send_blueprint()

    def _setup_abstractor(
        self,
        obs,
        top_result,
        analysis,
        seg_name_map,
        retrieval_viz_data,
        demo_episode=None,
    ) -> None:
        """Construct and initialise the RolloutAbstractor.

        For :data:`PerceptionMode.REAL_PERCEPTION` the live env's
        ``get_segmentation_name_mapping()`` is *not* called — that mode
        simulates a real-world scenario with no GT scene info. For GT modes
        we expose the RPC as a lookup closure consulted only after the VLM
        proposes relevant names (intersect, never enumerate).
        """
        assert self._abstractor_kwargs is not None
        with _profile(self.profiling, "setup/abstractor_setup"):
            mode: PerceptionMode = self._abstractor_kwargs.get(
                "perception_mode", PerceptionMode.GT_SEG_RECONSTRUCT
            )

            gt_name_lookup = None
            if mode != PerceptionMode.REAL_PERCEPTION:
                # Capture once after reset; demo HDF5 fallback if the env
                # server lacks the RPC.
                try:
                    cached_map = (
                        self.env.get_segmentation_name_mapping() or seg_name_map
                    )
                except Exception:
                    cached_map = seg_name_map
                gt_name_lookup = lambda: cached_map  # noqa: E731

            self.abstractor = RolloutAbstractor(
                top_result,
                analysis,
                gt_name_lookup=gt_name_lookup,
                env_spec=self.env_spec,
                subtraj_boundaries=getattr(
                    analysis, "selected_subtraj_boundaries", None
                ),
                demo_episode=demo_episode,
                **self._abstractor_kwargs,
            )
            camera_params = self._viz_handler.camera_params or {}
            ok = self.abstractor.setup(obs, str(self.task_description), camera_params)

        if not ok:
            # The abstractor's contract is "False = continue without
            # abstraction". A scene where perception or the VLM missed an
            # object is still a valid rollout of the policy: run it unguided
            # and say so in the metadata, rather than record a 0-step
            # failure that looks like the arm never moved.
            logger.error(
                "[EpisodeRunner] RolloutAbstractor.setup() failed; running the "
                "policy WITHOUT guidance for this episode (recorded as "
                "abstraction_setup_failed=true in the episode metadata)."
            )
            self._abstractor_setup_failed = True
            self.abstractor = None
            if self.recorder is not None:
                self.recorder.update_metadata("abstraction_setup_failed", True)
            return

        # Inject multi-demo uncertainty (std) into the abstractor
        _mds = retrieval_viz_data.get("multi_demo_stats")
        if _mds is not None and "std" in _mds and "pair_labels" in _mds:
            self.abstractor.set_ref_std(_mds["std"], _mds["pair_labels"])

        self._store_abstractor_metadata(analysis, retrieval_viz_data)

        with _profile(self.profiling, "setup/viz_logging"):
            if self.visualizer is not None:
                self.visualizer.log_abstraction_setup(
                    live_scene_image=self.abstractor.live_analysis.scene_image
                    if self.abstractor.live_analysis
                    else None,
                    live_seg_image=self.abstractor.live_analysis.seg_index_image
                    if self.abstractor.live_analysis
                    else None,
                    mapping=self.abstractor.mapping,
                    ref_traj=self.abstractor.ref_traj,
                    pair_labels=self.abstractor.pair_labels,
                    constraints=analysis.constraints if analysis is not None else None,
                )
            self._viz_handler.set_constraints(
                analysis.constraints if analysis is not None else None
            )

        # Alignment plotter
        if self.abstractor.ref_traj is not None:
            _pair_st_idx = [p.subtraj_idx for p in self.abstractor._translated_pairs]
            self._alignment_plotter = AlignmentPlotter(
                ref_traj=self.abstractor.ref_traj,
                pair_labels=self.abstractor.pair_labels,
                subtrajectories=self.abstractor._subtrajectories,
                pair_subtraj_idx=_pair_st_idx,
                ref_std=self.abstractor.ref_std,
            )

    def _store_abstractor_metadata(
        self, analysis, retrieval_viz_data: dict | None = None
    ) -> None:
        """Persist abstractor pair labels, ref trajectory, and mapping to recorder.

        Also rewrites the ``vlm_analysis`` metadata's ``distance_pairs`` entry
        with the post-equivalence-mapping pairs (live-scene IDs), so the
        rerun ``retrieval/vlm_analysis`` panel reflects the IDs the system
        actually uses.
        """
        if self.recorder is None or self.abstractor is None:
            return

        self.recorder.update_metadata(
            "abstraction_pair_labels", self.abstractor.pair_labels
        )
        if self.abstractor.ref_traj is not None:
            self.recorder.update_metadata(
                "abstraction_ref_traj", self.abstractor.ref_traj.tolist()
            )
        if self.abstractor.mapping is not None:
            _m = self.abstractor.mapping
            self.recorder.update_metadata(
                "equivalence_mapping",
                {
                    "retrieved_objects": {
                        str(k): v for k, v in _m.retrieved_objects.items()
                    },
                    "live_objects": {str(k): v for k, v in _m.live_objects.items()},
                    "id_map": {str(k): str(v) for k, v in _m.id_map.items()},
                    "unmapped_retrieved_ids": _m.unmapped_retrieved_ids,
                    "vlm_prompt_text": getattr(_m, "vlm_prompt_text", "") or "",
                    "vlm_response_json": getattr(_m, "vlm_response_json", "") or "",
                },
            )

            conv = getattr(_m, "conversation", None)
            if conv is not None:
                import base64 as _base64
                import io as _io

                from PIL import Image as _Image

                msgs = conv.messages
                user_images_b64: list[str] = []
                if len(msgs) > 1:
                    for img_arr in msgs[1].images:
                        pil_img = _Image.fromarray(img_arr.astype(np.uint8))
                        buf = _io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        user_images_b64.append(
                            _base64.b64encode(buf.getvalue()).decode("utf-8")
                        )
                self.recorder.update_metadata(
                    "equivalence_conversation",
                    {
                        "model": conv.model,
                        "system_prompt": msgs[0].text if msgs else "",
                        "user_text": msgs[1].text if len(msgs) > 1 else "",
                        "user_images_b64": user_images_b64,
                        "assistant_response": msgs[2].text if len(msgs) > 2 else "",
                    },
                )

        if analysis is not None and analysis.constraints:
            self.recorder.update_metadata(
                "abstraction_constraints",
                [
                    {
                        "kind": c.kind,
                        "subtraj_idx": c.subtraj_idx,
                        "label": c.label,
                        "description": c.description,
                        "expression": c.expression,
                        "pairs_affected": c.pairs_affected,
                        "reasoning": c.reasoning,
                    }
                    for c in analysis.constraints
                ],
            )
            # Stable column order for the per-step constraint_costs array.
            self._constraint_keys = [
                f"{'sg' if c.kind == 'sub_goal' else 'pc'}/{c.label}"
                for c in analysis.constraints
            ]
            self.recorder.update_metadata(
                "abstraction_constraint_keys", self._constraint_keys
            )
        import dataclasses as _dc

        translated_pair_dicts = [
            _dc.asdict(p) for p in self.abstractor._translated_pairs
        ]
        self.recorder.update_metadata("translated_pairs", translated_pair_dicts)

        # Rewrite vlm_analysis.distance_pairs with translated (live-ID) pairs
        # so the rerun "VLM Analysis" panel shows the IDs the system actually
        # uses after equivalence mapping. Pairs without a live counterpart
        # (dropped during translate_pairs) are excluded.
        if retrieval_viz_data is not None:
            vlm_analysis_meta = retrieval_viz_data.get("vlm_analysis")
            if isinstance(vlm_analysis_meta, dict):
                vlm_analysis_meta = dict(vlm_analysis_meta)
                vlm_analysis_meta["distance_pairs"] = translated_pair_dicts
                self.recorder.update_metadata("vlm_analysis", vlm_analysis_meta)
                if self.visualizer is not None:
                    self.visualizer.log_retrieval_info(vlm_analysis=vlm_analysis_meta)

    def _setup_interaction_abstractor(self, first_frame: np.ndarray) -> None:
        """Optionally construct and initialise the InteractionAbstractor."""
        if self.interaction_retriever is None or self.abstractor is None:
            return

        from policy_abstraction.evaluation.abstraction import InteractionAbstractor

        with _profile(self.profiling, "setup/interaction_retrieval"):
            top_interaction = None
            try:
                interaction_results = self.interaction_retriever.retrieve(
                    first_frame, top_k=1
                )
                if interaction_results:
                    top_interaction = interaction_results[0]
                    logger.info(
                        f"[EpisodeRunner] Top interaction: "
                        f"{top_interaction.get('interaction_chain')!r} "
                        f"(score={top_interaction.get('score', 0):.3f})"
                    )
            except Exception as exc:
                logger.error(f"[EpisodeRunner] Interaction retrieval failed: {exc}")
                raise

            if top_interaction is None:
                return

            self.interaction_abstractor = InteractionAbstractor(
                interaction_result=top_interaction,
                translated_pairs=self.abstractor._translated_pairs,
                int_name_map=self.abstractor._int_name_map,
                primary_camera=self._primary_camera,
                feature=self.abstractor._feature,
                mta_window=self.abstractor._mta_window,
                erode_px=self.abstractor._erode_px,
                dynamics_model=self.abstractor._dynamics_model,
            )
            ia_ok = self.interaction_abstractor.setup()
            if not ia_ok:
                logger.warning(
                    "[EpisodeRunner] InteractionAbstractor.setup() failed; "
                    "low-level guidance disabled."
                )
                self.interaction_abstractor = None
                return

            if self.recorder is not None:
                self.recorder.update_metadata(
                    "interaction_pair_labels",
                    self.interaction_abstractor.pair_labels,
                )
                if self.interaction_abstractor.ref_traj is not None:
                    self.recorder.update_metadata(
                        "interaction_ref_traj",
                        self.interaction_abstractor.ref_traj.tolist(),
                    )
                self.recorder.update_metadata(
                    "interaction_result",
                    {
                        k: v
                        for k, v in top_interaction.items()
                        if isinstance(v, (str, int, float))
                    },
                )

    def _send_blueprint(self) -> None:
        """Send the visualiser blueprint once after all static data is logged."""
        if self.visualizer is not None:
            self.visualizer.send_blueprint()

    # ------------------------------------------------------------------
    # Guidance construction
    # ------------------------------------------------------------------

    def _build_dynamic_guidance(
        self, obs: dict, abstraction_result
    ) -> tuple[dict | None, dict | None, object | None]:
        """Build guidance params from abstraction + interaction results.

        Returns:
            ``(dynamic_guidance, guidance_info, interaction_result)``
        """
        if not (
            self._use_distance_guidance
            and abstraction_result is not None
            and abstraction_result.pairs is not None
            and abstraction_result.entity_centroids is not None
            and self.abstractor is not None
            and self.abstractor.ref_traj is not None
        ):
            return None, None, None

        _chunk_size = (
            self._action_shape[0] * self._action_shape[1]
            if self._action_shape is not None
            else self.args.policy.replan_steps
        )

        # Compute effective guidance factor from HL relevances
        hl_relevances = abstraction_result.relevances
        total_hl_relevance = np.sum(
            [v for v in hl_relevances.values() if v is not None]
        )
        total_hl_relevance = (
            total_hl_relevance / len(hl_relevances) if hl_relevances else 0.0
        )

        _guidance_factor = self._hl_base_factor
        if not self._debug_guidance:
            _guidance_factor = _guidance_factor * total_hl_relevance

        guidance_info: dict = {
            "total_hl_relevance": total_hl_relevance,
            "guidance_factor": _guidance_factor,
            # The high-level alignment residual, reported alongside the
            # low-level one below. It is what the guidance factor is scaled
            # against, so a run whose corrections look wrong is read by
            # comparing the two: the visualiser has always had a place for
            # these, and nothing was filling it.
            "hl_mse": float(abstraction_result.alignment_mse),
            "hl_factor": float(self._hl_base_factor),
        }

        # Per-pair aligned_ts (apply offset)
        _aligned_ts_offset = abstraction_result.aligned_ts + self._aligned_t_offset

        # Dynamics-integrated reference distances for the guidance chunk
        _dyn_ref_chunk = self.abstractor.compute_dynamics_ref_chunk(
            _aligned_ts_offset, _chunk_size
        )
        _dyn_ref_std = self.abstractor.crop_ref_std(_aligned_ts_offset, _chunk_size)
        _zero_aligned_ts = np.zeros_like(_aligned_ts_offset)

        _t_guidance_build = 0.0

        _ee_ori_0 = quat2axisangle(obs[self.env_spec.robot.eef_quat_key])
        _gripper_0 = np.atleast_1d(
            self.env_spec.robot.gripper.aperture_from_qpos(
                obs[self.env_spec.robot.gripper_qpos_key]
            )
        )

        with _timed() as _t_hl:
            hl_guidance = build_distance_guidance_params(
                pairs=abstraction_result.pairs,
                live_distances=abstraction_result.live_distances,
                entity_centroids=abstraction_result.entity_centroids,
                object_pointclouds=abstraction_result.object_pointclouds,
                ref_traj=_dyn_ref_chunk,
                ref_std=_dyn_ref_std,
                pair_labels=self.abstractor.pair_labels,
                aligned_ts=_zero_aligned_ts,
                chunk_size=_chunk_size,
                int_name_map=getattr(self.abstractor, "_int_name_map", None),
                dt=self._control_dt,
                guidance_factor=_guidance_factor,
                guidance_weights=abstraction_result.guidance_weights,
                ee_ori_0=_ee_ori_0,
                gripper_0=_gripper_0,
                relation_filter=self._relation_filter,
            )
        _t_guidance_build += _t_hl.elapsed

        # Low-level interaction guidance
        ll_guidance = None
        interaction_result = None
        if self.interaction_abstractor is not None:
            interaction_result = self.interaction_abstractor.step(abstraction_result)
            if (
                interaction_result is not None
                and interaction_result.ref_traj is not None
            ):
                ll_mse = interaction_result.alignment_mse
                ll_factor = self._ll_base_factor * math.exp(-self._ll_decay_k * ll_mse)
                guidance_info["ll_mse"] = ll_mse
                guidance_info["ll_factor"] = ll_factor

                with _timed() as _t_ll:
                    _ll_aligned_t = (
                        interaction_result.aligned_t + self._aligned_t_offset
                    )
                    _ll_n_pairs = len(interaction_result.pair_labels)
                    _ll_aligned_ts = np.full(_ll_n_pairs, _ll_aligned_t, dtype=np.int32)
                    ll_guidance = build_distance_guidance_params(
                        pairs=interaction_result.pairs,
                        live_distances=interaction_result.live_distances,
                        entity_centroids=abstraction_result.entity_centroids,
                        object_pointclouds=abstraction_result.object_pointclouds,
                        ref_traj=interaction_result.ref_traj,
                        pair_labels=interaction_result.pair_labels,
                        aligned_ts=_ll_aligned_ts,
                        chunk_size=_chunk_size,
                        int_name_map=getattr(self.abstractor, "_int_name_map", None),
                        dt=self._control_dt,
                        guidance_factor=ll_factor,
                        guidance_weights=None,
                        ee_ori_0=_ee_ori_0,
                        gripper_0=_gripper_0,
                        relation_filter=self._relation_filter,
                    )
                _t_guidance_build += _t_ll.elapsed

        self.profiling["guidance_build"].append(_t_guidance_build)

        dynamic_guidance = merge_guidance_params(hl_guidance, ll_guidance)

        return dynamic_guidance, guidance_info, interaction_result

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record_timestep(
        self,
        *,
        images: dict[str, np.ndarray],
        depths: dict[str, np.ndarray] | None,
        robot_state: np.ndarray,
        robot_joints: np.ndarray,
        action: np.ndarray,
        t: int,
        reward: float,
        done: bool,
        action_chunks: np.ndarray | None,
        abstraction_result,
        interaction_result,
        frame_bundle: FrameBundle,
        guidance_info: dict | None,
    ) -> None:
        """Record all per-timestep data to the recorder.

        ``images`` is keyed by camera name. Every camera that produced a
        frame is written under its own ``observation/<name>_image`` stream,
        so RoboCasa (3 cameras) and LIBERO (2) both round-trip losslessly.
        Replay reads the same per-camera streams via the saved EnvSpec.
        """
        if self.recorder is None:
            return

        self.recorder.record(
            camera_image_streams=images,
            camera_depth_streams=depths,
            depth_timestep=t if depths else None,
            state=robot_state,
            robot_joints=robot_joints,
            action=action,
            camera_params=self._viz_handler.camera_params,
            reward=np.array([reward], dtype=np.float32),
            done=np.array([done], dtype=bool),
            timestep=np.array([t], dtype=np.int32),
        )

        if action_chunks is not None:
            self.recorder.record(action_chunks=action_chunks)
            # Pointclouds are variable-shape (M points per object varies); the
            # recorder routes them to a .pcd.npz sidecar at the replan cadence.
            # Always pass — even an empty dict — so the sidecar's per-replan
            # index stays aligned with ``t // replan_steps`` on replay.
            # Object centroids are recoverable from the points so we don't store
            # them, but the full entity_centroids dict (which also carries the
            # EE + gripper-finger tips) is needed to draw distance-pair lines
            # on replay, so it is recorded when present.
            pcd_record: dict = {
                "abstraction_object_pointclouds": (
                    abstraction_result.object_pointclouds
                    if abstraction_result is not None
                    and abstraction_result.object_pointclouds
                    else {}
                ),
            }
            if abstraction_result is not None:
                if abstraction_result.scene_pointcloud is not None:
                    pcd_record["abstraction_scene_pointcloud"] = (
                        abstraction_result.scene_pointcloud
                    )
                if abstraction_result.entity_centroids:
                    pcd_record["abstraction_entity_centroids"] = (
                        abstraction_result.entity_centroids
                    )
            self.recorder.record(**pcd_record)

        if abstraction_result is not None:
            self._record_abstraction(abstraction_result, frame_bundle)

        if interaction_result is not None:
            self._record_interaction(interaction_result)

        if guidance_info is not None:
            self._record_guidance_info(guidance_info)

        if abstraction_result is not None:
            if (
                abstraction_result.guidance_weights is not None
                and len(abstraction_result.guidance_weights) > 0
            ):
                self.recorder.record(
                    guidance_weights=abstraction_result.guidance_weights.astype(
                        np.float32
                    ),
                )
            if abstraction_result.relevances:
                rel_labels = abstraction_result.pair_labels
                self.recorder.record(
                    guidance_relevances=np.array(
                        [
                            abstraction_result.relevances.get(lbl, np.nan)
                            for lbl in rel_labels
                        ],
                        dtype=np.float32,
                    ),
                )

    def _record_abstraction(
        self, abstraction_result, frame_bundle: FrameBundle
    ) -> None:
        """Record high-level abstraction data for one timestep."""
        if self.recorder is None:
            return
        labels = abstraction_result.pair_labels
        extra: dict = {}
        if frame_bundle.ref_image_annotated is not None:
            extra["abstraction_ref_image"] = frame_bundle.ref_image_annotated
        # Pass all non-primary reference frames + chunk overlays through as
        # per-camera dicts; the recorder fans them out to per-camera streams.
        if frame_bundle.ref_images:
            extra["ref_camera_image_streams"] = {
                name: img
                for name, img in frame_bundle.ref_images.items()
                if name != frame_bundle.primary_name and img is not None
            }
        if frame_bundle.chunk_overlays:
            extra["camera_chunk_overlay_streams"] = {
                name: img
                for name, img in frame_bundle.chunk_overlays.items()
                if img is not None
            }
        if frame_bundle.live_image_annotated is not None:
            extra["abstraction_live_image"] = frame_bundle.live_image_annotated
        if frame_bundle.alignment_plot is not None:
            extra["abstraction_alignment_plot"] = frame_bundle.alignment_plot
        # Per-step constraint costs (column order set by abstraction_constraint_keys metadata).
        if abstraction_result.constraint_costs:
            constraint_keys = self._constraint_keys or sorted(
                abstraction_result.constraint_costs.keys()
            )
            extra["abstraction_constraint_costs"] = np.array(
                [
                    abstraction_result.constraint_costs.get(k, np.nan)
                    for k in constraint_keys
                ],
                dtype=np.float32,
            )
        # Perception (predicted-seg) artefacts: per-view overlays + history grids.
        # Index 0 = agentview, 1 = wrist (matches _pcd_from_predicted_seg_multiview).
        overlays = abstraction_result.perception_overlays
        if overlays:
            if len(overlays) > 0:
                extra["perception_overlay_agentview"] = overlays[0]
            if len(overlays) > 1:
                extra["perception_overlay_wrist"] = overlays[1]
        grids = abstraction_result.perception_history_grids
        if grids:
            if len(grids) > 0:
                extra["perception_history_agentview"] = grids[0]
            if len(grids) > 1:
                extra["perception_history_wrist"] = grids[1]
        debug_imgs = abstraction_result.perception_debug_images
        if debug_imgs:
            if len(debug_imgs) > 0:
                extra["perception_debug_agentview"] = debug_imgs[0]
            if len(debug_imgs) > 1:
                extra["perception_debug_wrist"] = debug_imgs[1]
        if abstraction_result.perception_timings_md is not None:
            self.recorder.update_metadata(
                "perception_timings_md", abstraction_result.perception_timings_md
            )

        self.recorder.record(
            abstraction_live_distances=np.array(
                [abstraction_result.live_distances.get(lbl, np.nan) for lbl in labels],
                dtype=np.float32,
            ),
            abstraction_aligned_t=np.array(
                [abstraction_result.aligned_t], dtype=np.int32
            ),
            abstraction_ref_distances=np.array(
                [abstraction_result.ref_distances.get(lbl, np.nan) for lbl in labels],
                dtype=np.float32,
            ),
            abstraction_errors=np.array(
                [abstraction_result.distance_errors.get(lbl, np.nan) for lbl in labels],
                dtype=np.float32,
            ),
            abstraction_progress=np.array(
                [abstraction_result.mta_progress], dtype=np.float32
            ),
            abstraction_alignment_mse_high=np.array(
                [abstraction_result.alignment_mse], dtype=np.float32
            ),
            **extra,
        )

    def _record_interaction(self, interaction_result) -> None:
        """Record low-level interaction data for one timestep."""
        if self.recorder is None:
            return
        ll_labels = interaction_result.pair_labels
        self.recorder.record(
            interaction_live_distances=np.array(
                [
                    interaction_result.live_distances.get(lbl, np.nan)
                    for lbl in ll_labels
                ],
                dtype=np.float32,
            ),
            interaction_aligned_t=np.array(
                [interaction_result.aligned_t], dtype=np.int32
            ),
            interaction_ref_distances=np.array(
                [
                    interaction_result.ref_distances.get(lbl, np.nan)
                    for lbl in ll_labels
                ],
                dtype=np.float32,
            ),
            abstraction_alignment_mse_low=np.array(
                [interaction_result.alignment_mse], dtype=np.float32
            ),
            interaction_mta_progress=np.array(
                [interaction_result.mta_progress], dtype=np.float32
            ),
        )

    def _record_guidance_info(self, guidance_info: dict) -> None:
        """Record guidance factor values for one timestep."""
        if self.recorder is None:
            return
        guidance_record = {
            "guidance_factor": np.array(
                [guidance_info["guidance_factor"]], dtype=np.float32
            ),
        }
        if "ll_factor" in guidance_info:
            guidance_record["guidance_ll_factor"] = np.array(
                [guidance_info["ll_factor"]], dtype=np.float32
            )
        self.recorder.record(**guidance_record)

    # ------------------------------------------------------------------
    # Profiling display
    # ------------------------------------------------------------------

    def _log_replan_profiling(self, timings: dict[str, float], pbar) -> list[str]:
        """Append per-replan timings and render ANSI status lines below the progress bar.

        Returns the profiling lines for final summary printing.
        """
        step_total = sum(timings.values())
        for key, val in timings.items():
            self.profiling[key].append(val)
        self.profiling["step_total"].append(step_total)

        prof_lines = [
            f"  \033[36mperception\033[0m {timings['perception'] * 1e3:6.0f}ms"
            f"| \033[36mabstraction\033[0m {timings['abstraction'] * 1e3:6.0f}ms"
            f"| \033[36malignment\033[0m {timings['alignment_plot'] * 1e3:6.0f}ms"
            f"| \033[36mguidance\033[0m {timings['guidance_build'] * 1e3:6.0f}ms"
            f"| \033[36mpolicy\033[0m {timings['policy_infer'] * 1e3:6.0f}ms"
            f"| \033[36mframe_bundle\033[0m {timings['frame_bundle'] * 1e3:6.0f}ms"
            f"| \033[36mvisualizer\033[0m {timings['visualizer'] * 1e3:6.0f}ms"
            f"| \033[1m total\033[0m {step_total * 1e3:6.0f}ms",
        ]
        # Use tqdm.write so the bar is cleared before writing and redrawn
        # after. The previous sticky-below-bar rendering was fragile to any
        # other stderr output during the loop (e.g. logger.warning from
        # abstraction helpers), which drifted the cursor and let the bar
        # overwrite the last profile line.
        for line in prof_lines:
            pbar.write(line, file=pbar.fp)
        return prof_lines

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _enrich_obs_with_gt_pointclouds(self, obs: dict) -> dict:
        """Fetch GT object point clouds from the simulator and add to *obs*.

        Called when :attr:`PerceptionMode.GT_POINTCLOUD` is active.  Adds
        ``"gt_object_pointclouds"`` (``{seg_id → [N, 3]}``) and
        ``"gt_object_colors"`` (``{seg_id → [3]}``) to the observation.
        """
        try:
            raw = self.env.get_gt_object_pointclouds()
            gt_pcds: dict[int, np.ndarray] = {}
            gt_colors: dict[int, np.ndarray] = {}
            for seg_id, data in raw.items():
                seg_id = int(seg_id)
                gt_pcds[seg_id] = data["points"]
                gt_colors[seg_id] = data["colors"]
            obs["gt_object_pointclouds"] = gt_pcds
            obs["gt_object_colors"] = gt_colors
        except Exception as exc:
            logger.warning(f"[EpisodeRunner] Failed to fetch GT pointclouds: {exc}")
        return obs

    def _extract_camera_image(self, obs: dict, camera: CameraSpec) -> np.ndarray:
        """Read ``obs[camera.image_key]`` and apply the camera's image transform.

        Returns a contiguous uint8 RGB array suitable for downstream
        rendering and policy preprocessing.
        """
        return np.ascontiguousarray(camera.apply_image_transform(obs[camera.image_key]))

    def _extract_camera_depth(self, obs: dict, camera: CameraSpec) -> np.ndarray | None:
        """Metric depth for *camera*, or ``None`` when it publishes none.

        The camera's image transform is applied here too: depth is registered
        to the colour frame, and a transform applied to one but not the other
        would silently unregister them.
        """
        key = camera.depth_key
        if not key or key not in obs:
            return None
        depth = obs[key]
        if depth is None:
            return None
        return np.ascontiguousarray(
            camera.apply_image_transform(np.asarray(depth, dtype=np.float32))
        )

    def _build_robot_joints(self, obs: dict) -> np.ndarray:
        """Concatenate arm joint positions and gripper qpos into a single vector."""
        return np.concatenate(
            [
                obs[self.env_spec.robot.joint_pos_key],
                obs[self.env_spec.robot.gripper_qpos_key],
            ]
        ).astype(np.float32)

    def _build_policy_input(
        self,
        obs: dict,
        images: dict[str, np.ndarray],
    ) -> dict:
        """This episode's policy input. See :func:`build_policy_input`."""
        return build_policy_input(
            self.env_spec,
            obs,
            images,
            task_description=self.task_description,
            rng_seed=int(self._episode_rng.integers(0, 2**31)),
        )

    def _policy_preprocess_imgs(
        self, images: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """This episode's policy images. See :func:`policy_preprocess_imgs`."""
        return policy_preprocess_imgs(images, self.args.camera.resize_size)

    def _query_policy(
        self,
        obs,
        images: dict[str, np.ndarray],
        guidance_fn=None,
        *,
        use_guidance: bool = True,
    ) -> np.ndarray:
        """
        Query the policy client and populate ``self.action_plan``.

        ``use_guidance=False`` sends the request bare — no ``__gp_*__`` params
        and no fallback to the static ``guidance_fn`` — which is how the
        guidance gate turns a guided run into an unguided query for one replan.

        Returns:
            ``action_chunks``: the raw model output (ndarray). Previously was
            uint8 RGB visualisation and *action_chunks* is the raw model output.
        """
        element = self._build_policy_input(obs, images)

        effective_guidance = (
            guidance_fn if guidance_fn is not None else self._guidance_fn
        )
        if not use_guidance:
            effective_guidance = None
        self.guidance_applied = effective_guidance is not None
        if effective_guidance is not None:
            if callable(effective_guidance):
                element = pack_guidance(element, effective_guidance)
            else:
                effective_guidance = dict(effective_guidance)
                effective_guidance.update(
                    guidance_request_flags(
                        debug=self._debug_guidance,
                        normalize=self._normalize_guidance,
                        line_search=self._use_line_search,
                        max_alpha=self._guidance_max_alpha,
                        multi_alpha=self._multi_alpha,
                        multi_alpha_combine=self._multi_alpha_combine,
                    )
                )
                element = pack_guidance_params(element, effective_guidance)

        action_chunks: np.ndarray = self.client.infer(element)["actions"]
        self.num_inferences += 1

        if self._action_shape is None and action_chunks.ndim == 3:
            self._action_shape = (
                action_chunks.shape
            )  # (num_chunks, horizon, action_dim)

        if action_chunks.ndim == 3:  # [num_chunks, action_horizon, action_dim]
            selected_idx = 0

            self.action_plan.extend(
                action_chunks[selected_idx, : self.args.policy.replan_steps, :]
            )

            if len(self.action_plan) < self.args.policy.replan_steps:
                logger.warning(
                    f"Expected at least {self.args.policy.replan_steps} actions, "
                    f"but got {len(self.action_plan)}"
                )
        else:  # legacy: [action_horizon, action_dim]
            if len(action_chunks) < self.args.policy.replan_steps:
                raise ValueError(
                    f"We want to replan every {self.args.policy.replan_steps} steps, "
                    f"but policy only predicts {len(action_chunks)} steps."
                )
            self.action_plan.extend(action_chunks[: self.args.policy.replan_steps, :])

        return action_chunks

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    @property
    def guidance_available(self) -> bool:
        """Whether a replan *could* be guided right now.

        True once the retrieved demo has been analysed into distance pairs and
        guidance is configured; False before retrieval, in a run launched
        unguided, or when the abstractor could not be set up on this scene.
        An operator display reads it to tell "guidance off" from "guidance
        impossible".
        """
        return bool(
            self._use_distance_guidance
            and self.abstractor is not None
            and self.abstractor.ref_traj is not None
        )

    def get_profiling_summary(self) -> dict:
        """Return aggregated profiling stats (seconds) for this episode."""
        summary: dict = {}
        for key, values in self.profiling.items():
            if not values:
                continue
            arr = np.array(values)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "total": float(arr.sum()),
                "count": len(values),
            }
        return summary

    # ------------------------------------------------------------------
    # Post-episode utilities
    # ------------------------------------------------------------------

    def save_video(self, video_path: str, episode_idx: int, success: bool) -> None:
        """
        Save episode replay as an MP4 video.

        The filename mirrors the episode ``.vla`` stem when ``episode_data_path``
        is set (e.g. ``task_00_pick..._episode_000_S.mp4``); otherwise it falls
        back to a ``task_description``-based name. A ``_S`` / ``_F`` tag is
        appended for success / failure to match the renamed ``.vla``/``.json``
        produced by :meth:`save_episode`.

        Args:
            video_path:  Directory where the video file will be written.
            episode_idx: Episode index, used only when no ``episode_data_path``.
            success:     Whether the episode succeeded — encoded as ``_S``/``_F``.
        """
        tag = "_S" if success else "_F"
        if self.episode_data_path is not None:
            stem = pathlib.Path(self.episode_data_path).stem
        else:
            task_segment = str(self.task_description).replace(" ", "_")
            stem = f"{task_segment}_episode_{episode_idx:03d}"
        # Ensure the success/failure tag is present even when save_episode()
        # was not called (and thus episode_data_path was not renamed).
        if not stem.endswith(tag):
            stem = f"{stem}{tag}"
        video_file = pathlib.Path(video_path) / f"{stem}.mp4"

        def _pad_even(frame: np.ndarray) -> np.ndarray:
            """Pad frame height/width to even numbers so libx264 is happy."""
            h, w = frame.shape[:2]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            return frame

        # Ensure all frames have the same size (alignment plot rendering can
        # produce slight size variations across timesteps).
        frames = [np.asarray(x) for x in self.replay_images]
        if frames:
            max_h = max(f.shape[0] for f in frames)
            max_w = max(f.shape[1] for f in frames)
            padded = []
            for f in frames:
                ph = max_h - f.shape[0]
                pw = max_w - f.shape[1]
                if ph or pw:
                    f = np.pad(f, ((0, ph), (0, pw), (0, 0)), mode="constant")
                padded.append(f)
            frames = padded

        imageio.mimwrite(
            video_file,
            [_pad_even(f) for f in frames],
            fps=VIDEO_FPS,
        )
        logger.info(f"Saved video: {video_file}")

        # Sidecar: per-episode perception segmentation video, when the
        # multi-view segmentation fn is an OpenPerceptionClient (or anything
        # exposing .save_segmentation_video). Stored alongside the rollout
        # video so it's discoverable as part of the same episode.
        seg_fn = getattr(self.abstractor, "_multi_view_segmentation_fn", None)
        save_perc_video = getattr(seg_fn, "save_segmentation_video", None)
        if callable(save_perc_video):
            perc_file = pathlib.Path(video_path) / f"perception_{stem}.mp4"
            try:
                out = save_perc_video(perc_file)
                if out is not None:
                    logger.info(f"Saved perception video: {out}")
            except Exception as exc:
                logger.warning(f"Failed to save perception video: {exc}")

    def save_episode(self, success: bool | None = None) -> None:
        """Delegate episode saving to the :class:`BaseRecorder`.

        When *success* is provided, rename the saved ``.vla`` and its ``.json``
        sidecar to append ``_S`` (success) or ``_F`` (failure). The runner's
        ``episode_data_path`` is updated in place so a subsequent
        :meth:`save_video` call picks up the same stem.
        """
        if self.recorder is not None:
            self.recorder.save()
        else:
            logger.debug("No recorder configured; skipping episode save.")

        if success is not None and self.episode_data_path is not None:
            tag = "_S" if success else "_F"
            old_vla = pathlib.Path(self.episode_data_path)
            new_vla = old_vla.with_name(f"{old_vla.stem}{tag}{old_vla.suffix}")
            for old in (old_vla, old_vla.with_suffix(".json")):
                if old.exists():
                    new = old.with_name(f"{old_vla.stem}{tag}{old.suffix}")
                    try:
                        old.rename(new)
                    except OSError as exc:
                        logger.warning(f"Failed to rename {old} -> {new}: {exc}")
            self.episode_data_path = str(new_vla)

        if self.visualizer is not None:
            self.visualizer.close()

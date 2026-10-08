"""Driving one experiment: rollouts on demand, against a policy loaded once.

:class:`ExperimentSession` sits between the GUI and
:class:`~policy_abstraction.evaluation.backends.EvalBackends`. It owns the
plan/store bookkeeping and runs each rollout on a worker thread so the operator's
window keeps painting — and keeps accepting the stop hotkey — while the arm is
moving.

Threading contract, because getting it wrong here strands a robot:

* Exactly one rollout thread exists at a time; :meth:`start_rollout` refuses if
  one is live.
* While a rollout is live the GUI thread only *sets* flags
  (:meth:`request_stop`) and *reads* snapshots (:meth:`live_frame`,
  :attr:`state`). The rollout thread alone drives the env.
* Between rollouts the GUI thread does poll the env, for the staging view
  (:meth:`refresh_preview`). One WebSocket answers in order, so a reply is
  identified only by its position in the stream: two threads mid-exchange at
  once and each gets the other's answer, permanently. The state check alone
  does not prevent that — the rollout thread can start just after it — so the
  preview also takes the client's ``rpc_lock``, and never waits for it.
* Every exit path runs :meth:`_finish`, which flushes the recorder, records the
  outcome, and homes the arm. A raised episode is still a recorded rollout.
"""

from __future__ import annotations

import dataclasses
import enum
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from loguru import logger

from policy_abstraction.data.recorder import RobodmRecorder
from policy_abstraction.evaluation.backends import EvalBackends
from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.outcomes import (
    ActualTargets,
    FailureMode,
    FailureModes,
    Outcome,
    RolloutRecord,
    StopReason,
    failure_modes_label,
    normalize_actual_targets,
    normalize_failure_modes,
)
from policy_abstraction.experiments.plan import PlannedRollout
from policy_abstraction.experiments.store import ExperimentStore

if TYPE_CHECKING:
    from policy_abstraction.experiments.reference import ReferenceFrames


def resolve_staging_camera(env_spec, camera_name: Optional[str] = None):
    """The camera the operator stages against.

    *camera_name* wins when given; otherwise the env's primary camera,
    which is the one the policy sees. ``None`` when the env published no
    spec at all.
    """
    if env_spec is None:
        return None
    if camera_name:
        return env_spec.camera_by_name(camera_name)
    return env_spec.primary_camera


def fetch_camera_params(env, camera, *, height: int, width: int) -> dict:
    """Intrinsics/extrinsics for *camera*, or ``{}`` if unavailable.

    Failure is not fatal: an uncalibrated cell still gets a live view,
    and the overlay says it cannot project. That is a much better failure
    than refusing to collect data.
    """
    if camera is None:
        return {}
    try:
        params = env.get_camera_params(
            camera_names=[camera.name], camera_height=height, camera_width=width
        )
    except Exception as exc:
        logger.warning(f"Could not fetch camera params: {exc}")
        return {}
    entry = (params or {}).get(camera.name)
    if not entry:
        logger.warning(
            f"Env published no intrinsics/extrinsics for camera "
            f"{camera.name!r}; the grid overlay will be unavailable. "
            "On the real cell this means --env.calibration-path is unset, "
            "or names no extrinsic for this camera."
        )
        return {}
    return {
        "intrinsic": np.asarray(entry["intrinsic"]),
        "extrinsic": np.asarray(entry["extrinsic"]),
    }


def frame_from_obs(obs: dict, camera) -> Optional[np.ndarray]:
    """The displayable RGB frame for *camera*, with its transform applied."""
    if camera is None:
        return None
    raw = obs.get(camera.image_key)
    if raw is None:
        return None
    return np.ascontiguousarray(camera.apply_image_transform(np.asarray(raw)))


class SessionState(enum.Enum):
    """What the session is doing, as far as the operator is concerned."""

    STAGING = "staging"
    """Idle, waiting for the operator to place the object and start."""
    RUNNING = "running"
    """A rollout is in flight."""
    STOPPING = "stopping"
    """A stop was requested; the rollout thread is winding down."""
    FINISHED = "finished"
    """Every planned rollout has a result."""


@dataclasses.dataclass
class LiveFrame:
    """A snapshot the GUI can render without touching the env."""

    image: Optional[np.ndarray] = None
    intrinsic: Optional[np.ndarray] = None
    extrinsic: Optional[np.ndarray] = None
    step: int = 0
    timestamp: float = 0.0
    # The wrist camera's frame from the same instant, when the env has one.
    # Shown beside the staging view, never drawn on: it has no grid to
    # project and moves with the arm.
    wrist: Optional[np.ndarray] = None


class ExperimentSession:
    """One operator, one policy setting, many rollouts."""

    def __init__(
        self,
        backends: EvalBackends,
        config: ExperimentConfig,
        store: ExperimentStore,
        *,
        task_id: int = 0,
        policy_setting: str = "",
        on_state_change: Optional[Callable[[], None]] = None,
        reference: Optional["ReferenceFrames"] = None,
    ) -> None:
        self.backends = backends
        self.config = config
        self.store = store
        self.task_id = task_id
        self.policy_setting = policy_setting
        self._on_state_change = on_state_change
        # The experiment this one was created from, when it was: its first
        # frames are ghosted under the staging view so the operator can
        # match the scene, not only the marks. See experiments/reference.py.
        self.reference = reference

        self.plan = store.load_plan()
        self.state = SessionState.STAGING
        self.current: Optional[PlannedRollout] = None
        self.last_record: Optional[RolloutRecord] = None
        self.last_error: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._stop_outcome: (
            tuple[Outcome, tuple[FailureMode, ...], StopReason] | None
        ) = None
        self._stop_actual_targets: tuple[str, ...] = ()
        self._stop_note = ""
        # The record the rollout thread is building, and whether it has
        # resolved its verdict yet. Guarded by `_verdict_lock`, because a late
        # classification from the GUI has to land in exactly one of three
        # places: the pending stop, this record before it is written, or a
        # superseding row once it is.
        self._verdict_lock = threading.Lock()
        self._inflight: RolloutRecord | None = None
        self._verdict_resolved = False
        # A rollout the budget ended, not the operator. It is recorded as a
        # TIMEOUT failure on its own, and offered here — exactly once — so the
        # GUI can ask what the arm was doing when the clock ran out.
        self._unclassified_timeout: RolloutRecord | None = None
        self._runner = None
        self._started_at = 0.0
        self._frame_lock = threading.Lock()
        self._frame = LiveFrame()

        self.camera = resolve_staging_camera(backends.env.env_spec, config.camera)
        self._camera_params: dict = {}
        # Streamed to the operator beside the staging view; None on an env
        # without one (a single-camera cell, or a simulator with no wrist
        # view), in which case the GUI shows no wrist panel at all.
        spec = backends.env.env_spec
        self.wrist_camera = spec.wrist_camera if spec is not None else None
        # "home" / "reset" / None until the first park tells us which.
        self._homes_with: Optional[str] = None
        # None until the first preview attempt says whether this env
        # can serve observations on demand.
        self._live_preview: Optional[bool] = None
        # False until a rollout has actually run the policy. The first
        # inference against a freshly loaded checkpoint compiles it, which on
        # hardware is minutes; the rollout that pays that is flagged so the
        # report does not average it in with the rest.
        self._warmed = False
        # How far the rollout in flight has got, for the window's step counter.
        # Updated from the rollout thread, read from the GUI thread; ints are
        # written atomically under the GIL, so neither takes the frame lock for
        # a number that is stale by a frame either way.
        self._live_step = 0
        self._step_limit = 0
        # When the arm actually started working, which is what `timeout_s`
        # is measured against. None until the first action is executed.
        self._task_started_at: Optional[float] = None
        # True only while `warm_up_policy` is in flight. Read from the GUI
        # thread so the window can say what the long silence is.
        self.warming_up = False

        self._advance_to_pending()

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def prepare_env(self) -> None:
        """Build the env once for the whole session and grab a preview frame.

        Separate from ``__init__`` because it talks to the robot: the caller
        decides when that is allowed to happen (after the GUI is up, so a slow
        homing move is visible rather than looking like a hang).
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
        self._save_reference_view()
        self.warm_up_policy()

    def warm_up_policy(self) -> Optional[float]:
        """Pay the policy's first-inference cost before the operator is waiting.

        Delegates to :meth:`EvalBackends.warm_up_policy`, which reads one
        observation and throws the actions away — the arm is never commanded.
        Doing it here, at the end of set-up, is what stops the first rollout
        from being a rollout in which nothing moves.

        The ``startup_phase`` binding is what the start-up window renders as
        its headline; the same line goes to the log for the headless path.
        """
        # The prompt the *next* rollout will send, not the raw template: a
        # disambiguation experiment's `task` still has its `{target_object}`
        # placeholder in it, and warming on a string no episode ever sends is
        # a warm-up one step further from the real thing than it needs to be.
        planned = self.current
        prompt = (
            self.config.task_for(planned.assignments)
            if planned is not None
            else self.config.task
        ) or self.backends.task_languages[self.task_id]
        self.warming_up = True
        self._notify()
        try:
            logger.bind(
                startup_phase="WARMING UP THE POLICY",
                startup_detail=(
                    "First inference compiles the checkpoint — this can take "
                    "several minutes. The robot is not commanded; the operator "
                    "window opens when it finishes."
                ),
            ).info(
                "Warming up the policy: one throw-away inference so the first "
                "rollout does not pay the compile. The arm is not commanded."
            )
            elapsed = self.backends.warm_up_policy(prompt)
        finally:
            self.warming_up = False
            self._notify()
        if elapsed is not None:
            # Marked here, not in the rollout: the compile is paid, and the
            # first rollout is now an ordinary one as far as the report is
            # concerned.
            self._warmed = True
            logger.info(f"Policy warm-up finished in {elapsed:.1f}s.")
        return elapsed

    def _save_reference_view(self) -> None:
        """Keep the first staging frame so the report can project onto it."""
        frame = self.live_frame()
        if frame.image is None:
            return
        from policy_abstraction.experiments.overlay import preprocessed_for

        self.store.save_reference_view(
            frame.image,
            intrinsic=frame.intrinsic,
            extrinsic=frame.extrinsic,
            camera_name=self.camera.name if self.camera else "",
            preprocessed=preprocessed_for(self.camera),
        )

    def _fetch_camera_params(self) -> dict:
        args = self.backends.args
        return fetch_camera_params(
            self.backends.env,
            self.camera,
            height=args.camera.height,
            width=args.camera.width,
        )

    @property
    def camera_params(self) -> dict:
        return self._camera_params

    @property
    def policy_warmed(self) -> bool:
        """Whether the policy's first-inference cost has already been paid."""
        return self._warmed

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def _publish_frame(self, obs: dict, step: int) -> None:
        image = frame_from_obs(obs, self.camera)
        if image is None:
            return
        with self._frame_lock:
            self._frame = LiveFrame(
                image=image,
                intrinsic=self._camera_params.get("intrinsic"),
                extrinsic=self._camera_params.get("extrinsic"),
                step=step,
                timestamp=time.time(),
                wrist=frame_from_obs(obs, self.wrist_camera),
            )

    def live_frame(self) -> LiveFrame:
        """The most recent frame. Safe to call from any thread."""
        with self._frame_lock:
            return self._frame

    def refresh_preview(self) -> bool:
        """Pull a fresh frame from the env. Only legal when no rollout is live.

        Returns False (rather than raising) when a rollout is running, so a GUI
        timer can call it unconditionally.

        Not every env serves observations on demand: the real cell does
        (``get_obs``), the simulator adapters do not. Because the env
        client is a generic RPC proxy, ``get_obs`` always *exists* and an
        unsupported one only says no on the wire — so an unconditional
        timer would spend a failed round-trip and a server-side
        traceback several times a second. It is probed once, and after
        that the last frame the rollout published simply stands.
        """
        if self.state in (SessionState.RUNNING, SessionState.STOPPING):
            return False
        if self._live_preview is False:
            return False

        # The state check above is not enough on its own: the rollout
        # thread can start between it and the call below, and both would
        # then be mid-flight on one connection. Take the client's lock so
        # the two cannot interleave -- but never wait for it, because this
        # runs on the GUI thread and the rollout thread may be holding it
        # for a multi-second park. A skipped tick just leaves the last
        # frame up for another 1/refresh_hz of a second.
        lock = getattr(self.backends.env, "rpc_lock", None)
        if lock is not None and not lock.acquire(blocking=False):
            return False
        try:
            obs = self.backends.env.get_obs()
        except Exception as exc:
            if self._live_preview is None:
                logger.info(
                    f"This env serves no on-demand observations ({exc}); "
                    "the staging view will hold the last rollout's frame "
                    "instead of streaming."
                )
            self._live_preview = False
            return False
        finally:
            if lock is not None:
                lock.release()
        self._live_preview = True
        self._publish_frame(obs, step=0)
        return True

    # ------------------------------------------------------------------
    # Plan navigation
    # ------------------------------------------------------------------

    def _advance_to_pending(self) -> None:
        self.current = self.store.next_pending()
        self.state = (
            SessionState.FINISHED if self.current is None else SessionState.STAGING
        )

    def skip_current(self) -> None:
        """Record the current condition as aborted and move on.

        Aborted rollouts do not count toward the plan, so the same condition is
        offered again later — this is "I mis-staged it", not "I am dropping this
        cell from the experiment".
        """
        if self.current is None or self.state is not SessionState.STAGING:
            return
        record = RolloutRecord(
            index=self.current.index,
            assignments=dict(self.current.assignments),
            outcome=Outcome.ABORTED,
            stop_reason=StopReason.OPERATOR,
            policy_setting=self.policy_setting,
            target_positions=self.config.target_positions(self.current.assignments),
            task=self.config.task_for(self.current.assignments),
            notes="skipped during staging",
            finished_at=time.time(),
        )
        self.store.append(record)
        self.last_record = record
        self._notify()

    def reconfigure(self, config: ExperimentConfig, store: ExperimentStore) -> None:
        """Point the session at an edited experiment, keeping the backends.

        The operator edits the grid from the GUI (see
        :mod:`policy_abstraction.experiments.revise`, which decides what the
        edit means for the results on disk and hands back the store to use).
        Rebinding here rather than restarting the process is the whole point:
        the policy server stays up, and a pi05 checkpoint costs about three
        minutes to load.

        Refused mid-rollout — the rollout thread owns the store it is about to
        append to, and swapping it underneath would file the result against the
        wrong plan.
        """
        if self.is_running:
            raise RuntimeError(
                "cannot change the experiment while a rollout is running"
            )

        # `self._warmed` is deliberately left alone: the point of editing the
        # experiment from the window rather than restarting is that the policy
        # server stays up, so the next rollout does not pay a warm-up again.
        camera_changed = config.camera != self.config.camera
        self.config = config
        self.store = store
        self.plan = store.load_plan()
        self.last_record = None
        self.last_error = None
        if camera_changed:
            self.camera = resolve_staging_camera(
                self.backends.env.env_spec, config.camera
            )
            self._camera_params = self._fetch_camera_params()

        # The new directory has no staging frame of its own yet, and the
        # report projects onto one. The scene in front of the camera is the
        # same scene it was a moment ago, so the current frame is the right
        # one to keep.
        self._save_reference_view()

        # Episodes and videos follow the store, so a rollout's .vla lands
        # beside the record that describes it.
        output = getattr(self.backends.args, "output", None)
        if output is not None:
            output.episode_data_path = str(store.episodes_dir)
            output.video_out_path = str(store.videos_dir)

        self._advance_to_pending()
        logger.info(
            f"Experiment is now {store.root} "
            f"({len(self.plan)} rollouts planned, {self.store.status()})"
        )
        self._notify()

    def progress(self) -> tuple[int, int]:
        """``(steps done, step budget)`` for the rollout in flight.

        The budget is 0 until the first rollout of the session has resolved it —
        it comes from the env's own task limit when the config does not set one,
        and that is only known once the backends are up.
        """
        return self._live_step, self._step_limit

    def reclassify(
        self,
        index: int,
        outcome: Outcome,
        failure_mode: FailureModes | None = None,
        *,
        note: str = "",
        actual_targets: ActualTargets | None = None,
    ) -> RolloutRecord:
        """Correct the verdict on a rollout that has already been recorded.

        Operators misclassify: a rollout called a failure turns out to have put
        the cable in the bag after all, or "grasp miss" was really "dropped".
        The correction is an *append*, not an edit — the JSONL stays the
        immutable log of what was pressed at the time, and
        :meth:`~policy_abstraction.experiments.store.ExperimentStore.results`
        takes the last scored row for each index, so the report follows without
        knowing anything about corrections.

        The new row copies the original, so the timing, the step count, the
        staged positions and the episode path all still describe the rollout
        that actually ran.

        Args:
            index: Plan index of the rollout to correct.
            outcome: The corrected verdict.
            failure_mode: One mode or several — a rollout that went for the
                wrong object and then dropped it failed both ways. Only
                meaningful with ``FAILURE``; forced to none otherwise, since a
                success cannot have failed a particular way.
            note: Appended to the record's notes, in the operator's words.
            actual_targets: Which named object(s) it actually went for — one
                id or several, in the order it reached for them. ``None``
                leaves whatever was recorded at the time alone — most
                corrections are about the verdict, not about what was reached
                — except that promoting a rollout to a success moves it onto
                the object the prompt named, which is what the success asserts.

        Returns:
            The superseding record, already on disk.

        Raises:
            RuntimeError: A rollout is in flight — the thread is about to append
                to this log, and the ordering of the two rows would decide the
                verdict.
            KeyError: No rollout has been recorded under *index*.
        """
        if self.is_running:
            raise RuntimeError("cannot reclassify while a rollout is running")

        original = self.record_for(index)
        if original is None:
            raise KeyError(f"no rollout has been recorded at index {index}")
        modes = (
            normalize_failure_modes(failure_mode) if outcome is Outcome.FAILURE else ()
        )

        target = self.config.target_factor()
        if actual_targets is None:
            targets = tuple(original.actual_targets)
            if outcome is Outcome.SUCCESS and target is not None:
                targets = normalize_actual_targets(
                    original.assignments.get(target.name, "")
                )
            elif outcome is Outcome.ABORTED:
                # A discarded rollout asserts nothing about what was reached.
                targets = ()
        else:
            targets = normalize_actual_targets(actual_targets)

        # Both columns of each pair are replaced: `replace` would otherwise
        # carry the original's list across, and the list is what the record
        # believes.
        corrected = dataclasses.replace(
            original,
            outcome=outcome,
            failure_mode=modes[0] if modes else FailureMode.NONE,
            failure_modes=list(modes),
            actual_target=targets[0] if targets else "",
            actual_targets=list(targets),
            notes=_append_note(
                original.notes,
                note
                or (
                    f"reclassified {original.outcome.value} -> {outcome.value}"
                    + (f" ({failure_modes_label(modes)})" if modes else "")
                ),
            ),
        )
        self.store.append(corrected)
        logger.info(
            f"Rollout {index + 1} reclassified: {_verdict(original)}"
            f" -> {_verdict(corrected)}"
        )
        # An aborted row promoted to a verdict completes that plan index, and a
        # verdict demoted to aborted re-opens it, so what is on deck can change.
        self.last_record = corrected
        self._advance_to_pending()
        self._notify()
        return corrected

    def record_for(self, index: int) -> Optional[RolloutRecord]:
        """The row that currently stands for *index*, correction included."""
        latest: Optional[RolloutRecord] = None
        for record in self.store.records():
            if record.index == index:
                latest = record
        return latest

    @property
    def status(self):
        return self.store.status()

    @property
    def is_running(self) -> bool:
        return self.state in (SessionState.RUNNING, SessionState.STOPPING)

    def instructions(self) -> list[str]:
        """Everything the operator must stage for the condition on deck.

        Usually one line per factor; a factor that stages several objects
        contributes one line each.
        """
        if self.current is None:
            return []
        assignments = dict(self.current.assignments)
        return [
            line
            for f in self.config.factors
            if f.name in assignments
            for line in f.staging_lines(assignments[f.name], assignments)
        ]

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    def start_rollout(self) -> bool:
        """Begin the pending rollout on a worker thread.

        Returns False when there is nothing to start or one is already running.
        """
        if self.state is not SessionState.STAGING or self.current is None:
            return False
        self._stop_requested.clear()
        self._stop_outcome = None
        self._stop_actual_targets = ()
        self._stop_note = ""
        self._verdict_resolved = False
        self.last_error = None
        self.state = SessionState.RUNNING
        self._started_at = time.time()
        self._task_started_at = None
        # Cleared here rather than at the end of the last rollout, so the
        # finished count stays on screen while the operator stages the next one.
        self._live_step = 0
        self._thread = threading.Thread(
            target=self._run_rollout,
            args=(self.current,),
            name=f"rollout-{self.current.index}",
            daemon=True,
        )
        self._thread.start()
        self._notify()
        return True

    def request_stop(
        self,
        outcome: Outcome,
        failure_mode: FailureModes | None = None,
        reason: StopReason = StopReason.OPERATOR,
        *,
        actual_targets: ActualTargets | None = None,
    ) -> bool:
        """Ask the running rollout to stop and be recorded as *outcome*.

        Returns immediately — the rollout ends at its next step boundary. The
        GUI reflects that via :attr:`state` going ``STOPPING`` then ``STAGING``.

        Args:
            failure_mode: How it failed — one mode or several. Ignored unless
                *outcome* is ``FAILURE``. The GUI stops with a placeholder and
                refines it through :meth:`classify_failure` once the operator
                has answered the dialog.
            actual_targets: Which named object(s) the policy actually went
                for, in a scene holding several — one id, or several in the
                order it reached for them. Only the operator watching can say,
                so it is asked with the failure mode and carried through to
                the record the rollout thread is about to write.
        """
        if self.state is not SessionState.RUNNING:
            return False
        self._stop_actual_targets = normalize_actual_targets(actual_targets)
        modes = (
            normalize_failure_modes(failure_mode) if outcome is Outcome.FAILURE else ()
        )
        with self._verdict_lock:
            self._stop_outcome = (outcome, modes, reason)
        self.state = SessionState.STOPPING
        self._stop_requested.set()
        self._notify()
        return True

    def pop_timeout_to_classify(self) -> Optional[RolloutRecord]:
        """The rollout the budget just ended, if it has not been offered yet.

        A timeout is recorded as a failure on its own — the operator is not
        made to press anything while the arm winds down — but *why* the task
        did not finish in time (never moved, missed the grasp, dropped it) is
        still theirs to say. This hands the record over exactly once, so the
        window asks once; the answer goes through :meth:`classify_failure`.
        """
        with self._verdict_lock:
            record, self._unclassified_timeout = self._unclassified_timeout, None
        return record

    def classify_failure(
        self,
        modes: FailureModes,
        *,
        note: str = "",
        actual_targets: ActualTargets | None = None,
    ) -> bool:
        """Say how the rollout stopped with :meth:`request_stop` actually failed.

        The stop is issued before the operator is asked, so the arm is already
        winding down while they decide — and by the time they answer, the
        rollout may be anywhere: still stepping, resolved but not yet on disk,
        or written and homing. The answer lands correctly in every case:

        * still stepping — it becomes the pending stop verdict;
        * resolved, not written — the in-flight record is amended in place;
        * written — a superseding row is appended, which is what every rate
          reads, so nothing double-counts.

        A rollout the budget ended (see :meth:`pop_timeout_to_classify`) is
        classified the same way; it keeps its derived ``TIMEOUT`` mode, so the
        report still counts it as a timeout, with the operator's modes added.

        Only a rollout that stopped as a failure is touched: a success, an
        abort, or one that raised keeps its verdict. Returns whether the
        classification was applied.
        """
        chosen = normalize_failure_modes(modes) or (FailureMode.OTHER,)
        targets = normalize_actual_targets(actual_targets)
        with self._verdict_lock:
            if not self._verdict_resolved:
                # Still stepping — or not even started: the thread may not
                # have built its record yet when the operator is this quick.
                pending = self._stop_outcome
                if pending is None or pending[0] is not Outcome.FAILURE:
                    return False
                self._stop_outcome = (Outcome.FAILURE, chosen, StopReason.OPERATOR)
                self._stop_actual_targets = targets
                self._stop_note = note
                return True
            record = self._inflight
            if record is not None:
                if not _classifiable(record):
                    return False
                record.set_failure_modes(_with_timeout(record, chosen))
                record.notes = note
                record.set_actual_targets(targets)
                return True
        record = self.last_record
        if record is None or not _classifiable(record):
            return False
        record.set_failure_modes(_with_timeout(record, chosen))
        record.notes = note
        record.set_actual_targets(targets)
        # The JSONL is append-only, so amending means appending a corrected
        # row that supersedes the old one. `ExperimentStore.results()` keeps
        # only the last scored row per index, which is what every rate is
        # computed from, so this cannot double-count.
        self.store.append(record)
        self._notify()
        return True

    def _task_clock(self) -> float:
        """When the timeout's clock started; latched on the first executed step.

        Before the arm has moved this is "now", so the elapsed time is zero and
        no amount of set-up can time the rollout out.
        """
        if self._task_started_at is not None:
            return self._task_started_at
        runner = self._runner
        if runner is not None and int(getattr(runner, "steps_executed", 0)) > 0:
            self._task_started_at = time.time()
            return self._task_started_at
        return time.time()

    def _should_stop(self) -> bool:
        """The predicate handed to ``EpisodeRunner``. Runs on the rollout thread.

        The timeout measures how long the *arm* has had, not how long the
        rollout has existed. Everything before the first executed action —
        the env reset, retrieval and demo analysis, and above all the first
        inference, which on a ``torch.compile``d checkpoint autotunes for
        several minutes — is set-up the operator is not budgeting for. Timing
        it from ``start_rollout`` meant the warm-up rollout spent its whole
        budget inside that first inference and was stopped on the very next
        iteration, having moved the arm exactly one step. The record still
        carries the wall-clock duration and its ``warmup`` flag, so nothing
        about that cost is hidden; it just no longer eats the task's time.
        """
        if self._stop_requested.is_set():
            return True
        timeout = self.config.timeout_s
        if timeout is not None and (time.time() - self._task_clock()) >= timeout:
            # A timeout is a failure with a known cause, so it is recorded
            # without waiting on the operator; what the arm was doing when
            # the clock ran out is asked afterwards (`pop_timeout_to_classify`).
            self._stop_outcome = (
                Outcome.FAILURE,
                (FailureMode.TIMEOUT,),
                StopReason.TIMEOUT,
            )
            self._stop_requested.set()
            logger.info(f"Rollout timed out after {timeout:.0f}s")
            return True
        return False

    def _run_rollout(self, planned: PlannedRollout) -> None:
        """Worker-thread body. Must never raise past this frame."""
        attempt = self.store.attempt_count(planned.index)
        episode_path = self.store.episode_path(planned, attempt=attempt)
        args = self.backends.args
        record = RolloutRecord(
            index=planned.index,
            assignments=dict(planned.assignments),
            outcome=Outcome.ABORTED,
            policy_setting=self.policy_setting,
            target_positions=self.config.target_positions(planned.assignments),
            # Filled from the template before anything can raise, so a rollout
            # that dies on the first step still records what it was asked to do.
            task=self.config.task_for(planned.assignments),
            started_at=self._started_at,
            warmup=not self._warmed,
        )
        with self._verdict_lock:
            self._inflight = record
        runner = None
        try:
            max_steps = (
                self.config.max_steps or self.backends.task_max_steps[self.task_id]
            )
            self._step_limit = int(max_steps)
            # Per-rollout, not per-experiment: a disambiguation experiment
            # varies exactly this and holds the scene fixed.
            task_description = (
                record.task
                or self.backends.env.language_instruction
                or self.backends.task_languages[self.task_id]
            )
            record.task = str(task_description)
            recorder = (
                RobodmRecorder() if self.backends.args.output.save_episodes else None
            )
            runner = self.backends.build_runner(
                task_description=task_description,
                initial_state=self._initial_state(),
                max_steps=max_steps,
                episode_data_path=(
                    str(episode_path) if args.output.save_episodes else None
                ),
                episode_idx=planned.index,
                recorder=recorder,
                stop_signal=self._should_stop,
            )
            self._runner = runner
            self._install_frame_tap(runner)

            done, inferences = runner.run()

            with self._verdict_lock:
                outcome, failure_modes, stop_reason = self._resolve_outcome(
                    runner, done
                )
                record.outcome = outcome
                record.set_failure_modes(failure_modes)
                record.stop_reason = stop_reason
                record.notes = self._stop_note
                record.set_actual_targets(self._resolve_actual_targets(record))
                record.timed_out = FailureMode.TIMEOUT in failure_modes
                if record.timed_out:
                    # Offered for classification the moment the verdict is
                    # in, while the episode is still being flushed: the
                    # answer lands on the in-flight record if it comes fast,
                    # or as a superseding row if it comes late.
                    self._unclassified_timeout = record
                self._verdict_resolved = True
            record.steps = int(runner.steps_executed)
            record.inferences = int(inferences)
            if inferences:
                # Marked after the fact, not before: a rollout that raised on
                # the way to the first inference has warmed nothing up, and the
                # next one would still pay for it.
                self._warmed = True
        except Exception as exc:
            # A raised episode is still data: it happened, the operator saw it,
            # and losing the row would silently bias the success rate upward.
            record.outcome = Outcome.ABORTED
            record.stop_reason = StopReason.ERROR
            record.error = f"{type(exc).__name__}: {exc}"
            self.last_error = record.error
            logger.error(f"Rollout {planned.index} raised: {exc}")
            logger.debug(traceback.format_exc())
        finally:
            self._finish(record, runner, episode_path)

    def _initial_state(self) -> Any:
        """The init state handed to the runner.

        Simulated envs index a saved-state array; the real cell publishes a
        placeholder per trial (there is no savable state — the operator *is*
        the initial-state generator). Both are addressed the same way, and an
        env that publishes none at all is tolerated.
        """
        try:
            states = self.backends.env.get_task_init_states(task_id=self.task_id)
        except Exception as exc:
            logger.debug(f"env published no init states ({exc}); using None")
            return None
        if states is None or not len(states):
            return None
        # Vary the sim's initial state across rollouts so a positional sweep in
        # simulation is not the same scene twelve times over. On hardware every
        # entry is the same "go home" placeholder, so this is a no-op there.
        idx = self.current.index % len(states) if self.current is not None else 0
        return states[idx]

    def _install_frame_tap(self, runner) -> None:
        """Publish each rendered frame to the GUI by wrapping the recorder hook.

        Wrapping ``_record_timestep`` rather than adding a callback parameter
        keeps the change to :class:`EpisodeRunner` to the single ``stop_signal``
        argument — the runner already computes exactly the images we want, once
        per step, and this borrows them without a second render.
        """
        original = runner._record_timestep

        def tapped(*args, **kwargs):
            # Before the image check: the counter must keep moving on a cell
            # whose staging camera serves no frames, which is exactly the case
            # where the operator has nothing else to tell a slow rollout from a
            # stalled one.
            self._live_step = int(kwargs.get("t", self._live_step))
            images = kwargs.get("images")
            if images and self.camera is not None:
                frame = images.get(self.camera.name)
                if frame is not None:
                    wrist = (
                        images.get(self.wrist_camera.name)
                        if self.wrist_camera is not None
                        else None
                    )
                    with self._frame_lock:
                        self._frame = LiveFrame(
                            image=np.ascontiguousarray(frame),
                            intrinsic=self._camera_params.get("intrinsic"),
                            extrinsic=self._camera_params.get("extrinsic"),
                            step=int(kwargs.get("t", 0)),
                            timestamp=time.time(),
                            wrist=(
                                np.ascontiguousarray(wrist)
                                if wrist is not None
                                else None
                            ),
                        )
            return original(*args, **kwargs)

        runner._record_timestep = tapped

    def _resolve_outcome(
        self, runner, done: bool
    ) -> tuple[Outcome, tuple[FailureMode, ...], StopReason]:
        """Turn how the loop ended into a verdict."""
        if self._stop_outcome is not None:
            return self._stop_outcome
        if done:
            # The env's own success check fired. Simulators have one; the real
            # UR5e never reports done unless --env.success-mode says so, in
            # which case it is the operator's verdict coming back the long way.
            return Outcome.SUCCESS, (), StopReason.ENV_DONE
        # Ran out of steps without the operator or the env calling it.
        return Outcome.FAILURE, (FailureMode.TIMEOUT,), StopReason.MAX_STEPS

    def _resolve_actual_targets(self, record: RolloutRecord) -> tuple[str, ...]:
        """Which named object(s) this rollout actually went for.

        A success went for the one the prompt named — that is what a success
        means here, and asking the operator to confirm it would be a keystroke
        that can only ever have one answer. A failure is whatever they said,
        and empty when they were not asked: an experiment with no named objects
        has nothing to confuse, and the report reads a blank as unrecorded
        rather than inventing a target.
        """
        if self._stop_actual_targets:
            return self._stop_actual_targets
        target = self.config.target_factor()
        if target is not None and record.outcome is Outcome.SUCCESS:
            return normalize_actual_targets(record.assignments.get(target.name, ""))
        return ()

    def _finish(self, record: RolloutRecord, runner, episode_path) -> None:
        """Flush, record, home. Runs on the rollout thread, always."""
        record.finished_at = time.time()
        record.duration_s = max(0.0, record.finished_at - record.started_at)

        if runner is not None:
            try:
                runner.save_episode(success=record.succeeded)
                # save_episode renames the file with an _S/_F tag; take the
                # name it settled on so the record points at a file that exists.
                record.episode_path = self._relative(runner.episode_data_path)
            except Exception as exc:
                logger.warning(f"Failed to save episode: {exc}")
            try:
                runner.save_video(
                    str(self.store.videos_dir), record.index, record.succeeded
                )
            except Exception as exc:
                logger.debug(f"No replay video written: {exc}")

        with self._verdict_lock:
            self.store.append(record)
            self.last_record = record
            self._inflight = None
            self._verdict_resolved = True
        self._runner = None

        try:
            self._home()
        except Exception as exc:
            logger.error(f"Could not return the robot home after the rollout: {exc}")
            self.last_error = f"homing failed: {exc}"

        logger.info(
            f"Rollout {record.index} -> {_verdict(record)}"
            f", {record.steps} steps, {record.duration_s:.1f}s"
            + (" (includes the policy server warm-up)" if record.warmup else "")
        )
        self._advance_to_pending()
        self.refresh_preview()
        self._notify()

    def _home(self) -> None:
        """Return the arm to its rest pose between rollouts.

        ``home`` is the real cell's explicit parking command; the
        simulators do not implement it and their ``reset`` is the
        equivalent. Which one this env supports is discovered once and
        remembered: the env client is a generic RPC proxy, so ``env.home``
        always *exists* and a simulator only says no on the wire — probing
        every rollout would mean a failed round-trip and a server-side
        traceback in the log each time.
        """
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

    def _relative(self, path) -> Optional[str]:
        if not path:
            return None
        try:
            import pathlib

            return str(pathlib.Path(path).relative_to(self.store.root))
        except ValueError:
            return str(path)

    def _notify(self) -> None:
        if self._on_state_change is not None:
            try:
                self._on_state_change()
            except Exception as exc:
                logger.debug(f"state-change callback raised: {exc}")

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop any live rollout, wait for it, park the arm, drop the backends.

        The wait matters: the rollout thread owns the env client, and tearing
        the servers down underneath it would abandon the arm mid-motion.
        """
        if self.is_running:
            logger.info("Stopping the in-flight rollout before shutting down …")
            self.request_stop(Outcome.ABORTED, None, StopReason.OPERATOR)
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
        self.backends.shutdown()


def _append_note(existing: str, addition: str) -> str:
    """Join notes without losing the one that was already there."""
    parts = [p for p in (existing.strip(), addition.strip()) if p]
    return "; ".join(parts)


def _classifiable(record: RolloutRecord) -> bool:
    """Whether :meth:`ExperimentSession.classify_failure` may amend *record*.

    Failures the operator called, and failures the budget called — never a
    success, an abort, or a rollout that raised.
    """
    return record.outcome is Outcome.FAILURE and record.stop_reason in (
        StopReason.OPERATOR,
        StopReason.TIMEOUT,
        StopReason.MAX_STEPS,
    )


def _with_timeout(
    record: RolloutRecord, chosen: tuple[FailureMode, ...]
) -> tuple[FailureMode, ...]:
    """The operator's modes, keeping ``TIMEOUT`` on a rollout the budget ended."""
    if record.timed_out and FailureMode.TIMEOUT not in chosen:
        return chosen + (FailureMode.TIMEOUT,)
    return chosen


def _verdict(record: RolloutRecord) -> str:
    """``failure/grasp_miss+dropped`` for a log line; just the outcome otherwise."""
    if not record.failure_modes:
        return record.outcome.value
    return record.outcome.value + "/" + "+".join(m.value for m in record.failure_modes)


__all__ = [
    "ExperimentSession",
    "LiveFrame",
    "SessionState",
    "fetch_camera_params",
    "frame_from_obs",
    "resolve_staging_camera",
]

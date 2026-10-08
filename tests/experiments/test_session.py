"""Session orchestration: threading, stop semantics, and recording.

These are the behaviours that decide whether an operator can trust the
stop button, so they are tested against fakes rather than left to be
discovered at the workcell.
"""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.factors import GridPositionFactor
from policy_abstraction.experiments.grid import TableGrid
from policy_abstraction.experiments.outcomes import (
    FailureMode,
    FailureModes,
    Outcome,
    RolloutRecord,
    StopReason,
)
from policy_abstraction.experiments.session import (
    ExperimentSession,
    SessionState,
    fetch_camera_params,
    frame_from_obs,
    resolve_staging_camera,
)
from policy_abstraction.experiments.store import ExperimentStore
from tests.experiments.conftest import FakeBackends, FakeEnv, FakeEnvWithoutHome


@pytest.fixture
def config():
    return ExperimentConfig(
        name="t",
        task="Put the cable in the box.",
        factors=[
            GridPositionFactor(
                "cable_position",
                TableGrid(
                    origin=(0, 0, 0),
                    u_axis=(0, 1, 0),
                    v_axis=(1, 0, 0),
                    rows=1,
                    cols=3,
                    name="cable",
                ),
            )
        ],
        repeats=1,
        max_steps=50,
    )


@pytest.fixture
def session_factory(tmp_path, config):
    made = []

    def build(backends=None, cfg=None, **kwargs):
        cfg = cfg or config
        backends = backends or FakeBackends()
        store = ExperimentStore.open(tmp_path / cfg.name, cfg)
        session = ExperimentSession(
            backends, cfg, store, policy_setting="test", **kwargs
        )
        made.append(session)
        return session

    yield build
    for s in made:
        s.shutdown(timeout=5)


def _wait_idle(session, timeout=10.0):
    deadline = time.time() + timeout
    while session.is_running and time.time() < deadline:
        time.sleep(0.01)
    assert not session.is_running, "rollout did not finish"


def _wait_running(session, min_steps=1, timeout=10.0):
    """Wait until the rollout has actually taken *min_steps* steps.

    An operator watches before deciding, so a stop lands mid-rollout.
    Stopping the instant the thread is spawned is a different (also
    valid) case, and it is the one the timeout tests cover.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        runner = session._runner
        if runner is not None and runner.steps_executed >= min_steps:
            return
        time.sleep(0.005)
    raise AssertionError("rollout never started stepping")


class TestLifecycle:
    def test_starts_at_the_first_pending_condition(self, session_factory):
        session = session_factory()
        assert session.state is SessionState.STAGING
        assert session.current.index == 0
        assert session.instructions() == ["Place the cable_position in cell A1"]

    def test_prepare_env_makes_the_env_once_and_saves_a_reference(
        self, session_factory
    ):
        backends = FakeBackends()
        session = session_factory(backends)
        session.prepare_env()
        assert len(backends.env.made) == 1
        assert session.store.reference_frame_path.exists()
        assert session.camera_params  # projection is available

    def test_prepare_env_warms_the_policy_without_commanding_the_arm(
        self, session_factory
    ):
        """The compile is paid at start-up, not in the first rollout.

        On the UR5e cell the server reported ready after 106 s and then spent
        449 s inside its first inference; left to rollout 0 that is a rollout
        in which nothing moves.
        """
        backends = FakeBackends()
        session = session_factory(backends)
        session.prepare_env()
        assert backends.warm_ups == ["Put the cable in the box."]
        assert session.policy_warmed
        assert not session.warming_up
        # A warm-up that stepped the env would have moved the robot.
        assert backends.env.steps == 0

    def test_the_window_can_say_the_policy_is_warming_up(self, session_factory):
        """`warming_up` is what the banner and the start-up headline read."""
        backends = FakeBackends()
        backends.release_warm_up = threading.Event()
        session = session_factory(backends)
        worker = threading.Thread(target=session.prepare_env, daemon=True)
        worker.start()
        assert backends.warming.wait(timeout=5)
        assert session.warming_up, "nothing for the operator to read during the wait"
        backends.release_warm_up.set()
        worker.join(timeout=5)
        assert not session.warming_up

    def test_a_failed_warm_up_is_not_fatal(self, session_factory):
        """Its only cost is the first rollout's old slow start."""
        backends = FakeBackends()
        backends.warm_up_seconds = None  # what EvalBackends returns on failure
        session = session_factory(backends)
        session.prepare_env()
        assert not session.policy_warmed
        assert not session.warming_up
        assert session.live_frame().image is not None

    def test_uncalibrated_camera_is_survivable(self, session_factory):
        env = FakeEnv()
        env.camera_params = {}
        session = session_factory(FakeBackends(env))
        session.prepare_env()
        assert session.camera_params == {}
        assert session.live_frame().image is not None

    def test_the_wrist_frame_rides_along_when_the_env_has_one(self, session_factory):
        """The staging view and the wrist view are one snapshot."""
        session = session_factory(FakeBackends(FakeEnv(wrist=True)))
        session.prepare_env()
        frame = session.live_frame()
        assert session.wrist_camera is not None
        assert frame.image is not None
        assert frame.wrist is not None
        assert frame.wrist.shape == (24, 32, 3)

    def test_no_wrist_camera_means_no_wrist_frame(self, session_factory):
        session = session_factory(FakeBackends(FakeEnv()))
        session.prepare_env()
        assert session.wrist_camera is None
        assert session.live_frame().image is not None
        assert session.live_frame().wrist is None

    def test_a_rollout_streams_the_wrist_frame_too(self, session_factory):
        """Mid-rollout the frames come from the runner's per-step images."""
        import numpy as np

        session = session_factory(FakeBackends(FakeEnv(wrist=True)))
        session.prepare_env()

        class Runner:
            def _record_timestep(self, **kwargs):
                return None

        runner = Runner()
        session._install_frame_tap(runner)
        runner._record_timestep(
            t=3,
            images={
                "agentview": np.zeros((32, 32, 3), np.uint8),
                "wrist": np.full((24, 32, 3), 7, np.uint8),
            },
        )
        frame = session.live_frame()
        assert frame.step == 3
        assert frame.wrist is not None and int(frame.wrist[0, 0, 0]) == 7

    def test_finished_when_the_plan_is_exhausted(self, session_factory):
        session = session_factory()
        for _ in range(3):
            assert session.start_rollout()
            _wait_idle(session)
        assert session.state is SessionState.FINISHED
        assert session.current is None
        assert not session.start_rollout()


class TestStopSemantics:
    def test_success_hotkey_records_a_success_and_advances(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 400}))
        session.start_rollout()
        _wait_running(session)
        assert session.request_stop(Outcome.SUCCESS, FailureMode.NONE)
        assert session.state is SessionState.STOPPING
        _wait_idle(session)
        record = session.last_record
        assert record.outcome is Outcome.SUCCESS
        assert record.stop_reason is StopReason.OPERATOR
        assert record.steps > 0, "steps taken before the stop must be recorded"
        assert session.current.index == 1

    def test_catastrophic_stop_is_a_failure_with_its_own_mode(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 400}))
        session.start_rollout()
        _wait_running(session)
        session.request_stop(Outcome.FAILURE, FailureMode.CATASTROPHIC)
        _wait_idle(session)
        assert session.last_record.outcome is Outcome.FAILURE
        assert session.last_record.failure_mode is FailureMode.CATASTROPHIC

    def test_stop_is_refused_when_nothing_is_running(self, session_factory):
        session = session_factory()
        assert not session.request_stop(Outcome.SUCCESS)

    def test_cannot_start_two_rollouts_at_once(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 400}))
        assert session.start_rollout()
        assert not session.start_rollout()
        session.request_stop(Outcome.ABORTED)
        _wait_idle(session)

    def test_env_done_is_a_success(self, session_factory):
        session = session_factory(
            FakeBackends(runner_kwargs={"steps": 3, "done": True})
        )
        session.start_rollout()
        _wait_idle(session)
        assert session.last_record.outcome is Outcome.SUCCESS
        assert session.last_record.stop_reason is StopReason.ENV_DONE

    def test_running_out_of_steps_is_a_timeout_failure(self, session_factory):
        session = session_factory(
            FakeBackends(runner_kwargs={"steps": 3, "done": False})
        )
        session.start_rollout()
        _wait_idle(session)
        assert session.last_record.outcome is Outcome.FAILURE
        assert session.last_record.failure_mode is FailureMode.TIMEOUT
        assert session.last_record.stop_reason is StopReason.MAX_STEPS

    def test_the_timeout_does_not_count_the_policy_warm_up(
        self, session_factory, config
    ):
        """The first inference compiles the checkpoint; that is not task time.

        Timed from ``start_rollout``, a warm-up longer than ``timeout_s`` spent
        the entire budget before the arm moved, and the rollout was stopped on
        the first poll after it — nothing on the robot ever moved.
        """
        config.timeout_s = 0.05
        session = session_factory(
            FakeBackends(
                runner_kwargs={
                    "steps": 3,
                    "tick": 0.0,
                    "warmup": 0.2,  # four times the whole budget
                    "done": True,
                }
            ),
            cfg=config,
        )
        session.start_rollout()
        _wait_idle(session)
        record = session.last_record
        assert record.steps == 3, "the rollout was cut short by the warm-up"
        assert record.outcome is Outcome.SUCCESS
        assert record.stop_reason is StopReason.ENV_DONE
        assert not record.timed_out
        # The wall-clock duration still tells the truth about what it cost.
        assert record.duration_s >= 0.2
        assert record.warmup

    def test_wall_clock_timeout_stops_without_asking(self, session_factory, config):
        config.timeout_s = 0.05
        session = session_factory(
            FakeBackends(runner_kwargs={"steps": 10_000}), cfg=config
        )
        session.start_rollout()
        _wait_idle(session)
        record = session.last_record
        assert record.outcome is Outcome.FAILURE
        assert record.failure_mode is FailureMode.TIMEOUT
        assert record.stop_reason is StopReason.TIMEOUT
        assert record.timed_out

    def test_a_timeout_is_offered_for_classification_once(
        self, session_factory, config
    ):
        config.timeout_s = 0.05
        session = session_factory(
            FakeBackends(runner_kwargs={"steps": 10_000}), cfg=config
        )
        session.start_rollout()
        _wait_idle(session)
        offered = session.pop_timeout_to_classify()
        assert offered is not None and offered.index == session.last_record.index
        assert session.pop_timeout_to_classify() is None, "asked once"

    def test_an_operator_stop_is_not_offered_as_a_timeout(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 400}))
        session.start_rollout()
        session.request_stop(Outcome.FAILURE, FailureMode.DROPPED)
        _wait_idle(session)
        assert session.pop_timeout_to_classify() is None

    def test_classifying_a_timeout_keeps_the_derived_mode(
        self, session_factory, config
    ):
        """The report still counts it as a timeout; now it also knows why."""
        config.timeout_s = 0.05
        session = session_factory(
            FakeBackends(runner_kwargs={"steps": 10_000}), cfg=config
        )
        session.start_rollout()
        _wait_idle(session)
        assert session.pop_timeout_to_classify() is not None
        assert session.classify_failure(FailureMode.NO_MOTION, note="never moved")
        record = session.record_for(0)
        assert record.outcome is Outcome.FAILURE
        assert record.failure_modes == [FailureMode.NO_MOTION, FailureMode.TIMEOUT]
        assert record.timed_out
        assert record.stop_reason is StopReason.TIMEOUT
        assert record.notes == "never moved"

    def test_running_out_of_steps_is_offered_and_classifiable_too(
        self, session_factory
    ):
        session = session_factory(
            FakeBackends(runner_kwargs={"steps": 3, "done": False})
        )
        session.start_rollout()
        _wait_idle(session)
        assert session.pop_timeout_to_classify() is not None
        assert session.classify_failure(FailureMode.GRASP_MISS)
        record = session.record_for(0)
        assert record.failure_modes == [FailureMode.GRASP_MISS, FailureMode.TIMEOUT]
        assert record.stop_reason is StopReason.MAX_STEPS


class TestRecording:
    def test_the_condition_and_its_metric_target_are_recorded(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 2}))
        session.start_rollout()
        _wait_idle(session)
        record = session.last_record
        assert record.assignments == {"cable_position": "A1"}
        assert len(record.target_positions["cable_position"]) == 3
        assert record.policy_setting == "test"
        assert record.duration_s > 0

    def test_a_raised_episode_is_still_recorded(self, session_factory):
        session = session_factory(
            FakeBackends(runner_kwargs={"raises": RuntimeError("arm fault")})
        )
        session.start_rollout()
        _wait_idle(session)
        record = session.last_record
        assert record.outcome is Outcome.ABORTED
        assert record.stop_reason is StopReason.ERROR
        assert "arm fault" in record.error
        # Aborted, so the condition is offered again rather than lost.
        assert session.current.index == 0

    def test_skip_logs_an_abort_and_re_offers_the_condition(self, session_factory):
        session = session_factory()
        session.skip_current()
        assert session.last_record.outcome is Outcome.ABORTED
        assert session.current.index == 0
        assert session.status.aborted == 1

    def test_skip_is_ignored_mid_rollout(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 400}))
        session.start_rollout()
        session.skip_current()
        assert session.status.aborted == 0
        session.request_stop(Outcome.ABORTED)
        _wait_idle(session)


class TestWarmUp:
    """The first inference compiles the checkpoint; the record says so."""

    def _run_one(self, session, outcome=Outcome.SUCCESS):
        session.start_rollout()
        _wait_running(session)
        session.request_stop(outcome, FailureMode.NONE)
        _wait_idle(session)
        return session.last_record

    def test_only_the_first_rollout_is_flagged(self, session_factory):
        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        session = session_factory(backends)
        assert self._run_one(session).warmup
        assert not self._run_one(session).warmup

    def test_a_rollout_that_never_inferred_does_not_count_as_the_warm_up(
        self, session_factory
    ):
        """It raised before the policy ran, so the next one still pays."""
        backends = FakeBackends(
            runner_kwargs={"raises": RuntimeError("policy server not up yet")}
        )
        session = session_factory(backends)
        session.start_rollout()
        _wait_idle(session)
        assert session.last_record is not None and session.last_record.warmup

        session.backends.runner_kwargs = {"steps": 10_000}
        assert self._run_one(session).warmup

    def test_the_flag_survives_reconfiguring(self, session_factory, config, tmp_path):
        """Editing the experiment does not restart the server, so nothing rewarms."""
        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        session = session_factory(backends)
        self._run_one(session)
        session.reconfigure(config, ExperimentStore.open(tmp_path / "again", config))
        assert not self._run_one(session).warmup


class TestParking:
    def test_the_arm_is_parked_after_every_rollout(self, session_factory):
        env = FakeEnv()
        session = session_factory(FakeBackends(env, runner_kwargs={"steps": 2}))
        session.start_rollout()
        _wait_idle(session)
        assert env.homed >= 1

    def test_parking_falls_back_to_reset_when_home_is_unsupported(
        self, session_factory
    ):
        """Simulators have no `home`; the session must not give up."""
        env = FakeEnvWithoutHome()
        session = session_factory(FakeBackends(env, runner_kwargs={"steps": 2}))
        session.start_rollout()
        _wait_idle(session)
        assert env.resets >= 1

    def test_unsupported_home_is_probed_once_not_every_rollout(self, session_factory):
        """The env client proxies any name, so probing costs a round-trip."""
        env = FakeEnvWithoutHome()
        calls = {"n": 0}
        original = env.home

        def counting_home():
            calls["n"] += 1
            return original()

        env.home = counting_home
        session = session_factory(FakeBackends(env, runner_kwargs={"steps": 1}))
        for _ in range(3):
            session.start_rollout()
            _wait_idle(session)
        assert calls["n"] == 1
        assert env.resets >= 3

    def test_a_failure_to_park_is_surfaced_not_swallowed(self, session_factory):
        env = FakeEnv()

        def broken_home():
            raise RuntimeError("e-stop engaged")

        env.home = broken_home
        env.reset = broken_home
        session = session_factory(FakeBackends(env, runner_kwargs={"steps": 1}))
        session.start_rollout()
        _wait_idle(session)
        assert "homing failed" in (session.last_error or "")


class TestReconfigure:
    """Swapping the experiment under a live session, as the settings editor does."""

    def _reshaped(self, config, cols):
        data = config.to_dict()
        data["factors"][0]["grid"]["cols"] = cols
        data["factors"][0]["cells"] = None
        return ExperimentConfig.from_dict(data)

    def test_the_new_plan_replaces_the_old(self, session_factory, config, tmp_path):
        session = session_factory()
        wider = self._reshaped(config, 5)
        store = ExperimentStore.open(tmp_path / "wider", wider)
        session.reconfigure(wider, store)
        assert len(session.plan) == 5
        assert session.store is store
        assert session.config is wider
        assert session.state is SessionState.STAGING

    def test_it_resumes_the_new_experiment_where_it_stopped(
        self, session_factory, config, tmp_path
    ):
        session = session_factory()
        store = ExperimentStore.open(tmp_path / "part_done", config)
        store.append(
            RolloutRecord(
                index=0,
                assignments={"cable_position": "A1"},
                outcome=Outcome.SUCCESS,
            )
        )
        session.reconfigure(config, store)
        assert session.current is not None
        assert session.current.index == 1

    def test_a_finished_plan_leaves_the_session_finished(
        self, session_factory, config, tmp_path
    ):
        session = session_factory()
        store = ExperimentStore.open(tmp_path / "all_done", config)
        for i in range(len(config.build_plan())):
            store.append(
                RolloutRecord(index=i, assignments={}, outcome=Outcome.SUCCESS)
            )
        session.reconfigure(config, store)
        assert session.state is SessionState.FINISHED

    def test_episodes_follow_the_new_directory(self, session_factory, config, tmp_path):
        session = session_factory()
        store = ExperimentStore.open(tmp_path / "moved", config)
        session.reconfigure(config, store)
        assert session.backends.args.output.episode_data_path == str(store.episodes_dir)

    def test_the_reference_frame_carries_over(self, session_factory, config, tmp_path):
        """The scene has not changed, so the new store gets the same view."""
        session = session_factory()
        session.prepare_env()
        store = ExperimentStore.open(tmp_path / "with_ref", config)
        session.reconfigure(config, store)
        assert store.reference_frame_path.exists()

    def test_it_is_refused_mid_rollout(self, session_factory, config, tmp_path):
        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        session = session_factory(backends)
        store = ExperimentStore.open(tmp_path / "nope", config)
        session.start_rollout()
        _wait_running(session)
        with pytest.raises(RuntimeError, match="while a rollout is running"):
            session.reconfigure(config, store)
        assert session.store is not store
        session.request_stop(Outcome.ABORTED)
        _wait_idle(session)


class TestPreviewSharesTheEnvSafely:
    """The GUI's preview timer and the rollout thread share one env client.

    Both are RPCs on a connection that answers in order, so the preview
    has to take the client's lock — but on the Qt thread, and never
    waiting for it: the rollout thread holds it across a park that can
    run for seconds.
    """

    def test_the_preview_takes_the_clients_lock(self, session_factory):
        import threading

        env = FakeEnv()
        env.rpc_lock = threading.RLock()
        held = []
        original = env.get_obs
        env.get_obs = lambda: (held.append(env.rpc_lock._is_owned()), original())[1]

        session = session_factory(FakeBackends(env))
        assert session.refresh_preview()
        assert held == [True]

    def test_the_preview_skips_its_turn_rather_than_waiting(self, session_factory):
        import threading

        env = FakeEnv()
        env.rpc_lock = threading.RLock()
        env.get_obs = lambda: pytest.fail("polled while the lock was held")

        session = session_factory(FakeBackends(env))
        acquired = threading.Event()
        release = threading.Event()

        def hold():
            with env.rpc_lock:
                acquired.set()
                release.wait(timeout=5)

        blocker = threading.Thread(target=hold, daemon=True)
        blocker.start()
        assert acquired.wait(timeout=5)
        try:
            assert session.refresh_preview() is False
        finally:
            release.set()
            blocker.join(timeout=5)

    def test_an_env_client_without_a_lock_still_previews(self, session_factory):
        """Test doubles and older clients have no ``rpc_lock``."""
        env = FakeEnv()
        assert not hasattr(env, "rpc_lock")
        session = session_factory(FakeBackends(env))
        assert session.refresh_preview()


class TestShutdown:
    def test_shutdown_stops_a_live_rollout_and_releases_the_backends(
        self, tmp_path, config
    ):
        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        store = ExperimentStore.open(tmp_path / "shutdown", config)
        session = ExperimentSession(backends, config, store)
        session.start_rollout()
        backends.building.wait(timeout=5)
        session.shutdown(timeout=10)
        assert not session.is_running
        assert backends.shutdown_calls == 1
        # The interrupted rollout is on disk, as aborted.
        assert store.records()[-1].outcome is Outcome.ABORTED

    def test_shutdown_is_safe_with_nothing_running(self, tmp_path, config):
        backends = FakeBackends()
        store = ExperimentStore.open(tmp_path / "quiet", config)
        session = ExperimentSession(backends, config, store)
        session.shutdown(timeout=5)
        assert backends.shutdown_calls == 1


class TestResume:
    def test_a_second_session_continues_where_the_first_stopped(self, tmp_path, config):
        store = ExperimentStore.open(tmp_path / "resume", config)
        first = ExperimentSession(
            FakeBackends(runner_kwargs={"steps": 1, "done": True}),
            config,
            store,
        )
        first.start_rollout()
        _wait_idle(first)
        first.shutdown(timeout=5)

        reopened = ExperimentStore.open(tmp_path / "resume", config)
        second = ExperimentSession(FakeBackends(), config, reopened)
        try:
            assert second.current.index == 1
            assert second.status.completed == 1
        finally:
            second.shutdown(timeout=5)


class TestStagingHelpers:
    """The camera helpers ``--calibrate`` uses without a session.

    Calibration runs *before* an experiment directory exists — there
    is no plan and no store yet — so these have to work standalone.
    They did not, and ``--calibrate`` died on a missing plan.json.
    """

    def test_primary_camera_is_the_default(self):
        spec = FakeEnv().env_spec
        assert resolve_staging_camera(spec, None).name == "agentview"
        assert resolve_staging_camera(spec, "agentview").name == "agentview"

    def test_no_env_spec_is_not_fatal(self):
        assert resolve_staging_camera(None, None) is None

    def test_params_and_frame_without_a_session_or_store(self):
        env = FakeEnv()
        camera = resolve_staging_camera(env.env_spec, None)
        params = fetch_camera_params(env, camera, height=32, width=32)
        assert set(params) == {"intrinsic", "extrinsic"}
        image = frame_from_obs(env.get_obs(), camera)
        assert image is not None and image.ndim == 3

    def test_uncalibrated_camera_yields_empty_params(self):
        env = FakeEnv()
        env.camera_params = {}
        camera = resolve_staging_camera(env.env_spec, None)
        assert fetch_camera_params(env, camera, height=32, width=32) == {}

    def test_a_raising_env_yields_empty_params(self):
        env = FakeEnv()

        def boom(**_kwargs):
            raise RuntimeError("no such command")

        env.get_camera_params = boom
        camera = resolve_staging_camera(env.env_spec, None)
        assert fetch_camera_params(env, camera, height=32, width=32) == {}

    def test_missing_image_key_is_none_not_a_crash(self):
        camera = resolve_staging_camera(FakeEnv().env_spec, None)
        assert frame_from_obs({}, camera) is None
        assert frame_from_obs({"agentview_image": None}, camera) is None


class TestProgress:
    """The live step count, which is what tells slow from stalled."""

    def test_the_budget_is_the_configs_max_steps(self, session_factory):
        session = session_factory()
        session.start_rollout()
        _wait_running(session, min_steps=2)
        step, limit = session.progress()
        assert limit == 50, "config.max_steps is the budget when it is set"
        assert step >= 1
        session.request_stop(Outcome.SUCCESS)
        _wait_idle(session)

    def test_the_env_budget_is_used_when_the_config_sets_none(
        self, session_factory, config
    ):
        cfg = dataclasses.replace(config, max_steps=None)
        session = session_factory(cfg=cfg)
        session.start_rollout()
        _wait_running(session, min_steps=2)
        assert session.progress()[1] == 500, "falls back to the task's own limit"
        session.request_stop(Outcome.SUCCESS)
        _wait_idle(session)

    def test_it_climbs_with_the_rollout_and_resets_on_the_next(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 10_000}))
        session.start_rollout()
        _wait_running(session, min_steps=3)
        assert session.progress()[0] >= 3
        session.request_stop(Outcome.SUCCESS)
        _wait_idle(session)
        # Still readable while the operator stages the next scene …
        assert session.progress()[0] >= 3
        session.start_rollout()
        _wait_running(session, min_steps=1)
        # … and starts again from the bottom rather than jumping.
        assert session.progress()[0] < 3 or session.progress()[0] >= 1
        session.request_stop(Outcome.SUCCESS)
        _wait_idle(session)

    def test_there_is_no_budget_before_the_first_rollout(self, session_factory):
        assert session_factory().progress() == (0, 0)


class TestClassifyFailure:
    """The F key stops first and asks second; the answer must land wherever
    the rollout has got to by the time it arrives."""

    def _stop_as_other(self, session):
        session.start_rollout()
        _wait_running(session)
        assert session.request_stop(Outcome.FAILURE, FailureMode.OTHER)

    def test_answered_while_still_stepping(self, session_factory, monkeypatch):
        session = session_factory()
        self._stop_as_other(session)
        # Hold the thread before it resolves, as a slow env step would.
        gate = threading.Event()
        original = session._resolve_outcome

        def slow_resolve(runner, done):
            gate.wait(5)
            return original(runner, done)

        monkeypatch.setattr(session, "_resolve_outcome", slow_resolve)
        assert session.classify_failure(
            (FailureMode.DROPPED, FailureMode.WRONG_TARGET),
            note="let go over the wrong bag",
            actual_targets="black_cable",
        )
        gate.set()
        _wait_idle(session)
        record = session.record_for(0)
        assert record.failure_modes == [FailureMode.DROPPED, FailureMode.WRONG_TARGET]
        assert record.notes == "let go over the wrong bag"
        assert record.actual_target == "black_cable"
        assert len(session.store.records()) == 1, "landed in the stop, no correction"

    def test_answered_before_the_thread_has_built_its_record(self, session_factory):
        """Stop and classify the instant the rollout starts."""
        session = session_factory()
        session.start_rollout()
        assert session.request_stop(Outcome.FAILURE, FailureMode.OTHER)
        assert session.classify_failure(FailureMode.GRASP_MISS, note="quick")
        _wait_idle(session)
        record = session.record_for(0)
        assert record.failure_modes == [FailureMode.GRASP_MISS]
        assert record.notes == "quick"
        assert len(session.store.records()) == 1

    def test_answered_after_the_row_is_on_disk(self, session_factory):
        """Rollout over and the arm homing: the answer supersedes the row."""
        session = session_factory()
        self._stop_as_other(session)
        _wait_idle(session)
        assert session.record_for(0).failure_modes == [FailureMode.OTHER]
        assert session.classify_failure([FailureMode.COLLISION], note="hit the rail")
        record = session.record_for(0)
        assert record.failure_modes == [FailureMode.COLLISION]
        assert record.notes == "hit the rail"
        assert len(session.store.records()) == 2, "appended, never edited"
        assert len(session.store.results()) == 1

    def test_answered_between_resolving_and_writing(self, session_factory, monkeypatch):
        session = session_factory()
        self._stop_as_other(session)
        resolved = threading.Event()
        gate = threading.Event()
        original = session._finish

        def slow_finish(record, runner, episode_path):
            resolved.set()
            gate.wait(5)
            original(record, runner, episode_path)

        monkeypatch.setattr(session, "_finish", slow_finish)
        assert resolved.wait(5)
        assert session.classify_failure(FailureMode.NO_MOTION)
        gate.set()
        _wait_idle(session)
        assert session.record_for(0).failure_modes == [FailureMode.NO_MOTION]
        assert len(session.store.records()) == 1

    def test_a_success_is_left_alone(self, session_factory):
        session = session_factory()
        session.start_rollout()
        _wait_running(session)
        session.request_stop(Outcome.SUCCESS)
        _wait_idle(session)
        assert not session.classify_failure(FailureMode.DROPPED)
        assert session.record_for(0).outcome is Outcome.SUCCESS
        assert len(session.store.records()) == 1

    def test_nothing_ticked_reads_as_other(self, session_factory):
        session = session_factory()
        self._stop_as_other(session)
        _wait_idle(session)
        assert session.classify_failure([], note="no idea")
        record = session.record_for(0)
        assert record.failure_modes == [FailureMode.OTHER]
        assert record.notes == "no idea"


class TestReclassify:
    """Correcting a verdict after the fact, which operators need often."""

    def _collect(
        self,
        session,
        outcome=Outcome.FAILURE,
        mode: FailureModes = FailureMode.GRASP_MISS,
    ):
        session.start_rollout()
        _wait_running(session)
        session.request_stop(outcome, mode)
        _wait_idle(session)
        return session.last_record

    def test_a_failure_can_become_a_success(self, session_factory):
        session = session_factory()
        self._collect(session)
        assert session.status.successes == 0

        corrected = session.reclassify(0, Outcome.SUCCESS)
        assert corrected.outcome is Outcome.SUCCESS
        assert corrected.failure_mode is FailureMode.NONE
        assert session.status.successes == 1
        assert session.status.failures == 0

    def test_the_original_row_is_kept(self, session_factory):
        session = session_factory()
        self._collect(session)
        session.reclassify(0, Outcome.SUCCESS)
        rows = session.store.records()
        assert len(rows) == 2, "the correction is appended, never an edit"
        assert rows[0].outcome is Outcome.FAILURE
        assert rows[1].outcome is Outcome.SUCCESS
        # And the rate follows the last one, so the rollout is not double-counted.
        assert len(session.store.results()) == 1

    def test_everything_but_the_verdict_survives_the_correction(self, session_factory):
        session = session_factory()
        original = self._collect(session)
        corrected = session.reclassify(0, Outcome.SUCCESS)
        for field in ("index", "steps", "duration_s", "started_at", "task", "warmup"):
            assert getattr(corrected, field) == getattr(original, field), field
        assert corrected.assignments == original.assignments
        assert corrected.episode_path == original.episode_path

    def test_only_a_failure_carries_a_failure_mode(self, session_factory):
        session = session_factory()
        self._collect(session)
        corrected = session.reclassify(0, Outcome.SUCCESS, FailureMode.DROPPED)
        assert corrected.failure_mode is FailureMode.NONE

    def test_the_mode_alone_can_be_corrected(self, session_factory):
        session = session_factory()
        self._collect(session, Outcome.FAILURE, FailureMode.GRASP_MISS)
        corrected = session.reclassify(0, Outcome.FAILURE, FailureMode.DROPPED)
        assert corrected.outcome is Outcome.FAILURE
        assert corrected.failure_mode is FailureMode.DROPPED
        assert corrected.failure_modes == [FailureMode.DROPPED]
        assert session.status.failures == 1

    def test_a_failure_can_carry_several_modes(self, session_factory):
        """Went for the wrong object and dropped it: both are kept, and both
        survive the trip through the JSONL."""
        session = session_factory()
        record = self._collect(
            session, Outcome.FAILURE, (FailureMode.DROPPED, FailureMode.WRONG_TARGET)
        )
        assert record.failure_modes == [FailureMode.DROPPED, FailureMode.WRONG_TARGET]
        assert record.failure_mode is FailureMode.DROPPED
        reloaded = session.store.results()[0]
        assert reloaded.failure_modes == record.failure_modes
        assert reloaded.failure_mode is FailureMode.DROPPED
        assert session.status.failures == 1, "one rollout, however many ways it failed"

    def test_a_correction_replaces_every_mode(self, session_factory):
        """Re-scoring does not merge with what was ticked before."""
        session = session_factory()
        self._collect(
            session, Outcome.FAILURE, (FailureMode.GRASP_MISS, FailureMode.COLLISION)
        )
        corrected = session.reclassify(0, Outcome.FAILURE, [FailureMode.NO_MOTION])
        assert corrected.failure_modes == [FailureMode.NO_MOTION]
        assert "No Motion" in corrected.notes
        promoted = session.reclassify(0, Outcome.SUCCESS, FailureMode.DROPPED)
        assert promoted.failure_modes == []
        assert promoted.failure_mode is FailureMode.NONE

    def test_what_happened_is_written_into_the_notes(self, session_factory):
        session = session_factory()
        self._collect(session)
        assert "failure -> success" in session.reclassify(0, Outcome.SUCCESS).notes
        assert (
            "moved the bag"
            in session.reclassify(
                0, Outcome.FAILURE, FailureMode.OTHER, note="moved the bag"
            ).notes
        )

    def test_discarding_a_result_offers_its_condition_again(self, session_factory):
        session = session_factory()
        self._collect(session, Outcome.SUCCESS, FailureMode.NONE)
        assert session.current.index == 1, "the plan moved on"
        session.reclassify(0, Outcome.ABORTED)
        assert session.status.completed == 0
        assert session.current.index == 0, "rollout 0 is on deck again"

    def test_promoting_an_aborted_row_completes_that_index(self, session_factory):
        session = session_factory()
        session.skip_current()
        assert session.status.completed == 0
        session.reclassify(0, Outcome.FAILURE, FailureMode.NO_MOTION)
        assert session.status.completed == 1
        assert session.current.index == 1

    def test_correcting_a_correction_reads_as_the_latest(self, session_factory):
        session = session_factory()
        self._collect(session)
        session.reclassify(0, Outcome.SUCCESS)
        session.reclassify(0, Outcome.FAILURE, FailureMode.DROPPED)
        assert session.status.successes == 0
        assert session.record_for(0).failure_mode is FailureMode.DROPPED
        assert len(session.store.records()) == 3

    def test_an_unrecorded_rollout_cannot_be_corrected(self, session_factory):
        with pytest.raises(KeyError):
            session_factory().reclassify(2, Outcome.SUCCESS)

    def test_it_is_refused_while_a_rollout_is_running(self, session_factory):
        session = session_factory(FakeBackends(runner_kwargs={"steps": 10_000}))
        self._collect(session)
        session.start_rollout()
        _wait_running(session)
        with pytest.raises(RuntimeError, match="running"):
            session.reclassify(0, Outcome.SUCCESS)
        session.request_stop(Outcome.SUCCESS)
        _wait_idle(session)

    def test_the_window_is_told(self, session_factory):
        session = session_factory()
        self._collect(session)
        seen = []
        session._on_state_change = lambda: seen.append(1)
        session.reclassify(0, Outcome.SUCCESS)
        assert seen, "the GUI must repaint: the success rate just moved"


class TestActedOnObject:
    """Which object the policy went for — the confusion matrix's raw material."""

    @pytest.fixture
    def objects_config(self):
        from policy_abstraction.experiments.factors import (
            ObjectLayoutFactor,
            TargetObjectFactor,
        )

        return ExperimentConfig(
            name="objects",
            task="Put the {target_object} in the bag.",
            factors=[
                ObjectLayoutFactor(
                    "object_layout",
                    TableGrid(
                        origin=(0.25, -0.2, 0.0),
                        u_axis=(0, 1, 0),
                        v_axis=(1, 0, 0),
                        rows=1,
                        cols=1,
                        cell_size_u=0.4,
                        cell_size_v=0.3,
                        name="objects",
                    ),
                    ["red_mug", "black_cable"],
                    layouts=1,
                ),
                TargetObjectFactor("target_object", layout="object_layout"),
            ],
            repeats=1,
            max_steps=50,
        )

    def _run(self, session, outcome, mode=FailureMode.NONE, acted_on=""):
        session.start_rollout()
        _wait_running(session)
        session.request_stop(outcome, mode, actual_targets=acted_on)
        _wait_idle(session)
        return session.last_record

    def test_a_success_went_for_the_object_it_was_asked_for(
        self, session_factory, objects_config
    ):
        """So the operator is never asked a question with one possible answer."""
        session = session_factory(cfg=objects_config)
        record = self._run(session, Outcome.SUCCESS)
        assert record.actual_target == record.assignments["target_object"]

    def test_a_failure_records_what_the_operator_saw(
        self, session_factory, objects_config
    ):
        session = session_factory(cfg=objects_config)
        record = self._run(
            session, Outcome.FAILURE, FailureMode.WRONG_TARGET, acted_on="black_cable"
        )
        assert record.actual_target == "black_cable"

    def test_never_committing_is_recorded_as_such(
        self, session_factory, objects_config
    ):
        from policy_abstraction.experiments.outcomes import NO_OBJECT

        session = session_factory(cfg=objects_config)
        record = self._run(
            session, Outcome.FAILURE, FailureMode.NO_MOTION, acted_on=NO_OBJECT
        )
        assert record.actual_target == NO_OBJECT, "distinct from 'nobody was asked'"

    def test_an_experiment_with_no_objects_records_nothing(self, session_factory):
        """A positional experiment stages one thing; there is nothing to confuse."""
        session = session_factory()
        assert self._run(session, Outcome.SUCCESS).actual_target == ""

    def test_the_answer_does_not_leak_into_the_next_rollout(
        self, session_factory, objects_config
    ):
        session = session_factory(cfg=objects_config)
        self._run(
            session, Outcome.FAILURE, FailureMode.WRONG_TARGET, acted_on="black_cable"
        )
        record = self._run(session, Outcome.FAILURE, FailureMode.GRASP_MISS)
        assert record.actual_target == "", "stale answers would fake a confusion"

    def test_re_scoring_to_a_success_moves_it_onto_the_named_object(
        self, session_factory, objects_config
    ):
        session = session_factory(cfg=objects_config)
        self._run(
            session, Outcome.FAILURE, FailureMode.WRONG_TARGET, acted_on="black_cable"
        )
        corrected = session.reclassify(0, Outcome.SUCCESS)
        assert corrected.actual_target == corrected.assignments["target_object"]

    def test_re_scoring_can_correct_the_object_too(
        self, session_factory, objects_config
    ):
        session = session_factory(cfg=objects_config)
        self._run(
            session, Outcome.FAILURE, FailureMode.WRONG_TARGET, acted_on="black_cable"
        )
        corrected = session.reclassify(
            0, Outcome.FAILURE, FailureMode.WRONG_TARGET, actual_targets="red_mug"
        )
        assert corrected.actual_target == "red_mug"

    def test_a_verdict_only_correction_leaves_the_object_alone(
        self, session_factory, objects_config
    ):
        session = session_factory(cfg=objects_config)
        self._run(
            session, Outcome.FAILURE, FailureMode.WRONG_TARGET, acted_on="black_cable"
        )
        corrected = session.reclassify(0, Outcome.FAILURE, FailureMode.DROPPED)
        assert corrected.actual_target == "black_cable"

    def test_discarding_asserts_nothing_about_what_was_reached(
        self, session_factory, objects_config
    ):
        session = session_factory(cfg=objects_config)
        self._run(
            session, Outcome.FAILURE, FailureMode.WRONG_TARGET, acted_on="black_cable"
        )
        discarded = session.reclassify(0, Outcome.ABORTED)
        assert discarded.actual_target == ""
        assert discarded.actual_targets == []

    def test_going_for_several_objects_records_each_in_order(
        self, session_factory, objects_config
    ):
        """A policy that hesitates between two wrong objects made two mistakes."""
        session = session_factory(cfg=objects_config)
        record = self._run(
            session,
            Outcome.FAILURE,
            FailureMode.WRONG_TARGET,
            acted_on=("black_cable", "red_mug"),
        )
        assert record.actual_targets == ["black_cable", "red_mug"]
        assert record.actual_target == "black_cable", "the single column is the first"
        reloaded = session.store.results()[0]
        assert reloaded.actual_targets == ["black_cable", "red_mug"]

    def test_a_late_classification_carries_several_objects(
        self, session_factory, objects_config
    ):
        session = session_factory(cfg=objects_config)
        session.start_rollout()
        _wait_running(session)
        session.request_stop(Outcome.FAILURE, FailureMode.OTHER)
        session.classify_failure(
            FailureMode.WRONG_TARGET, actual_targets=["red_mug", "black_cable"]
        )
        _wait_idle(session)
        assert session.last_record.actual_targets == ["red_mug", "black_cable"]

    def test_re_scoring_can_name_several_objects(self, session_factory, objects_config):
        session = session_factory(cfg=objects_config)
        self._run(
            session, Outcome.FAILURE, FailureMode.WRONG_TARGET, acted_on="black_cable"
        )
        corrected = session.reclassify(
            0,
            Outcome.FAILURE,
            FailureMode.WRONG_TARGET,
            actual_targets=("black_cable", "red_mug"),
        )
        assert corrected.actual_targets == ["black_cable", "red_mug"]
        # And a later verdict-only correction keeps the whole list.
        again = session.reclassify(0, Outcome.FAILURE, FailureMode.DROPPED)
        assert again.actual_targets == ["black_cable", "red_mug"]

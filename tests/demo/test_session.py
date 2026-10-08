"""The demo session, without a robot, a policy or a display.

What the window's safety rests on is here: Start runs on a thread, Stop lands
and parks the arm, the guidance switch is what the runner polls, and the
snapshots the observer publishes are complete and consistent.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from policy_abstraction.demo.session import (
    DemoSession,
    DemoState,
    OverlayMode,
    ReferenceTrends,
)
from policy_abstraction.vlm.types import DistancePair
from tests.experiments.conftest import FakeBackends, FakeEnv


def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


@pytest.fixture
def session():
    backends = FakeBackends(FakeEnv(wrist=True), runner_kwargs={"steps": 10_000})
    s = DemoSession(backends, guided_warmup=False)
    s.prepare_env()
    yield s
    s.shutdown()


class TestLifecycle:
    def test_prepare_env_warms_up_and_previews(self, session):
        assert session.backends.warm_ups == [session.task]
        view = session.live_view()
        assert view.image is not None and view.image.shape == (32, 32, 3)
        assert view.wrist is not None
        assert session.state is DemoState.IDLE

    def test_start_then_stop_homes_and_returns_idle(self, session):
        assert session.start_rollout()
        assert not session.start_rollout(), "one rollout at a time"
        _wait(lambda: session.backends.last_runner is not None)
        assert session.state is DemoState.RUNNING
        assert session.request_stop()
        _wait(lambda: session.state is DemoState.IDLE)
        assert session.backends.env.homed == 1
        assert session.rollouts == 1
        assert session.backends.last_runner.stopped_early

    def test_the_runner_gets_the_gate_and_the_observer(self, session):
        session.start_rollout()
        _wait(lambda: session.backends.built)
        kwargs = session.backends.built[-1]
        assert kwargs["guidance_gate"]() is False
        session.guided = True
        assert kwargs["guidance_gate"]() is True
        assert kwargs["on_step"] == session._on_step
        assert callable(kwargs["stop_signal"])
        session.request_stop()
        _wait(lambda: session.state is DemoState.IDLE)

    def test_shutdown_stops_a_live_rollout(self):
        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        s = DemoSession(backends, guided_warmup=False)
        s.prepare_env()
        s.start_rollout()
        _wait(lambda: backends.last_runner is not None)
        s.shutdown()
        assert backends.shutdown_calls == 1
        assert backends.last_runner.stopped_early
        assert s.state is DemoState.IDLE

    def test_a_runner_that_raises_is_reported_and_the_arm_still_homes(self):
        backends = FakeBackends(
            runner_kwargs={"raises": RuntimeError("policy server gone")}
        )
        s = DemoSession(backends, guided_warmup=False)
        s.prepare_env()
        s.start_rollout()
        _wait(lambda: s.state is DemoState.IDLE)
        assert "policy server gone" in (s.last_error or "")
        assert backends.env.homed == 1


class TestSwitches:
    def test_guidance_toggle_notifies(self, session):
        calls = []
        session._on_state_change = lambda: calls.append(1)
        assert session.toggle_guidance() is True
        assert session.guided
        session.guided = True  # no change, no notification
        assert len(calls) == 1
        session.toggle_guidance()
        assert not session.guided
        assert len(calls) == 2

    def test_overlay_mode_cycles(self, session):
        assert session.overlay_mode is OverlayMode.AUTO
        assert session.cycle_overlay_mode() is OverlayMode.ON
        assert session.cycle_overlay_mode() is OverlayMode.OFF
        assert session.cycle_overlay_mode() is OverlayMode.AUTO
        assert OverlayMode.AUTO.shows(True) and not OverlayMode.AUTO.shows(False)
        assert OverlayMode.ON.shows(False) and not OverlayMode.OFF.shows(True)

    def test_guidance_availability_reads_the_runner(self, session):
        assert not session.guidance_available
        session._runner = type("R", (), {"guidance_available": True})()
        assert session.guidance_available

    def test_viewer_layout_without_a_visualizer(self, session):
        assert session.viewer_layout is None
        assert session.set_viewer_layout("scene") is False

    def test_viewer_layout_with_a_visualizer(self, session):
        class Viz:
            layout = "full"
            first_time = False

            def set_layout(self, layout):
                self.layout = layout

            def close(self):
                pass

        session.visualizer = Viz()
        assert session.set_viewer_layout("scene")
        assert session.viewer_layout == "scene"


class _Abstractor:
    """The slice of RolloutAbstractor the session reads."""

    def __init__(self, pairs, labels):
        self.ref_traj = np.linspace(0.3, 0.05, 40, dtype=np.float32)[:, None].repeat(
            len(labels), axis=1
        )
        self.ref_std = None
        self.pair_labels = list(labels)
        self._translated_pairs = pairs
        self._subtrajectories = [(0, 20), (20, 40)]
        self._latest_agentview_predicted_seg = np.zeros((32, 32), dtype=np.int32)
        self._latest_agentview_predicted_seg[8:16, 8:16] = 3
        self._int_name_map = {3: "cup"}


class _Result:
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.pair_labels = list(labels)
        self.live_distances = {labels[0]: 0.21}
        self.ref_distances = {labels[0]: 0.2}
        self.distance_errors = {labels[0]: 0.01}
        self.entity_centroids = {"ee": np.array([0.4, 0.0, 0.3]), 3: np.zeros(3)}
        self.aligned_t = 7
        self.aligned_ts = np.array([7], dtype=np.int32)
        self.current_subtraj_idx = 0
        self.guidance_weights = np.ones(1, dtype=np.float32)
        self.relevances = {"cup": 0.9}
        self.perception_overlays = [np.full((32, 32, 3), 7, dtype=np.uint8)]
        self.mta_progress = 0.2


class TestObserver:
    def test_a_step_with_an_analysis_publishes_everything(self, session):
        pairs = [DistancePair("ee", 3, "ee_to_cup", subtraj_idx=0)]
        labels = ["ee_to_cup"]
        session._runner = type("R", (), {"abstractor": _Abstractor(pairs, labels)})()
        result = _Result(pairs, labels)
        frame = np.full((32, 32, 3), 50, dtype=np.uint8)
        session._on_step(
            images={"agentview": frame, "wrist": frame},
            t=12,
            abstraction_result=result,
            frame_bundle=None,
            guidance_info={"guidance_factor": 0.0, "gated": True},
            replan=True,
            guided=False,
        )
        view = session.live_view()
        assert view.step == 12 and view.replan and not view.guided
        assert view.analysed and view.abstraction is result
        assert view.names == {3: "cup"}
        assert view.seg is not None and (view.seg == 3).any()
        assert np.allclose(view.ee_pos, [0.4, 0.0, 0.3])
        assert view.perception_overlay is not None
        assert view.colors["ee_to_cup"]
        assert view.guidance_info["gated"]

        trends = session.trends
        assert isinstance(trends, ReferenceTrends)
        assert trends.pair_labels == ("ee_to_cup",)
        assert trends.subtrajectories == ((0, 20), (20, 40))
        assert trends.colors == view.colors
        history = session.trend_history()
        assert len(history) == 1
        sample = history[0]
        assert sample.step == 12 and sample.replan and not sample.guided
        assert sample.live[0] == pytest.approx(0.21)
        assert sample.aligned_t == 7

        # A later step without a frame keeps the last one.
        session._on_step(
            images=None, t=13, abstraction_result=result, replan=False, guided=True
        )
        view = session.live_view()
        assert view.image is not None and view.step == 13 and view.guided
        assert session.trend_count == 2
        assert session.trend_history(since=1)[0].step == 13

    def test_a_step_without_analysis_streams_the_frame_only(self, session):
        frame = np.full((32, 32, 3), 9, dtype=np.uint8)
        session._on_step(images={"agentview": frame}, t=3, replan=True, guided=False)
        view = session.live_view()
        assert view.step == 3 and not view.analysed
        assert view.image[0, 0, 0] == 9
        assert session.trends is None and session.trend_count == 0

    def test_a_new_rollout_clears_the_history(self, session):
        pairs = [DistancePair("ee", 3, "ee_to_cup", subtraj_idx=0)]
        session._runner = type(
            "R", (), {"abstractor": _Abstractor(pairs, ["ee_to_cup"])}
        )()
        session._on_step(
            images=None,
            t=1,
            abstraction_result=_Result(pairs, ["ee_to_cup"]),
            replan=True,
            guided=True,
        )
        assert session.trend_count == 1
        session.start_rollout()
        assert session.trend_count == 0 and session.trends is None
        assert not session.live_view().analysed
        session.request_stop()
        _wait(lambda: session.state is DemoState.IDLE)


class TestHeatmapAndSaving:
    def test_heatmap_toggle_is_dead_without_positions(self, session):
        assert session.training_heatmap is None and not session.show_heatmap
        assert session.toggle_heatmap() is False

    def test_heatmap_toggle_with_positions(self):
        backends = FakeBackends()
        marker = object()
        s = DemoSession(backends, guided_warmup=False, training_heatmap=marker)
        assert s.training_heatmap is marker and not s.show_heatmap
        assert s.toggle_heatmap() is True
        assert s.toggle_heatmap() is False
        s2 = DemoSession(
            backends, guided_warmup=False, training_heatmap=marker, show_heatmap=True
        )
        assert s2.show_heatmap

    def test_the_arm_homes_before_the_episode_is_written(self):
        """Encoding the episode can take a minute; the operator who pressed
        Stop is watching the arm, so it parks first and the save follows."""
        order: list[str] = []
        env = FakeEnv()
        original_home = env.home

        def home():
            order.append("home")
            original_home()

        env.home = home  # type: ignore[method-assign]
        backends = FakeBackends(env, runner_kwargs={"steps": 10_000})
        s = DemoSession(backends, guided_warmup=False)
        s.prepare_env()
        s.start_rollout()
        _wait(lambda: backends.last_runner is not None)
        runner = backends.last_runner
        original_save = runner.save_episode

        def save(success=None):
            order.append("save")
            assert s.state is DemoState.IDLE, "IDLE before the save starts"
            assert s.saving
            original_save(success)

        runner.save_episode = save  # type: ignore[method-assign]
        s.request_stop()
        _wait(lambda: s.state is DemoState.IDLE and not s.saving and len(order) == 2)
        assert order == ["home", "save"]
        s.shutdown()


class TestPause:
    def test_pause_only_while_running(self, session):
        assert session.toggle_pause() is False and not session.paused
        session.start_rollout()
        _wait(lambda: session.backends.built)
        assert session.backends.built[-1]["pause_signal"]() is False
        assert session.toggle_pause() is True and session.paused
        assert session.backends.built[-1]["pause_signal"]() is True
        assert session.toggle_pause() is False
        session.toggle_pause()
        session.request_stop()  # a stop clears the pause so the loop can end
        assert not session.paused
        _wait(lambda: session.state is DemoState.IDLE)

    def test_a_paused_frame_keeps_the_last_analysis(self, session):
        pairs = [DistancePair("ee", 3, "ee_to_cup", subtraj_idx=0)]
        session._runner = type(
            "R", (), {"abstractor": _Abstractor(pairs, ["ee_to_cup"])}
        )()
        result = _Result(pairs, ["ee_to_cup"])
        session._on_step(
            images={"agentview": np.zeros((32, 32, 3), np.uint8)},
            t=5,
            abstraction_result=result,
            replan=True,
            guided=True,
        )
        session._on_step(
            images={"agentview": np.full((32, 32, 3), 7, np.uint8)},
            t=5,
            replan=False,
            guided=True,
            paused=True,
        )
        view = session.live_view()
        assert view.image[0, 0, 0] == 7, "the picture moved"
        assert view.abstraction is result and view.guided, "the analysis stayed"
        assert session.trend_count == 1, "a held frame is not a sample"


class TestNoRecording:
    def test_without_recording_the_runner_still_gets_a_recorder(self):
        """Retrieval and the demo analysis only run with a recorder, so an
        unrecorded demo hands the runner one that keeps nothing."""
        from policy_abstraction.demo.session import NullRecorder

        backends = FakeBackends(runner_kwargs={"steps": 5})
        backends.args.output.save_episodes = False
        s = DemoSession(backends, guided_warmup=False)
        s.prepare_env()
        s.start_rollout()
        _wait(lambda: s.state is DemoState.IDLE and not s.saving)
        kwargs = backends.built[-1]
        assert isinstance(kwargs["recorder"], NullRecorder)
        assert kwargs["episode_data_path"] is None
        assert kwargs["render_overlays"] is True
        assert s.episode_paths == []
        rec = NullRecorder()
        rec.initialize("x", {"a": 1})
        rec.record(camera_image_streams={})
        rec.update_metadata("k", 2)
        rec.save()
        assert rec.metadata == {"a": 1, "k": 2}
        s.shutdown()

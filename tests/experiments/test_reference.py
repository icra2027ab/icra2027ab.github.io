"""The reference experiment's frames, ghosted under the staging view."""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.factors import GridPositionFactor
from policy_abstraction.experiments.grid import TableGrid
from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord
from policy_abstraction.experiments.plan import PlannedRollout
from policy_abstraction.experiments.reference import (
    GHOST_LEVELS,
    ReferenceFrames,
    ghost,
)
from policy_abstraction.experiments.store import ExperimentStore


@pytest.fixture
def config():
    return ExperimentConfig(
        name="ref_test",
        task="Put the cable in the box.",
        factors=[
            GridPositionFactor(
                "cable_position",
                TableGrid(
                    origin=(0, 0, 0),
                    u_axis=(0, 1, 0),
                    v_axis=(1, 0, 0),
                    rows=1,
                    cols=2,
                    name="cable",
                ),
            )
        ],
        repeats=2,
    )


@pytest.fixture
def reference(tmp_path, config):
    """A reference experiment with rollouts 0 and 1 scored, 2 and 3 not."""
    store = ExperimentStore.open(tmp_path / "ref", config)
    plan = store.load_plan()
    for planned in plan[:2]:
        store.append(
            RolloutRecord(
                index=planned.index,
                assignments=dict(planned.assignments),
                outcome=Outcome.SUCCESS,
                episode_path=f"episodes/rollout_{planned.index}.vla",
            )
        )
    return store


def _loader_returning(color):
    calls = []

    def load(path):
        calls.append(path)
        return np.full((8, 12, 3), color, dtype=np.uint8), 0

    load.calls = calls  # type: ignore[attr-defined]
    return load


class TestMatching:
    def test_the_same_condition_gets_the_reference_frame(self, reference):
        frames = ReferenceFrames(reference, loader=_loader_returning(200))
        planned = reference.load_plan()[0]
        frame = frames.frame_for(planned)
        assert frame is not None
        assert frame.index == 0
        assert frame.assignments == planned.assignments
        assert frame.image.shape == (8, 12, 3)

    def test_a_condition_the_reference_never_scored_has_no_frame(self, reference):
        frames = ReferenceFrames(reference, loader=_loader_returning(200))
        # Index 2 is the second cell's first repeat; the reference scored
        # only the first cell's two repeats.
        assert frames.frame_for(reference.load_plan()[2]) is None
        assert frames.frame_for(None) is None

    def test_an_edited_plan_matches_on_assignments(self, reference):
        """A plan renumbered from the GUI still finds the scene it staged."""
        frames = ReferenceFrames(reference, loader=_loader_returning(200))
        planned = reference.load_plan()[0]
        moved = PlannedRollout(index=7, assignments=planned.assignments, repeat=0)
        frame = frames.frame_for(moved)
        assert frame is not None and frame.index == 0

    def test_frames_are_read_once(self, reference):
        loader = _loader_returning(200)
        frames = ReferenceFrames(reference, loader=loader)
        planned = reference.load_plan()[0]
        frames.frame_for(planned)
        frames.frame_for(planned)
        assert len(loader.calls) == 1

    def test_an_unreadable_recording_is_a_missing_frame_not_a_crash(self, reference):
        def broken(path):
            raise OSError("no such recording")

        frames = ReferenceFrames(reference, loader=broken)
        assert frames.frame_for(reference.load_plan()[0]) is None


class TestGhost:
    def test_levels_run_off_to_reference_only(self):
        assert GHOST_LEVELS[0] == 0.0 and GHOST_LEVELS[-1] == 1.0

    def test_zero_is_the_live_frame(self):
        live = np.zeros((4, 4, 3), np.uint8)
        ref = np.full((4, 4, 3), 255, np.uint8)
        assert ghost(live, ref, 0.0) is live

    def test_one_is_the_reference(self):
        live = np.zeros((4, 4, 3), np.uint8)
        ref = np.full((4, 4, 3), 255, np.uint8)
        assert (ghost(live, ref, 1.0) == 255).all()

    def test_half_blends(self):
        live = np.zeros((4, 4, 3), np.uint8)
        ref = np.full((4, 4, 3), 200, np.uint8)
        out = ghost(live, ref, 0.5)
        assert out.shape == live.shape and abs(int(out[0, 0, 0]) - 100) <= 1

    def test_a_reference_of_another_size_is_fitted_to_the_live_frame(self):
        live = np.zeros((6, 8, 3), np.uint8)
        ref = np.full((12, 16, 3), 200, np.uint8)
        assert ghost(live, ref, 0.5).shape == (6, 8, 3)


class TestWindow:
    """The window offers the ghost only when there is a reference to show."""

    @pytest.fixture
    def qapp(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    def _window(self, qapp, tmp_path, config, reference_store):
        from policy_abstraction.experiments.gui.main_window import ExperimentWindow
        from policy_abstraction.experiments.session import ExperimentSession
        from tests.experiments.conftest import FakeBackends

        frames = (
            None
            if reference_store is None
            else ReferenceFrames(reference_store, loader=_loader_returning(255))
        )
        store = ExperimentStore.open(tmp_path / "arm", config)
        session = ExperimentSession(
            FakeBackends(runner_kwargs={"steps": 10_000}),
            config,
            store,
            policy_setting="t",
            reference=frames,
        )
        session.prepare_env()
        return ExperimentWindow(session, config), session

    def test_no_reference_no_ghost_action(self, qapp, tmp_path, config):
        win, session = self._window(qapp, tmp_path, config, None)
        try:
            assert "ghost" not in {a.id for a in win.actions_list}
        finally:
            win._timer.stop()
            session.shutdown(timeout=10)

    def test_the_ghost_cycles_and_only_shows_while_staging(
        self, qapp, tmp_path, config, reference
    ):
        win, session = self._window(qapp, tmp_path, config, reference)
        try:
            assert "ghost" in {a.id for a in win.actions_list}
            live = np.zeros((8, 12, 3), np.uint8)
            planned = session.current
            assert planned is not None
            faint, notes = win._ghosted(live, planned)
            assert notes and "30%" in notes[0]
            assert 60 <= int(faint[0, 0, 0]) <= 90
            win._cycle_ghost()  # -> strong
            strong, notes = win._ghosted(live, planned)
            assert "60%" in notes[0] and 140 <= int(strong[0, 0, 0]) <= 170
            win._cycle_ghost()  # -> reference only
            full, _ = win._ghosted(live, planned)
            assert (full == 255).all()
            win._cycle_ghost()  # -> off
            same, notes = win._ghosted(live, planned)
            assert same is live and notes == []
            win._cycle_ghost()  # -> faint again, but the arm is moving
            from policy_abstraction.experiments.session import SessionState

            session.state = SessionState.RUNNING
            same, _ = win._ghosted(live, planned)
            assert same is live
        finally:
            win._timer.stop()
            session.shutdown(timeout=10)

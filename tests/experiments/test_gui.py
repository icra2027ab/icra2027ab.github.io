"""The operator window, driven offscreen.

Qt runs headless under ``QT_QPA_PLATFORM=offscreen``, so the window can
be built, painted and grabbed in CI. What is worth testing here is the
wiring an operator's safety depends on: that every action has both a
button and a hotkey, that they enable together, and that a stop
actually reaches the session.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from policy_abstraction.experiments.config import ExperimentConfig  # noqa: E402
from policy_abstraction.experiments.factors import (  # noqa: E402
    GridPositionFactor,
)
from policy_abstraction.experiments.grid import TableGrid  # noqa: E402
from policy_abstraction.experiments.outcomes import (  # noqa: E402
    OPERATOR_FAILURE_MODES,
    FailureMode,
    Outcome,
)
from policy_abstraction.experiments.session import (  # noqa: E402
    ExperimentSession,
    SessionState,
)
from policy_abstraction.experiments.store import ExperimentStore  # noqa: E402
from tests.experiments.conftest import FakeBackends  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def config():
    return ExperimentConfig(
        name="gui_test",
        task="Put the cable in the box.",
        factors=[
            GridPositionFactor(
                "cable_position",
                TableGrid(
                    origin=(0, 0, 0),
                    u_axis=(0, 1, 0),
                    v_axis=(1, 0, 0),
                    rows=2,
                    cols=2,
                    name="cable",
                ),
            )
        ],
        repeats=1,
        view_scale=1.0,
    )


@pytest.fixture
def window(qapp, tmp_path, config):
    from policy_abstraction.experiments.gui.main_window import (
        ExperimentWindow,
    )

    backends = FakeBackends(runner_kwargs={"steps": 10_000})
    store = ExperimentStore.open(tmp_path / "gui", config)
    session = ExperimentSession(backends, config, store, policy_setting="t")
    session.prepare_env()
    win = ExperimentWindow(session, config)
    yield win
    win._timer.stop()
    session.shutdown(timeout=10)
    win.deleteLater()


class TestActionWiring:
    def test_every_action_has_a_button_and_a_shortcut(self, window):
        for action in window.actions_list:
            assert action.id in window._buttons
            assert action.id in window._qactions
            assert action.shortcut
            assert action.shortcut in window._buttons[action.id].text()

    def test_shortcuts_are_unique(self, window):
        keys = [a.shortcut for a in window.actions_list]
        assert len(set(keys)) == len(keys)

    def test_the_documented_quick_actions_all_exist(self, window):
        ids = {a.id for a in window.actions_list}
        assert {
            "start",
            "stop_success",
            "stop_catastrophic",
            "quit",
        } <= ids

    def test_button_and_hotkey_enable_together(self, window):
        """A live hotkey behind a greyed-out button is how people get hurt."""
        for state in (SessionState.STAGING, SessionState.RUNNING):
            window.session.state = state
            window._update_buttons()
            for action in window.actions_list:
                button = window._buttons[action.id]
                qaction = window._qactions[action.id]
                assert button.isEnabled() == qaction.isEnabled()

    def test_start_is_offered_only_when_idle(self, window):
        window.session.state = SessionState.STAGING
        window._update_buttons()
        assert window._buttons["start"].isEnabled()
        for state in (
            SessionState.RUNNING,
            SessionState.STOPPING,
            SessionState.FINISHED,
        ):
            window.session.state = state
            window._update_buttons()
            assert not window._buttons["start"].isEnabled()

    def test_stops_are_offered_only_while_running(self, window):
        for stop in ("stop_success", "stop_failure", "stop_catastrophic"):
            window.session.state = SessionState.STAGING
            window._update_buttons()
            assert not window._buttons[stop].isEnabled()
            window.session.state = SessionState.RUNNING
            window._update_buttons()
            assert window._buttons[stop].isEnabled()

    def test_quit_is_always_available(self, window):
        for state in SessionState:
            window.session.state = state
            window._update_buttons()
            assert window._buttons["quit"].isEnabled()

    def test_buttons_do_not_take_focus(self, window):
        """Otherwise Space re-fires the last clicked button, not Start."""
        from PyQt6.QtCore import Qt

        for button in window._buttons.values():
            assert button.focusPolicy() is Qt.FocusPolicy.NoFocus


class TestStopReachesTheSession:
    def test_success_hotkey_stops_the_rollout(self, window):
        window.session.start_rollout()
        window._stop_success()
        assert window.session.state is SessionState.STOPPING
        assert window.session._stop_outcome[0] is Outcome.SUCCESS

    def test_catastrophic_stop_asks_nothing(self, window):
        """No dialog: every millisecond is arm still moving."""
        window.session.start_rollout()
        window._stop_catastrophic()
        assert window.session.state is SessionState.STOPPING
        outcome, modes, _ = window.session._stop_outcome
        assert outcome is Outcome.FAILURE
        assert modes == (FailureMode.CATASTROPHIC,)

    def test_the_failure_dialog_can_record_several_modes(self, window, monkeypatch):
        """Wrong object *and* dropped it: both observations reach the record."""
        from policy_abstraction.experiments.gui import main_window as mw

        monkeypatch.setattr(
            mw,
            "FailureModeDialog",
            _FakeModeDialog(
                (FailureMode.WRONG_TARGET, FailureMode.DROPPED), "grabbed the mug"
            ),
        )
        window.session.start_rollout()
        window._stop_failure()
        if window.session._thread is not None:
            window.session._thread.join(timeout=10)
        record = window.session.last_record
        assert record.outcome is Outcome.FAILURE
        assert record.failure_modes == [FailureMode.DROPPED, FailureMode.WRONG_TARGET]
        assert record.failure_mode is FailureMode.DROPPED, (
            "the first, in taxonomy order"
        )
        assert window.session.record_for(0).failure_modes == record.failure_modes

    def test_a_timeout_opens_the_failure_dialog_once(self, window, monkeypatch):
        """Recorded as a failure on its own, but the operator still says why."""
        import time

        from policy_abstraction.experiments.gui import main_window as mw

        fake = _FakeModeDialog(FailureMode.NO_MOTION, "sat there")
        monkeypatch.setattr(mw, "FailureModeDialog", fake)
        window.session.config.timeout_s = 0.05
        window.session.start_rollout()
        deadline = time.time() + 10
        while window.session.is_running and time.time() < deadline:
            time.sleep(0.01)
        assert not window.session.is_running
        window._tick()
        assert "recorded as a failure" in fake.kwargs["heading"]
        record = window.session.record_for(0)
        assert record.outcome is Outcome.FAILURE
        assert record.failure_modes == [FailureMode.NO_MOTION, FailureMode.TIMEOUT]
        assert record.notes == "sat there"
        assert record.timed_out
        # A second tick must not ask again.
        fake.kwargs = None
        window._tick()
        assert fake.kwargs is None

    def test_cancelling_the_timeout_dialog_leaves_a_plain_timeout(
        self, window, monkeypatch
    ):
        import time

        from policy_abstraction.experiments.gui import main_window as mw

        monkeypatch.setattr(
            mw,
            "FailureModeDialog",
            _FakeModeDialog(FailureMode.OTHER, "", accept=False),
        )
        window.session.config.timeout_s = 0.05
        window.session.start_rollout()
        deadline = time.time() + 10
        while window.session.is_running and time.time() < deadline:
            time.sleep(0.01)
        window._tick()
        record = window.session.record_for(0)
        assert record.failure_modes == [FailureMode.TIMEOUT]


class TestRendering:
    def test_the_window_paints_with_a_live_frame(self, window):
        window._refresh()
        pixmap = window.grab()
        assert pixmap.width() > 0 and pixmap.height() > 0

    def test_it_paints_before_any_frame_has_arrived(self, window):
        window.session._frame = type(window.session._frame)()
        window._refresh()
        assert window.grab().width() > 0

    def test_the_wrist_panel_is_hidden_without_a_wrist_camera(self, window):
        assert window.session.wrist_camera is None
        assert window.wrist_panel.isHidden()

    def test_the_wrist_panel_streams_the_wrist_camera(self, qapp, tmp_path, config):
        from policy_abstraction.experiments.gui.main_window import (
            ExperimentWindow,
        )
        from tests.experiments.conftest import FakeEnv

        backends = FakeBackends(FakeEnv(wrist=True), runner_kwargs={"steps": 10_000})
        store = ExperimentStore.open(tmp_path / "gui_wrist", config)
        session = ExperimentSession(backends, config, store, policy_setting="t")
        session.prepare_env()
        win = ExperimentWindow(session, config)
        try:
            assert not win.wrist_panel.isHidden()
            win._last_frame_stamp = -1.0
            win._update_view()
            assert win.wrist_panel.view._pixmap is not None
            assert win.grab().width() > 0
        finally:
            win._timer.stop()
            session.shutdown(timeout=10)
            win.deleteLater()

    def test_a_broken_overlay_still_shows_the_camera(self, window, monkeypatch):
        """The operator must never lose the live view of a moving arm."""
        import policy_abstraction.experiments.gui.main_window as mw

        def explode(*_args, **_kwargs):
            raise RuntimeError("overlay bug")

        monkeypatch.setattr(mw, "render_staging_view", explode)
        window._last_frame_stamp = -1.0
        window._update_view()
        assert window.grab().width() > 0

    def test_the_banner_names_the_state(self, window):
        window.session.state = SessionState.STAGING
        assert "STAGE" in window._banner_text()
        window.session.state = SessionState.RUNNING
        assert "RUNNING" in window._banner_text()
        window.session.state = SessionState.STOPPING
        assert "STOPPING" in window._banner_text()
        window.session.state = SessionState.FINISHED
        assert "COMPLETE" in window._banner_text()


class TestWarmUpIsVisible:
    """A multi-minute compile must not look like a hung arm.

    The policy's first inference autotunes the checkpoint — measured at 449 s
    on the UR5e cell — and until it returns nothing moves and nothing is
    logged. Both surfaces that an operator can see during that wait are
    checked here.
    """

    def test_the_banner_says_so_while_the_warm_up_is_in_flight(self, window):
        window.session.warming_up = True
        assert "WARMING UP" in window._banner_text()
        assert "not commanded" in window._banner_text()

    def test_the_banner_says_so_at_step_zero_of_an_unwarmed_rollout(self, window):
        """The fallback, for a start-up warm-up that was skipped or failed."""
        window.session.warming_up = False
        window.session.state = SessionState.RUNNING
        window.session._warmed = False
        window.session._live_step = 0
        assert "warming up the policy" in window._banner_text()

        window.session._warmed = True
        assert "warming up the policy" not in window._banner_text()

    def test_the_start_up_window_re_headlines_on_a_phase(self, qapp):
        from policy_abstraction.experiments.gui.startup import StartupWindow

        startup = StartupWindow("exp")
        try:
            startup.set_phase("WARMING UP THE POLICY", "compiling the checkpoint")
            assert "WARMING UP" in startup.status.text()
            assert "compiling the checkpoint" in startup.detail.text()

            # Once start-up is over the headline is owned by whatever ended it.
            startup.mark_finished()
            startup.set_phase("TOO LATE", "ignored")
            assert "TOO LATE" not in startup.status.text()
        finally:
            startup.deleteLater()

    def test_a_bound_log_record_carries_the_phase_across_the_thread(self):
        """The wiring: `logger.bind(startup_phase=...)` -> the headline.

        The session emits the phase as an extra on an ordinary log record
        rather than through a callback threaded down into the session factory,
        so this is the join that has to hold.
        """
        from loguru import logger

        seen: list[tuple[str, str]] = []

        def sink(message) -> None:
            record = message.record
            phase = record["extra"].get("startup_phase")
            if phase:
                seen.append((phase, record["extra"].get("startup_detail", "")))

        sink_id = logger.add(sink, level="INFO", format="{message}")
        try:
            logger.info("an ordinary line")
            logger.bind(startup_phase="WARMING UP", startup_detail="d").info("x")
        finally:
            logger.remove(sink_id)

        assert seen == [("WARMING UP", "d")], (
            "only a phase-bound record may re-headline the start-up window"
        )


class TestFailureModeDialog:
    def test_offers_every_operator_selectable_mode(self, qapp):
        from policy_abstraction.experiments.gui.widgets import (
            FailureModeDialog,
        )

        dialog = FailureModeDialog()
        assert set(dialog._buttons) == set(OPERATOR_FAILURE_MODES)
        # Timeout is derived, never chosen.
        assert FailureMode.TIMEOUT not in dialog._buttons
        assert dialog.selected() == (FailureMode.GRASP_MISS,)
        dialog.deleteLater()

    def test_several_modes_can_be_ticked_at_once(self, qapp):
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog()
        dialog._buttons[FailureMode.DROPPED].setChecked(True)
        dialog._buttons[FailureMode.COLLISION].setChecked(True)
        assert dialog.selected() == (
            FailureMode.GRASP_MISS,
            FailureMode.DROPPED,
            FailureMode.COLLISION,
        ), "every ticked box, in taxonomy order, the default included"
        dialog.deleteLater()

    def test_nothing_ticked_records_other(self, qapp):
        """Enter on an untouched dialog still yields a verdict."""
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog()
        dialog._buttons[FailureMode.GRASP_MISS].setChecked(False)
        assert dialog.selected() == (FailureMode.OTHER,)
        dialog.deleteLater()

    def test_it_starts_from_the_modes_it_is_given(self, qapp):
        """Re-scoring opens on what was recorded, all of it."""
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog(
            default=(FailureMode.NO_MOTION, FailureMode.COLLISION)
        )
        assert dialog.selected() == (FailureMode.NO_MOTION, FailureMode.COLLISION)
        assert not dialog._buttons[FailureMode.GRASP_MISS].isChecked()
        dialog.deleteLater()


class TestSettingsDialog:
    """The form itself: what it reads back, and what it refuses."""

    @pytest.fixture
    def dialog(self, qapp, config):
        from policy_abstraction.experiments.gui.settings import SettingsDialog

        dlg = SettingsDialog(config, "cable_baseline")
        yield dlg
        dlg.deleteLater()

    def test_it_starts_showing_the_current_settings(self, dialog, config):
        assert dialog.result_name() == "cable_baseline"
        assert dialog.result_config().to_dict() == config.to_dict()

    def test_editing_the_grid_reshapes_the_plan(self, dialog):
        widgets = dialog._factor_widgets["cable_position"]
        widgets["rows"].setValue(4)
        widgets["cols"].setValue(3)
        edited = dialog.result_config()
        assert edited.factors[0].grid.rows == 4
        assert edited.factors[0].grid.cols == 3
        assert len(edited.build_plan()) == 12

    def test_repeats_multiply_the_plan(self, dialog):
        dialog._repeats.setValue(3)
        assert len(dialog.result_config().build_plan()) == 12

    def test_the_grouping_is_read_back(self, dialog):
        dialog._repeats.setValue(2)
        dialog._grouping.setCurrentText("sweep")
        edited = dialog.result_config()
        assert edited.grouping == "sweep"
        plan = edited.build_plan()
        assert plan[0].assignments != plan[1].assignments

    def test_a_blank_cell_list_means_every_cell(self, dialog):
        widgets = dialog._factor_widgets["cable_position"]
        assert widgets["cells"].text() == ""
        widgets["rows"].setValue(3)
        assert len(dialog.result_config().factors[0].cells) == 6

    def test_a_cell_subset_is_kept(self, dialog):
        dialog._factor_widgets["cable_position"]["cells"].setText("A1, B2")
        assert dialog.result_config().factors[0].cells == ((0, 0), (1, 1))

    def test_cells_outside_a_shrunken_grid_are_dropped_not_fatal(self, dialog):
        widgets = dialog._factor_widgets["cable_position"]
        widgets["cells"].setText("A1, B2")
        widgets["rows"].setValue(1)
        assert dialog.result_config().factors[0].cells == ((0, 0),)

    def test_an_empty_cell_list_is_rejected_with_a_readable_message(self, dialog):
        widgets = dialog._factor_widgets["cable_position"]
        widgets["cells"].setText("D4")
        widgets["rows"].setValue(1)
        widgets["cols"].setValue(1)
        with pytest.raises(ValueError, match="clear the field"):
            dialog.result_config()

    def test_zero_means_no_limit_for_the_optional_caps(self, dialog):
        dialog._timeout.setValue(0)
        dialog._max_steps.setValue(0)
        edited = dialog.result_config()
        assert edited.timeout_s is None
        assert edited.max_steps is None

    def test_the_summary_tracks_the_form(self, dialog):
        dialog._repeats.setValue(5)
        assert "20" in dialog.summary.text()

    def test_a_bad_value_shows_in_the_summary_rather_than_raising(self, dialog):
        dialog._factor_widgets["cable_position"]["cells"].setText("not-a-cell")
        assert "malformed" in dialog.summary.text()


class TestSettingsFlow:
    """The window's side: applying an edit, and refusing to guess."""

    @pytest.fixture
    def window_with_reviser(self, qapp, tmp_path, config):
        from policy_abstraction.experiments.gui.main_window import ExperimentWindow
        from policy_abstraction.experiments.revise import ExperimentReviser
        from policy_abstraction.experiments.store import experiment_root

        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        root = experiment_root("run", "gui_test", base=tmp_path)
        store = ExperimentStore.open(root, config)
        session = ExperimentSession(backends, config, store, policy_setting="t")
        session.prepare_env()
        reviser = ExperimentReviser(
            run="run",
            name="gui_test",
            config=config,
            store=store,
            base=tmp_path,
            source_config=tmp_path / "configs" / "src.yaml",
        )
        win = ExperimentWindow(session, config, reviser=reviser)
        yield win
        win._timer.stop()
        session.shutdown(timeout=10)
        win.deleteLater()

    def test_the_action_appears_only_with_a_reviser(self, window, window_with_reviser):
        assert "settings" not in {a.id for a in window.actions_list}
        assert "settings" in {a.id for a in window_with_reviser.actions_list}

    def test_it_is_offered_only_between_rollouts(self, window_with_reviser):
        win = window_with_reviser
        win.session.state = SessionState.RUNNING
        win._update_buttons()
        assert not win._buttons["settings"].isEnabled()
        win.session.state = SessionState.STAGING
        win._update_buttons()
        assert win._buttons["settings"].isEnabled()

    def _reshape(self, config, cols):
        from policy_abstraction.experiments.config import ExperimentConfig

        data = config.to_dict()
        data["factors"][0]["grid"]["cols"] = cols
        data["factors"][0]["cells"] = None
        return ExperimentConfig.from_dict(data)

    def test_applying_an_edit_rebinds_the_window_and_the_session(
        self, window_with_reviser, config, monkeypatch
    ):
        from policy_abstraction.experiments.gui import main_window as mw

        monkeypatch.setattr(mw.QMessageBox, "information", lambda *a, **k: None)
        win = window_with_reviser
        win._apply_settings(self._reshape(config, 4), "gui_test")
        assert len(win.session.plan) == 8
        assert win.config.factors[0].grid.cols == 4
        assert win.session.store.root.name == "gui_test"

    def test_an_edit_over_results_asks_instead_of_deleting(
        self, window_with_reviser, config, monkeypatch
    ):
        from policy_abstraction.experiments.gui import main_window as mw
        from policy_abstraction.experiments.outcomes import RolloutRecord

        win = window_with_reviser
        win.session.store.append(
            RolloutRecord(index=0, assignments={}, outcome=Outcome.SUCCESS)
        )
        asked = []
        monkeypatch.setattr(
            mw.ExperimentWindow,
            "_resolve_conflict",
            lambda self, revision, cfg: asked.append(revision),
        )
        win._apply_settings(self._reshape(config, 4), "gui_test")
        assert len(asked) == 1
        assert asked[0].records == 1
        # Nothing was applied while the question is outstanding.
        assert win.session.store.record_count() == 1
        assert len(win.session.plan) == 4

    def test_the_new_experiment_route_keeps_the_old_results(
        self, window_with_reviser, config, monkeypatch
    ):
        from policy_abstraction.experiments.gui import main_window as mw
        from policy_abstraction.experiments.outcomes import RolloutRecord

        win = window_with_reviser
        old_root = win.session.store.root
        win.session.store.append(
            RolloutRecord(index=0, assignments={}, outcome=Outcome.SUCCESS)
        )
        monkeypatch.setattr(mw.QMessageBox, "information", lambda *a, **k: None)
        monkeypatch.setattr(mw, "ConflictDialog", _FakeConflict(overwrite=False))
        win._apply_settings(self._reshape(config, 4), "gui_test")
        assert win.session.store.root != old_root
        assert ExperimentStore(old_root).record_count() == 1
        assert len(win.session.plan) == 8

    def test_the_overwrite_route_needs_the_confirmations(
        self, window_with_reviser, config, monkeypatch
    ):
        from policy_abstraction.experiments.gui import main_window as mw
        from policy_abstraction.experiments.outcomes import RolloutRecord

        win = window_with_reviser
        win.session.store.append(
            RolloutRecord(index=0, assignments={}, outcome=Outcome.SUCCESS)
        )
        monkeypatch.setattr(mw, "ConflictDialog", _FakeConflict(overwrite=True))
        monkeypatch.setattr(mw, "confirm_destroy", lambda *a: False)
        win._apply_settings(self._reshape(config, 4), "gui_test")
        assert win.session.store.record_count() == 1, "declining must keep the data"

        monkeypatch.setattr(mw.QMessageBox, "information", lambda *a, **k: None)
        monkeypatch.setattr(mw, "confirm_destroy", lambda *a: True)
        win._apply_settings(self._reshape(config, 4), "gui_test")
        assert win.session.store.record_count() == 0
        assert len(win.session.plan) == 8


class _FakeConflict:
    """Stands in for the modal choice, answering it the same way every time."""

    def __init__(self, *, overwrite: bool) -> None:
        self._overwrite = overwrite

    def __call__(self, revision, suggested, parent=None):
        self.revision = revision
        self.suggested = suggested
        return self

    def exec(self):
        return 1

    @property
    def overwrite(self):
        return self._overwrite

    def new_name(self):
        return self.suggested


class TestConfirmDestroy:
    """The two confirmations in front of deleting an experiment's results.

    The conflict dialog's overwrite route mocks this out, so exercise it here:
    it is the last thing standing between a mis-click and a lost afternoon of
    collection, and both questions have to be answered to get past it.
    """

    @pytest.fixture
    def revision(self, tmp_path):
        from policy_abstraction.experiments.revise import Revision, RevisionAction

        return Revision(
            name="gui_test",
            root=tmp_path / "gui_test",
            action=RevisionAction.DESTRUCTIVE,
            plan_changed=True,
            name_changed=False,
            config_changed=True,
            old_total=4,
            new_total=8,
            records=3,
            scored=3,
        )

    def _answers(self, monkeypatch, *, first, typed=None, accepted=True):
        """Stand in for both modals, recording whether the second was reached."""
        from PyQt6.QtWidgets import QMessageBox

        from policy_abstraction.experiments.gui import settings as st

        asked = {"name": False}
        monkeypatch.setattr(
            st.QMessageBox,
            "warning",
            lambda *a, **k: (
                QMessageBox.StandardButton.Yes
                if first
                else QMessageBox.StandardButton.Cancel
            ),
        )
        monkeypatch.setattr(st.QMessageBox, "information", lambda *a, **k: None)

        def _get_text(*_a, **_k):
            asked["name"] = True
            return (typed or "", accepted)

        monkeypatch.setattr(st.QInputDialog, "getText", _get_text)
        return asked

    def test_declining_the_first_question_never_asks_the_second(
        self, qapp, revision, monkeypatch
    ):
        from policy_abstraction.experiments.gui.settings import confirm_destroy

        asked = self._answers(monkeypatch, first=False)
        assert confirm_destroy(None, revision) is False
        assert not asked["name"], "cancelling the warning must end it there"

    def test_the_name_has_to_be_typed_back(self, qapp, revision, monkeypatch):
        from policy_abstraction.experiments.gui.settings import confirm_destroy

        self._answers(monkeypatch, first=True, typed="gui_tes")
        assert confirm_destroy(None, revision) is False

    def test_cancelling_the_second_question_keeps_the_data(
        self, qapp, revision, monkeypatch
    ):
        from policy_abstraction.experiments.gui.settings import confirm_destroy

        self._answers(monkeypatch, first=True, typed="gui_test", accepted=False)
        assert confirm_destroy(None, revision) is False

    def test_both_answers_are_needed_to_delete(self, qapp, revision, monkeypatch):
        from policy_abstraction.experiments.gui.settings import confirm_destroy

        asked = self._answers(monkeypatch, first=True, typed="  gui_test  ")
        assert confirm_destroy(None, revision) is True
        assert asked["name"]


class TestLayoutRegionSettings:
    """Moving and resizing the patch of table a disambiguation scene uses."""

    @pytest.fixture
    def layout_config(self):
        from policy_abstraction.experiments.config import ExperimentConfig
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
                        origin=(0.25, -0.20, 0.0),
                        u_axis=(0, 1, 0),
                        v_axis=(1, 0, 0),
                        rows=1,
                        cols=1,
                        cell_size_u=0.40,
                        cell_size_v=0.30,
                        name="objects",
                    ),
                    ["red_mug", "black_cable", "white_box"],
                    layouts=2,
                ),
                TargetObjectFactor("target_object", layout="object_layout"),
            ],
            repeats=1,
        )

    @pytest.fixture
    def dialog(self, qapp, layout_config):
        from policy_abstraction.experiments.gui.settings import SettingsDialog

        dlg = SettingsDialog(layout_config, "objects_baseline")
        yield dlg
        dlg.deleteLater()

    def test_the_region_is_offered_in_metres_not_cells(self, dialog):
        widgets = dialog._factor_widgets["object_layout"]
        assert widgets["region_size_u"].value() == pytest.approx(0.40)
        assert widgets["region_size_v"].value() == pytest.approx(0.30)
        assert "rows" not in widgets, "a region has no cells to number"

    def test_it_starts_showing_the_current_settings(self, dialog, layout_config):
        assert dialog.result_config().to_dict() == layout_config.to_dict()

    def test_resizing_the_region_moves_the_objects(self, dialog):
        widgets = dialog._factor_widgets["object_layout"]
        before = dialog.result_config().factors[0].positions("L1")
        widgets["region_size_u"].setValue(0.60)
        widgets["region_size_v"].setValue(0.50)
        factor = dialog.result_config().factors[0]
        assert factor.region.extent_u == pytest.approx(0.60)
        assert factor.region.extent_v == pytest.approx(0.50)
        after = factor.positions("L1")
        assert any(not np.allclose(before[k], after[k]) for k in before), (
            "a bigger region has to scatter them differently"
        )

    def test_moving_the_region_moves_the_objects_with_it(self, dialog):
        widgets = dialog._factor_widgets["object_layout"]
        before = dialog.result_config().factors[0].positions("L1")["red_mug"]
        widgets["region_origin_x"].setValue(0.35)
        after = dialog.result_config().factors[0].positions("L1")["red_mug"]
        assert after[0] == pytest.approx(before[0] + 0.10, abs=1e-6)

    def test_the_cell_counts_underneath_are_left_alone(self, dialog):
        """A region written as several cells keeps them; only its size moves."""
        widgets = dialog._factor_widgets["object_layout"]
        widgets["region_size_u"].setValue(0.60)
        region = dialog.result_config().to_dict()["factors"][0]["region"]
        assert (region["rows"], region["cols"]) == (1, 1)
        assert region["cell_size_u"] == pytest.approx(0.60)

    def test_more_layouts_lengthen_the_plan(self, dialog, layout_config):
        assert len(layout_config.build_plan()) == 6  # 2 layouts x 3 objects
        dialog._factor_widgets["object_layout"]["layouts"].setValue(5)
        assert len(dialog.result_config().build_plan()) == 15

    def test_a_region_too_small_to_stage_is_reported_not_raised(self, dialog):
        widgets = dialog._factor_widgets["object_layout"]
        widgets["region_size_u"].setValue(0.05)
        widgets["region_size_v"].setValue(0.05)
        with pytest.raises(ValueError):
            dialog.result_config()
        # And the operator sees why, rather than the window vanishing.
        dialog._recount()
        assert "color:#c0392b" in dialog.summary.text()

    def test_the_separation_and_the_margin_are_editable_too(self, dialog):
        widgets = dialog._factor_widgets["object_layout"]
        widgets["min_separation"].setValue(0.15)
        widgets["margin"].setValue(0.02)
        factor = dialog.result_config().factors[0]
        assert factor.min_separation == pytest.approx(0.15)
        assert factor.margin == pytest.approx(0.02)

    def test_the_target_factor_is_not_editable_from_here(self, dialog):
        """Its levels come from the layout factor's object list, not the form."""
        assert dialog._factor_widgets["target_object"] == {}


class TestHistoryDialog:
    """The list of what has been collected, and re-scoring a row from it."""

    @pytest.fixture
    def window_with_records(self, window):
        from policy_abstraction.experiments.outcomes import RolloutRecord

        store = window.session.store
        plan = window.session.plan
        store.append(
            RolloutRecord(
                index=plan[0].index,
                assignments=dict(plan[0].assignments),
                outcome=Outcome.FAILURE,
                failure_mode=FailureMode.GRASP_MISS,
                steps=310,
                duration_s=64.0,
            )
        )
        store.append(
            RolloutRecord(
                index=plan[1].index,
                assignments=dict(plan[1].assignments),
                outcome=Outcome.SUCCESS,
                steps=200,
                duration_s=41.0,
            )
        )
        window.session._advance_to_pending()
        return window

    @pytest.fixture
    def dialog(self, qapp, window_with_records):
        from policy_abstraction.experiments.gui.history import HistoryDialog

        dlg = HistoryDialog(window_with_records.session)
        yield dlg
        dlg.deleteLater()

    def test_it_lists_every_rollout_recorded(self, dialog):
        assert dialog.table.rowCount() == 2
        assert dialog.table.item(0, 0).text() == "1"
        assert dialog.table.item(0, 2).text() == "failure"
        assert dialog.table.item(0, 3).text() == "Grasp Miss"
        assert dialog.table.item(1, 2).text() == "success"

    def test_it_shows_what_was_staged(self, dialog):
        assert "cable_position=" in dialog.table.item(0, 1).text()

    def test_the_buttons_need_a_selection(self, dialog):
        dialog.table.clearSelection()
        dialog._update_buttons()
        assert not dialog.success_button.isEnabled()
        dialog.table.selectRow(0)
        assert dialog.success_button.isEnabled()

    def test_marking_a_failure_a_success_moves_the_rate(self, dialog):
        session = dialog.session
        assert session.status.successes == 1
        dialog.table.selectRow(0)
        dialog._mark_success()
        assert session.status.successes == 2
        assert session.status.failures == 0
        # And the list repaints to the corrected verdict rather than going stale.
        assert dialog.table.item(0, 2).text() == "success"
        assert dialog.table.item(0, 3).text() == "—"

    def test_re_scoring_a_failure_asks_which_mode(self, dialog, monkeypatch):
        from policy_abstraction.experiments.gui import history as hist

        monkeypatch.setattr(
            hist, "FailureModeDialog", _FakeModeDialog(FailureMode.DROPPED, "let go")
        )
        dialog.table.selectRow(1)
        dialog._mark_failure()
        record = dialog.session.record_for(1)
        assert record.outcome is Outcome.FAILURE
        assert record.failure_mode is FailureMode.DROPPED
        assert "let go" in record.notes

    def test_re_scoring_can_give_several_modes(self, dialog, monkeypatch):
        from policy_abstraction.experiments.gui import history as hist

        fake = _FakeModeDialog((FailureMode.GRASP_MISS, FailureMode.COLLISION), "")
        monkeypatch.setattr(hist, "FailureModeDialog", fake)
        dialog.table.selectRow(0)
        dialog._mark_failure()
        assert fake.kwargs["default"] == (FailureMode.GRASP_MISS,), (
            "the dialog opens on what was recorded"
        )
        record = dialog.session.record_for(0)
        assert record.failure_modes == [FailureMode.GRASP_MISS, FailureMode.COLLISION]
        assert dialog.table.item(0, 3).text() == "Grasp Miss + Collision"

    def test_cancelling_the_mode_dialog_changes_nothing(self, dialog, monkeypatch):
        from policy_abstraction.experiments.gui import history as hist

        monkeypatch.setattr(
            hist,
            "FailureModeDialog",
            _FakeModeDialog(FailureMode.DROPPED, "", accept=False),
        )
        dialog.table.selectRow(1)
        dialog._mark_failure()
        assert dialog.session.record_for(1).outcome is Outcome.SUCCESS

    def test_discarding_asks_first_and_reopens_the_condition(self, dialog, monkeypatch):
        from policy_abstraction.experiments.gui import history as hist

        monkeypatch.setattr(
            hist.QMessageBox,
            "question",
            lambda *a, **k: hist.QMessageBox.StandardButton.Cancel,
        )
        dialog.table.selectRow(1)
        dialog._mark_aborted()
        assert dialog.session.status.completed == 2, "cancel keeps the result"

        monkeypatch.setattr(
            hist.QMessageBox,
            "question",
            lambda *a, **k: hist.QMessageBox.StandardButton.Yes,
        )
        dialog._mark_aborted()
        assert dialog.session.status.completed == 1
        assert dialog.session.current.index == 1, "and it is offered again"

    def test_a_discarded_rollout_is_still_listed(self, dialog, monkeypatch):
        """Or an operator who discarded the wrong one could not put it back."""
        from policy_abstraction.experiments.gui import history as hist

        monkeypatch.setattr(
            hist.QMessageBox,
            "question",
            lambda *a, **k: hist.QMessageBox.StandardButton.Yes,
        )
        dialog.table.selectRow(1)
        dialog._mark_aborted()
        assert dialog.table.rowCount() == 2
        assert dialog.table.item(1, 2).text() == "aborted"

    def test_a_correction_is_refused_mid_rollout(self, dialog):
        dialog.session.state = SessionState.RUNNING
        dialog.table.selectRow(0)
        with pytest.raises(RuntimeError):
            dialog.session.reclassify(0, Outcome.SUCCESS)
        dialog.session.state = SessionState.STAGING


class TestHistoryAction:
    def test_it_is_offered_only_between_rollouts(self, window):
        assert "history" in {a.id for a in window.actions_list}
        window.session.state = SessionState.RUNNING
        window._update_buttons()
        assert not window._buttons["history"].isEnabled()
        window.session.state = SessionState.STAGING
        window._update_buttons()
        assert window._buttons["history"].isEnabled()

    def test_it_says_so_when_nothing_has_been_collected(self, window, monkeypatch):
        from policy_abstraction.experiments.gui import main_window as mw

        seen = []
        monkeypatch.setattr(
            mw.QMessageBox, "information", lambda _p, title, *a, **k: seen.append(title)
        )
        window._history()
        assert seen == ["Nothing collected yet"]


class _FakeModeDialog:
    """Stands in for the failure-mode picker, answering the same way each time."""

    def __init__(self, mode, note, *, accept: bool = True, acted_on: str = "") -> None:
        self._mode, self._note, self._accept = mode, note, accept
        self._acted_on = acted_on

    def __call__(self, parent=None, **kwargs):
        self.kwargs = kwargs
        return self

    def exec(self):
        return 1 if self._accept else 0

    def selected(self):
        return self._mode

    def note(self):
        return self._note

    def acted_on(self):
        return self._acted_on


class TestStepCounter:
    """The step count on the banner: what tells slow from stalled."""

    def test_the_banner_counts_steps_while_running(self, window):
        window.session.state = SessionState.RUNNING
        window.session._live_step = 143
        window.session._step_limit = 1250
        assert "step 143/1250" in window._banner_text()

    def test_it_still_counts_before_a_budget_is_known(self, window):
        window.session.state = SessionState.RUNNING
        window.session._live_step = 7
        window.session._step_limit = 0
        assert "step 7" in window._banner_text()

    def test_the_staging_banner_does_not_carry_one(self, window):
        window.session.state = SessionState.STAGING
        assert "step" not in window._banner_text()


class TestActedOnPrompt:
    """The failure dialog asks which object, but only when there is a choice."""

    def test_a_positional_experiment_is_never_asked(self, qapp):
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog(objects=())
        assert dialog.acted_on() == (), "one object on the table, nothing to confuse"
        dialog.deleteLater()

    def test_it_offers_every_object_plus_none(self, qapp):
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog(
            objects=[("red_mug", "red mug"), ("black_cable", "black cable")]
        )
        offered = [box.text() for box in dialog._objects]
        assert any("red mug" in o for o in offered)
        assert any("black cable" in o for o in offered)
        assert any("no object" in o for o in offered)
        dialog.deleteLater()

    def test_it_starts_on_the_object_the_prompt_named(self, qapp):
        """So the commonest failure -- right object, fumbled -- needs no answer."""
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog(
            objects=[("red_mug", "red mug"), ("black_cable", "black cable")],
            acted_on="black_cable",
        )
        assert dialog.acted_on() == ("black_cable",)
        dialog.deleteLater()

    def test_none_is_a_distinct_answer_from_not_asked(self, qapp):
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog
        from policy_abstraction.experiments.outcomes import NO_OBJECT

        dialog = FailureModeDialog(objects=[("red_mug", "red mug")])
        dialog._object_boxes[NO_OBJECT].setChecked(True)
        assert dialog.acted_on() == (NO_OBJECT,)
        dialog.deleteLater()

    def test_several_wrong_objects_are_recorded_in_the_order_ticked(self, qapp):
        """The policy went for the mug, then the cable: both are the finding."""
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog(
            objects=[
                ("red_mug", "red mug"),
                ("black_cable", "black cable"),
                ("white_cable", "white cable"),
            ],
            acted_on="white_cable",
        )
        dialog._object_boxes["white_cable"].setChecked(False)
        dialog._object_boxes["black_cable"].setChecked(True)
        dialog._object_boxes["red_mug"].setChecked(True)
        assert dialog.acted_on() == ("black_cable", "red_mug")
        dialog.deleteLater()

    def test_no_object_and_a_named_object_exclude_each_other(self, qapp):
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog
        from policy_abstraction.experiments.outcomes import NO_OBJECT

        dialog = FailureModeDialog(
            objects=[("red_mug", "red mug"), ("black_cable", "black cable")],
            acted_on="black_cable",
        )
        dialog._object_boxes[NO_OBJECT].setChecked(True)
        assert dialog.acted_on() == (NO_OBJECT,)
        assert not dialog._object_boxes["black_cable"].isChecked()
        dialog._object_boxes["red_mug"].setChecked(True)
        assert dialog.acted_on() == ("red_mug",)
        assert not dialog._object_boxes[NO_OBJECT].isChecked()
        dialog.deleteLater()

    def test_ticking_nothing_is_unrecorded_not_no_object(self, qapp):
        from policy_abstraction.experiments.gui.widgets import FailureModeDialog

        dialog = FailureModeDialog(objects=[("red_mug", "red mug")], acted_on="red_mug")
        dialog._object_boxes["red_mug"].setChecked(False)
        assert dialog.acted_on() == ()
        dialog.deleteLater()

    def test_the_history_list_shows_what_it_went_for(self, qapp, window):
        from policy_abstraction.experiments.gui.history import HistoryDialog
        from policy_abstraction.experiments.outcomes import RolloutRecord

        plan = window.session.plan
        window.session.store.append(
            RolloutRecord(
                index=plan[0].index,
                assignments=dict(plan[0].assignments),
                outcome=Outcome.FAILURE,
                failure_mode=FailureMode.WRONG_TARGET,
                actual_target="black_cable",
            )
        )
        dialog = HistoryDialog(window.session)
        header = [
            dialog.table.horizontalHeaderItem(i).text()
            for i in range(dialog.table.columnCount())
        ]
        column = header.index("went for")
        # No object list on this positional config, so the id is shown as-is
        # rather than being silently dropped.
        assert dialog.table.item(0, column).text() == "black_cable"
        dialog.deleteLater()

    def test_the_history_list_shows_every_object_it_went_for(self, qapp, window):
        from policy_abstraction.experiments.gui.history import HistoryDialog
        from policy_abstraction.experiments.outcomes import RolloutRecord

        plan = window.session.plan
        window.session.store.append(
            RolloutRecord(
                index=plan[0].index,
                assignments=dict(plan[0].assignments),
                outcome=Outcome.FAILURE,
                failure_mode=FailureMode.WRONG_TARGET,
                actual_targets=["black_cable", "red_mug"],
            )
        )
        dialog = HistoryDialog(window.session)
        header = [
            dialog.table.horizontalHeaderItem(i).text()
            for i in range(dialog.table.columnCount())
        ]
        column = header.index("went for")
        assert dialog.table.item(0, column).text() == "black_cable → red_mug"
        dialog.deleteLater()

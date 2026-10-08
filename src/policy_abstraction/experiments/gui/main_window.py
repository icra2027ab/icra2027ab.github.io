"""The operator window: live view with the staging overlay, and the quick actions.

Threading, restated because it is the thing that breaks GUIs driving robots: the
rollout runs on a worker thread inside
:class:`~policy_abstraction.experiments.session.ExperimentSession`, and Qt
widgets may only be touched from the GUI thread. Nothing here is called from the
worker. The window polls on a timer and the session hands it immutable snapshots
(:meth:`~policy_abstraction.experiments.session.ExperimentSession.live_frame`),
so there is no cross-thread widget access to get wrong.

The stop actions deliberately do not block. Pressing stop sets a flag; the
rollout ends at its next step boundary and the session records it and homes the
arm. The window shows ``STOPPING`` in between, so the operator can see that the
press registered.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.gui.actions import (
    STOP_CATASTROPHIC,
    STOP_SUCCESS,
    Action,
    build_actions,
)
from policy_abstraction.experiments.gui.history import HistoryDialog
from policy_abstraction.experiments.gui.settings import (
    ConflictDialog,
    SettingsDialog,
    confirm_destroy,
)
from policy_abstraction.experiments.gui.widgets import (
    BigStatus,
    CameraPanel,
    CameraView,
    FailureModeDialog,
    InfoPanel,
    style_button,
)
from policy_abstraction.experiments.outcomes import FailureMode, Outcome, StopReason
from policy_abstraction.experiments.overlay import render_staging_view
from policy_abstraction.experiments.reference import GHOST_LEVELS, ghost
from policy_abstraction.experiments.revise import (
    AppliedRevision,
    ExperimentConflict,
    ExperimentReviser,
    Revision,
)
from policy_abstraction.experiments.session import ExperimentSession, SessionState

# Colour per session state, for the big status strip.
_STATE_COLORS = {
    SessionState.STAGING: "#2b6cb0",
    SessionState.RUNNING: "#27923f",
    SessionState.STOPPING: "#b7791f",
    SessionState.FINISHED: "#4a4a56",
}


class _SessionBridge(QObject):
    """Marshals the session's worker-thread callback onto the GUI thread.

    A Qt signal emitted from another thread is delivered as a queued connection,
    which is the one supported way to get from the rollout thread into the
    window. Everything else here runs on the GUI thread.
    """

    changed = pyqtSignal()


class ExperimentWindow(QMainWindow):
    """Live camera + overlay, quick actions, and the running tally."""

    def __init__(
        self,
        session: ExperimentSession,
        config: ExperimentConfig,
        *,
        refresh_hz: float = 15.0,
        on_report=None,
        reviser: "ExperimentReviser | None" = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.config = config
        self._on_report = on_report
        self._reviser = reviser
        self._quitting = False
        self._last_frame_stamp = -1.0
        # Index into GHOST_LEVELS. Starts at the faint level when there is a
        # reference to show: the operator asked for it by creating the
        # experiment from one.
        self._ghost_level = 1 if session.reference is not None else 0

        self.setWindowTitle(
            f"Experiment — {reviser.name if reviser is not None else config.name}"
        )
        self.resize(1180, 780)
        self.setStyleSheet(
            "QMainWindow { background: #101014; } QLabel { color: #d8d8e0; }"
        )

        self._build_ui()
        self._install_actions()

        self._bridge = _SessionBridge()
        # Queued: the emit comes from the rollout thread, and a direct
        # connection would run _refresh — and so every widget call in it
        # — off the GUI thread. PyQt6's stubs only declare connect(slot).
        self._bridge.changed.connect(
            self._refresh,
            Qt.ConnectionType.QueuedConnection,  # pyright: ignore[reportCallIssue]
        )
        session._on_state_change = self._bridge.changed.emit

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / max(1.0, refresh_hz)))
        self._refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # -- left: camera + status + buttons
        left = QVBoxLayout()
        left.setSpacing(10)
        self.view = CameraView()
        left.addWidget(self.view, stretch=1)

        self.status = BigStatus()
        left.addWidget(self.status)

        self.button_row = QHBoxLayout()
        self.button_row.setSpacing(8)
        left.addLayout(self.button_row)
        root.addLayout(left, stretch=3)

        # -- right: the numbers
        right = QVBoxLayout()
        right.setSpacing(10)
        # The wrist camera, live, above the numbers. Hidden on an env with
        # no wrist camera rather than left as an empty box.
        self.wrist_panel = CameraPanel("WRIST CAMERA")
        self.wrist_panel.setVisible(self.session.wrist_camera is not None)
        right.addWidget(self.wrist_panel)
        self.staging_panel = InfoPanel("STAGE THIS")
        self.progress_panel = InfoPanel("PROGRESS")
        self.last_panel = InfoPanel("LAST ROLLOUT")
        self.help_panel = InfoPanel("KEYS")
        for panel in (
            self.staging_panel,
            self.progress_panel,
            self.last_panel,
            self.help_panel,
        ):
            right.addWidget(panel)
        right.addStretch(1)
        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet("color: #6a6a74; font-size: 11px;")
        right.addWidget(self.footer)
        root.addLayout(right, stretch=2)

    def _install_actions(self) -> None:
        self.actions_list: tuple[Action, ...] = build_actions(
            self.session,
            on_start=self._start,
            on_success=self._stop_success,
            on_catastrophic=self._stop_catastrophic,
            on_failure=self._stop_failure,
            on_skip=self._skip,
            on_quit=self._quit,
            on_report=self._report,
            on_settings=None if self._reviser is None else self._settings,
            on_history=self._history,
            on_ghost=None if self.session.reference is None else self._cycle_ghost,
        )
        self._buttons: dict[str, QPushButton] = {}
        self._qactions: dict[str, QAction] = {}
        for action in self.actions_list:
            button = QPushButton(action.button_text)
            button.setToolTip(action.tooltip)
            button.clicked.connect(action.callback)
            # Keyboard focus would let Space activate whichever button was last
            # clicked instead of the Start action, which on a running rollout is
            # exactly the wrong surprise.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            style_button(button, action.style)
            self.button_row.addWidget(button)
            self._buttons[action.id] = button

            qaction = QAction(action.label, self)
            qaction.setShortcut(QKeySequence(action.shortcut))
            qaction.triggered.connect(action.callback)
            self.addAction(qaction)
            self._qactions[action.id] = qaction

        self.help_panel.set_rows([(a.shortcut, a.label) for a in self.actions_list])

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _start(self) -> None:
        if not self.session.start_rollout():
            return
        self._refresh()

    def _stop_success(self) -> None:
        self.session.request_stop(*STOP_SUCCESS)
        self._refresh()

    def _stop_catastrophic(self) -> None:
        # No dialog: this is the safety action, and anything between the press
        # and the stop is time the arm is still moving.
        self.session.request_stop(*STOP_CATASTROPHIC)
        self._refresh()

    def _stop_failure(self) -> None:
        """Stop first, classify second.

        The stop is issued before the dialog opens so the arm is already winding
        down while the operator decides what to call it. The dialog then
        annotates the record the session is about to write, or — if it already
        wrote it — the record on disk is amended in place.
        """
        if not self.session.request_stop(
            Outcome.FAILURE, FailureMode.OTHER, StopReason.OPERATOR
        ):
            return
        self._refresh()
        dialog = FailureModeDialog(
            self, objects=self.config.scene_objects(), acted_on=self._named_target()
        )
        if dialog.exec():
            # The session works out where the answer lands — the pending
            # stop, the record in flight, or a superseding row on disk — so
            # racing the dialog against the rollout thread is its problem and
            # not a reason to keep the operator away from the stop button.
            self.session.classify_failure(
                dialog.selected(),
                note=dialog.note(),
                actual_targets=dialog.acted_on(),
            )
        self._refresh()

    def _named_target(self, assignments: dict[str, str] | None = None) -> str:
        """The object the prompt asked for in the rollout being classified.

        Pre-selects the failure dialog, so "reached the right object and
        fumbled it" — much the commonest failure — needs no answer at all.
        Defaults to the rollout on deck; a finished rollout passes its own
        *assignments*, since the plan has moved on by then.
        """
        factor = self.config.target_factor()
        if factor is None:
            return ""
        if assignments is None:
            planned = self.session.current
            if planned is None:
                return ""
            assignments = planned.assignments
        return assignments.get(factor.name, "")

    def _cycle_ghost(self) -> None:
        self._ghost_level = (self._ghost_level + 1) % len(GHOST_LEVELS)
        # The live frame has not changed, but the picture has.
        self._last_frame_stamp = -1.0
        self._refresh()

    def _skip(self) -> None:
        self.session.skip_current()
        self._refresh()

    def _quit(self) -> None:
        """Ask the window to close. ``QWidget.close`` returns a bool,
        which the action contract does not carry."""
        self.close()

    # ------------------------------------------------------------------
    # Editing the experiment
    # ------------------------------------------------------------------

    def _settings(self) -> None:
        """Open the settings form, and apply whatever comes back.

        Refused mid-rollout: the rollout thread is about to append to the store
        this would swap out. The action is disabled then too — this is the
        belt for that braces.
        """
        if self._reviser is None:
            return
        if self.session.is_running:
            QMessageBox.information(
                self,
                "Rollout in progress",
                "Stop the rollout before changing the experiment settings.",
            )
            return
        dialog = SettingsDialog(
            self.config,
            self._reviser.name,
            self,
            collected=self.session.store.record_count(),
        )
        if not dialog.exec():
            return
        try:
            config = dialog.result_config()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Settings are not valid", str(exc))
            return
        self._apply_settings(config, dialog.result_name())

    def _history(self) -> None:
        """Show what has been collected, and let a verdict be corrected.

        Refused mid-rollout for the same reason the settings editor is: the
        rollout thread is about to append to the log this rewrites the reading
        of. The action is disabled then too — this is the belt for that braces.
        """
        if self.session.is_running:
            QMessageBox.information(
                self,
                "Rollout in progress",
                "Stop the rollout before re-scoring an earlier one.",
            )
            return
        if not self.session.store.record_count():
            QMessageBox.information(
                self,
                "Nothing collected yet",
                "No rollouts have been recorded in this experiment.",
            )
            return
        HistoryDialog(self.session, self).exec()
        # A correction moves the success rate, and may reopen a plan index.
        self._refresh()

    def _apply_settings(self, config, name: str, *, destroy: bool = False) -> None:
        if self._reviser is None:
            return
        try:
            applied = self._reviser.apply(config, name, destroy=destroy)
        except ExperimentConflict as conflict:
            self._resolve_conflict(conflict.revision, config)
            return
        except Exception as exc:
            logger.exception("Could not apply the edited settings")
            QMessageBox.critical(self, "Could not apply the settings", str(exc))
            return
        self._adopt(applied)

    def _resolve_conflict(self, revision: "Revision", config) -> None:
        """Put the destructive choice to the operator. Never decides for them."""
        if self._reviser is None:
            return
        dialog = ConflictDialog(
            revision, self._reviser.suggest_name(revision.name), self
        )
        if not dialog.exec():
            return
        if dialog.overwrite:
            if confirm_destroy(self, revision):
                self._apply_settings(config, revision.name, destroy=True)
            return
        new_name = dialog.new_name()
        if not new_name:
            QMessageBox.warning(
                self, "Name required", "The new experiment needs a name."
            )
            return
        # May conflict again if that name is also taken by a different plan,
        # in which case the operator is asked again with the new numbers.
        self._apply_settings(config, new_name)

    def _adopt(self, applied: "AppliedRevision") -> None:
        """Rebind the session and the window to the revised experiment."""
        try:
            self.session.reconfigure(applied.config, applied.store)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Could not switch experiment", str(exc))
            return
        self.config = applied.config
        self.setWindowTitle(f"Experiment — {applied.name}")
        # The overlay is drawn from the config, so the frame on screen is now
        # stale even though the camera has not produced a new one.
        self._last_frame_stamp = -1.0
        self._refresh()
        QMessageBox.information(self, "Settings applied", applied.describe())

    def _report(self) -> None:
        if self._on_report is None:
            return
        try:
            summary = self._on_report()
        except Exception as exc:
            QMessageBox.warning(self, "Report failed", str(exc))
            return
        totals = summary["totals"]
        QMessageBox.information(
            self,
            "Report written",
            f"{totals['successes']}/{totals['collected']} succeeded "
            f"({totals['success_rate']:.0%})\n\n"
            f"{self.session.store.report_dir / 'report.md'}",
        )

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Timer: repaint the frame, and poll the env when nothing is running."""
        if not self.session.is_running:
            self.session.refresh_preview()
        self._update_view()
        self._update_buttons()
        timed_out = self.session.pop_timeout_to_classify()
        if timed_out is not None:
            self._classify_timeout(timed_out)

    def _classify_timeout(self, record) -> None:
        """Ask what a rollout the budget ended was doing.

        The record is already a ``TIMEOUT`` failure — nothing here changes the
        verdict, and cancelling the dialog leaves it exactly as recorded. The
        modes the operator picks are added alongside ``TIMEOUT``, so the report
        still counts the timeout *and* knows what kind of not-finishing it was.
        """
        limit = self.config.timeout_s
        how = (
            f"timed out after {limit:.0f}s"
            if record.stop_reason is StopReason.TIMEOUT and limit
            else "ran out of steps"
        )
        dialog = FailureModeDialog(
            self,
            heading=(
                f"Rollout {record.index + 1} {how} and is recorded as a "
                "failure. What was it doing? The robot is returning home."
            ),
            objects=self.config.scene_objects(),
            acted_on=self._named_target(record.assignments),
        )
        if dialog.exec():
            self.session.classify_failure(
                dialog.selected(),
                note=dialog.note(),
                actual_targets=dialog.acted_on(),
            )
        self._refresh()

    def _refresh(self) -> None:
        """Full repaint, after anything that changes session state."""
        self._update_view()
        self._update_buttons()
        self._update_panels()

    def _update_view(self) -> None:
        frame = self.session.live_frame()
        if frame.image is None or frame.timestamp == self._last_frame_stamp:
            return
        self._last_frame_stamp = frame.timestamp
        planned = self.session.current
        assignments = dict(planned.assignments) if planned is not None else {}
        image, notes = self._ghosted(frame.image, planned)
        try:
            annotated = render_staging_view(
                image,
                self.config.factors,
                assignments,
                intrinsic=frame.intrinsic,
                extrinsic=frame.extrinsic,
                camera=self.session.camera,
                scale=self.config.view_scale,
                banner=self._banner_text(),
                banner_color=_banner_rgb(self.session.state),
                notes=notes,
            )
        except Exception as exc:
            # A broken overlay must never cost the operator their live view of a
            # moving robot.
            logger.warning(f"overlay failed, showing the raw frame: {exc}")
            annotated = frame.image
        self.view.set_image(annotated)
        if frame.wrist is not None:
            self.wrist_panel.set_image(frame.wrist)

    def _ghosted(self, image, planned) -> tuple[Any, list[str]]:
        """The live frame with the reference's scene blended in, and a caption.

        Only while staging: once the arm moves the operator needs the live
        picture alone, and the ghost would read as a second robot.
        """
        reference = self.session.reference
        alpha = GHOST_LEVELS[self._ghost_level]
        if (
            reference is None
            or alpha <= 0.0
            or self.session.state is not SessionState.STAGING
        ):
            return image, []
        ref = reference.frame_for(planned)
        if ref is None:
            return image, [f"reference {reference.name}: no frame for this condition"]
        try:
            blended = ghost(image, ref.image, alpha)
        except Exception as exc:
            logger.warning(f"reference ghost failed, showing the live frame: {exc}")
            return image, []
        return blended, [
            f"reference {reference.name} rollout {ref.index + 1} "
            f"ghosted at {alpha:.0%}  [G]"
        ]

    def _banner_text(self) -> str:
        session = self.session
        # One read of each: the rollout thread can advance the plan
        # between two attribute lookups, and a banner that dereferences
        # a `current` that just became None would take the window down
        # mid-rollout.
        state, planned = session.state, session.current
        if session.warming_up:
            return (
                "WARMING UP THE POLICY - compiling the checkpoint, "
                "the robot is not commanded"
            )
        if state is SessionState.FINISHED or planned is None:
            return "EXPERIMENT COMPLETE - press R for the report, Esc to close"
        if state is SessionState.RUNNING:
            # The step counter is what tells a slow rollout from a stalled one,
            # which from across the cell is otherwise the same picture.
            step, limit = session.progress()
            budget = f" - step {step}/{limit}" if limit else f" - step {step}"
            # Normally the warm-up is paid at start-up and this never shows.
            # It does when that was skipped or failed, and then step 0 is a
            # multi-minute compile that otherwise reads as a stalled arm.
            if step == 0 and not session.policy_warmed:
                budget += " - warming up the policy"
            return (
                f"RUNNING rollout {planned.index + 1}{budget}"
                " - S success / F fail / C STOP"
            )
        if state is SessionState.STOPPING:
            return "STOPPING - waiting for the rollout to finish and the arm to home"
        return (
            f"STAGE rollout {planned.index + 1}/{len(session.plan)} "
            "- press SPACE to start"
        )

    def _update_buttons(self) -> None:
        for action in self.actions_list:
            enabled = action.enabled_when(self.session)
            self._buttons[action.id].setEnabled(enabled)
            self._qactions[action.id].setEnabled(enabled)
        state = self.session.state
        self.status.set_state(
            self._banner_text() or state.value.upper(), _STATE_COLORS[state]
        )

    def _update_panels(self) -> None:
        session, status = self.session, self.session.status
        planned = session.current

        if planned is None:
            self.staging_panel.set_text(
                "<b>Every planned rollout has a result.</b><br>"
                "Press <b>R</b> to write the report."
            )
        else:
            rows = [("rollout", f"{planned.index + 1} of {len(session.plan)}")]
            rows += [
                (f.name, f.level(planned.assignments[f.name]).label)
                for f in self.config.factors
                if f.name in planned.assignments
            ]
            rows.append(("repeat", str(planned.repeat + 1)))
            # The instructions below the rows: a factor that stages several
            # objects has more to say than its level's name.
            self.staging_panel.set_rows(rows, self.session.instructions())

        rows = [
            ("collected", f"{status.completed} / {status.total}"),
            ("success", f"{status.successes}  ({status.success_rate:.0%})"),
            ("failure", str(status.failures)),
            ("aborted", str(status.aborted)),
            ("remaining", str(status.remaining)),
        ]
        # Completion time is recorded per rollout anyway; showing the running
        # mean turns it into the number the operator actually wants at 4pm —
        # whether the rest of the plan fits before they have to stop. The
        # warm-up rollout is left out, or the first estimate of the afternoon
        # would be the one that includes compiling the checkpoint.
        times = [
            r.duration_s
            for r in session.store.results()
            if r.duration_s > 0 and not r.warmup
        ]
        if times:
            mean = sum(times) / len(times)
            rows.append(("mean time", f"{mean:.0f}s"))
            if status.remaining:
                rows.append(("~time left", _minutes(mean * status.remaining)))
        self.progress_panel.set_rows(rows)

        record = session.last_record
        if record is None:
            self.last_panel.set_text("<i>none yet</i>")
        else:
            rows = [
                ("outcome", record.outcome.value),
                ("steps", str(record.steps)),
                ("duration", f"{record.duration_s:.0f}s"),
            ]
            if record.failure_modes:
                rows.insert(1, ("mode", record.failure_label))
            if record.error:
                rows.append(("error", record.error[:80]))
            self.last_panel.set_rows(rows)

        footer = [str(session.store.root)]
        if session.last_error:
            footer.append(f"⚠ {session.last_error}")
        if not session.camera_params:
            footer.append(
                "⚠ no camera calibration — the grid cannot be projected onto "
                "the view. Placements are still recorded."
            )
        self.footer.setText("\n".join(footer))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802
        """Interrupt the experiment: stop, park, report, close.

        Guarded against re-entry because ``close()`` is also the quit action's
        callback, and a second Esc while the arm is still homing must not start
        a second teardown.
        """
        if self._quitting:
            event.ignore()
            return
        if self.session.is_running:
            answer = QMessageBox.question(
                self,
                "Interrupt experiment?",
                "A rollout is still running. It will be stopped and recorded as "
                "aborted, and the robot will be returned home.\n\nInterrupt?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer is not QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._quitting = True
        self._timer.stop()
        self.status.set_state("SHUTTING DOWN - parking the robot …", "#b7791f")
        self.repaint()
        event.accept()


def _minutes(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _banner_rgb(state: SessionState) -> tuple[int, int, int]:
    hexcolor = _STATE_COLORS[state].lstrip("#")
    return tuple(int(hexcolor[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


__all__ = ["ExperimentWindow"]

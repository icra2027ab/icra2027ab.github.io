"""The rollouts already collected, and correcting the verdict on one.

Operators misclassify. A rollout called a failure turns out to have dropped the
cable in the bag on the way past; "grasp miss" was really "dropped"; the success
hotkey was pressed on the wrong rollout. Until now the only fix was to edit
``rollouts.jsonl`` by hand, which is exactly the file nobody should be editing by
hand mid-experiment.

So: a list, and two buttons. :class:`HistoryDialog` shows what is on disk and
hands a correction to
:meth:`~policy_abstraction.experiments.session.ExperimentSession.reclassify`,
which appends a superseding row rather than rewriting anything. The log stays the
record of what was pressed at the time; the report follows the correction.

Nothing here talks to the robot. It reads the store and calls one session
method.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from policy_abstraction.experiments.gui.widgets import FailureModeDialog
from policy_abstraction.experiments.outcomes import (
    ActualTargets,
    FailureMode,
    FailureModes,
    Outcome,
    RolloutRecord,
)

COLUMNS = ("#", "condition", "outcome", "mode", "went for", "steps", "time", "notes")

_OUTCOME_COLORS = {
    Outcome.SUCCESS: "#27923f",
    Outcome.FAILURE: "#c0392b",
    Outcome.ABORTED: "#6a6a74",
}


def _condition(record: RolloutRecord) -> str:
    """The staged condition, as ``factor=level`` pairs on one line."""
    if not record.assignments:
        return "—"
    return "  ".join(f"{k}={v}" for k, v in sorted(record.assignments.items()))


def _clock(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


class HistoryDialog(QDialog):
    """Every rollout collected so far, with its verdict editable.

    Shows the *standing* verdict per plan index — a rollout corrected twice
    appears once, as it currently reads — because the question the operator is
    answering is "is this right now", not "what was it before". The full history
    including every superseded row is still in ``rollouts.jsonl``.
    """

    def __init__(self, session, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Rollouts collected")
        self.setMinimumSize(820, 460)
        self._records: list[RolloutRecord] = []

        layout = QVBoxLayout(self)

        blurb = QLabel(
            "Select a rollout and re-score it. The correction is appended to "
            "the log — the original row is kept, and the report uses the "
            "latest verdict."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #8f8fa0;")
        layout.addWidget(blurb)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Qt types both headers as optional — they are not, for a table built
        # here, but the checked path costs one `if` and the table is still
        # readable without the sizing if some platform ever disagrees.
        rows_header = self.table.verticalHeader()
        if rows_header is not None:
            rows_header.setVisible(False)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            # The condition and the notes take whatever is left; everything
            # else is a short number that should not be padded out.
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(
                len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
            )
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.doubleClicked.connect(self._mark_failure)
        layout.addWidget(self.table, stretch=1)

        row = QHBoxLayout()
        self.success_button = QPushButton("Mark &success")
        self.success_button.clicked.connect(self._mark_success)
        self.failure_button = QPushButton("Mark &failure…")
        self.failure_button.clicked.connect(self._mark_failure)
        self.discard_button = QPushButton("&Discard (aborted)")
        self.discard_button.setToolTip(
            "Exclude this rollout from every rate, and offer its condition "
            "again — for a rollout that was mis-staged rather than failed."
        )
        self.discard_button.clicked.connect(self._mark_aborted)
        for button in (
            self.success_button,
            self.failure_button,
            self.discard_button,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #8f8fa0;")
        layout.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.reload()

    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read the store and repaint. Keeps the selected rollout selected."""
        selected = self.selected_index()
        self._records = _standing_records(self.session.store)
        self.table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            cells = (
                str(record.index + 1),
                _condition(record),
                record.outcome.value,
                record.failure_label or "—",
                self._object_label(record),
                str(record.steps or "—"),
                _clock(record.duration_s),
                record.notes or "",
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 2:
                    item.setForeground(
                        _color(_OUTCOME_COLORS.get(record.outcome, "#d8d8e0"))
                    )
                self.table.setItem(row, col, item)
            if record.index == selected:
                self.table.selectRow(row)

        scored = [r for r in self._records if r.outcome.counts_toward_rate]
        successes = sum(r.outcome is Outcome.SUCCESS for r in scored)
        self.summary.setText(
            f"{len(self._records)} rollouts recorded — {successes}/{len(scored)} "
            f"successful"
            + (
                f", {len(self._records) - len(scored)} discarded"
                if len(scored) != len(self._records)
                else ""
            )
        )
        self._update_buttons()

    def _object_label(self, record: RolloutRecord) -> str:
        """The acted-on object(s) in the operator's words, not their ids.

        Several are joined in the order the policy went for them —
        ``"black cable → red cable"`` — so a row reads as the sequence the
        operator watched.
        """
        if not record.actual_targets:
            return "—"
        labels = dict(self.session.config.scene_objects())
        return " → ".join(labels.get(t, t) for t in record.actual_targets)

    def selected_index(self) -> Optional[int]:
        """Plan index of the selected rollout, or ``None``."""
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        if 0 <= row < len(self._records):
            return self._records[row].index
        return None

    def selected_record(self) -> Optional[RolloutRecord]:
        index = self.selected_index()
        if index is None:
            return None
        return next((r for r in self._records if r.index == index), None)

    def _update_buttons(self) -> None:
        record = self.selected_record()
        for button in (
            self.success_button,
            self.failure_button,
            self.discard_button,
        ):
            button.setEnabled(record is not None)

    # ------------------------------------------------------------------

    def _mark_success(self) -> None:
        self._apply(Outcome.SUCCESS, FailureMode.NONE)

    def _mark_aborted(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        confirm = QMessageBox.question(
            self,
            "Discard this rollout?",
            f"Rollout {record.index + 1} stops counting toward any rate, and "
            "its condition is offered again for collection.\n\n"
            "The row stays in the log. Discard?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm is not QMessageBox.StandardButton.Yes:
            return
        self._apply(Outcome.ABORTED, FailureMode.NONE)

    def _mark_failure(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        # What was recorded, minus the derived mode nobody can tick; the
        # commonest failure when nothing is left to start from.
        default: FailureModes = tuple(
            m for m in record.failure_modes if m is not FailureMode.TIMEOUT
        ) or (FailureMode.GRASP_MISS,)
        dialog = FailureModeDialog(
            self,
            default=default,
            heading=(
                f"Re-scoring rollout {record.index + 1}, recorded as "
                f"{record.outcome.value}. The robot is not moving."
            ),
            objects=self.session.config.scene_objects(),
            # What was recorded at the time, or the object the prompt named if
            # nothing was: re-scoring a success to a failure has to start
            # somewhere, and "went for the one it was asked for" is the guess
            # that matches the rollout it is being re-scored from.
            acted_on=record.actual_targets or self._named_target(record),
        )
        if not dialog.exec():
            return
        self._apply(
            Outcome.FAILURE,
            dialog.selected(),
            note=dialog.note(),
            acted_on=dialog.acted_on(),
        )

    def _named_target(self, record: RolloutRecord) -> str:
        factor = self.session.config.target_factor()
        if factor is None:
            return ""
        return record.assignments.get(factor.name, "")

    def _apply(
        self,
        outcome: Outcome,
        mode: FailureModes | None,
        *,
        note: str = "",
        acted_on: ActualTargets | None = None,
    ) -> None:
        index = self.selected_index()
        if index is None:
            return
        try:
            self.session.reclassify(
                index, outcome, mode, note=note, actual_targets=acted_on
            )
        except (RuntimeError, KeyError) as exc:
            QMessageBox.warning(self, "Could not re-score that rollout", str(exc))
            return
        self.reload()


def _standing_records(store) -> list[RolloutRecord]:
    """The verdict that currently stands for each plan index, in plan order.

    Unlike :meth:`~policy_abstraction.experiments.store.ExperimentStore.results`
    this keeps the aborted ones: a rollout discarded by mistake is exactly one
    an operator needs to find in this list to put back.
    """
    latest: dict[int, RolloutRecord] = {}
    for record in store.records():
        latest[record.index] = record
    return [latest[i] for i in sorted(latest)]


def _color(hexcolor: str):
    from PyQt6.QtGui import QColor

    return QColor(hexcolor)


__all__ = ["HistoryDialog"]

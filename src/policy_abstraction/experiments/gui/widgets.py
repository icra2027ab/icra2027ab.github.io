"""Reusable Qt widgets for operator-facing experiment tools.

Kept generic on purpose: the next experiment (orientation, box position,
appearance) reuses these unchanged and only swaps the factors behind them.
Nothing here knows what a grid is — :class:`CameraView` is handed a finished
RGB array, and the panels are handed strings.
"""

from __future__ import annotations

import html
from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from policy_abstraction.experiments.outcomes import (
    NO_OBJECT,
    OPERATOR_FAILURE_MODES,
    ActualTargets,
    FailureMode,
    FailureModes,
    normalize_actual_targets,
    normalize_failure_modes,
)

# Button palettes, keyed by Action.style.
BUTTON_STYLES = {
    "default": ("#3a3a44", "#f0f0f4"),
    "primary": ("#2b6cb0", "#ffffff"),
    "success": ("#27923f", "#ffffff"),
    "danger": ("#c0392b", "#ffffff"),
}


def style_button(button: QPushButton, style: str) -> None:
    bg, fg = BUTTON_STYLES.get(style, BUTTON_STYLES["default"])
    button.setStyleSheet(
        f"QPushButton {{ background: {bg}; color: {fg}; border: none; "
        f"border-radius: 6px; padding: 10px 14px; font-weight: 600; }}"
        f"QPushButton:disabled {{ background: #2a2a31; color: #6a6a74; }}"
        f"QPushButton:hover:enabled {{ background: {bg}; opacity: 0.9; }}"
    )


class CameraView(QLabel):
    """Displays an RGB frame, scaled to fit while keeping its aspect ratio.

    The frame is already annotated by the time it arrives — the overlay is drawn
    with OpenCV against the camera's real intrinsics, not with Qt painter calls,
    so exactly the same image can be written to a file for a regression test.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        minimum_size: tuple[int, int] = (480, 360),
        placeholder: str = "waiting for the first camera frame …",
    ) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(*minimum_size)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #0e0e12; border-radius: 8px;")
        self.setText(placeholder)
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, image: np.ndarray) -> None:
        """Show *image*, an ``[H, W, 3]`` uint8 RGB array."""
        arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        h, w = arr.shape[:2]
        # QImage does not copy the buffer, so the array must outlive the QImage
        # — `.copy()` on the QImage takes ownership and makes that moot.
        qimage = QImage(
            arr.data,  # type: ignore[arg-type]  — QImage takes a buffer
            w,
            h,
            3 * w,
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._rescale()

    def resizeEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        super().setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class CameraPanel(QFrame):
    """A titled, fixed-height camera view for the side column.

    Sized for a secondary stream — the wrist camera beside the staging
    view — so it takes one panel's worth of the column and never squeezes
    the numbers below it out of sight.
    """

    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None,
        *,
        height: int = 220,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background: #1a1a20; border-radius: 8px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(4)
        self._title = QLabel(title)
        self._title.setStyleSheet(
            "color: #8f8fa0; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
        )
        layout.addWidget(self._title)
        self.view = CameraView(
            minimum_size=(160, height),
            placeholder="waiting for the wrist camera …",
        )
        self.view.setMaximumHeight(height)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.view)

    def set_image(self, image: np.ndarray) -> None:
        self.view.set_image(image)


class InfoPanel(QFrame):
    """A titled block of key/value rows, refreshed wholesale."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #1a1a20; border-radius: 8px; }"
            "QLabel { color: #d8d8e0; }"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 12)
        self._layout.setSpacing(4)
        self._title = QLabel(title)
        self._title.setStyleSheet(
            "color: #8f8fa0; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
        )
        self._layout.addWidget(self._title)
        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.TextFormat.RichText)
        self._layout.addWidget(self._body)

    def set_rows(
        self, rows: Sequence[tuple[str, str]], lines: Sequence[str] = ()
    ) -> None:
        """Key/value rows, optionally followed by plain lines.

        The plain lines carry text that has no key — a factor's staging
        instructions, which are already whole sentences and read worse split
        into a label and a value.
        """
        body = [
            f"<span style='color:#8f8fa0'>{k}</span>&nbsp;&nbsp;"
            f"<span style='color:#f0f0f4'>{v}</span>"
            for k, v in rows
        ]
        if lines:
            body.append("")
            body += [
                f"<span style='color:#f0f0f4'>{html.escape(line)}</span>".replace(
                    "  ", "&nbsp;&nbsp;"
                )
                for line in lines
            ]
        self._body.setText("<br>".join(body))

    def set_text(self, text: str) -> None:
        self._body.setText(text)


class BigStatus(QLabel):
    """The one line the operator reads from across the cell."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(56)
        self.set_state("", "#3a3a44")

    def set_state(self, text: str, color: str) -> None:
        self.setText(text)
        self.setStyleSheet(
            f"background: {color}; color: white; border-radius: 8px; "
            f"font-size: 19px; font-weight: 700; padding: 10px;"
        )


class FailureModeDialog(QDialog):
    """Pick how a rollout failed — one way or several — plus an optional note.

    Modal and keyboard-first: the modes are numbered, so ``1``–``7`` toggles
    each and Enter records the failure without the operator's hand leaving the
    keyboard. Several can be ticked at once, because a rollout that went for
    the wrong object and then dropped it failed in both ways and the second
    observation is worth keeping. The robot is already stopped and homing by
    the time this opens — nothing here delays the safety action.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        default: FailureModes = FailureMode.GRASP_MISS,
        heading: str = "The rollout is stopped and the robot is returning home.",
        objects: Sequence[tuple[str, str]] = (),
        acted_on: ActualTargets | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("How did it fail?")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        # Overridable because the same taxonomy is offered when re-scoring a
        # rollout collected an hour ago, where "returning home" is a lie.
        label = QLabel(heading)
        label.setWordWrap(True)
        layout.addWidget(label)

        hint = QLabel("Tick every way it failed (1–7 toggle, Enter records).")
        hint.setStyleSheet("color: #8a8a94;")
        layout.addWidget(hint)

        preselected = set(normalize_failure_modes(default))
        self._buttons: dict[FailureMode, QCheckBox] = {}
        grid = QGridLayout()
        for i, mode in enumerate(OPERATOR_FAILURE_MODES):
            box = QCheckBox(f"&{i + 1}  {mode.label}")
            box.setChecked(mode in preselected)
            self._buttons[mode] = box
            grid.addWidget(box, i, 0)
        layout.addLayout(grid)

        # Which object(s) it went for, when the scene held several.
        # Pre-ticked to the one the prompt named, so the common failure —
        # reached the right object and fumbled it — is already answered and
        # the operator only touches this when the policy went somewhere else.
        # Several can be ticked, because a policy that cannot resolve the
        # reference does not always commit to one wrong object: it can go for
        # the black cable, hesitate, and then the red one, and each of those
        # is a confusion worth recording. That is the whole disambiguation
        # result, and nothing but a person watching can supply it.
        self._object_boxes: dict[str, QCheckBox] = {}
        self._objects: list[QCheckBox] = []
        if objects:
            preselected = set(normalize_actual_targets(acted_on))
            layout.addWidget(QLabel("Which object(s) did it go for?"))
            hint = QLabel("Tick every one it went for, in any order (A–Z toggle).")
            hint.setStyleSheet("color: #8a8a94;")
            layout.addWidget(hint)
            grid = QGridLayout()
            for i, (object_id, label) in enumerate(objects):
                key = chr(ord("A") + i) if i < 26 else ""
                box = QCheckBox(f"&{key}  {label}" if key else label)
                box.setChecked(object_id in preselected)
                box.toggled.connect(self._object_ticked)
                self._object_boxes[object_id] = box
                self._objects.append(box)
                grid.addWidget(box, i, 0)
            none_box = QCheckBox("&0  no object — never committed to one")
            none_box.setChecked(NO_OBJECT in preselected)
            none_box.toggled.connect(self._none_ticked)
            self._object_boxes[NO_OBJECT] = none_box
            self._objects.append(none_box)
            grid.addWidget(none_box, len(objects), 0)
            layout.addLayout(grid)
            # The order the operator ticks the boxes in is the order the
            # policy went for the objects — that is how it is recorded.
            self._tick_order: list[str] = [
                t for t in normalize_actual_targets(acted_on) if t in self._object_boxes
            ]

        layout.addWidget(QLabel("Note (optional)"))
        self._note = QTextEdit()
        self._note.setMaximumHeight(70)
        layout.addWidget(self._note)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def selected(self) -> tuple[FailureMode, ...]:
        """Every ticked mode, in taxonomy order.

        Never empty: a failure with no box ticked is recorded as ``OTHER``,
        so accepting the dialog without touching it still yields a verdict.
        """
        chosen = tuple(m for m, box in self._buttons.items() if box.isChecked())
        return normalize_failure_modes(chosen) or (FailureMode.OTHER,)

    def note(self) -> str:
        return self._note.toPlainText().strip()

    def _object_ticked(self, checked: bool) -> None:
        """Naming an object contradicts "no object", so the two exclude each other."""
        box = self.sender()
        object_id = next((i for i, b in self._object_boxes.items() if b is box), None)
        if object_id is None:
            return
        if checked:
            if object_id not in self._tick_order:
                self._tick_order.append(object_id)
            none_box = self._object_boxes.get(NO_OBJECT)
            if none_box is not None and none_box.isChecked():
                none_box.blockSignals(True)
                none_box.setChecked(False)
                none_box.blockSignals(False)
                self._tick_order = [t for t in self._tick_order if t != NO_OBJECT]
        else:
            self._tick_order = [t for t in self._tick_order if t != object_id]

    def _none_ticked(self, checked: bool) -> None:
        if checked:
            for object_id, box in self._object_boxes.items():
                if object_id != NO_OBJECT and box.isChecked():
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.blockSignals(False)
            self._tick_order = [NO_OBJECT]
        else:
            self._tick_order = [t for t in self._tick_order if t != NO_OBJECT]

    def acted_on(self) -> tuple[str, ...]:
        """Object ids the operator says it went for, in the order they ticked them.

        ``()`` when nothing is ticked or the question was not asked. The
        report reads an empty answer as unrecorded, not as "no object" — that
        is its own box.
        """
        if not self._object_boxes:
            return ()
        ticked = [t for t in self._tick_order if self._object_boxes[t].isChecked()]
        # Anything ticked programmatically (setChecked from a test, say)
        # without passing through the slot still counts, after the ordered ones.
        ticked += [
            t
            for t, box in self._object_boxes.items()
            if box.isChecked() and t not in ticked
        ]
        return normalize_actual_targets(ticked)


__all__ = [
    "BUTTON_STYLES",
    "BigStatus",
    "CameraView",
    "FailureModeDialog",
    "InfoPanel",
    "style_button",
]

"""Editing the experiment's definition from the operator window.

The grid is wrong until it has been looked at. An operator standing at the cell
finds out that the reachable area is 4x3 rather than 3x4, or that three repeats
will not fit in the afternoon — and until now fixing that meant closing the
window, editing YAML, and paying another three-minute checkpoint load.

So: a form. :class:`SettingsDialog` edits a copy of the
:class:`~policy_abstraction.experiments.config.ExperimentConfig` and hands back a
new one; :mod:`policy_abstraction.experiments.revise` decides what applying it
means for the results already on disk; and when that decision is destructive,
:class:`ConflictDialog` puts the choice to the operator instead of guessing.

Nothing here talks to the robot, the store or the session. It is a form, a
choice, and two confirmations.
"""

from __future__ import annotations

import copy
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.grid import cell_label, parse_cell_label
from policy_abstraction.experiments.revise import Revision

# Cap the spin boxes well above anything sane rather than at what is sane: a
# grid bigger than this is a typo, and a limit the operator can hit while
# meaning it is worse than no limit at all.
MAX_ROWS_COLS = 26
MAX_REPEATS = 100


def _spin(value: int, low: int, high: int, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(int(value))
    if suffix:
        box.setSuffix(suffix)
    return box


def _dspin(
    value: float, low: float, high: float, decimals: int = 3, step: float = 0.01
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setValue(float(value))
    return box


class SettingsDialog(QDialog):
    """Edit the experiment definition: name, repeats, grid shape, limits.

    Edits a *copy*. The caller gets :meth:`result_config` and
    :meth:`result_name` only after OK, and applying them is somebody else's
    decision — this dialog never writes anything.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_name: str,
        parent: Optional[QWidget] = None,
        *,
        collected: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Experiment settings")
        self.setMinimumWidth(560)
        self._config = copy.deepcopy(config)
        self._source = config
        self._collected = collected
        self._factor_widgets: dict[str, dict] = {}

        outer = QVBoxLayout(self)

        body = QWidget()
        self._form_layout = QVBoxLayout(body)
        self._form_layout.setContentsMargins(0, 0, 8, 0)
        self._build_experiment_group()
        for factor in self._config.factors:
            self._build_factor_group(factor)
        self._form_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(self.summary)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._name.setText(experiment_name)
        self._recount()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_experiment_group(self) -> None:
        cfg = self._config
        group = QGroupBox("Experiment")
        form = QFormLayout(group)

        self._name = QLineEdit()
        self._name.setToolTip(
            "The directory under data/<run>/experiments/, and the name this "
            "experiment resumes by. Give each policy setting its own."
        )
        form.addRow("Experiment name", self._name)

        self._title = QLineEdit(cfg.name)
        self._title.setToolTip("Heading on the report. Not used for resume.")
        form.addRow("Report title", self._title)

        self._task = QLineEdit(cfg.task)
        self._task.setToolTip(
            "The prompt handed to the policy. On hardware this must be the "
            "instruction the checkpoint was trained on, verbatim."
        )
        form.addRow("Task", self._task)

        self._repeats = _spin(cfg.repeats, 1, MAX_REPEATS)
        self._repeats.setToolTip("How many rollouts to run at each cell.")
        self._repeats.valueChanged.connect(self._recount)
        form.addRow("Repetitions per cell", self._repeats)

        self._order = QComboBox()
        self._order.addItems(["systematic", "random"])
        self._order.setCurrentText(cfg.order)
        self._order.setToolTip(
            "systematic visits the cells in order — shorter object moves. "
            "random shuffles them, guarding against drift."
        )
        form.addRow("Order", self._order)

        self._grouping = QComboBox()
        self._grouping.addItems(["condition", "sweep"])
        self._grouping.setCurrentText(cfg.grouping)
        self._grouping.setToolTip(
            "condition runs every repeat of a cell back to back — the object "
            "is staged once per cell. sweep visits every cell once, then "
            "again, so a run stopped early still covers the whole grid."
        )
        form.addRow("Repeat grouping", self._grouping)

        self._seed = _spin(cfg.seed, 0, 10_000_000)
        self._seed.setToolTip(
            "Seeds the shuffle. Two policy settings with the same seed get the "
            "same order, which is what makes them comparable."
        )
        form.addRow("Seed", self._seed)

        self._max_steps = _spin(cfg.max_steps or 0, 0, 100_000, " steps")
        self._max_steps.setSpecialValueText("env default")
        form.addRow("Max steps", self._max_steps)

        self._timeout = _dspin(cfg.timeout_s or 0.0, 0.0, 3600.0, decimals=0, step=10.0)
        self._timeout.setSpecialValueText("no limit")
        self._timeout.setSuffix(" s")
        self._timeout.setToolTip(
            "Wall clock per rollout. Exceeding it records a TIMEOUT failure, "
            "then asks what the arm was doing."
        )
        form.addRow("Timeout", self._timeout)

        self._view_scale = _dspin(cfg.view_scale, 0.5, 8.0, decimals=1, step=0.5)
        form.addRow("View scale", self._view_scale)

        self._notes = QLineEdit(cfg.notes)
        form.addRow("Notes", self._notes)

        self._form_layout.addWidget(group)

    def _build_factor_group(self, factor) -> None:
        grid = getattr(factor, "grid", None)
        region = getattr(factor, "region", None)
        group = QGroupBox(f"{factor.name}  ({factor.kind})")
        form = QFormLayout(group)
        widgets: dict = {}

        if grid is not None:
            widgets["rows"] = _spin(grid.rows, 1, MAX_ROWS_COLS)
            widgets["cols"] = _spin(grid.cols, 1, MAX_ROWS_COLS)
            shape = QHBoxLayout()
            shape.addWidget(QLabel("rows"))
            shape.addWidget(widgets["rows"])
            shape.addWidget(QLabel("columns"))
            shape.addWidget(widgets["cols"])
            shape.addStretch(1)
            holder = QWidget()
            holder.setLayout(shape)
            form.addRow("Grid size", holder)

            widgets["cell_u"] = _dspin(grid.cell_size_u, 0.005, 2.0)
            widgets["cell_v"] = _dspin(grid.cell_size_v, 0.005, 2.0)
            size = QHBoxLayout()
            size.addWidget(QLabel("along u"))
            size.addWidget(widgets["cell_u"])
            size.addWidget(QLabel("along v"))
            size.addWidget(widgets["cell_v"])
            size.addStretch(1)
            holder = QWidget()
            holder.setLayout(size)
            form.addRow("Cell size (m)", holder)

            origin = QHBoxLayout()
            for axis, value in zip("xyz", grid.origin, strict=True):
                spin = _dspin(value, -5.0, 5.0, decimals=4, step=0.005)
                widgets[f"origin_{axis}"] = spin
                origin.addWidget(QLabel(axis))
                origin.addWidget(spin)
            origin.addStretch(1)
            holder = QWidget()
            holder.setLayout(origin)
            holder.setToolTip(
                "Outer corner of cell A1 in the robot base frame, metres. "
                "Check it with --calibrate before collecting."
            )
            form.addRow("Origin (m)", holder)

            widgets["marker"] = _dspin(grid.marker_height, 0.0, 1.0, decimals=4)
            form.addRow("Marker height (m)", widgets["marker"])

            covers_all = set(factor.cells) == set(grid.cells())
            widgets["cells"] = QLineEdit(
                "" if covers_all else ", ".join(grid.label(c) for c in factor.cells)
            )
            widgets["cells"].setPlaceholderText("blank = every cell")
            widgets["cells"].setToolTip(
                "Restrict the sweep to a subset, e.g. 'A1, A2, B1'. Cells "
                "outside the grid are dropped when the grid shrinks."
            )
            form.addRow("Cells", widgets["cells"])

            for key in ("rows", "cols"):
                widgets[key].valueChanged.connect(self._recount)
            widgets["cells"].textChanged.connect(self._recount)

        elif region is not None:
            self._build_region_rows(factor, region, form, widgets)

        elif hasattr(factor, "angles_deg"):
            widgets["angles"] = QLineEdit(
                ", ".join(f"{a:g}" for a in factor.angles_deg)
            )
            widgets["angles"].setToolTip("Degrees from the anchor grid's u axis.")
            widgets["angles"].textChanged.connect(self._recount)
            form.addRow("Angles (deg)", widgets["angles"])

        elif hasattr(factor, "variants"):
            widgets["variants"] = QLineEdit(", ".join(factor.variants))
            widgets["variants"].textChanged.connect(self._recount)
            form.addRow("Variants", widgets["variants"])

        else:
            form.addRow(QLabel("<i>not editable from here</i>"))

        self._factor_widgets[factor.name] = widgets
        self._form_layout.addWidget(group)

    def _build_region_rows(self, factor, region, form, widgets: dict) -> None:
        """The patch of table an object-layout factor scatters its objects over.

        Sized in metres rather than in cells, because a region has none: it is
        one rectangle, and the operator measuring it has a tape rather than a
        grid. The cell counts underneath are preserved and the cell size derived
        from them, so a region written as 2x3 cells stays 2x3.
        """
        widgets["region_size_u"] = _dspin(region.extent_u, 0.02, 4.0, step=0.01)
        widgets["region_size_v"] = _dspin(region.extent_v, 0.02, 4.0, step=0.01)
        size = QHBoxLayout()
        size.addWidget(QLabel("across (u)"))
        size.addWidget(widgets["region_size_u"])
        size.addWidget(QLabel("deep (v)"))
        size.addWidget(widgets["region_size_v"])
        size.addStretch(1)
        holder = QWidget()
        holder.setLayout(size)
        holder.setToolTip(
            "The whole rectangle the objects are scattered inside, in metres. "
            "Shrinking it below what the separation and margin need makes the "
            "layouts unsamplable, and the summary below says so."
        )
        form.addRow("Region size (m)", holder)

        origin = QHBoxLayout()
        for axis, value in zip("xyz", region.origin, strict=True):
            spin = _dspin(value, -5.0, 5.0, decimals=4, step=0.005)
            widgets[f"region_origin_{axis}"] = spin
            origin.addWidget(QLabel(axis))
            origin.addWidget(spin)
        origin.addStretch(1)
        holder = QWidget()
        holder.setLayout(origin)
        holder.setToolTip(
            "Outer corner of the region in the robot base frame, metres. "
            "Check it with --calibrate before collecting."
        )
        form.addRow("Origin (m)", holder)

        widgets["region_marker"] = _dspin(region.marker_height, 0.0, 1.0, decimals=4)
        form.addRow("Marker height (m)", widgets["region_marker"])

        widgets["layouts"] = _spin(getattr(factor, "num_layouts", 4), 1, 100)
        widgets["layouts"].setToolTip(
            "How many scenes the objects are scattered into. Every layout is "
            "swept for every object, so this multiplies the rollout count."
        )
        form.addRow("Layouts", widgets["layouts"])

        widgets["min_separation"] = _dspin(
            getattr(factor, "min_separation", 0.10), 0.0, 2.0, step=0.01
        )
        widgets["min_separation"].setToolTip(
            "Closest two objects may be placed, in metres. Far enough apart "
            "that the operator, and the policy, can tell which is which."
        )
        form.addRow("Min separation (m)", widgets["min_separation"])

        widgets["margin"] = _dspin(getattr(factor, "margin", 0.05), 0.0, 1.0, step=0.01)
        widgets["margin"].setToolTip("Keep-out band inside the region's edge, metres.")
        form.addRow("Edge margin (m)", widgets["margin"])

        for key in (
            "region_size_u",
            "region_size_v",
            "region_marker",
            "min_separation",
            "margin",
        ):
            widgets[key].valueChanged.connect(self._recount)
        widgets["layouts"].valueChanged.connect(self._recount)

    # ------------------------------------------------------------------
    # Reading the form back
    # ------------------------------------------------------------------

    def result_name(self) -> str:
        return self._name.text().strip()

    def result_config(self) -> ExperimentConfig:
        """The edited config. Raises ``ValueError`` on input it cannot parse."""
        data = copy.deepcopy(self._source.to_dict())
        data["name"] = self._title.text().strip() or self._source.name
        data["task"] = self._task.text()
        data["repeats"] = self._repeats.value()
        data["order"] = self._order.currentText()
        data["grouping"] = self._grouping.currentText()
        data["seed"] = self._seed.value()
        data["max_steps"] = self._max_steps.value() or None
        data["timeout_s"] = self._timeout.value() or None
        data["view_scale"] = self._view_scale.value()
        data["notes"] = self._notes.text()

        for entry in data.get("factors", []):
            widgets = self._factor_widgets.get(entry.get("name"), {})
            if not widgets:
                continue
            if "rows" in widgets:
                grid = entry["grid"]
                grid["rows"] = widgets["rows"].value()
                grid["cols"] = widgets["cols"].value()
                grid["cell_size_u"] = widgets["cell_u"].value()
                grid["cell_size_v"] = widgets["cell_v"].value()
                grid["origin"] = [widgets[f"origin_{a}"].value() for a in "xyz"]
                grid["marker_height"] = widgets["marker"].value()
                entry["cells"] = _parse_cells(
                    widgets["cells"].text(),
                    grid["rows"],
                    grid["cols"],
                    first_col=int(grid.get("first_col", 1)),
                )
            elif "region_size_u" in widgets:
                region = entry["region"]
                # The rectangle keeps its cell counts; only its metric size and
                # its corner move, so a region defined as several cells stays
                # several cells.
                rows = max(1, int(region.get("rows", 1)))
                cols = max(1, int(region.get("cols", 1)))
                region["cell_size_u"] = widgets["region_size_u"].value() / cols
                region["cell_size_v"] = widgets["region_size_v"].value() / rows
                region["origin"] = [
                    widgets[f"region_origin_{a}"].value() for a in "xyz"
                ]
                region["marker_height"] = widgets["region_marker"].value()
                entry["layouts"] = widgets["layouts"].value()
                entry["min_separation"] = widgets["min_separation"].value()
                entry["margin"] = widgets["margin"].value()
            elif "angles" in widgets:
                entry["angles_deg"] = _parse_numbers(
                    widgets["angles"].text(), entry["name"], "angles"
                )
            elif "variants" in widgets:
                entry["variants"] = _parse_list(
                    widgets["variants"].text(), entry["name"], "variants"
                )
        return ExperimentConfig.from_dict(data)

    # ------------------------------------------------------------------

    def _recount(self) -> None:
        """Keep the consequence of the current form in front of the operator."""
        try:
            config = self.result_config()
            total = len(config.build_plan())
        except (ValueError, KeyError) as exc:
            self.summary.setText(f"<span style='color:#c0392b'>{exc}</span>")
            return
        text = f"<b>{total}</b> rollouts planned"
        if config.timeout_s:
            text += (
                f" — up to {total * config.timeout_s / 3600:.1f} h at the "
                f"{config.timeout_s:.0f}s timeout"
            )
        if self._collected:
            text += (
                f"<br><span style='color:#b7791f'>{self._collected} rollout"
                f"{'' if self._collected == 1 else 's'} already recorded under "
                "the current settings.</span>"
            )
        self.summary.setText(text)

    def _accept_if_valid(self) -> None:
        if not self.result_name():
            QMessageBox.warning(
                self, "Name required", "The experiment needs a name to resume by."
            )
            return
        try:
            self.result_config()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Settings are not valid", str(exc))
            return
        self.accept()


def _parse_list(text: str, factor: str, what: str) -> list[str]:
    items = [part.strip() for part in text.replace("\n", ",").split(",")]
    items = [item for item in items if item]
    if not items:
        raise ValueError(f"factor {factor!r} needs at least one {what[:-1]}")
    return items


def _parse_numbers(text: str, factor: str, what: str) -> list[float]:
    out = []
    for item in _parse_list(text, factor, what):
        try:
            out.append(float(item))
        except ValueError:
            raise ValueError(f"{factor}: {item!r} is not a number") from None
    return out


def _parse_cells(
    text: str, rows: int, cols: int, *, first_col: int = 1
) -> Optional[list[str]]:
    """Parse a cell subset. Blank means every cell, expressed as ``None``.

    Cells that fall outside a shrunken grid are dropped rather than rejected:
    shrinking the grid is exactly when a subset written for the old one stops
    fitting, and refusing the edit would leave the operator to work out which
    entry offends.
    """
    if not text.strip():
        return None
    kept: list[str] = []
    for item in _parse_list(text, "grid", "cells"):
        try:
            row, col = parse_cell_label(item, first_col=first_col)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if 0 <= row < rows and 0 <= col < cols:
            kept.append(cell_label(row, col, first_col=first_col))
    if not kept:
        raise ValueError(
            f"none of those cells are inside the {rows}x{cols} grid; "
            "clear the field to sweep every cell"
        )
    return kept


class ConflictDialog(QDialog):
    """The choice when edited settings would orphan recorded rollouts.

    Two ways out, and cancel. Starting a new experiment is the default and the
    pre-selected one: it keeps both sets of results, costs nothing but a name,
    and is almost always what was meant.
    """

    def __init__(
        self,
        revision: Revision,
        suggested_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("These settings do not match the results on disk")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)

        explanation = QLabel(revision.describe())
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self._new = QRadioButton("Start a new experiment under a new name")
        self._new.setChecked(True)
        layout.addWidget(self._new)
        self._new_name = QLineEdit(suggested_name)
        self._new_name.setToolTip("The name the new experiment resumes by.")
        layout.addWidget(self._new_name)

        keep = QLabel(
            f"    The {revision.records} recorded rollout"
            f"{'' if revision.records == 1 else 's'} at {revision.root.name} "
            "stay where they are, and that experiment can still be resumed."
        )
        keep.setWordWrap(True)
        keep.setStyleSheet("color: #6a6a74;")
        layout.addWidget(keep)

        self._overwrite = QRadioButton(
            f"Overwrite — permanently delete the {revision.records} recorded "
            f"rollout{'' if revision.records == 1 else 's'} and start over"
        )
        self._overwrite.setStyleSheet("color: #c0392b;")
        layout.addWidget(self._overwrite)
        destroy = QLabel(
            "    Deletes the whole directory: records, episodes, videos and "
            "report. There is no undo, and you will be asked to confirm twice."
        )
        destroy.setWordWrap(True)
        destroy.setStyleSheet("color: #6a6a74;")
        layout.addWidget(destroy)

        self._new.toggled.connect(self._new_name.setEnabled)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def overwrite(self) -> bool:
        return self._overwrite.isChecked()

    def new_name(self) -> str:
        return self._new_name.text().strip()


def confirm_destroy(parent: Optional[QWidget], revision: Revision) -> bool:
    """Ask twice before deleting an experiment's results.

    The first question states what goes; the second asks for the experiment's
    name to be typed back. Two clicks in the same place cannot do it, which is
    the point — everything else in this window is one keystroke, and this is the
    one action that cannot be undone.
    """
    first = QMessageBox.warning(
        parent,
        "Delete these results?",
        f"This permanently deletes {revision.root}\n\n"
        f"  {revision.records} rollout record"
        f"{'' if revision.records == 1 else 's'} "
        f"({revision.scored} scored)\n"
        "  plus the episodes, videos and report in that directory\n\n"
        "There is no undo. Delete?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    if first is not QMessageBox.StandardButton.Yes:
        return False

    typed, ok = QInputDialog.getText(
        parent,
        "Confirm deletion",
        f"Type the experiment name to confirm:\n\n    {revision.name}",
    )
    if not ok or typed.strip() != revision.name:
        if ok:
            QMessageBox.information(
                parent,
                "Not deleted",
                "That did not match the experiment name, so nothing was deleted.",
            )
        return False
    return True


__all__ = [
    "ConflictDialog",
    "SettingsDialog",
    "confirm_destroy",
]

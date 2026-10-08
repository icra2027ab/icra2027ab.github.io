"""PyQt6 operator interface for running experiments at the workcell.

Imported on demand — everything below :mod:`policy_abstraction.experiments` is
headless and must stay importable without a display.
"""

from policy_abstraction.experiments.gui.actions import Action, build_actions
from policy_abstraction.experiments.gui.app import run_gui
from policy_abstraction.experiments.gui.history import HistoryDialog
from policy_abstraction.experiments.gui.main_window import ExperimentWindow
from policy_abstraction.experiments.gui.settings import (
    ConflictDialog,
    SettingsDialog,
    confirm_destroy,
)
from policy_abstraction.experiments.gui.widgets import (
    BigStatus,
    CameraView,
    FailureModeDialog,
    InfoPanel,
)

__all__ = [
    "Action",
    "BigStatus",
    "CameraView",
    "ConflictDialog",
    "ExperimentWindow",
    "FailureModeDialog",
    "HistoryDialog",
    "InfoPanel",
    "SettingsDialog",
    "build_actions",
    "confirm_destroy",
    "run_gui",
]

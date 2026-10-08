"""PyQt6 window for the live guidance demo.

Imported on demand — :mod:`policy_abstraction.demo` itself is headless and
must stay importable without a display.
"""

from policy_abstraction.demo.gui.app import run_demo_gui
from policy_abstraction.demo.gui.trends import TrendsPanel
from policy_abstraction.demo.gui.window import DemoAction, DemoWindow

__all__ = ["DemoAction", "DemoWindow", "TrendsPanel", "run_demo_gui"]

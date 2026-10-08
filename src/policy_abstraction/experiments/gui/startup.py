"""The window that stands in while the servers come up.

Loading and compiling a pi05 checkpoint costs about three minutes, and until
recently that was three minutes of a terminal with nothing moving in it, before
any window appeared. The operator could not tell a slow load from a hang, and
could not tell either from a GUI that was never going to open at all.

So this window opens *first* — before the servers, before the checkpoint — and
the startup work runs on a thread behind it. It shows the log as it happens, so
"still compiling" and "wedged on a dead socket" look different from across the
cell. When the backends are ready it is replaced by the operator window.

Closing it is a cancel, not a kill: the servers are mid-launch and a process
that walks away from them leaves them running. The window says it is cancelling
and stays up until the startup finishes and can be shut down cleanly.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from policy_abstraction.experiments.gui.widgets import BigStatus

MAX_LOG_LINES = 800
"""Enough to hold a whole start-up, bounded so a retry loop cannot grow
without limit."""


class StartupWindow(QMainWindow):
    """A status line and the start-up log, up from the first second."""

    def __init__(
        self,
        experiment: str,
        *,
        on_cancel: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_cancel = on_cancel
        self._cancelled = False
        self._finished = False

        self.setWindowTitle(f"Experiment — {experiment}")
        self.resize(900, 560)
        self.setStyleSheet(
            "QMainWindow { background: #101014; } QLabel { color: #d8d8e0; }"
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.status = BigStatus()
        self.status.set_state("STARTING UP", "#2b6cb0")
        layout.addWidget(self.status)

        self.detail = QLabel(
            "Starting the env and policy servers. Loading and compiling the "
            "checkpoint takes about three minutes; the operator window opens "
            "by itself when it is done.\n"
            "Nothing is staged and the robot is not commanded until then."
        )
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #8f8fa0; font-size: 12px;")
        layout.addWidget(self.detail)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_LINES)
        self.log.setFont(QFont("monospace", 9))
        self.log.setStyleSheet(
            "QPlainTextEdit { background: #16161c; color: #c8c8d2; "
            "border: none; border-radius: 8px; padding: 8px; }"
        )
        layout.addWidget(self.log, stretch=1)

        self.footer = QLabel("Close this window to cancel.")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet("color: #6a6a74; font-size: 11px;")
        layout.addWidget(self.footer)

    # ------------------------------------------------------------------

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def append(self, line: str) -> None:
        """Add one log line. Call from the GUI thread only."""
        for part in str(line).rstrip().splitlines() or [""]:
            self.log.appendPlainText(part)
        bar = self.log.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def set_status(self, text: str, color: str = "#2b6cb0") -> None:
        self.status.set_state(text, color)

    def set_phase(self, phase: str, detail: str = "") -> None:
        """Re-headline the window for a named stage of start-up.

        Start-up is not one wait but several, and the longest of them — the
        policy warm-up, which compiles the checkpoint on its first inference —
        looks from across the cell exactly like a hang. The headline says which
        wait this is; the log underneath keeps scrolling either way.
        """
        if self._finished:
            return
        self.set_status(phase, "#7d5bbe")
        if detail:
            self.detail.setText(detail)

    def show_failure(self, text: str) -> None:
        """Report that start-up died, and let the window be closed.

        The message stays on screen rather than the process exiting silently:
        this is the one place an operator finds out *why* nothing opened.
        """
        self._finished = True
        self.set_status("START-UP FAILED", "#c0392b")
        self.detail.setText(
            "The servers did not come up. Nothing was staged and the robot was "
            "not commanded. Close this window; the same message is in the "
            "terminal and in the experiment's evaluation.log."
        )
        self.append("")
        self.append(text)
        self.footer.setText("Close this window to exit.")

    def mark_finished(self) -> None:
        """Start-up is over, so closing no longer has to wait for anything."""
        self._finished = True

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802
        """A close during start-up is a cancel, and cancelling has to wait.

        The servers are mid-launch. Tearing the process down now would leave
        them running with nothing owning them, so the window stays up, says so,
        and the launcher shuts everything down as soon as start-up returns.
        """
        if self._finished:
            event.accept()
            return
        event.ignore()
        if self._cancelled:
            return
        self._cancelled = True
        self.set_status("CANCELLING …", "#b7791f")
        self.detail.setText(
            "Waiting for start-up to finish so the servers can be shut down "
            "cleanly, rather than left running with nothing owning them. This "
            "window closes itself."
        )
        self.footer.setText("")
        if self._on_cancel is not None:
            self._on_cancel()


__all__ = ["MAX_LOG_LINES", "StartupWindow"]

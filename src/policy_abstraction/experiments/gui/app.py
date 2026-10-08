"""Bootstrapping the operator window, and shutting it down safely.

Two things this adds over "make a window and run the event loop".

**Teardown discipline.** A GUI driving a physical arm must park the robot on
*every* exit path — the window closing, Ctrl-C in the terminal, an unhandled
exception in a Qt slot — because the alternative is a 6-axis arm left holding a
servo target while the process that was feeding it is gone.

**Deferred start-up.** :func:`run_gui` accepts a callable that *builds* the
session as well as a session already built. Given the callable, a window is on
screen within a second — :class:`~policy_abstraction.experiments.gui.startup.StartupWindow`,
showing the log — and the servers and the three-minute checkpoint load run on a
worker thread behind it, with the operator window taking over when they are
ready. The alternative, and what this replaces, is three minutes of a terminal
in which a slow load and a hang look identical.
"""

from __future__ import annotations

import dataclasses
import signal
import sys
import threading
import traceback
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from loguru import logger

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.session import ExperimentSession

if TYPE_CHECKING:  # PyQt is imported lazily, so this module stays importable
    from PyQt6.QtWidgets import QWidget  # on a machine with no display.

SessionSource = Union[ExperimentSession, Callable[[], ExperimentSession]]


@dataclasses.dataclass
class _Launch:
    """What the signal handler and the teardown need, as start-up moves it.

    The window and the session both arrive late in the deferred case, and both
    handlers can fire before they do — so they are read from here rather than
    captured, and ``None`` is a state both handlers cope with.
    """

    window: Optional["QWidget"] = None
    session: Optional[ExperimentSession] = None
    cancelled: bool = False
    failed: bool = False
    bridge: Any = None
    """The signal carrier the start-up thread emits through. Held only to keep
    it alive: a garbage-collected QObject drops its connections, and the worker
    would then be emitting into nothing."""


def run_gui(
    session: SessionSource,
    config: ExperimentConfig,
    *,
    on_report: Optional[Callable[[], dict]] = None,
    refresh_hz: float = 15.0,
    reviser=None,
) -> int:
    """Show the window and run until the operator closes it.

    Args:
        session: A live session, or a zero-argument callable that builds one.
            The callable is what lets the window open *before* the servers: it
            runs on a worker thread behind a start-up screen, and the operator
            window replaces that screen when it returns. Anything it raises is
            shown in the window rather than only in the terminal.
        reviser: Lets the operator edit the experiment from the window (see
            :mod:`policy_abstraction.experiments.revise`). Omitted, the
            settings action is simply not offered.

    Returns the Qt exit code, or 1 if start-up failed. The session is shut down
    — rollout stopped, arm parked, servers terminated — before this returns,
    whatever ended the loop.
    """
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(f"experiment:{config.name}")

    launch = _Launch()
    window_kwargs = dict(refresh_hz=refresh_hz, on_report=on_report, reviser=reviser)
    # isinstance rather than callable(): it narrows the union in both
    # directions, and a session is never callable anyway.
    if isinstance(session, ExperimentSession):
        launch.session = session
        launch.window = _open_operator_window(session, config, window_kwargs)
    else:
        _begin_deferred(app, session, config, launch, window_kwargs)

    # Qt's event loop swallows SIGINT: without this, Ctrl-C in the terminal does
    # nothing until the window is closed by hand, which is not what anyone
    # reaching for Ctrl-C next to a moving robot expects. The idle timer gives
    # the interpreter a chance to run the handler between events.
    def _sigint(_signum, _frame):
        logger.warning("SIGINT — closing the experiment window …")
        window = launch.window
        if window is not None:
            # During start-up this is a cancel that the window defers until the
            # servers exist to be shut down; closing it does not quit yet.
            window.close()
        if launch.session is not None or launch.failed:
            app.quit()

    previous = signal.signal(signal.SIGINT, _sigint)
    keepalive = QTimer()
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)

    try:
        code = app.exec()
    finally:
        signal.signal(signal.SIGINT, previous)
        # Unconditional: an exception escaping the event loop must still park
        # the arm and stop the servers.
        if launch.session is not None:
            launch.session.shutdown()
    return 1 if launch.failed else int(code)


def _open_operator_window(
    session: ExperimentSession, config: ExperimentConfig, window_kwargs: dict
):
    from policy_abstraction.experiments.gui.main_window import ExperimentWindow

    window = ExperimentWindow(session, config, **window_kwargs)
    window.show()
    return window


def _begin_deferred(
    app,
    build_session: Callable[[], ExperimentSession],
    config: ExperimentConfig,
    launch: _Launch,
    window_kwargs: dict,
) -> None:
    """Put the start-up window up now and build the session behind it."""
    from PyQt6.QtCore import QObject, Qt, pyqtSignal

    from policy_abstraction.experiments.gui.startup import StartupWindow

    class _Bridge(QObject):
        """Marshals the worker thread's news onto the GUI thread.

        Qt widgets may only be touched from the thread that made them, and a
        signal emitted from another thread is the one supported way across.
        """

        line = pyqtSignal(str)
        phase = pyqtSignal(str, str)
        ready = pyqtSignal(object)
        failed = pyqtSignal(str)

    startup = StartupWindow(config.name)
    launch.window = startup
    startup.show()

    bridge = _Bridge()
    launch.bridge = bridge  # see the field's docstring
    queued = Qt.ConnectionType.QueuedConnection

    def _on_line(text: str) -> None:
        startup.append(text)

    def _on_ready(session) -> None:
        logger.remove(sink_id)
        if startup.cancelled:
            # The operator gave up while the servers were still coming up. They
            # exist now, so shut them down properly rather than walking away.
            startup.append("Start-up finished; shutting the servers down.")
            launch.cancelled = True
            launch.session = session
            startup.mark_finished()
            startup.close()
            app.quit()
            return
        launch.session = session
        launch.window = _open_operator_window(session, config, window_kwargs)
        startup.mark_finished()
        startup.close()

    def _on_failed(text: str) -> None:
        logger.remove(sink_id)
        launch.failed = True
        startup.show_failure(text)

    def _on_phase(phase: str, detail: str) -> None:
        startup.set_phase(phase, detail)

    bridge.line.connect(_on_line, queued)  # pyright: ignore[reportCallIssue]
    bridge.phase.connect(_on_phase, queued)  # pyright: ignore[reportCallIssue]
    bridge.ready.connect(_on_ready, queued)  # pyright: ignore[reportCallIssue]
    bridge.failed.connect(_on_failed, queued)  # pyright: ignore[reportCallIssue]

    # The progress the window shows *is* the start-up log — there is no second
    # channel to invent, and everything that already logs its way through a
    # server launch is exactly what the operator wants to see.
    def _sink(message) -> None:
        # A record may additionally name the *stage* of start-up it belongs to,
        # bound at the call site as `startup_phase`. That is how a long silent
        # step re-headlines the window instead of being one more log line
        # scrolling past — see ExperimentSession.warm_up_policy.
        record = message.record
        bridge.line.emit(record["message"])
        phase = record["extra"].get("startup_phase")
        if phase:
            bridge.phase.emit(
                str(phase), str(record["extra"].get("startup_detail", ""))
            )

    sink_id = logger.add(
        _sink,
        level="INFO",
        format="{message}",
        enqueue=False,
    )

    def _work() -> None:
        try:
            session = build_session()
        except BaseException as exc:  # noqa: BLE001 — reported, never swallowed
            logger.error(f"Start-up failed: {exc}")
            logger.debug(traceback.format_exc())
            bridge.failed.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )
            return
        bridge.ready.emit(session)

    threading.Thread(target=_work, name="backend-startup", daemon=True).start()


__all__ = ["SessionSource", "run_gui"]

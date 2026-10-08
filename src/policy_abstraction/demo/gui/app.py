"""Bootstrapping the demo window, and shutting it down safely.

Two things this adds over "make a window and run the event loop", both
borrowed from the experiment GUI because they are about the arm and not the
experiment.

**Teardown discipline.** The session is shut down — rollout stopped, arm
parked, servers terminated — on *every* exit path: the window closing, Ctrl-C
in the terminal, an exception escaping a Qt slot. The alternative is a 6-axis
arm left holding a servo target while the process feeding it is gone.

**Deferred start-up.** Given a callable that *builds* the session, a window is
on screen within a second — the same
:class:`~policy_abstraction.experiments.gui.startup.StartupWindow`, streaming
the log — while the servers and the checkpoint load run on a worker thread
behind it. The demo window takes over when they are ready, so a slow compile
and a hang look different from across the cell.

**Dark palette.** The application gets a dark Fusion palette before any
window exists — the window's own stylesheets set their colours explicitly,
but the dialogs, the scroll bars and anything else Qt draws from the palette
would otherwise come up in the desktop's light theme beside them.
"""

from __future__ import annotations

import dataclasses
import signal
import sys
import threading
import traceback
from typing import Any, Callable, Optional, Union

from loguru import logger
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QWidget

from policy_abstraction.demo.gui.window import DemoWindow
from policy_abstraction.demo.session import DemoSession
from policy_abstraction.experiments.gui.startup import StartupWindow

SessionSource = Union[DemoSession, Callable[[], DemoSession]]


@dataclasses.dataclass
class _Launch:
    """What the signal handler and the teardown need, as start-up moves it.

    In the deferred case both the window and the session arrive late, and
    both handlers can fire before they do — so they read from here, and
    ``None`` is a state both cope with.
    """

    window: Optional[QWidget] = None
    session: Optional[DemoSession] = None
    cancelled: bool = False
    failed: bool = False
    bridge: Any = None
    """Held only to keep it alive: a garbage-collected QObject drops its
    connections, and the worker would then be emitting into nothing."""


class _StartupBridge(QObject):
    """Marshals the start-up thread's news onto the GUI thread — the one
    supported way across, since widgets belong to the thread that made them."""

    line = pyqtSignal(str)
    phase = pyqtSignal(str, str)
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)


def run_demo_gui(
    session_source: SessionSource,
    *,
    title: str = "Guidance demo",
    refresh_hz: float = 20.0,
    show_trends: bool = True,
    fullscreen: bool = True,
) -> int:
    """Show the demo window and run until the operator closes it.

    *session_source* is a live :class:`DemoSession`, or a zero-argument
    callable that builds one — on a worker thread, behind the start-up
    screen, with anything it raises shown in that window rather than only in
    the terminal. *fullscreen* opens the window maximised (F11 in the window
    toggles true full screen); ``False`` opens it at its default size.
    Returns the Qt exit code, or 1 if start-up failed. The session is shut
    down before this returns, whatever ended the loop.
    """
    if QApplication.instance() is None:
        _prepare_web_engine()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(title)
    install_dark_palette(app)
    launch = _Launch()
    opts = _WindowOptions(title, refresh_hz, show_trends, fullscreen)
    if isinstance(session_source, DemoSession):
        launch.session = session_source
        launch.window = _open_window(session_source, opts)
    else:
        _begin_deferred(app, session_source, opts, launch)

    # Qt's event loop swallows SIGINT: without this, Ctrl-C does nothing until
    # the window is closed by hand. The idle timer gives the interpreter a
    # chance to run the handler between events.
    def _sigint(_signum, _frame):
        logger.warning("SIGINT — closing the demo window …")
        if launch.window is not None:
            # During start-up this is a cancel the window defers until the
            # servers exist to be shut down; closing it does not quit yet.
            launch.window.close()
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


@dataclasses.dataclass(frozen=True)
class _WindowOptions:
    """What ``DemoWindow`` is opened with, carried through the deferred start."""

    title: str
    refresh_hz: float
    show_trends: bool
    fullscreen: bool


def _prepare_web_engine() -> None:
    """What QtWebEngine needs done *before* the QApplication exists.

    The 3-D panel embeds the Rerun web viewer in a ``QWebEngineView``, and
    Chromium's GL contexts must be shareable across threads — an attribute
    that is read when the application is constructed, not after. Importing
    the module first is the same story: QtWebEngine initialises against the
    application, so it must be loaded before one exists. Neither is fatal
    when missing; the window then shows a placeholder instead of the scene.
    """
    if hasattr(Qt.ApplicationAttribute, "AA_ShareOpenGLContexts"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401 — imported for its side effect
    except Exception as exc:  # noqa: BLE001 — optional wheel
        logger.warning(
            f"QtWebEngine is unavailable; the 3-D panel will be empty: {exc}"
        )


PALETTE_WINDOW = "#101014"
PALETTE_ALTERNATE = "#16161c"
PALETTE_TEXT = "#d8d8e0"
PALETTE_BUTTON = "#3a3a44"
PALETTE_HIGHLIGHT = "#2b6cb0"


def dark_palette() -> QPalette:
    """The window's colours as a ``QPalette``: the background the stylesheets
    use, the text they use, a blue highlight."""
    palette = QPalette()
    window, alternate = QColor(PALETTE_WINDOW), QColor(PALETTE_ALTERNATE)
    text, button = QColor(PALETTE_TEXT), QColor(PALETTE_BUTTON)
    highlight = QColor(PALETTE_HIGHLIGHT)
    role = QPalette.ColorRole
    palette.setColor(role.Window, window)
    palette.setColor(role.Base, window)
    palette.setColor(role.AlternateBase, alternate)
    palette.setColor(role.ToolTipBase, alternate)
    palette.setColor(role.ToolTipText, text)
    palette.setColor(role.Text, text)
    palette.setColor(role.WindowText, text)
    palette.setColor(role.ButtonText, text)
    palette.setColor(role.Button, button)
    palette.setColor(role.Highlight, highlight)
    palette.setColor(role.HighlightedText, QColor("white"))
    palette.setColor(role.Link, highlight)
    disabled = QPalette.ColorGroup.Disabled
    dim = QColor(PALETTE_TEXT)
    dim.setAlpha(110)
    for r in (role.Text, role.WindowText, role.ButtonText):
        palette.setColor(disabled, r, dim)
    return palette


def install_dark_palette(app) -> None:
    """Fusion with :func:`dark_palette`, on *app*, before any window.

    Fusion, because the platform style may paint from its own theme and
    ignore the palette. The window's stylesheets set every colour they care
    about explicitly, so they are unchanged by this; what changes is what
    they leave to Qt — dialogs, scroll bars, tooltips, the Rerun viewer's
    host widget — which now match them rather than the desktop.
    """
    if hasattr(app, "setStyle"):
        app.setStyle("Fusion")
    app.setPalette(dark_palette())


def _open_window(session: DemoSession, opts: _WindowOptions) -> DemoWindow:
    window = DemoWindow(
        session,
        refresh_hz=opts.refresh_hz,
        title=opts.title,
        show_trends=opts.show_trends,
        fullscreen=opts.fullscreen,
    )
    if opts.fullscreen:
        window.showMaximized()
    else:
        window.show()
    return window


def _begin_deferred(
    app,
    build_session: Callable[[], DemoSession],
    opts: _WindowOptions,
    launch: _Launch,
) -> None:
    """Put the start-up window up now and build the session behind it."""
    startup = StartupWindow(opts.title)
    launch.window = startup
    startup.show()
    bridge = _StartupBridge()
    launch.bridge = bridge
    queued = Qt.ConnectionType.QueuedConnection

    def _on_ready(session) -> None:
        logger.remove(sink_id)
        launch.session = session
        if startup.cancelled:
            # The operator gave up while the servers were coming up. They
            # exist now, so run_demo_gui's teardown shuts them down properly.
            startup.append("Start-up finished; shutting the servers down.")
            launch.cancelled = True
            startup.mark_finished()
            startup.close()
            app.quit()
            return
        launch.window = _open_window(session, opts)
        startup.mark_finished()
        startup.close()

    def _on_failed(text: str) -> None:
        logger.remove(sink_id)
        launch.failed = True
        startup.show_failure(text)

    bridge.line.connect(startup.append, queued)  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]
    bridge.phase.connect(startup.set_phase, queued)  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]
    bridge.ready.connect(_on_ready, queued)  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]
    bridge.failed.connect(_on_failed, queued)  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]

    # The progress the window shows *is* the start-up log. A record bound
    # with `startup_phase` (see DemoSession.warm_up_policy) re-headlines the
    # window instead of scrolling past as one more line.
    def _sink(message) -> None:
        record = message.record
        bridge.line.emit(record["message"])
        phase = record["extra"].get("startup_phase")
        if phase:
            bridge.phase.emit(
                str(phase), str(record["extra"].get("startup_detail", ""))
            )

    sink_id = logger.add(_sink, level="INFO", format="{message}", enqueue=False)

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

    threading.Thread(target=_work, name="demo-startup", daemon=True).start()


__all__ = ["SessionSource", "dark_palette", "install_dark_palette", "run_demo_gui"]

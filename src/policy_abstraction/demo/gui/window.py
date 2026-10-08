"""The demo window: the exterior view with the pairs drawn on, the switch, the plot.

What it is for: someone standing next to the cell presses SPACE, the arm runs
the policy, and while it runs they flip guidance on and off with G. The
window has to make the difference visible — the banner says which mode the
executing chunk came from, the overlay draws the tracked objects and the
distance pairs when guidance is on, the 3-D panel beside the camera shows
the corrected chunk in the scene, and the bottom row — the wrist camera, the
retrieved demo's frame, the trend plots and the DISTANCE PAIRS panel that is
their legend — shows the live distances against the retrieved demo's with the
guided stretches shaded. When the chunk on the arm is unguided, everything
that only guidance feeds — the demo frame, the plots, the pairs, the
detector's masks, the GUIDANCE numbers — sits under a :class:`Dimmer`, so
the switch's position is visible from across the room. Each rollout starts
those panes fresh: the previous rollout's demo frame and masks are cleared
before the new one's arrive.

Threading, once: the rollout runs on a worker thread inside
:class:`~policy_abstraction.demo.session.DemoSession`, and Qt widgets may only
be touched from the GUI thread. Nothing here is called from the worker. The
window polls the session's immutable snapshots on a timer, and the session's
state-change callback crosses threads as a queued Qt signal (``_SessionBridge``)
— the one supported way in.

Every action is declared once as a :class:`DemoAction` and rendered as both a
button and a ``QAction`` shortcut, so the two cannot drift: a greyed-out
button has a hotkey that does nothing, and vice versa.

Sizing, once: nothing in this window may grow it. The window is shown
maximised on a cell display of about 1900x1000, and a layout that asks for
more than that pushes the window off the screen. So every widget that shows
content — the camera views, the status strip, the info panels' bodies — has
an ``Ignored`` horizontal size policy and a small explicit minimum, which
tells the layout to size them and never the other way round. The side column
has a fixed width and scrolls when it is taller than the window.

The bottom row is tall on purpose — it starts at about 42 % of the height
and never drops under :data:`_BOTTOM_MIN_HEIGHT` — because the trend plots
are what the audience reads, and at a strip's height their traces cannot be
told apart. The two small camera views in it scale with that height.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

from loguru import logger
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QKeySequence, QPainter
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from policy_abstraction.demo.gui.trends import TrendsPanel
from policy_abstraction.demo.overlay import (
    COLOR_GUIDED,
    COLOR_IDLE,
    COLOR_STOPPING,
    COLOR_UNAVAILABLE,
    COLOR_UNGUIDED,
    render_live_view,
)
from policy_abstraction.demo.session import DemoSession, DemoState, LiveView
from policy_abstraction.experiments.gui.widgets import (
    BigStatus,
    CameraPanel,
    CameraView,
    InfoPanel,
    style_button,
)

_TITLE_STYLE = "color: #8f8fa0; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
"""The small caps title above a view — the same as ``CameraPanel``'s."""

# The big status strip, per mode.
STATUS_GUIDED = "#27923f"
STATUS_UNGUIDED = "#2b6cb0"
STATUS_STOPPING = "#b7791f"
STATUS_IDLE = "#4a4a56"
STATUS_WARMUP = "#7d5bbe"

_SIDE_WIDTH = 340
"""The side column's width. Fixed: the camera and the 3-D panel take
whatever the window has left, the column never bargains for more."""
_SIDE_CAMERA_HEIGHT = 150
_VIEW_MIN = (160, 120)
"""The exterior view's minimum. Small on purpose — the layout sizes the
view, and the frame is scaled to fit it, never the reverse."""
_WEB_MIN = (320, 240)
_WINDOW_MIN = (960, 600)
_BOTTOM_MIN_HEIGHT = 260
"""The bottom row's floor: below it the two trend plots are strips."""
_BOTTOM_SHARE = 0.42
"""The bottom row's initial share of the height under the status strip."""
_BOTTOM_SPLIT = (20, 20, 45, 15)
"""Wrist view, retrieved demo, trends, pairs — initial shares of the row's
width, %. The pairs pane is clamped to :data:`_PAIRS_WIDTH` either way."""
_PAIRS_WIDTH = (240, 280, 320)
"""The DISTANCE PAIRS pane's minimum, preferred and maximum width."""
_HANDLE_WIDTH_BOTTOM = 12
_HANDLE_WIDTH_CENTRE = 10
"""Splitter handles, painted in the window's background: the gap between
the wrist image, the demo frame and the plots."""
DEMO_PLACEHOLDER = "retrieved demo frame appears once guidance is set up"
PERCEPTION_PLACEHOLDER = "waiting for the detector …"
PAIRS_PLACEHOLDER = (
    "<i>pairs appear once the rollout has retrieved its demo and "
    "perception has found the objects</i>"
)
DIM_CAPTION = "guidance off"
DIM_FILL = QColor(16, 16, 20, 165)

NO_VIEWER_TEXT = (
    "3-D view: Rerun is not serving in-process (launch with --viz.serve, the default)"
)

_PREFER_DARK_JS = """
(function () {
  const orig = window.matchMedia ? window.matchMedia.bind(window) : null;
  const fake = (q, matches) => ({ matches, media: q, onchange: null,
    addListener() {}, removeListener() {}, addEventListener() {},
    removeEventListener() {}, dispatchEvent() { return false; } });
  window.matchMedia = function (q) {
    if (typeof q === 'string' && q.indexOf('prefers-color-scheme') >= 0) {
      return fake(q, q.indexOf('dark') >= 0);
    }
    return orig ? orig(q) : fake(q, false);
  };
})();
"""
"""Answer ``prefers-color-scheme: dark`` as true inside the embedded browser.

The Rerun web viewer draws its whole UI on a canvas and picks light or dark
from ``window.matchMedia('(prefers-color-scheme: dark)')`` — there is no
CSS to override. QtWebEngine's Chromium reports *light* for that query
whatever the Qt palette says (verified on the cell machine: a dark
``QPalette`` restyles the Fusion widgets and leaves the viewer white). So
the query is answered at document creation, before the viewer's script
runs, and the scene comes up in the viewer's dark theme like the rest of
the window.
"""


@dataclasses.dataclass(frozen=True)
class DemoAction:
    """One operator action: a button and a hotkey from the same definition."""

    id: str
    label: str
    shortcut: str
    """Qt key sequence, e.g. ``"Space"``, ``"G"``, ``"Esc"``."""
    tooltip: str
    callback: Callable[[], None]
    enabled_when: Callable[[DemoSession], bool]
    style: str = "default"
    """``default`` / ``primary`` / ``success`` / ``danger``, see ``style_button``."""
    label_when: Optional[Callable[[DemoSession], str]] = None
    """A label that follows the session — ``Guidance: ON`` — refreshed each tick."""
    style_when: Optional[Callable[[DemoSession], str]] = None

    def current_label(self, session: DemoSession) -> str:
        return self.label_when(session) if self.label_when else self.label

    def current_style(self, session: DemoSession) -> str:
        return self.style_when(session) if self.style_when else self.style

    def button_text(self, session: DemoSession) -> str:
        # "&" is Qt's mnemonic marker on a button: "Stop & home" would show
        # as "Stop _home". Doubling it shows the character.
        label = self.current_label(session).replace("&", "&&")
        return f"{label}  [{self.shortcut}]"


class TaskEdit(QLineEdit):
    """The instruction over the picture: a line edit that owns the keyboard
    while it has focus.

    Every hotkey in the window is a ``QAction`` shortcut, and Qt offers each
    key to the shortcut map before the focused widget unless that widget
    claims it. Typing "Stop the arm" into a field that lets S stop the arm
    would be a disaster, so this claims every key while editing: Enter
    commits, Escape reverts, and both hand the keyboard back.
    """

    committed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._before = ""
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip(
            "The instruction the policy is conditioned on. Click to edit; "
            "Enter applies it (at the next replan if a rollout is running), "
            "Escape reverts."
        )
        self.setStyleSheet(
            "QLineEdit { background: rgba(16, 16, 20, 150); color: #f4f4f8; "
            "border: 1px solid rgba(255, 255, 255, 40); border-radius: 10px; "
            "padding: 8px 18px; font-size: 26px; font-weight: 700; }"
            "QLineEdit:focus { background: rgba(16, 16, 20, 215); "
            "border: 1px solid #2b6cb0; }"
        )

    def event(self, e) -> bool:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        if e.type() == QEvent.Type.ShortcutOverride and self.hasFocus():
            e.accept()  # ours, not the window's hotkeys
            return True
        return super().event(e)

    def focusInEvent(self, e) -> None:  # type: ignore[override]  # noqa: N802  # ty: ignore[invalid-method-override]
        self._before = self.text()
        super().focusInEvent(e)

    def keyPressEvent(self, e) -> None:  # type: ignore[override]  # noqa: N802  # ty: ignore[invalid-method-override]
        key = e.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.committed.emit(self.text())
            self.clearFocus()
            return
        if key == Qt.Key.Key_Escape:
            self.setText(self._before)
            self.clearFocus()
            return
        super().keyPressEvent(e)


class _OverlaidView(QWidget):
    """The camera view with the instruction laid over its top edge.

    The picture fills the widget; the field floats over it and follows the
    width, so the layout still sizes only the picture.
    """

    _MARGIN = 14

    def __init__(self, view: CameraView, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.view = view
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view)
        self.task_edit = TaskEdit(self)
        self.task_edit.raise_()
        self.setMinimumSize(view.minimumSize())
        self.setSizePolicy(view.sizePolicy())

    def resizeEvent(self, e) -> None:  # type: ignore[override]  # noqa: N802  # ty: ignore[invalid-method-override]
        super().resizeEvent(e)
        self.place_overlay()

    def place_overlay(self) -> None:
        """Lay the field across the top of the *picture*: the label is wider
        than a 4:3 frame and letterboxes it, and a bar across the empty
        margins reads as a second widget rather than a caption."""
        m = self._MARGIN
        h = self.task_edit.sizeHint().height()
        pix = self.view.pixmap()
        if pix is not None and not pix.isNull():
            pw, ph = pix.width(), pix.height()
            x0 = (self.view.width() - pw) // 2 + self.view.x()
            y0 = (self.view.height() - ph) // 2 + self.view.y()
        else:
            pw, x0, y0 = self.width(), 0, 0
        self.task_edit.setGeometry(x0 + m, y0 + m, max(pw - 2 * m, 50), h)


class _OptionsPanel(QFrame):
    """A titled side-column block holding checkboxes, styled like InfoPanel."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #1a1a20; border-radius: 8px; }"
            "QLabel { color: #d8d8e0; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setStyleSheet(
            "color: #8f8fa0; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
        )
        layout.addWidget(title_label)
        self.body_layout = layout


class _SessionBridge(QObject):
    """Marshals the session's worker-thread callback onto the GUI thread.

    A Qt signal emitted from another thread is delivered as a queued
    connection, which is the one supported way to get from the rollout
    thread into the window. Everything else here runs on the GUI thread.
    """

    changed = pyqtSignal()


class Dimmer(QWidget):
    """A translucent dark sheet with a caption, laid over a widget that the
    current mode leaves idle.

    A child of its target, transparent to the mouse, kept on top and resized
    with it through an event filter — so the target's own layout never knows
    it is there. :meth:`set_active` shows and hides it; the window decides
    when, see :attr:`DemoWindow.dimmed`.
    """

    def __init__(self, target: QWidget, caption: str = DIM_CAPTION) -> None:
        super().__init__(target)
        self._target = target
        self._caption = caption
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setGeometry(target.rect())
        target.installEventFilter(self)
        self.hide()

    @property
    def target(self) -> QWidget:
        return self._target

    @property
    def caption(self) -> str:
        return self._caption

    def set_active(self, active: bool) -> None:
        if active:
            self.setGeometry(self._target.rect())
            self.raise_()
            self.show()
        else:
            self.hide()

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]  # noqa: N802  # ty: ignore[invalid-method-override]
        if obj is self._target and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self.setGeometry(self._target.rect())
            if not self.isHidden():
                self.raise_()
        return False

    def paintEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802  # ty: ignore[invalid-method-override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(DIM_FILL)
        painter.drawRoundedRect(self.rect(), 8, 8)
        font = QFont(self.font())
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 210, 220))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._caption)
        painter.end()


def _make_web_view() -> QWidget:
    """Build the embedded browser the 3-D panel shows the Rerun web viewer in.

    Imported here and not at module level: QtWebEngine is a separate wheel,
    and a machine without it — or an offscreen test run where Chromium
    cannot start — must still get the rest of the window. Raises whatever
    the import or the construction raises; the caller shows the reason.
    """
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    return QWebEngineView()


def _prefer_dark(view: QWidget) -> None:
    """Install :data:`_PREFER_DARK_JS` on *view*'s page, to run in every
    frame at document creation. A view without a page — or a QtWebEngine
    without scripts — keeps its light theme; the scene matters more."""
    try:
        from PyQt6.QtWebEngineCore import QWebEngineScript

        script = QWebEngineScript()
        script.setName("prefer-dark")
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setRunsOnSubFrames(True)
        script.setSourceCode(_PREFER_DARK_JS)
        page = view.page()  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        page.scripts().insert(script)
    except Exception as exc:  # noqa: BLE001 — a light viewer beats no viewer
        logger.warning(f"3-D panel: could not ask the viewer for its dark theme: {exc}")


class DemoWindow(QMainWindow):
    """Live exterior view, the 3-D scene, the switch, the side panels, the plot."""

    def __init__(
        self,
        session: DemoSession,
        *,
        refresh_hz: float = 20.0,
        title: str = "Guidance demo",
        show_trends: bool = True,
        fullscreen: bool = True,
    ) -> None:
        super().__init__()
        self.session = session
        self._quitting = False
        self._last_frame_stamp = -1.0
        self._dirty = True
        """The frame has not changed but the picture has (a toggle flipped)."""
        self._trends_seen = None
        self._trend_count = 0
        self._start_maximised = fullscreen
        """Where F11 returns to: maximised (the launcher's default) or normal."""
        self._web: Optional[QWidget] = None
        """The embedded browser, once the viewer URL is known and it could be built."""
        self._web_failed = False
        """The browser could not be built; the placeholder says why. Not retried."""
        self._viewer_url_loaded: Optional[str] = None
        self._balanced_once = False
        self._was_running = False
        """Whether the last repaint saw a rollout: the IDLE→RUNNING edge is
        where the previous rollout's pictures are cleared."""
        self._dimmed = False

        self.setWindowTitle(title)
        self.setMinimumSize(*_WINDOW_MIN)
        self.resize(1400, 960)
        self.setStyleSheet(
            "QMainWindow { background: #101014; } QLabel { color: #d8d8e0; }"
            "QSplitter::handle { background: #101014; }"
            "QScrollArea { background: transparent; border: none; }"
        )

        self._build_ui()
        self.bottom_row.setVisible(show_trends)
        self._install_actions()

        self._bridge = _SessionBridge()
        # Queued: the emit comes from the rollout thread, and a direct
        # connection would run _refresh — and so every widget call in it
        # — off the GUI thread. PyQt6's stubs only declare connect(slot).
        self._bridge.changed.connect(
            self._refresh,
            Qt.ConnectionType.QueuedConnection,  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]
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
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # The status strip wraps rather than widening the window: a QLabel
        # without word wrap makes its text width its minimum width, and the
        # RUNNING line at 19px bold is wider than the camera view.
        self.status = BigStatus()
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(200)
        self.status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        root.addWidget(self.status)

        # Centre: the exterior view, the 3-D scene, and the side column.
        self.view = CameraView(minimum_size=_VIEW_MIN)
        _size_by_layout(self.view, _VIEW_MIN)
        # The instruction, big, over the top of the picture — and editable:
        # click it, type, Enter. It is the prompt the policy is conditioned
        # on, so changing it is part of the demonstration.
        self.view_box = _OverlaidView(self.view)
        self.task_edit = self.view_box.task_edit
        self.task_edit.setText(self.session.task)
        self.task_edit.committed.connect(self._task_committed)

        self.viewer_panel = self._build_viewer_panel()

        side = QWidget()
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        self.perception_panel = CameraPanel("PERCEPTION", height=_SIDE_CAMERA_HEIGHT)
        self.perception_panel.view.setText(PERCEPTION_PLACEHOLDER)
        self.guidance_panel = InfoPanel("GUIDANCE")
        self.help_panel = InfoPanel("KEYS")
        # The pixmap must not size the label: a pixmap scaled to the label's
        # own size would otherwise become its minimum.
        _size_by_layout(
            self.perception_panel.view, (0, _SIDE_CAMERA_HEIGHT), vertical="fixed"
        )
        self.perception_panel.view.setWordWrap(True)
        column.addWidget(self.perception_panel)
        for panel in (self.guidance_panel, self.help_panel):
            _body_by_layout(panel)
            column.addWidget(panel)
        self.options_panel = _OptionsPanel("OVERLAY OPTIONS")
        column.addWidget(self.options_panel)
        column.addStretch(1)

        # Fixed width, and it scrolls: the column's stack of panels is taller
        # than a small display, and a plain widget would make the window so.
        self.side_scroll = QScrollArea()
        self.side_scroll.setWidget(side)
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFixedWidth(_SIDE_WIDTH)
        self.side_scroll.setMinimumHeight(120)
        self.side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.side_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        side.setStyleSheet("background: transparent;")

        self.centre_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.centre_splitter.setHandleWidth(_HANDLE_WIDTH_CENTRE)
        self.centre_splitter.addWidget(self.view_box)
        self.centre_splitter.addWidget(self.viewer_panel)
        self.centre_splitter.addWidget(self.side_scroll)
        self.centre_splitter.setStretchFactor(0, 3)
        self.centre_splitter.setStretchFactor(1, 2)
        self.centre_splitter.setStretchFactor(2, 0)
        self.centre_splitter.setCollapsible(0, False)
        self.centre_splitter.setCollapsible(2, False)

        # Bottom row, the full width: the wrist camera, the retrieved demo's
        # frame, the trend plots, and the DISTANCE PAIRS panel that is their
        # legend. The T key shows and hides the whole row. The two views are
        # plain CameraViews sized by the layout, so they follow the row's
        # height and keep their aspect.
        self.wrist_view = CameraView(
            minimum_size=_VIEW_MIN, placeholder="waiting for the wrist camera …"
        )
        self.wrist_box = _titled_view("WRIST", self.wrist_view)
        self.wrist_box.setVisible(self.session.wrist_camera is not None)
        self.demo_view = CameraView(
            minimum_size=_VIEW_MIN, placeholder=DEMO_PLACEHOLDER
        )
        self.demo_box = _titled_view("RETRIEVED DEMO", self.demo_view)
        self.trends = TrendsPanel()
        # The pairs pane: the trend plot's legend, in the plot's colours,
        # with the numbers. Clamped in width so a long label wraps inside
        # it and the plots keep the rest of the row.
        self.pairs_panel = InfoPanel("DISTANCE PAIRS")
        _body_by_layout(self.pairs_panel)
        self.pairs_panel._body.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.pairs_panel._layout.addStretch(1)
        self.pairs_panel.setMinimumWidth(_PAIRS_WIDTH[0])
        self.pairs_panel.setMaximumWidth(_PAIRS_WIDTH[2])
        self.pairs_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored
        )

        self.bottom_row = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_row.setHandleWidth(_HANDLE_WIDTH_BOTTOM)
        self.bottom_row.addWidget(self.wrist_box)
        self.bottom_row.addWidget(self.demo_box)
        self.bottom_row.addWidget(self.trends)
        self.bottom_row.addWidget(self.pairs_panel)
        for i, share in enumerate(_BOTTOM_SPLIT):
            self.bottom_row.setStretchFactor(i, share)
            self.bottom_row.setCollapsible(i, False)
        self.bottom_row.setMinimumHeight(_BOTTOM_MIN_HEIGHT)

        # What only guidance feeds goes dark when guidance is off.
        self.dimmers: tuple[Dimmer, ...] = tuple(
            Dimmer(target)
            for target in (
                self.demo_box,
                self.trends,
                self.pairs_panel,
                self.perception_panel,
                self.guidance_panel,
            )
        )

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.centre_splitter)
        self.main_splitter.addWidget(self.bottom_row)
        self.main_splitter.setStretchFactor(0, 58)
        self.main_splitter.setStretchFactor(1, 42)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        root.addWidget(self.main_splitter, stretch=1)

        self.button_row = QHBoxLayout()
        self.button_row.setSpacing(8)
        root.addLayout(self.button_row)

    def _build_viewer_panel(self) -> QStackedWidget:
        """The 3-D panel: a placeholder until the viewer URL is known, then
        the embedded browser — or the placeholder saying why there is none."""
        panel = QStackedWidget()
        panel.setMinimumSize(*_WEB_MIN)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewer_placeholder = QLabel(NO_VIEWER_TEXT)
        self.viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_placeholder.setWordWrap(True)
        self.viewer_placeholder.setMinimumSize(*_WEB_MIN)
        self.viewer_placeholder.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.viewer_placeholder.setStyleSheet(
            "background: #0e0e12; border-radius: 8px; color: #8f8fa0; padding: 16px;"
        )
        panel.addWidget(self.viewer_placeholder)
        panel.setVisible(self.session.viewer_url is not None)
        return panel

    def _install_actions(self) -> None:
        self.actions_list: tuple[DemoAction, ...] = self._declare_actions()
        self._buttons: dict[str, QPushButton] = {}
        self._qactions: dict[str, QAction] = {}
        for action in self.actions_list:
            button = QPushButton(action.button_text(self.session))
            button.setToolTip(action.tooltip)
            button.clicked.connect(action.callback)
            # Keyboard focus would let Space activate whichever button was
            # last clicked instead of Start, which with the arm moving is
            # exactly the wrong surprise.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            style_button(button, action.current_style(self.session))
            # A push button's minimum is its text width, and nine of them
            # side by side ask for more than the display. The row shares
            # the width it is given instead, each button in proportion to
            # its label, and a tight window clips a label rather than grow.
            weight = max(1, button.sizeHint().width())
            button.setMinimumWidth(48)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self.button_row.addWidget(button, stretch=weight)
            self._buttons[action.id] = button

            qaction = QAction(action.label, self)
            qaction.setShortcut(QKeySequence(action.shortcut))
            qaction.triggered.connect(action.callback)
            self.addAction(qaction)
            self._qactions[action.id] = qaction

        self._install_option_boxes()

        rows = [(a.shortcut, a.label) for a in self.actions_list]
        rows.append(("H", "Training heat map on/off"))
        rows.append(("N", "Distance numbers on/off"))
        self.help_panel.set_rows(rows)

    def _install_option_boxes(self) -> None:
        """The two controls that are states, not events: checkboxes in the
        side column under the keys, each with a hotkey so they stay
        keyboard-reachable like everything else."""
        self.heatmap_box = QCheckBox("Training heat map  [H]")
        self.heatmap_box.setToolTip(
            "Overlay where the training demos grasped the object, as a density "
            "on the table. Needs --demo.training-positions."
        )
        self.heatmap_box.toggled.connect(self._heatmap_toggled)
        self.readouts_box = QCheckBox("Distance numbers  [N]")
        self.readouts_box.setToolTip(
            "Write each pair's live / reference distance on its line. Off, the "
            "lines and masks stay and the picture is quieter."
        )
        self.readouts_box.setChecked(self.session.show_readouts)
        self.readouts_box.toggled.connect(self._readouts_toggled)
        for box in (self.heatmap_box, self.readouts_box):
            box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            box.setStyleSheet(
                "QCheckBox { color: #d8d8e0; font-weight: 600; padding: 4px 2px; }"
                "QCheckBox:disabled { color: #6a6a74; }"
            )
            self.options_panel.body_layout.addWidget(box)

        heat_action = QAction("Training heat map", self)
        heat_action.setShortcut(QKeySequence("H"))
        heat_action.triggered.connect(self._toggle_heatmap)
        self.addAction(heat_action)
        self._qactions["heatmap"] = heat_action
        readouts_action = QAction("Distance numbers", self)
        readouts_action.setShortcut(QKeySequence("N"))
        readouts_action.triggered.connect(self._toggle_readouts)
        self.addAction(readouts_action)
        self._qactions["readouts"] = readouts_action
        self._sync_heatmap_box()

    def _declare_actions(self) -> tuple[DemoAction, ...]:
        return (
            DemoAction(
                id="start",
                label="Start",
                shortcut="Space",
                tooltip=(
                    "Run the policy from the parked pose; while it runs, Space "
                    "pauses between two steps (the arm holds) and resumes."
                ),
                callback=self._start_or_pause,
                enabled_when=lambda s: (
                    (s.state is DemoState.IDLE and not s.warming_up)
                    or s.state is DemoState.RUNNING
                ),
                style="primary",
                label_when=lambda s: (
                    "Resume"
                    if s.state is DemoState.RUNNING and s.paused
                    else "Pause"
                    if s.state is DemoState.RUNNING
                    else "Start"
                ),
                style_when=lambda s: (
                    "success"
                    if s.state is DemoState.RUNNING and s.paused
                    else "default"
                    if s.state is DemoState.RUNNING
                    else "primary"
                ),
            ),
            DemoAction(
                id="stop",
                label="Stop & home",
                shortcut="S",
                tooltip="End the rollout at its next step and return the arm home.",
                callback=self._stop,
                enabled_when=lambda s: s.state is DemoState.RUNNING,
                style="danger",
            ),
            DemoAction(
                id="guidance",
                label="Guidance on/off",
                shortcut="G",
                tooltip=(
                    "The switch. Takes effect at the next replan; the retrieved "
                    "demo, the tracker and the alignment keep running either way."
                ),
                callback=self._toggle_guidance,
                enabled_when=lambda s: True,
                label_when=lambda s: f"Guidance: {'ON' if s.guided else 'OFF'}",
                style_when=lambda s: "success" if s.guided else "default",
            ),
            DemoAction(
                id="overlays",
                label="Overlays auto/on/off",
                shortcut="O",
                tooltip=(
                    "What the exterior view draws: auto (masks and pairs while "
                    "guidance is on), always on, or off."
                ),
                callback=self._cycle_overlays,
                enabled_when=lambda s: True,
                label_when=lambda s: f"Overlays: {s.overlay_mode.value}",
            ),
            DemoAction(
                id="trends",
                label="Trends row",
                shortcut="T",
                tooltip=(
                    "Show or hide the bottom row: the wrist camera, the "
                    "retrieved demo frame and the distance trend plots."
                ),
                callback=self._toggle_trends,
                enabled_when=lambda s: True,
                label_when=lambda s: (
                    "Trends row: shown"
                    if not self.bottom_row.isHidden()
                    else "Trends row: hidden"
                ),
            ),
            DemoAction(
                id="trend_window",
                label="Trend window: snapshot/full",
                shortcut="W",
                tooltip=(
                    "What the trend plots show: a snapshot around now (a few "
                    "chunks either side of the aligned time, the last few "
                    "chunks of the rollout) or the full trajectory."
                ),
                callback=self._toggle_trend_window,
                enabled_when=lambda s: True,
                label_when=lambda s: (
                    "Trend window: snapshot"
                    if self.trends.windowed
                    else "Trend window: full"
                ),
            ),
            DemoAction(
                id="viewer",
                label="3-D view",
                shortcut="V",
                tooltip=(
                    "Show or hide the 3-D panel — the Rerun scene served from "
                    "this process, beside the camera view."
                ),
                callback=self._toggle_viewer_panel,
                enabled_when=lambda s: True,
                label_when=lambda s: (
                    "3-D view: shown"
                    if not self.viewer_panel.isHidden()
                    else "3-D view: hidden"
                ),
            ),
            DemoAction(
                id="layout",
                label="Viewer layout scene/full",
                shortcut="L",
                tooltip=(
                    "Re-lay out the Rerun viewer: the 3-D scene alone, or the "
                    "full analysis layout with the cameras and the plots."
                ),
                callback=self._toggle_viewer_layout,
                enabled_when=lambda s: s.viewer_layout is not None,
                label_when=lambda s: f"Layout: {s.viewer_layout or 'none'}",
            ),
            DemoAction(
                id="fullscreen",
                label="Full screen",
                shortcut="F11",
                tooltip="Toggle true full screen; back to the maximised window.",
                callback=self._toggle_fullscreen,
                enabled_when=lambda s: True,
                label_when=lambda s: (
                    "Full screen: on" if self.isFullScreen() else "Full screen: off"
                ),
            ),
            DemoAction(
                id="quit",
                label="Quit",
                shortcut="Esc",
                tooltip="Stop any running rollout, park the arm, and close.",
                callback=self._quit,
                enabled_when=lambda s: True,
            ),
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _start_or_pause(self) -> None:
        """Space: start when idle, pause or resume while running."""
        session = self.session
        if session.state is DemoState.RUNNING:
            session.toggle_pause()
        elif not session.start_rollout():
            return
        self._refresh()

    def _start(self) -> None:  # kept for tests and callers that mean "start"
        if self.session.start_rollout():
            self._refresh()

    def _stop(self) -> None:
        self.session.request_stop()
        self._refresh()

    def _toggle_guidance(self) -> None:
        self.session.toggle_guidance()
        self._refresh()

    def _task_committed(self, text: str) -> None:
        """Enter in the instruction field: hand the prompt to the session and
        show what it settled on (whitespace squeezed, blank ignored)."""
        self.session.set_task(text)
        self.task_edit.setText(self.session.task)
        self._refresh()

    def _toggle_readouts(self) -> None:
        self.session.toggle_readouts()
        self._sync_readouts_box()
        self._refresh()

    def _readouts_toggled(self, checked: bool) -> None:
        if checked != self.session.show_readouts:
            self.session.toggle_readouts()
            self._refresh()

    def _sync_readouts_box(self) -> None:
        if self.readouts_box.isChecked() != self.session.show_readouts:
            self.readouts_box.blockSignals(True)
            self.readouts_box.setChecked(self.session.show_readouts)
            self.readouts_box.blockSignals(False)

    def _toggle_heatmap(self) -> None:
        self.session.toggle_heatmap()
        self._sync_heatmap_box()
        self._refresh()

    def _heatmap_toggled(self, checked: bool) -> None:
        if checked != self.session.show_heatmap:
            self.session.toggle_heatmap()
            self._sync_heatmap_box()
            self._refresh()

    def _sync_heatmap_box(self) -> None:
        """The checkbox follows the session, never the other way round."""
        available = self.session.training_heatmap is not None
        self.heatmap_box.setEnabled(available)
        self._qactions["heatmap"].setEnabled(available)
        if self.heatmap_box.isChecked() != self.session.show_heatmap:
            self.heatmap_box.blockSignals(True)
            self.heatmap_box.setChecked(self.session.show_heatmap)
            self.heatmap_box.blockSignals(False)
        if not available:
            self.heatmap_box.setToolTip(
                "No training positions were given (--demo.training-positions), "
                "so there is no heat map to show."
            )

    def _cycle_overlays(self) -> None:
        self.session.cycle_overlay_mode()
        self._refresh()

    def _toggle_trends(self) -> None:
        # isHidden, not isVisible: see _toggle_viewer_panel.
        if self.bottom_row.isHidden():
            self.bottom_row.setVisible(True)
            self._balance_bottom()
        else:
            self.bottom_row.hide()
        self._refresh()

    def _toggle_trend_window(self) -> None:
        self.trends.set_windowed(not self.trends.windowed)
        self._update_buttons()

    def _toggle_viewer_panel(self) -> None:
        # isHidden, not isVisible: before the window is shown every child
        # reports invisible, and the toggle must still mean what it says.
        if self.viewer_panel.isHidden():
            self._show_viewer_panel()
        else:
            self.viewer_panel.hide()
        self._refresh()

    def _show_viewer_panel(self) -> None:
        """Show the 3-D panel and give it its share of the centre.

        A splitter hands a widget that was hidden when it was laid out a
        zero share, and shows it later at its minimum width — the 3-D scene
        would come up as a 320 px strip beside a camera view taking the
        rest. So the shares are set outright: the camera 3, the scene 2,
        the side column its fixed width. Dragging the handle afterwards is
        kept across window resizes, since the splitter scales what it has.
        """
        self.viewer_panel.setVisible(True)
        self._balance_centre()

    def _balance_centre(self) -> None:
        if self.viewer_panel.isHidden():
            return
        total = self.centre_splitter.width()
        if total <= 0:
            return
        handles = self.centre_splitter.handleWidth() * 2
        free = max(0, total - _SIDE_WIDTH - handles)
        web = max(_WEB_MIN[0], free * 2 // 5)
        camera = max(_VIEW_MIN[0], free - web)
        self.centre_splitter.setSizes([camera, web, _SIDE_WIDTH])

    def _balance_bottom(self) -> None:
        """Give the bottom row its share of the height, and its four
        panes their shares of the width.

        Set outright, for the same reason as :meth:`_balance_centre`: a
        splitter laid out before the window has its size hands out
        minimums, and the trend plots would come up as a strip.
        """
        height = self.main_splitter.height()
        if height > 0 and not self.bottom_row.isHidden():
            bottom = max(_BOTTOM_MIN_HEIGHT, int(height * _BOTTOM_SHARE))
            self.main_splitter.setSizes([max(1, height - bottom), bottom])
        width = self.bottom_row.width()
        if width > 0:
            shares: list[int] = list(_BOTTOM_SPLIT)
            if self.wrist_box.isHidden():
                shares[0] = 0
            handles = self.bottom_row.handleWidth() * (len(shares) - 1)
            pairs = min(
                _PAIRS_WIDTH[2],
                max(_PAIRS_WIDTH[0], (width - handles) * shares[3] // 100),
            )
            rest = max(0, width - handles - pairs)
            denominator = max(1, sum(shares[:3]))
            sizes = [rest * s // denominator for s in shares[:3]] + [pairs]
            self.bottom_row.setSizes(sizes)

    def _toggle_viewer_layout(self) -> None:
        current = self.session.viewer_layout
        if current is None:
            return
        self.session.set_viewer_layout("full" if current == "scene" else "scene")
        self._refresh()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            if self._start_maximised:
                self.showMaximized()
            else:
                self.showNormal()
        else:
            self.showFullScreen()
        self._update_buttons()

    def _quit(self) -> None:
        """Ask the window to close. ``QWidget.close`` returns a bool, which
        the action contract does not carry."""
        self.close()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Timer: poll the env when nothing runs, repaint what changed."""
        if not self.session.is_running:
            self.session.refresh_preview()
        self._watch_rollout_edge()
        self._update_view()
        self._update_viewer()
        self._update_buttons()
        self._update_trends()
        self._update_panels()
        self._update_dimmers()

    def _refresh(self) -> None:
        """Full repaint, after anything that changes session state.

        A toggle changes the picture without changing the frame, so the
        view is marked dirty rather than waiting for a new timestamp.
        """
        self._dirty = True
        self._watch_rollout_edge()
        self._update_view()
        self._update_viewer()
        self._update_buttons()
        self._update_trends()
        self._update_panels()
        self._update_dimmers()

    def _watch_rollout_edge(self) -> None:
        """Clear the previous rollout's pictures the moment a new one starts.

        The session blanks its own snapshot at ``start_rollout`` — but a
        blank field is exactly what :meth:`_update_view` leaves alone, so
        the demo frame and the detector's masks of the last rollout would
        stay on screen until this rollout's first replan. They are reset here
        instead, on the IDLE→RUNNING edge, whichever repaint sees it first.
        """
        running = self.session.is_running
        if running and not self._was_running:
            self._start_fresh()
        self._was_running = running

    def _start_fresh(self) -> None:
        """Every pane that shows the rollout's own analysis, back to its
        placeholder. The wrist and exterior views keep streaming."""
        _reset_view(self.demo_view, DEMO_PLACEHOLDER)
        _reset_view(self.perception_panel.view, PERCEPTION_PLACEHOLDER)
        self.pairs_panel.set_text(PAIRS_PLACEHOLDER)
        self.trends.clear()
        self._trends_seen = None
        self._trend_count = 0
        self._dirty = True

    def _update_view(self) -> None:
        view = self.session.live_view()
        if view.image is None:
            return
        if view.timestamp == self._last_frame_stamp and not self._dirty:
            return
        self._last_frame_stamp = view.timestamp
        self._dirty = False
        banner, color = self._frame_banner(view)
        try:
            annotated = render_live_view(
                view,
                show_overlays=self.session.overlay_mode.shows(view.guided),
                banner=banner,
                banner_color=color,
                badges=self._badges(view),
                badge_start_line=2,  # the instruction field covers the top
                preprocessed=bool(
                    getattr(self.session.camera, "pixels_mirrored", False)
                ),
                heatmap=(
                    self.session.training_heatmap if self.session.show_heatmap else None
                ),
                readouts=self.session.show_readouts,
            )
        except Exception as exc:
            # A broken overlay must never cost the operator their live view
            # of a moving robot.
            logger.warning(f"live overlay failed, showing the raw frame: {exc}")
            annotated = view.image
        self.view.set_image(annotated)
        self.view_box.place_overlay()
        if view.wrist is not None:
            self.wrist_view.set_image(view.wrist)
        if view.ref_frame is not None:
            self.demo_view.set_image(view.ref_frame)
        if view.perception_overlay is not None:
            self.perception_panel.set_image(view.perception_overlay)

    def _update_viewer(self) -> None:
        """Load the Rerun web viewer into the 3-D panel once its URL is known.

        The URL can appear late — the visualiser is built with the session,
        behind the start-up screen — so this is checked every tick until it
        has been loaded once. Without a URL the panel keeps its placeholder.
        """
        url = self.session.viewer_url
        if url is None or url == self._viewer_url_loaded or self._web_failed:
            return
        if self._web is None:
            try:
                self._web = _make_web_view()
                _prefer_dark(self._web)
            except Exception as exc:  # noqa: BLE001 — shown, never fatal
                self._web_failed = True
                logger.warning(f"3-D panel: the embedded browser is unavailable: {exc}")
                self.viewer_placeholder.setText(
                    "3-D view unavailable: the embedded browser could not be "
                    f"created ({type(exc).__name__}: {exc}). The scene is still "
                    f"served at {url} — open it in a browser."
                )
                self._show_viewer_panel()
                return
            self._web.setMinimumSize(*_WEB_MIN)
            self._web.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self.viewer_panel.addWidget(self._web)
        self._viewer_url_loaded = url
        self._web.setUrl(QUrl(url))  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        # Placeholder → browser: the first page load sometimes comes up on a
        # blank canvas when the widget was hidden while loading; a reload
        # once it is the shown page settles it.
        self.viewer_panel.setCurrentWidget(self._web)
        self._show_viewer_panel()
        self._web.reload()  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]

    def _frame_banner(self, view: LiveView) -> tuple[str, tuple[int, int, int]]:
        """The short mode word drawn along the bottom of the frame."""
        session = self.session
        if session.state is DemoState.STOPPING:
            return "STOPPING", COLOR_STOPPING
        if session.state is DemoState.IDLE:
            return "IDLE", COLOR_IDLE
        if session.paused:
            return "PAUSED", COLOR_STOPPING
        if view.guided:
            return "GUIDED", COLOR_GUIDED
        if session.guided and self._guidance_unavailable():
            return "GUIDANCE UNAVAILABLE", COLOR_UNAVAILABLE
        return "UNGUIDED", COLOR_UNGUIDED

    def _guidance_unavailable(self) -> bool:
        """The switch is on and there is nothing for it to open — for good,
        not merely because the abstractor is still setting up."""
        session = self.session
        return not session.guidance_configured or (
            not session.guidance_available and session.abstraction_failed
        )

    def _badges(self, view: LiveView) -> list[str]:
        session = self.session
        badges: list[str] = []
        if not session.guidance_configured:
            badges.append("launched unguided — the switch has nothing to open")
        elif (
            session.is_running
            and not session.guidance_available
            and session.abstraction_failed
        ):
            badges.append("perception found no pairs on this scene — running unguided")
        return badges

    def _status_text(self) -> tuple[str, str]:
        session = self.session
        if session.warming_up:
            return "WARMING UP THE POLICY — the arm is not commanded", STATUS_WARMUP
        state = session.state
        if state is DemoState.STOPPING:
            return "STOPPING — ending the step, then homing", STATUS_STOPPING
        if state is DemoState.RUNNING and session.paused:
            return (
                f"PAUSED — the arm holds at step {session.live_step}; "
                "SPACE resumes, S stops and homes",
                STATUS_STOPPING,
            )
        if state is DemoState.RUNNING:
            view = session.live_view()
            mode = "GUIDED" if view.guided else "UNGUIDED"
            unavailable = session.guided and self._guidance_unavailable()
            if unavailable:
                mode = "GUIDANCE UNAVAILABLE"
            limit = session.step_limit
            steps = (
                f"step {session.live_step}/{limit}"
                if limit
                else f"step {session.live_step}"
            )
            text = f"RUNNING · {mode} · {steps} · {session.elapsed:.1f} s"
            if session.guided != view.guided and not unavailable:
                # The switch moved; the chunk on the arm is still the old kind.
                text += " · switching at next replan"
            return text, (STATUS_GUIDED if view.guided else STATUS_UNGUIDED)
        text = "IDLE — press SPACE to run"
        if getattr(session, "saving", False):
            text += "  ·  writing the last episode to disk (the arm is parked)"
        if session.rollouts:
            text += f"  ·  {session.rollouts} rollout{'s' if session.rollouts != 1 else ''} so far"
        if session.last_error:
            text += f"  ·  ⚠ {session.last_error}"
        return text, STATUS_IDLE

    def _update_buttons(self) -> None:
        session = self.session
        for action in self.actions_list:
            enabled = action.enabled_when(session)
            button, qaction = self._buttons[action.id], self._qactions[action.id]
            button.setEnabled(enabled)
            qaction.setEnabled(enabled)
            text = action.button_text(session)
            if button.text() != text:
                button.setText(text)
            if action.style_when is not None:
                style_button(button, action.current_style(session))
        text, color = self._status_text()
        self.status.set_state(text, color)

    def _sync_task_edit(self) -> None:
        if not self.task_edit.hasFocus() and self.task_edit.text() != self.session.task:
            self.task_edit.setText(self.session.task)

    def _update_panels(self) -> None:
        self._sync_task_edit()
        view = self.session.live_view()
        result = view.abstraction
        mode = self.trends.mode_text
        if result is None:
            self.pairs_panel.set_text(
                f"{PAIRS_PLACEHOLDER}<br><br><span style='color:#8f8fa0'>{mode}</span>"
            )
        else:
            rows = []
            for pair in result.pairs or []:
                color = _hex(view.colors.get(pair.label, (150, 150, 158)))
                live = result.live_distances.get(pair.label)
                ref = result.ref_distances.get(pair.label)
                err = result.distance_errors.get(pair.label)
                value = f"{_m(live)} / {_m(ref)} m"
                if err is not None:
                    value += f" ({err:+.3f})"
                rows.append(
                    (f"<span style='color:{color}'>&#9679;</span> {pair.label}", value)
                )
            self.pairs_panel.set_rows(rows or [("pairs", "none")], lines=[mode])

        info = view.guidance_info or {}
        rows = [("switch", "ON" if self.session.guided else "OFF")]
        if self.session.is_running:
            rows.append(("applied to this chunk", "yes" if view.guided else "no"))
        factor = info.get("guidance_factor")
        if factor is not None:
            text = f"{float(factor):.3f}"
            if info.get("gated") and info.get("requested_factor") is not None:
                text += f" (withheld {float(info['requested_factor']):.3f})"
            rows.append(("factor", text))
        if result is not None:
            rows.append(("alignment", f"{100 * float(result.mta_progress):.0f}%"))
            idx = result.current_subtraj_idx
            rows.append(("sub-trajectory", "—" if idx is None else str(idx)))
            rel = [v for v in (result.relevances or {}).values() if v is not None]
            if rel:
                rows.append(("relevance", f"{sum(rel) / len(rel):.2f}"))
        if not self.session.guidance_configured:
            rows.append(("pipeline", "launched unguided"))
        self.guidance_panel.set_rows(rows)

    def _update_trends(self) -> None:
        session = self.session
        trends = session.trends
        if trends is not self._trends_seen:
            self._trends_seen = trends
            self._trend_count = 0
            if trends is None:
                self.trends.clear()  # a new rollout starts from nothing
            else:
                self.trends.set_reference(trends)
        if not self.bottom_row.isVisible():
            return  # counters untouched, so the next visible tick catches up
        count = session.trend_count
        if count != self._trend_count:
            self._trend_count = count
            self.trends.update_history(session.trend_history())

    # ------------------------------------------------------------------
    # Dimming
    # ------------------------------------------------------------------

    @property
    def dimmed(self) -> bool:
        """Whether the guidance panes are under their dimmers right now."""
        return self._dimmed

    def _should_dim(self, view: LiveView) -> bool:
        """The guidance panes go dark when guidance is *off*, by the measure
        the audience sees: the chunk on the arm during a rollout, the
        switch while idle.

        Never while the executing chunk is guided. And not in the half
        second after the switch is flipped on while the chunk on the arm is
        still the old kind — the picture is about to change, and a flicker
        would say the flip did nothing.
        """
        session = self.session
        if session.is_running:
            return not view.guided and not session.guided
        return not session.guided

    def _update_dimmers(self) -> None:
        dim = self._should_dim(self.session.live_view())
        if dim == self._dimmed and all(d.isHidden() != dim for d in self.dimmers):
            return
        self._dimmed = dim
        for dimmer in self.dimmers:
            dimmer.set_active(dim)

    def showEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802  # ty: ignore[invalid-method-override]
        super().showEvent(event)
        if not self._balanced_once:
            self._balanced_once = True
            self._balance_centre()
            self._balance_bottom()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]  # noqa: N802  # ty: ignore[invalid-method-override]
        """Stop, park, close.

        Shutting the backends down is the launcher's job (``run_demo_gui``'s
        ``finally``), which is what parks the arm on every exit path; the
        window only makes sure a running rollout is asked to stop first, and
        that a second Esc while it winds down does not start a second teardown.
        """
        if self._quitting:
            event.ignore()
            return
        if self.session.is_running:
            answer = QMessageBox.question(
                self,
                "Quit the demo?",
                "A rollout is running. Stop it, park the arm and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer is not QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.session.request_stop()
        self._quitting = True
        self._timer.stop()
        self.status.set_state("SHUTTING DOWN — parking the arm …", STATUS_STOPPING)
        self.repaint()
        event.accept()


def _size_by_layout(
    view: QLabel, minimum: tuple[int, int], *, vertical: str = "ignored"
) -> None:
    """Let the layout size a picture label, never its pixmap.

    A QLabel showing a pixmap reports the pixmap's size as its hint, and
    ``CameraView`` scales its pixmap to the label's *current* size — so an
    Expanding policy ratchets: whatever size the label reaches becomes what
    it asks for. ``Ignored`` breaks the loop; the explicit minimum is all the
    layout is promised.
    """
    view.setMinimumSize(*minimum)
    view.setSizePolicy(
        QSizePolicy.Policy.Ignored,
        QSizePolicy.Policy.Fixed if vertical == "fixed" else QSizePolicy.Policy.Ignored,
    )


def _body_by_layout(panel: InfoPanel) -> None:
    """Let the layout size an info panel's body, never its text.

    The body wraps already; an ``Ignored`` horizontal policy stops a long
    unbreakable pair label from widening its column anyway.
    """
    panel._body.setMinimumWidth(0)
    panel._body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)


def _reset_view(view: CameraView, text: str) -> None:
    """Back to a placeholder: the pixmap dropped, so a resize does not
    bring the old frame back, and the text in its place."""
    view._pixmap = None
    view.setText(text)


def _titled_view(title: str, view: CameraView) -> QWidget:
    """A small-caps title over a camera view that the layout sizes.

    Not a ``CameraPanel``: that one has a fixed height for a side column,
    and these views must follow the bottom row's height instead. The side
    margins keep the picture off the splitter handles.
    """
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.setSpacing(4)
    label = QLabel(title)
    label.setStyleSheet(_TITLE_STYLE)
    layout.addWidget(label)
    _size_by_layout(view, _VIEW_MIN)
    view.setWordWrap(True)
    layout.addWidget(view, stretch=1)
    return box


def _m(value) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _hex(color) -> str:
    r, g, b = (int(c) for c in color[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


__all__ = [
    "DEMO_PLACEHOLDER",
    "DIM_CAPTION",
    "NO_VIEWER_TEXT",
    "PAIRS_PLACEHOLDER",
    "PERCEPTION_PLACEHOLDER",
    "STATUS_GUIDED",
    "STATUS_IDLE",
    "STATUS_STOPPING",
    "STATUS_UNGUIDED",
    "STATUS_WARMUP",
    "DemoAction",
    "DemoWindow",
    "Dimmer",
]

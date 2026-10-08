"""The operator window for camera calibration.

One window, one procedure: the step list on the left says where you are, the
instructions under it say what to do, the controls under those do it, and
the big view on the right shows the camera — live during capture, the first
frame with a magnifier when picking the point, and the tracked-versus-predicted
overlay when checking the result.

The window owns nothing but widgets. All state lives in
:class:`~policy_abstraction.calibration.session.CalibrationSession`, which the
window polls on a timer and drives from its buttons, so the same session runs
under the tests with fake hardware and no display.
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional, Sequence

import numpy as np
from loguru import logger
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from policy_abstraction.calibration.geometry import CalibrationMode
from policy_abstraction.calibration.motion import MotionConfig, SafeBox
from policy_abstraction.calibration.session import (
    CalibrationResult,
    CalibrationSession,
    Step,
)
from policy_abstraction.calibration.suite import CalibrationSuite, load_saved_extrinsic

GREEN = QColor(60, 220, 90)
MAGENTA = QColor(255, 60, 220)
AMBER = QColor(255, 190, 40)
RED = QColor(240, 70, 70)
WHITE = QColor(255, 255, 255)

STEP_TITLES = {
    Step.REFERENCE: "Record the reference mark",
    Step.CAPTURE: "Capture poses",
    Step.SELECT_POINT: "Pick the tracked point",
    Step.TRACK: "Track and solve",
    Step.VERIFY: "Check the result",
    Step.DONE: "Saved",
}


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class Overlay:
    """A marker to draw over the image, in image coordinates."""

    __slots__ = ("uv", "color", "kind", "label")

    def __init__(self, uv, color: QColor, kind: str = "ring", label: str = "") -> None:
        self.uv = (float(uv[0]), float(uv[1]))
        self.color = color
        self.kind = kind
        self.label = label


class ImageView(QWidget):
    """Aspect-fit image with overlays, click-to-image-coordinates and a magnifier."""

    clicked = pyqtSignal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._image: Optional[QImage] = None
        self._overlays: list[Overlay] = []
        self._click_enabled = False
        self._magnifier = False
        self._cursor: Optional[QPointF] = None
        self._banner = ""
        self._banner_color = WHITE

    # -- state ----------------------------------------------------------------

    def set_image(
        self, rgb: Optional[np.ndarray], overlays: Sequence[Overlay] = ()
    ) -> None:
        self._image = None if rgb is None else to_qimage(rgb)
        self._overlays = list(overlays)
        self.update()

    def set_overlays(self, overlays: Sequence[Overlay]) -> None:
        self._overlays = list(overlays)
        self.update()

    def set_click_enabled(self, enabled: bool, magnifier: bool = False) -> None:
        self._click_enabled = enabled
        self._magnifier = magnifier and enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def set_banner(self, text: str, color: QColor = WHITE) -> None:
        self._banner = text
        self._banner_color = color
        self.update()

    @property
    def image_size(self) -> Optional[tuple[int, int]]:
        return (
            None if self._image is None else (self._image.width(), self._image.height())
        )

    # -- geometry -------------------------------------------------------------

    def _fit_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        iw, ih = self._image.width(), self._image.height()
        W, H = self.width(), self.height()
        scale = min(W / iw, H / ih)
        w, h = iw * scale, ih * scale
        return QRectF((W - w) / 2, (H - h) / 2, w, h)

    def to_image(self, pos: QPointF) -> Optional[tuple[float, float]]:
        r = self._fit_rect()
        if self._image is None or not r.contains(pos):
            return None
        u = (pos.x() - r.x()) / r.width() * self._image.width()
        v = (pos.y() - r.y()) / r.height() * self._image.height()
        return u, v

    def to_widget(self, uv) -> QPointF:
        r = self._fit_rect()
        assert self._image is not None
        return QPointF(
            r.x() + uv[0] / self._image.width() * r.width(),
            r.y() + uv[1] / self._image.height() * r.height(),
        )

    # -- events -----------------------------------------------------------------

    def mouseMoveEvent(self, a0: Optional[QMouseEvent]) -> None:  # noqa: N802
        if a0 is None:
            return
        self._cursor = a0.position()
        if self._magnifier:
            self.update()

    def leaveEvent(self, a0: Optional[QEvent]) -> None:  # noqa: N802
        self._cursor = None
        self.update()

    def mousePressEvent(self, a0: Optional[QMouseEvent]) -> None:  # noqa: N802
        if (
            a0 is None
            or not self._click_enabled
            or a0.button() != Qt.MouseButton.LeftButton
        ):
            return
        uv = self.to_image(a0.position())
        if uv is not None:
            self.clicked.emit(*uv)

    def paintEvent(self, a0: Optional[QPaintEvent]) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 24, 28))
        if self._image is None:
            p.setPen(QColor(160, 160, 170))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no image")
            p.end()
            return
        r = self._fit_rect()
        p.drawImage(r, self._image)
        for ov in self._overlays:
            self._draw_marker(p, ov)
        if self._magnifier and self._cursor is not None:
            self._draw_magnifier(p, self._cursor)
        if self._banner:
            p.setPen(self._banner_color)
            f = QFont()
            f.setPointSize(12)
            f.setBold(True)
            p.setFont(f)
            p.drawText(
                QRectF(r.x() + 10, r.y() + 6, r.width() - 20, 28),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._banner,
            )
        p.end()

    def _draw_marker(self, p: QPainter, ov: Overlay) -> None:
        if not all(np.isfinite(ov.uv)):
            return
        c = self.to_widget(ov.uv)
        pen = QPen(ov.color, 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if ov.kind == "ring":
            p.drawEllipse(c, 9, 9)
        elif ov.kind == "cross":
            p.drawLine(QPointF(c.x() - 12, c.y()), QPointF(c.x() + 12, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 12), QPointF(c.x(), c.y() + 12))
        else:  # crosshair with a gap, for the click
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p.drawLine(
                    QPointF(c.x() + 5 * dx, c.y() + 5 * dy),
                    QPointF(c.x() + 18 * dx, c.y() + 18 * dy),
                )
            p.drawEllipse(c, 2, 2)
        if ov.label:
            p.drawText(QPointF(c.x() + 12, c.y() - 12), ov.label)

    def _draw_magnifier(self, p: QPainter, cursor: QPointF) -> None:
        uv = self.to_image(cursor)
        if uv is None or self._image is None:
            return
        zoom, size = 6, 200
        src_w = size / zoom
        src = QRectF(uv[0] - src_w / 2, uv[1] - src_w / 2, src_w, src_w)
        r = self._fit_rect()
        # Top-right corner, unless the cursor is there.
        x = r.right() - size - 12
        y = r.top() + 12
        if cursor.x() > x - 20 and cursor.y() < y + size + 20:
            x = r.left() + 12
        dst = QRectF(x, y, size, size)
        p.fillRect(dst.adjusted(-2, -2, 2, 2), QColor(0, 0, 0))
        p.drawImage(dst, self._image, src)
        p.setPen(QPen(GREEN, 1))
        cx, cy = dst.center().x(), dst.center().y()
        p.drawLine(QPointF(cx - 30, cy), QPointF(cx - 6, cy))
        p.drawLine(QPointF(cx + 6, cy), QPointF(cx + 30, cy))
        p.drawLine(QPointF(cx, cy - 30), QPointF(cx, cy - 6))
        p.drawLine(QPointF(cx, cy + 6), QPointF(cx, cy + 30))
        p.setPen(WHITE)
        p.drawText(
            QPointF(dst.left() + 4, dst.bottom() - 4),
            f"({uv[0]:.1f}, {uv[1]:.1f})  x{zoom}",
        )


class SafeAreaView(QWidget):
    """The safe box in the robot base frame: top-down x-y plot plus a z bar.

    Shows the box, the live TCP, the planned poses (hollow) and the visited
    ones (filled), so the operator can see where the arm is going to go before
    it goes there — the camera image cannot show this, since the point of the
    exercise is that the camera's pose is still unknown.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(210)
        self.setMaximumHeight(260)
        self.box: Optional[SafeBox] = None
        self.margin_m = 0.02
        self.tcp: Optional[np.ndarray] = None
        self.targets: list[np.ndarray] = []
        self.visited = 0
        self.current: Optional[int] = None
        self.reference: Optional[np.ndarray] = None
        self.fault = False

    def update_state(
        self,
        *,
        box: Optional[SafeBox] = None,
        tcp=None,
        targets: Optional[list[np.ndarray]] = None,
        visited: Optional[int] = None,
        current: Optional[int] = None,
        reference=None,
        fault: Optional[bool] = None,
    ) -> None:
        if box is not None:
            self.box = box
        if tcp is not None:
            self.tcp = np.asarray(tcp, dtype=np.float64)[:3]
        if targets is not None:
            self.targets = list(targets)
        if current is not None or targets is not None:
            self.current = current
        if visited is not None:
            self.visited = visited
        if reference is not None:
            self.reference = np.asarray(reference, dtype=np.float64)[:3]
        if fault is not None:
            self.fault = fault
        self.update()

    def paintEvent(self, a0: Optional[QPaintEvent]) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        if self.box is None:
            p.setPen(QColor(160, 160, 170))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "safe box: waiting for the arm",
            )
            p.end()
            return
        W, H = self.width(), self.height()
        bar_w = 46
        plot = QRectF(34, 16, W - bar_w - 60, H - 40)
        lo, hi = self.box.lo, self.box.hi
        pad = 0.08
        xr = (lo[0] - pad, hi[0] + pad)
        yr = (lo[1] - pad, hi[1] + pad)
        scale = min(plot.width() / (xr[1] - xr[0]), plot.height() / (yr[1] - yr[0]))

        def to_plot(x: float, y: float) -> QPointF:
            # x to the right, y up: a top-down view of the base frame.
            return QPointF(
                plot.center().x() + (x - (xr[0] + xr[1]) / 2) * scale,
                plot.center().y() - (y - (yr[0] + yr[1]) / 2) * scale,
            )

        edge = RED if self.fault else GREEN
        tl, br = to_plot(lo[0], hi[1]), to_plot(hi[0], lo[1])
        p.setPen(QPen(edge, 2))
        p.setBrush(QColor(edge.red(), edge.green(), edge.blue(), 28))
        p.drawRect(QRectF(tl, br))
        m = self.margin_m
        p.setPen(
            QPen(
                QColor(edge.red(), edge.green(), edge.blue(), 110),
                1,
                Qt.PenStyle.DashLine,
            )
        )
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(to_plot(lo[0] - m, hi[1] + m), to_plot(hi[0] + m, lo[1] - m)))
        p.setPen(QColor(170, 170, 180))
        p.drawText(
            QPointF(tl.x(), tl.y() - 4),
            f"x {lo[0]:+.2f}..{hi[0]:+.2f}  y {lo[1]:+.2f}..{hi[1]:+.2f} m",
        )
        p.drawText(QPointF(6, H - 6), "top-down, robot base frame (x right, y up)")

        for i, t in enumerate(self.targets):
            c = to_plot(float(t[0]), float(t[1]))
            if i < self.visited:
                p.setPen(QPen(GREEN, 1))
                p.setBrush(GREEN)
            elif self.current is not None and i == self.current:
                p.setPen(QPen(AMBER, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(QPen(QColor(150, 150, 160), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, 3.5, 3.5)
        if self.reference is not None:
            c = to_plot(float(self.reference[0]), float(self.reference[1]))
            p.setPen(QPen(MAGENTA, 2))
            p.drawLine(QPointF(c.x() - 6, c.y()), QPointF(c.x() + 6, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 6), QPointF(c.x(), c.y() + 6))
        if self.tcp is not None:
            c = to_plot(float(self.tcp[0]), float(self.tcp[1]))
            inside = self.box.contains(self.tcp, self.margin_m)
            p.setPen(QPen(WHITE, 1))
            p.setBrush(GREEN if inside else RED)
            p.drawEllipse(c, 6, 6)

        bar = QRectF(W - bar_w - 8, 16, bar_w, H - 40)
        zr = (lo[2] - pad, hi[2] + pad)

        def to_bar(z: float) -> float:
            return bar.bottom() - (z - zr[0]) / (zr[1] - zr[0]) * bar.height()

        p.setPen(QPen(edge, 2))
        p.setBrush(QColor(edge.red(), edge.green(), edge.blue(), 28))
        p.drawRect(
            QRectF(
                bar.left() + 10,
                to_bar(hi[2]),
                bar.width() - 20,
                to_bar(lo[2]) - to_bar(hi[2]),
            )
        )
        p.setPen(QColor(170, 170, 180))
        p.drawText(QPointF(bar.left(), to_bar(hi[2]) - 4), f"z {hi[2]:+.2f}")
        p.drawText(QPointF(bar.left(), to_bar(lo[2]) + 12), f"{lo[2]:+.2f}")
        for i, t in enumerate(self.targets):
            y = to_bar(float(t[2]))
            p.setPen(QPen(GREEN if i < self.visited else QColor(150, 150, 160), 1))
            p.drawLine(QPointF(bar.left() + 12, y), QPointF(bar.right() - 12, y))
        if self.tcp is not None:
            y = to_bar(float(self.tcp[2]))
            inside = self.box.contains(self.tcp, self.margin_m)
            p.setPen(QPen(WHITE, 1))
            p.setBrush(GREEN if inside else RED)
            p.drawEllipse(QPointF(bar.center().x(), y), 6, 6)
        p.end()


class TrackingWorker(QThread):
    line = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, session: CalibrationSession) -> None:
        super().__init__()
        self._session = session

    def run(self) -> None:
        try:
            result = self._session.run_tracking(log=self.line.emit)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
            return
        self.succeeded.emit(result)


def _instructions(session: CalibrationSession, step: Step) -> str:
    cfg = session.config
    wrist = session.mode is CalibrationMode.WRIST
    setup = (
        "<p><b>Set-up</b></p><ul>"
        f"<li>Camera <code>{cfg.camera_name}</code> ({cfg.camera_device}) streams at "
        f"{cfg.width}x{cfg.height}; its intrinsics come from the RealSense.</li>"
        f"<li>Robot {cfg.robot_ip or '(fake)'}: joints and TCP are <i>read</i> over RTDE. "
        "<b>Nothing is ever commanded.</b></li>"
        "<li>Close the gripper fully and leave it closed. The point the controller "
        "reports as the TCP is the tip where the two pads meet.</li>"
        "<li>Put the arm in <b>freedrive</b>.</li></ul>"
    )
    if step is Step.REFERENCE:
        return setup + (
            "<p><b>Reference mark</b></p>"
            "<p>Pick a fixed, high-contrast mark on the table that will not move: a "
            "tape corner, a printed cross, a bolt head. Its position in the robot "
            "frame is needed, and there are two ways to get it.</p>"
            "<p><b>Pick it in the exterior image</b> (needs the exterior camera "
            "calibrated): the exterior view is frozen, you click the mark's centre, "
            "and its depth plus the exterior calibration place it in the robot frame. "
            "Quick, and good to a few millimetres.</p>"
            "<p><b>Or touch it</b>: in freedrive, bring the <b>closed fingertips "
            "exactly onto the mark</b>, hold still, and press <b>Record</b>. The "
            "controller's TCP is the position. Most accurate.</p>"
            "<p>Do not move the mark afterwards. Then park the arm with the wrist "
            "camera looking at the mark from 20–40 cm.</p>"
        )
    if step is Step.CAPTURE:
        keep = (
            "Keep the <b>mark inside the wrist image</b> the whole time."
            if wrist
            else "Keep the <b>fingertips visible</b> the whole time."
        )
        return (setup if not wrist else "") + (
            "<p><b>Capture</b> — two ways.</p>"
            "<p><b>Autonomous:</b> the arm moves itself through poses spread over the "
            "<b>safe box</b> drawn below (robot base frame, top-down plus height). "
            "Park the arm inside the box with the point in view, adjust the ranges, "
            "put the robot in <b>Remote Control</b>, then <b>Start autonomous capture</b> "
            "with the e-stop in your hand. <b>STOP MOTION</b> or <kbd>Esc</kbd> halts it.</p>"
            "<p><b>Manual:</b> in freedrive, move the arm slowly through the camera's "
            "view — near and far, side to side, high and low, rotating the wrist — "
            "pausing about a second at each pose.</p>"
            f"<p>{keep} A still pose is stored once the TCP has moved at least "
            f"{cfg.keyframe_min_move_m * 100:.0f} cm since the previous one; frames while "
            f"moving are stored too, for the tracker. Aim for {cfg.target_keyframes} still "
            "poses, then press <b>Finish capture</b>.</p>"
        )
    if step is Step.SELECT_POINT:
        what = (
            "the <b>exact centre of the reference mark</b>"
            if wrist
            else "the <b>exact spot where the closed pads meet</b> — the point the "
            "controller reports as the TCP"
        )
        return (
            "<p>This is the first captured frame. Click " + what + ". The magnifier "
            "follows the cursor; click again to correct. Then press "
            "<b>Track and solve</b>.</p>"
            "<p>Kalib tracks this one point through every frame, so the click must "
            "be on the physical point whose position the robot knows.</p>"
        )
    if step is Step.TRACK:
        n = session.dataset.n_frames if session.dataset else 0
        return (
            f"<p>Kalib is tracking the point through {n} frames with SpaTracker on the "
            "GPU. This takes a few minutes; its output appears in the log below.</p>"
            "<p>The camera pose is then solved by RANSAC PnP over the still frames and "
            "checked against the depth stream and against Kalib's own PnP.</p>"
        )
    if step is Step.VERIFY:
        return (
            "<p>Scrub through the frames. <span style='color:#3cdc5a'><b>Green ring</b></span>: "
            "where the tracker followed the point. <span style='color:#ff3cdc'><b>Magenta "
            "cross</b></span>: where the solved camera pose predicts it from the robot's "
            "own TCP. They should coincide in every still frame.</p>"
            "<p>If a stretch of frames drifts apart the tracker slipped there; pick the "
            "point again or capture again. <b>Save</b> when the verdict is good.</p>"
        )
    return (
        f"<p>Saved to <code>{cfg.output_path}</code>.</p>"
        "<p>The env server loads it through <code>--env.calibration-path</code>; the "
        "launch scripts default to <code>configs/ur5_camera_calibration.json</code>."
        + (
            " The wrist entry is parented to the TCP and composed with the live pose "
            "at every replan."
            if wrist
            else ""
        )
        + "</p>"
    )


class CalibrationWindow(QMainWindow):
    """One window for one camera, or for both in turn (a :class:`CalibrationSuite`)."""

    def __init__(
        self,
        target: CalibrationSuite | CalibrationSession,
        *,
        poll_hz: float = 15.0,
    ) -> None:
        super().__init__()
        if isinstance(target, CalibrationSession):
            target = CalibrationSuite(
                [target], calibration_path=target.config.output_path
            )
        self.suite = target
        self._worker: Optional[TrackingWorker] = None
        self._verify_index = 0
        self._flash_until = 0.0
        self._mark_mode = False
        self._mark_frame: Optional[tuple[np.ndarray, Optional[np.ndarray]]] = None
        self._set_title()
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / poll_hz))
        self._timer.timeout.connect(self._on_poll)
        self._timer.start()
        self._refresh()

    @property
    def session(self) -> CalibrationSession:
        return self.suite.active

    def _set_title(self) -> None:
        self.setWindowTitle(
            f"UR5e camera calibration — {self.session.config.camera_name} "
            f"({self.session.mode.value})"
        )

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        self.view = ImageView()
        self.view.clicked.connect(self._on_click)

        self.stage_label = QLabel()
        f = QFont()
        f.setBold(True)
        self.stage_label.setFont(f)
        self.stage_label.setVisible(len(self.suite.stages) > 1)

        self.steps_list = QListWidget()
        self.steps_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.steps_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._rebuild_steps()

        self.title = QLabel()
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        self.title.setFont(f)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setTextFormat(Qt.TextFormat.RichText)
        self.instructions.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.controls = QStackedWidget()
        self._pages: dict[Step, QWidget] = {}
        self._pages[Step.REFERENCE] = self._page_reference()
        self._pages[Step.CAPTURE] = self._page_capture()
        self._pages[Step.SELECT_POINT] = self._page_select()
        self._pages[Step.TRACK] = self._page_track()
        self._pages[Step.VERIFY] = self._page_verify()
        self._pages[Step.DONE] = self._page_done()
        for page in self._pages.values():
            self.controls.addWidget(page)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(110)
        self.log.setMaximumHeight(180)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.TypeWriter)
        mono.setPointSize(9)
        self.log.setFont(mono)

        # Everything the operator reads and presses lives in one scrollable
        # column, so a small screen never squeezes widgets on top of each other.
        column = QWidget()
        clay = QVBoxLayout(column)
        clay.addWidget(self.stage_label)
        clay.addWidget(self.steps_list)
        clay.addWidget(self.title)
        clay.addWidget(self.controls)
        clay.addWidget(self.instructions)
        clay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(column)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        left = QWidget()
        lay = QVBoxLayout(left)
        lay.addWidget(scroll, 1)
        lay.addWidget(self.status)
        lay.addWidget(self.log)
        left.setMinimumWidth(400)
        left.setMaximumWidth(560)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.view)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        for key, slot in (
            ("R", self._record_reference),
            ("S", self._start_capture),
            ("F", self._finish_capture),
            ("T", self._start_tracking),
            ("A", self._start_autonomous),
            ("M", self._pick_mark),
            ("Escape", self._escape),
            ("Return", self._save),
            ("Left", lambda: self._step_verify(-1)),
            ("Right", lambda: self._step_verify(+1)),
        ):
            QShortcut(QKeySequence(key), self).activated.connect(slot)

    def _page_reference(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.btn_pick_mark = QPushButton("Pick the mark in the exterior image  (M)")
        self.btn_pick_mark.clicked.connect(self._pick_mark)
        lay.addWidget(self.btn_pick_mark)
        self.btn_record = QPushButton("Record by touching it with the fingertips  (R)")
        self.btn_record.clicked.connect(self._record_reference)
        lay.addWidget(self.btn_record)
        self.btn_mark_cancel = QPushButton("Back to the wrist camera")
        self.btn_mark_cancel.clicked.connect(self._cancel_mark)
        self.btn_mark_cancel.setVisible(False)
        lay.addWidget(self.btn_mark_cancel)
        self.ref_label = QLabel("No reference recorded.")
        self.ref_label.setWordWrap(True)
        lay.addWidget(self.ref_label)
        return w

    def _page_capture(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        auto = QGroupBox("Autonomous motion (the arm moves itself)")
        alay = QVBoxLayout(auto)
        self.area = SafeAreaView()
        alay.addWidget(self.area)
        grid = QGridLayout()
        self.box_spins: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        for row, axis in enumerate("xyz"):
            grid.addWidget(QLabel(f"{axis} (m)"), row, 0)
            pair = []
            for col in (1, 2):
                sp = QDoubleSpinBox()
                sp.setRange(-1.5, 1.5)
                sp.setDecimals(3)
                sp.setSingleStep(0.01)
                sp.valueChanged.connect(self._box_edited)
                grid.addWidget(sp, row, col)
                pair.append(sp)
            self.box_spins[axis] = (pair[0], pair[1])
        self.btn_centre = QPushButton("Centre box on TCP")
        self.btn_centre.clicked.connect(self._centre_box)
        grid.addWidget(self.btn_centre, 0, 3, 3, 1)
        alay.addLayout(grid)
        self.spin_poses = QSpinBox()
        self.spin_poses.setRange(12, 200)
        self.spin_poses.setValue(self.session.config.target_keyframes)
        self.spin_tilt = QDoubleSpinBox()
        self.spin_tilt.setRange(0.0, 45.0)
        self.spin_tilt.setValue(20.0)
        self.spin_tilt.setSuffix("°")
        self.spin_roll = QDoubleSpinBox()
        self.spin_roll.setRange(0.0, 90.0)
        self.spin_roll.setValue(30.0)
        self.spin_roll.setSuffix("°")
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.01, 0.25)
        self.spin_speed.setDecimals(3)
        self.spin_speed.setSingleStep(0.01)
        self.spin_speed.setValue(0.08)
        self.spin_speed.setSuffix(" m/s")
        row = QHBoxLayout()
        for label, spin in (
            ("poses", self.spin_poses),
            ("tilt", self.spin_tilt),
            ("roll", self.spin_roll),
            ("speed", self.spin_speed),
        ):
            row.addWidget(QLabel(label))
            row.addWidget(spin)
        alay.addLayout(row)
        brow = QHBoxLayout()
        self.btn_auto = QPushButton("Start autonomous capture  (A)")
        self.btn_auto.clicked.connect(self._start_autonomous)
        self.btn_stop = QPushButton("STOP MOTION  (Esc)")
        self.btn_stop.setStyleSheet(
            "QPushButton { background: #c62828; color: white; font-weight: bold; "
            "padding: 6px; } QPushButton:disabled { background: #5a3030; color: #bbb; }"
        )
        self.btn_stop.clicked.connect(self._stop_motion)
        self.btn_stop.setEnabled(False)
        brow.addWidget(self.btn_auto)
        brow.addWidget(self.btn_stop)
        alay.addLayout(brow)
        self.motion_label = QLabel("")
        self.motion_label.setWordWrap(True)
        alay.addWidget(self.motion_label)
        lay.addWidget(auto)

        manual = QGroupBox("Manual (you freedrive the arm)")
        mlay = QVBoxLayout(manual)
        self.btn_start = QPushButton("Start manual capture  (S)")
        self.btn_start.clicked.connect(self._start_capture)
        mlay.addWidget(self.btn_start)
        lay.addWidget(manual)

        self.btn_finish = QPushButton("Finish capture  (F)")
        self.btn_finish.clicked.connect(self._finish_capture)
        lay.addWidget(self.btn_finish)
        self.progress = QProgressBar()
        self.progress.setRange(0, self.session.config.target_keyframes)
        self.progress.setFormat("%v / %m still poses")
        lay.addWidget(self.progress)
        self.capture_label = QLabel()
        self.capture_label.setWordWrap(True)
        lay.addWidget(self.capture_label)
        self.btn_load = QPushButton("Load a previous capture instead…")
        self.btn_load.clicked.connect(self._load_existing)
        lay.addWidget(self.btn_load)
        self.btn_skip = QPushButton("Skip this camera — keep its saved calibration")
        self.btn_skip.clicked.connect(self._skip_stage)
        self.btn_skip.setVisible(False)
        lay.addWidget(self.btn_skip)
        self._box_user_edited = False
        self._box_updating = False
        return w

    def _page_select(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.click_label = QLabel("No point picked yet.")
        lay.addWidget(self.click_label)
        row = QHBoxLayout()
        self.btn_track = QPushButton("Track and solve  (T)")
        self.btn_track.clicked.connect(self._start_tracking)
        self.btn_recapture = QPushButton("Capture again")
        self.btn_recapture.clicked.connect(self._recapture)
        row.addWidget(self.btn_track)
        row.addWidget(self.btn_recapture)
        lay.addLayout(row)
        return w

    def _page_track(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.track_label = QLabel("Tracking…")
        lay.addWidget(self.track_label)
        busy = QProgressBar()
        busy.setRange(0, 0)
        lay.addWidget(busy)
        return w

    def _page_verify(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.verdict_label = QLabel()
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.verdict_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._show_verify_frame)
        lay.addWidget(self.slider)
        self.frame_label = QLabel()
        self.frame_label.setWordWrap(True)
        lay.addWidget(self.frame_label)
        row = QHBoxLayout()
        self.btn_save = QPushButton("Save calibration  (Enter)")
        self.btn_save.clicked.connect(self._save)
        self.btn_reclick = QPushButton("Pick the point again")
        self.btn_reclick.clicked.connect(self._reclick)
        self.btn_recapture2 = QPushButton("Capture again")
        self.btn_recapture2.clicked.connect(self._recapture)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_reclick)
        row.addWidget(self.btn_recapture2)
        lay.addLayout(row)
        return w

    def _page_done(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.done_label = QLabel()
        self.done_label.setWordWrap(True)
        lay.addWidget(self.done_label)
        row = QHBoxLayout()
        self.btn_next = QPushButton("Next camera  (N)")
        self.btn_next.clicked.connect(self._next_stage)
        btn_again = QPushButton("Calibrate this camera again")
        btn_again.clicked.connect(self._recapture)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_next)
        row.addWidget(btn_again)
        row.addWidget(btn_close)
        lay.addLayout(row)
        QShortcut(QKeySequence("N"), self).activated.connect(self._next_stage)
        return w

    # -- state -> widgets -------------------------------------------------------

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)
        logger.info(text)

    def _rebuild_steps(self) -> None:
        self.steps_list.clear()
        for step in self.session.steps():
            QListWidgetItem(STEP_TITLES[step], self.steps_list)
        self.steps_list.setFixedHeight(20 + 22 * self.steps_list.count())

    def _refresh(self) -> None:
        step = self.session.step
        steps = self.session.steps()
        self.stage_label.setText(self.suite.label(self.suite.active_index))
        for i, s in enumerate(steps):
            item = self.steps_list.item(i)
            if item is None:
                continue
            done = steps.index(step) > i
            item.setText(
                ("✓  " if done else ("▶  " if s is step else "    ")) + STEP_TITLES[s]
            )
            f = item.font()
            f.setBold(s is step)
            item.setFont(f)
        self.title.setText(STEP_TITLES[step])
        self.instructions.setText(_instructions(self.session, step))
        self.controls.setCurrentWidget(self._pages[step])

        live = step in (Step.REFERENCE, Step.CAPTURE) and not self._mark_mode
        if live and not self._timer.isActive():
            self._timer.start()
        if not live and self._timer.isActive():
            self._timer.stop()
        self.view.set_click_enabled(
            step is Step.SELECT_POINT or self._mark_mode, magnifier=True
        )
        if not self._mark_mode:
            self.view.set_banner("")

        if step is Step.REFERENCE:
            can_pick = (
                self.suite.exterior_camera() is not None
                and self.suite.exterior_extrinsic() is not None
            )
            self.btn_pick_mark.setEnabled(can_pick and not self._mark_mode)
            self.btn_pick_mark.setToolTip(
                "" if can_pick else "Needs a calibrated exterior camera in this suite."
            )
            self.btn_record.setEnabled(not self._mark_mode)
            self.btn_mark_cancel.setVisible(self._mark_mode)

        if step is Step.CAPTURE:
            capturing = self.session.capturing
            motion = self.session.motion
            moving = motion is not None and motion.active
            self.btn_start.setEnabled(not capturing)
            self.btn_auto.setEnabled(not capturing)
            self.btn_stop.setEnabled(moving)
            self.btn_centre.setEnabled(not moving)
            self.btn_finish.setEnabled(
                capturing
                and not moving
                and self.session.status.n_keyframes >= self.session.config.min_keyframes
            )
            self.btn_load.setEnabled(not capturing)
            saved = load_saved_extrinsic(
                self.suite.calibration_path, self.session.config.camera_name
            )
            self.btn_skip.setVisible(
                len(self.suite.stages) > 1 and saved is not None and not capturing
            )
            self.progress.setRange(0, self.session.config.target_keyframes)
            self._update_area()
        elif step is Step.SELECT_POINT:
            self.btn_track.setEnabled(self.session.click_uv is not None)
            self._show_first_frame()
        elif step is Step.VERIFY and self.session.result is not None:
            self._populate_verify(self.session.result)
        elif step is Step.DONE:
            self.done_label.setText(
                "Kept the saved calibration."
                if self.session.skipped
                else f"Written to {self.session.saved_path}"
            )
            self.btn_next.setVisible(self.suite.has_next)
            if self.suite.has_next:
                nxt = self.suite.stages[self.suite.active_index + 1]
                self.btn_next.setText(f"Next: {nxt.config.camera_name}  (N)")

    def _on_poll(self) -> None:
        try:
            sample = self.session.poll()
        except Exception as exc:
            self._timer.stop()
            self._log(f"ERROR reading hardware: {exc}")
            QMessageBox.critical(
                self, "Hardware", f"Reading the camera or arm failed:\n{exc}"
            )
            return
        st = self.session.status
        overlays: list[Overlay] = []
        if self.session.capturing:
            if sample.accepted == "keyframe":
                self._flash_until = sample.timestamp + 0.4
            colour = GREEN if sample.settled else AMBER
            state = "still" if sample.settled else "moving"
            self.view.set_banner(
                f"{st.n_keyframes} still poses · {st.n_frames} frames · arm {state}"
                + ("  ●" if sample.timestamp < self._flash_until else ""),
                colour,
            )
            self.progress.setValue(min(st.n_keyframes, self.progress.maximum()))
            ext = st.extent_m() * 100
            self.capture_label.setText(
                f"TCP travelled {st.travel_since_keyframe_m * 100:.1f} cm since the last "
                f"still pose · coverage x {ext[0]:.0f} cm, y {ext[1]:.0f} cm, z {ext[2]:.0f} cm"
                + ("  — target reached, finish when ready" if st.target_reached else "")
                + ("  — frame limit reached" if st.full else "")
            )
            motion = self.session.motion
            moving = motion is not None and motion.active
            self.btn_finish.setEnabled(
                not moving and st.n_keyframes >= self.session.config.min_keyframes
            )
            if motion is not None:
                self._update_motion_label(motion)
        else:
            self.view.set_banner(
                "arm still" if sample.settled else "arm moving",
                GREEN if sample.settled else AMBER,
            )
        tcp = sample.tcp_pose
        if self.session.step is Step.CAPTURE:
            if not self._box_user_edited and not self.session.capturing:
                self._set_box(SafeBox.around(tcp[:3]), user=False)
            self._update_area(tcp)
        self.status.setText(
            f"TCP  x {tcp[0]:+.3f}  y {tcp[1]:+.3f}  z {tcp[2]:+.3f} m   "
            f"|q̇|max {np.max(np.abs(sample.joint_velocities)):.3f} rad/s"
        )
        self.view.set_image(sample.rgb, overlays)

    # -- actions ----------------------------------------------------------------

    def _pick_mark(self) -> None:
        """Freeze the exterior camera's view and let the operator click the mark."""
        if self.session.step is not Step.REFERENCE or self._mark_mode:
            return
        if (
            self.suite.exterior_extrinsic() is None
            or self.suite.exterior_camera() is None
        ):
            QMessageBox.information(
                self,
                "Exterior camera needed",
                "Picking the mark in an image needs a calibrated exterior camera in "
                "this suite. Touch the mark with the fingertips instead.",
            )
            return
        try:
            self._mark_frame = self.suite.exterior_frame()
        except Exception as exc:
            QMessageBox.warning(self, "Exterior camera", str(exc))
            return
        self._mark_mode = True
        self._refresh()
        self.view.set_image(self._mark_frame[0], [])
        self.view.set_banner(
            "EXTERIOR CAMERA — click the centre of the table mark", MAGENTA
        )
        self.ref_label.setText(
            "Click the mark in the exterior image. Its depth and the exterior "
            "calibration give its position in the robot frame."
        )

    def _cancel_mark(self) -> None:
        self._mark_mode = False
        self._mark_frame = None
        self._refresh()

    def _on_mark_click(self, u: float, v: float) -> None:
        try:
            point, depth = self.suite.mark_from_exterior_click(u, v, self._mark_frame)
            self.session.set_reference_point(point, source="clicked in exterior image")
        except Exception as exc:
            QMessageBox.warning(self, "Mark", str(exc))
            return
        assert self._mark_frame is not None
        self.view.set_image(
            self._mark_frame[0], [Overlay((u, v), MAGENTA, "crosshair", "mark")]
        )
        self.ref_label.setText(
            f"Mark at ({u:.0f}, {v:.0f}) px, depth {depth:.3f} m -> "
            f"[{point[0]:+.4f}, {point[1]:+.4f}, {point[2]:+.4f}] m in the base frame."
        )
        self._log(
            f"reference mark from the exterior image: {np.round(point, 4).tolist()} m "
            f"(depth {depth:.3f} m)"
        )
        self._mark_mode = False
        self._mark_frame = None
        self._refresh()

    def _next_stage(self) -> None:
        if self.session.step is not Step.DONE or not self.suite.has_next:
            return
        self.suite.advance()
        self._box_user_edited = False
        self._mark_mode = False
        # Per-stage widgets are shared; do not show the last camera's status.
        for label in (
            self.motion_label,
            self.capture_label,
            self.click_label,
            self.frame_label,
            self.verdict_label,
        ):
            label.setText("")
        self.ref_label.setText("No reference recorded.")
        self.progress.setValue(0)
        self._rebuild_steps()
        self._set_title()
        self._log(self.suite.label(self.suite.active_index))
        self._refresh()

    def _skip_stage(self) -> None:
        if self.session.step is not Step.CAPTURE or self.session.capturing:
            return
        ans = QMessageBox.question(
            self,
            "Skip this camera",
            f"Keep the saved calibration of {self.session.config.camera_name} and go "
            "on to the next camera?",
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self.session.skip()
        self._refresh()

    def _escape(self) -> None:
        if self._mark_mode:
            self._cancel_mark()
        else:
            self._stop_motion()

    def _record_reference(self) -> None:
        if self.session.step is not Step.REFERENCE or self._mark_mode:
            return
        try:
            p = self.session.record_reference()
        except Exception as exc:
            QMessageBox.warning(self, "Reference", str(exc))
            return
        self.ref_label.setText(f"Reference at {np.round(p, 4).tolist()} m (base frame)")
        self._log(f"reference mark at {np.round(p, 4).tolist()} m")
        self._refresh()

    # -- autonomous motion --------------------------------------------------------

    def _current_box(self) -> Optional[SafeBox]:
        try:
            return SafeBox(
                (self.box_spins["x"][0].value(), self.box_spins["x"][1].value()),
                (self.box_spins["y"][0].value(), self.box_spins["y"][1].value()),
                (self.box_spins["z"][0].value(), self.box_spins["z"][1].value()),
            )
        except ValueError:
            return None

    def _set_box(self, box: SafeBox, *, user: bool) -> None:
        self._box_updating = True
        for axis, (lo, hi) in zip("xyz", (box.x, box.y, box.z), strict=True):
            self.box_spins[axis][0].setValue(lo)
            self.box_spins[axis][1].setValue(hi)
        self._box_updating = False
        self._box_user_edited = user
        self._update_area()

    def _box_edited(self) -> None:
        if not self._box_updating:
            self._box_user_edited = True
            self._update_area()

    def _centre_box(self) -> None:
        s = self.session.last_sample
        if s is None:
            return
        self._set_box(SafeBox.around(s.tcp_pose[:3]), user=True)

    def _update_area(self, tcp=None) -> None:
        box = self._current_box()
        motion = self.session.motion
        if tcp is None and self.session.last_sample is not None:
            tcp = self.session.last_sample.tcp_pose
        self.area.update_state(
            box=box if motion is None or not motion.active else motion.config.box,
            tcp=tcp,
            targets=motion.targets if motion is not None else [],
            visited=motion.index if motion is not None else 0,
            current=(motion.index if motion is not None and motion.active else None),
            reference=self.session.reference_point_base,
            fault=motion is not None and motion.phase == "fault",
        )

    def _update_motion_label(self, motion) -> None:
        n = len(motion.targets)
        if motion.active:
            text = f"pose {min(motion.index + 1, n)}/{n} · {motion.phase}"
            colour = "#3cdc5a"
        elif motion.phase == "done":
            text = f"motion finished — {motion.message}. Press Finish capture."
            colour = "#3cdc5a"
        elif motion.phase == "fault":
            text = "MOTION STOPPED: " + motion.message
            colour = "#f04646"
        else:
            text = f"motion {motion.phase}: {motion.message}"
            colour = "#ffbe28"
        self.motion_label.setText(f"<span style='color:{colour}'>{text}</span>")
        if not motion.active:
            self.btn_stop.setEnabled(False)
            self.btn_finish.setEnabled(
                self.session.status.n_keyframes >= self.session.config.min_keyframes
            )

    def _motion_config(self) -> Optional[MotionConfig]:
        box = self._current_box()
        if box is None:
            QMessageBox.warning(self, "Safe box", "Each box range must be increasing.")
            return None
        return MotionConfig(
            box=box,
            n_targets=self.spin_poses.value(),
            speed=self.spin_speed.value(),
            tilt_deg=self.spin_tilt.value(),
            roll_deg=self.spin_roll.value(),
        )

    def _start_autonomous(self, *, confirm: bool = True) -> None:
        if self.session.step is not Step.CAPTURE or self.session.capturing:
            return
        cfg = self._motion_config()
        if cfg is None:
            return
        if confirm:
            b = cfg.box
            ans = QMessageBox.question(
                self,
                "The arm will move",
                "The arm will move by itself through "
                f"{cfg.n_targets} poses inside\n"
                f"  x {b.x[0]:+.3f} .. {b.x[1]:+.3f} m\n"
                f"  y {b.y[0]:+.3f} .. {b.y[1]:+.3f} m\n"
                f"  z {b.z[0]:+.3f} .. {b.z[1]:+.3f} m\n"
                f"at {cfg.speed:.2f} m/s. The robot must be in Remote Control and out "
                "of freedrive.\n\nKeep the e-stop in your hand. Start?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        try:
            run = self.session.start_autonomous_capture(cfg)
        except Exception as exc:
            self._log(f"autonomous capture not started: {exc}")
            QMessageBox.warning(self, "Autonomous capture", str(exc))
            self._refresh()
            return
        self._log(
            f"autonomous capture: {len(run.targets)} poses, {cfg.speed} m/s, box "
            f"x{cfg.box.x} y{cfg.box.y} z{cfg.box.z}"
        )
        self._refresh()

    def _stop_motion(self) -> None:
        motion = self.session.motion
        if motion is None or not motion.active:
            return
        self.session.stop_motion()
        self._log("motion stopped by operator")
        self._refresh()

    def _start_capture(self) -> None:
        if self.session.step is not Step.CAPTURE or self.session.capturing:
            return
        try:
            ds = self.session.start_capture()
        except Exception as exc:
            QMessageBox.warning(self, "Capture", str(exc))
            return
        self._log(f"capturing to {ds.root}")
        self._refresh()

    def _finish_capture(self) -> None:
        if not self.session.capturing:
            return
        try:
            self.session.finish_capture()
        except Exception as exc:
            QMessageBox.information(self, "Not yet", str(exc))
            return
        st = self.session.status
        self._log(
            f"capture finished: {st.n_frames} frames, {st.n_keyframes} still poses"
        )
        self._refresh()

    def _load_existing(self) -> None:
        start = str(self.session.config.dataset_root.parent)
        root = QFileDialog.getExistingDirectory(self, "Previous capture folder", start)
        if not root:
            return
        try:
            ds = self.session.use_existing_dataset(root)
        except Exception as exc:
            QMessageBox.warning(self, "Load capture", str(exc))
            return
        self._log(f"loaded {ds.n_frames} frames from {ds.root}")
        self._refresh()

    def _show_first_frame(self) -> None:
        try:
            rgb = self.session.first_frame()
        except Exception as exc:
            self._log(f"cannot show first frame: {exc}")
            return
        overlays = []
        if self.session.click_uv is not None:
            overlays.append(
                Overlay(self.session.click_uv, GREEN, "crosshair", "tracked point")
            )
        self.view.set_image(rgb, overlays)

    def _on_click(self, u: float, v: float) -> None:
        if self._mark_mode:
            self._on_mark_click(u, v)
            return
        if self.session.step is not Step.SELECT_POINT:
            return
        try:
            self.session.set_click(u, v)
        except Exception as exc:
            QMessageBox.warning(self, "Point", str(exc))
            return
        self.click_label.setText(f"Point at ({u:.1f}, {v:.1f}) px")
        self._log(f"tracked point picked at ({u:.1f}, {v:.1f})")
        self._refresh()

    def _start_tracking(self) -> None:
        if self.session.step is not Step.SELECT_POINT or self.session.click_uv is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self.session.step = Step.TRACK
        self._refresh()
        self._worker = TrackingWorker(self.session)
        self._worker.line.connect(self._log)
        self._worker.succeeded.connect(self._on_tracked)
        self._worker.failed.connect(self._on_tracking_failed)
        self._worker.start()

    def _on_tracked(self, result: CalibrationResult) -> None:
        self._log(f"solved: verdict {result.verdict.level}")
        self._refresh()

    def _on_tracking_failed(self, message: str) -> None:
        self._log("tracking failed: " + message.splitlines()[0])
        logger.error(message)
        QMessageBox.critical(self, "Tracking failed", message.split("\n\n")[0])
        self._refresh()

    def _populate_verify(self, result: CalibrationResult) -> None:
        n = len(result.records)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        s = result.solution
        colour = {"good": "#3cdc5a", "check": "#ffbe28", "bad": "#f04646"}[
            result.verdict.level
        ]
        lines = [
            f"<b style='color:{colour}'>Verdict: {result.verdict.level.upper()}</b>",
            "<br>".join("• " + r for r in result.verdict.reasons),
            f"Inliers {s.n_used}/{s.n_candidates} still frames · mean {s.mean_px:.2f} px, "
            f"median {s.median_px:.2f} px, max {s.max_inlier_px:.2f} px",
        ]
        if result.depth is not None:
            lines.append(
                f"Depth vs solved distance: median {result.depth.median_m * 100:+.1f} cm "
                f"(|·| {result.depth.median_abs_m * 100:.1f} cm) over {result.depth.n_valid} readings"
            )
        if result.kalib_difference is not None:
            d = result.kalib_difference
            lines.append(
                f"Kalib's own PnP differs by {d.translation_m * 100:.1f} cm / {d.rotation_deg:.2f}°"
            )
        pos = s.T_parent_cam[:3, 3]
        lines.append(
            f"Camera origin in the {self.session.parent_frame} frame: "
            f"[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m"
        )
        self.verdict_label.setText("<br>".join(lines))
        self.btn_save.setEnabled(True)
        self._show_verify_frame(0)

    def _show_verify_frame(self, i: int) -> None:
        result = self.session.result
        if result is None or self.session.dataset is None or not result.records:
            return
        i = int(np.clip(i, 0, len(result.records) - 1))
        self._verify_index = i
        rec = result.records[i]
        rgb = self.session.dataset.load_rgb(rec.index)
        tracked = result.tracking.tracks[i]
        predicted = result.reprojected()[i]
        overlays = []
        if np.all(np.isfinite(tracked)):
            overlays.append(Overlay(tracked, GREEN, "ring"))
        if np.all(np.isfinite(predicted)):
            overlays.append(Overlay(predicted, MAGENTA, "cross"))
        self.view.set_image(rgb, overlays)
        err = result.solution.reproj_err_px[i]
        used = bool(result.solution.used[i])
        dres = (
            result.depth.residual_m[i]
            if result.depth is not None and i < len(result.depth.residual_m)
            else float("nan")
        )
        parts = [
            f"frame {i + 1}/{len(result.records)}",
            "still" if rec.settled else "moving",
            "inlier" if used else ("excluded" if rec.settled else "not used"),
            f"error {err:.1f} px" if np.isfinite(err) else "not tracked",
        ]
        if np.isfinite(dres):
            parts.append(f"depth residual {dres * 100:+.1f} cm")
        self.frame_label.setText(" · ".join(parts))
        self.view.set_banner(
            " · ".join(parts), GREEN if used else (AMBER if rec.settled else WHITE)
        )

    def _step_verify(self, delta: int) -> None:
        if self.session.step is Step.VERIFY:
            self.slider.setValue(self.slider.value() + delta)

    def _save(self) -> None:
        if self.session.step is not Step.VERIFY or self.session.result is None:
            return
        if not self.session.result.verdict.ok:
            ans = QMessageBox.question(
                self,
                "Verdict is bad",
                "The checks flag this calibration as bad. Save it anyway?",
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        try:
            path = self.session.save()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._log(f"saved to {path}")
        self._refresh()

    def _reclick(self) -> None:
        if self.session.step in (Step.VERIFY, Step.DONE):
            self.session.step = Step.SELECT_POINT
            self._refresh()

    def _recapture(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self.session.motion is not None and self.session.motion.active:
            self.session.stop_motion("capture restarted")
        self.session.motion = None
        self.session.capturing = False
        self.session.step = (
            Step.REFERENCE
            if self.session.mode is CalibrationMode.WRIST
            else Step.CAPTURE
        )
        self.session.result = None
        self.session.click_uv = None
        self._refresh()

    def closeEvent(self, a0: Optional[QCloseEvent]) -> None:  # noqa: N802
        self._timer.stop()
        if self.session.motion is not None and self.session.motion.active:
            self.session.stop_motion("window closed")
        if self._worker is not None and self._worker.isRunning():
            cancel = getattr(self.session.tracker, "cancel", None)
            if cancel is not None:
                cancel()
            self._worker.wait(3000)
        self.suite.close()
        super().closeEvent(a0)


def run_gui(target: CalibrationSuite | CalibrationSession) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = CalibrationWindow(target)
    window.resize(1500, 960)
    window.show()
    return app.exec()


__all__ = [
    "CalibrationWindow",
    "ImageView",
    "Overlay",
    "SafeAreaView",
    "TrackingWorker",
    "run_gui",
]

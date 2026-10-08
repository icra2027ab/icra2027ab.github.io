"""The demo window, driven offscreen.

What an operator's safety rests on: every action is a button *and* a hotkey
that enable together, Space starts and S stops a rollout that then homes the
arm, and the switch is what the window says it is. Pixels are not tested;
that the overlay path runs on an analysed view is.
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is not installed")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import (
    QFocusEvent,  # noqa: E402
    QPalette,  # noqa: E402
)
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402

from policy_abstraction.demo.gui import app as app_module  # noqa: E402
from policy_abstraction.demo.gui import window as window_module  # noqa: E402
from policy_abstraction.demo.gui.window import (  # noqa: E402
    DEMO_PLACEHOLDER,
    NO_VIEWER_TEXT,
    PERCEPTION_PLACEHOLDER,
    DemoWindow,
)
from policy_abstraction.demo.session import (  # noqa: E402
    DemoSession,
    DemoState,
    LiveView,
    OverlayMode,
)
from policy_abstraction.vlm.types import DistancePair  # noqa: E402
from tests.experiments.conftest import FakeBackends, FakeEnv  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_windows(qapp):
    """Leave no window standing.

    Closed, not ``deleteLater``'d: a deferred deletion of a window holding a
    pyqtgraph scene is processed by whichever event loop runs next — another
    test module's ``app.exec()`` — and segfaults there. The window fixture
    deletes its own window synchronously instead.
    """
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
    qapp.processEvents()


@pytest.fixture
def session():
    backends = FakeBackends(FakeEnv(wrist=True), runner_kwargs={"steps": 10_000})
    s = DemoSession(backends, guided_warmup=False)
    s.prepare_env()
    yield s
    s.shutdown()


@pytest.fixture
def window(qapp, session):
    win = DemoWindow(session, refresh_hz=50.0)
    win.show()
    qapp.processEvents()
    yield win
    win._timer.stop()
    # A window closed over a running rollout asks "stop and quit?" in a
    # modal dialog, which offscreen nobody answers: stop the rollout first so
    # a failed test cannot hang the whole run.
    if session.is_running:
        session.request_stop()
        _pump(qapp, lambda: not session.is_running, timeout=10.0)
    win.close()
    # Delete the C++ side now, while no event loop runs, rather than leaving
    # it to a deferred deletion inside some later test's exec().
    from PyQt6 import sip

    sip.delete(win)
    qapp.processEvents()


def _pump(qapp, predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def _trigger(window, action_id: str) -> None:
    window._qactions[action_id].trigger()


class TestWiring:
    def test_every_action_has_a_button_and_a_hotkey_that_agree(self, window):
        for action in window.actions_list:
            button = window._buttons[action.id]
            qaction = window._qactions[action.id]
            assert qaction.shortcut().toString() == action.shortcut.replace(
                "Esc", "Esc"
            )
            assert button.isEnabled() == qaction.isEnabled()
            assert action.shortcut in button.text()
        ids = {a.id for a in window.actions_list}
        assert {
            "start",
            "stop",
            "guidance",
            "overlays",
            "trends",
            "trend_window",
            "viewer",
            "layout",
            "fullscreen",
            "quit",
        } <= ids
        shortcuts = {a.id: a.shortcut for a in window.actions_list}
        assert shortcuts["viewer"] == "V"
        assert shortcuts["layout"] == "L"
        assert shortcuts["fullscreen"] == "F11"
        assert shortcuts["trend_window"] == "W"
        assert "V" in window.help_panel._body.text()
        assert "F11" in window.help_panel._body.text()

    def test_start_and_stop_enable_in_opposition(self, window, session, qapp):
        assert window._buttons["start"].isEnabled()
        assert not window._buttons["stop"].isEnabled()
        _trigger(window, "start")
        _pump(qapp, lambda: session.state is DemoState.RUNNING)
        window._tick()
        # Space stays live while running: it is now the pause key.
        assert window._buttons["start"].isEnabled()
        assert "Pause" in window._buttons["start"].text()
        assert window._buttons["stop"].isEnabled()
        _trigger(window, "stop")
        _pump(qapp, lambda: session.state is DemoState.IDLE)
        assert session.backends.env.homed == 1
        window._tick()
        assert window._buttons["start"].isEnabled()

    def test_the_switch_is_what_the_button_says(self, window, session):
        assert not session.guided
        assert "OFF" in window._buttons["guidance"].text()
        _trigger(window, "guidance")
        assert session.guided
        assert "ON" in window._buttons["guidance"].text()
        _trigger(window, "guidance")
        assert not session.guided

    def test_overlays_cycle(self, window, session):
        assert session.overlay_mode is OverlayMode.AUTO
        _trigger(window, "overlays")
        assert session.overlay_mode is OverlayMode.ON
        assert "on" in window._buttons["overlays"].text()

    def test_t_hides_the_whole_bottom_row(self, window, qapp):
        assert window.bottom_row.isVisible()
        assert window.wrist_view.isVisible()
        assert window.demo_view.isVisible()
        assert window.trends.isVisible()
        assert window.pairs_panel.isVisible()
        assert "shown" in window._buttons["trends"].text()
        _trigger(window, "trends")
        assert not window.bottom_row.isVisible()
        assert not window.trends.isVisible()
        assert not window.wrist_view.isVisible()
        assert not window.pairs_panel.isVisible()
        assert "hidden" in window._buttons["trends"].text()
        _trigger(window, "trends")
        qapp.processEvents()
        assert window.bottom_row.isVisible()

    def test_w_toggles_the_trend_window(self, window):
        assert window.trends.windowed, "a snapshot around now by default"
        assert "snapshot" in window._buttons["trend_window"].text()
        assert window._qactions["trend_window"].shortcut().toString() == "W"
        _trigger(window, "trend_window")
        assert not window.trends.windowed
        assert "full" in window._buttons["trend_window"].text()
        window._tick()
        assert "full trajectory" in window.pairs_panel._body.text(), (
            "the pairs panel is the legend, and carries the mode line"
        )
        _trigger(window, "trend_window")
        assert window.trends.windowed
        assert "snapshot" in window._buttons["trend_window"].text()
        window._tick()
        assert "window:" in window.pairs_panel._body.text()

    def test_layout_switch_is_dead_without_a_viewer(self, window, session):
        assert session.viewer_layout is None
        assert not window._buttons["layout"].isEnabled()
        assert not window._qactions["layout"].isEnabled()

    def test_layout_switch_flips_the_layout(self, window, session):
        session.visualizer = FakeViz()
        window._tick()
        assert window._buttons["layout"].isEnabled()
        _trigger(window, "layout")
        assert session.viewer_layout == "full"
        _trigger(window, "layout")
        assert session.viewer_layout == "scene"

    def test_fullscreen_toggles_and_returns_to_maximised(self, window, qapp):
        assert not window.isFullScreen()
        _trigger(window, "fullscreen")
        qapp.processEvents()
        assert window.isFullScreen()
        assert "on" in window._buttons["fullscreen"].text()
        _trigger(window, "fullscreen")
        qapp.processEvents()
        assert not window.isFullScreen()
        assert window.isMaximized(), "F11 off returns to the maximised window"


class FakeViz:
    """Stands in for the Rerun visualiser: a layout, and a web viewer URL
    when it serves in-process."""

    first_time = False

    def __init__(self, url=None):
        self.layout = "scene"
        self.web_viewer_url = url

    def set_layout(self, layout):
        self.layout = layout

    def close(self):
        pass


class FakeScripts:
    def __init__(self):
        self.inserted = []

    def insert(self, script):
        self.inserted.append(script)


class FakePage:
    def __init__(self):
        self._scripts = FakeScripts()

    def scripts(self):
        return self._scripts


class FakeWebView(QLabel):
    """A widget with a ``QWebEngineView``'s surface, minus Chromium.

    Offscreen tests must not start a browser process, and deleting one
    synchronously is its own hazard; the wiring is what is under test.
    """

    def __init__(self):
        super().__init__("fake browser")
        self.urls: list[str] = []
        self.reloads = 0
        self._page = FakePage()

    def setUrl(self, url):  # noqa: N802
        self.urls.append(url.toString())

    def reload(self):
        self.reloads += 1

    def page(self):
        return self._page


class TestViewerPanel:
    def test_hidden_with_a_placeholder_when_nothing_is_served(self, window, session):
        assert session.viewer_url is None
        assert window.viewer_panel.isHidden()
        assert window.viewer_placeholder.text() == NO_VIEWER_TEXT
        assert window.viewer_panel.currentWidget() is window.viewer_placeholder
        assert "hidden" in window._buttons["viewer"].text()

    def test_v_shows_and_hides_the_panel(self, window, qapp):
        assert window.viewer_panel.isHidden()
        _trigger(window, "viewer")
        qapp.processEvents()
        assert not window.viewer_panel.isHidden()
        assert "shown" in window._buttons["viewer"].text()
        _trigger(window, "viewer")
        assert window.viewer_panel.isHidden()

    def test_a_late_url_loads_into_the_browser_once(
        self, window, session, qapp, monkeypatch
    ):
        made: list[FakeWebView] = []

        def factory():
            made.append(FakeWebView())
            return made[-1]

        monkeypatch.setattr(window_module, "_make_web_view", factory)
        window._tick()
        assert not made, "no URL yet, no browser"
        url = "http://localhost:9090/?url=rerun+http://127.0.0.1:9876/proxy"
        session.visualizer = FakeViz(url)
        window._tick()
        qapp.processEvents()
        assert len(made) == 1
        web = made[0]
        assert web.urls == [url]
        assert web.reloads == 1, "the placeholder→browser transition reloads"
        assert window.viewer_panel.currentWidget() is web
        assert not window.viewer_panel.isHidden()
        assert web.minimumSize().width() == 320
        assert web.minimumSize().height() == 240
        window._tick()
        window._tick()
        assert web.urls == [url], "loaded once, not on every tick"

    def test_the_viewer_is_asked_for_its_dark_theme(
        self, window, session, qapp, monkeypatch
    ):
        """The Rerun viewer draws its UI on a canvas and reads the colour
        scheme from a media query QtWebEngine answers as light; the script
        installed at document creation answers it as dark instead."""
        pytest.importorskip("PyQt6.QtWebEngineCore")
        assert "prefers-color-scheme" in window_module._PREFER_DARK_JS
        made: list[FakeWebView] = []

        def factory():
            made.append(FakeWebView())
            return made[-1]

        monkeypatch.setattr(window_module, "_make_web_view", factory)
        session.visualizer = FakeViz("http://localhost:9090/")
        window._tick()
        scripts = made[0].page().scripts().inserted
        assert [s.name() for s in scripts] == ["prefer-dark"]
        script = scripts[0]
        assert "prefers-color-scheme" in script.sourceCode()
        assert script.runsOnSubFrames()
        from PyQt6.QtWebEngineCore import QWebEngineScript

        assert (
            script.injectionPoint() is QWebEngineScript.InjectionPoint.DocumentCreation
        )

    def test_an_uncreatable_browser_leaves_a_placeholder_that_says_why(
        self, window, session, qapp, monkeypatch
    ):
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            raise ImportError("No module named 'PyQt6.QtWebEngineWidgets'")

        monkeypatch.setattr(window_module, "_make_web_view", factory)
        url = "http://localhost:9090/?url=rerun+http://127.0.0.1:9876/proxy"
        session.visualizer = FakeViz(url)
        window._tick()
        qapp.processEvents()
        assert calls == 1
        assert window.viewer_panel.currentWidget() is window.viewer_placeholder
        text = window.viewer_placeholder.text()
        assert "unavailable" in text
        assert "QtWebEngineWidgets" in text
        assert url in text, "the operator can still open the page by hand"
        assert not window.viewer_panel.isHidden()
        window._tick()
        assert calls == 1, "not retried every tick"


class TestPainting:
    def test_the_preview_frame_reaches_the_view(self, window, qapp):
        window._tick()
        qapp.processEvents()
        assert window.view._pixmap is not None
        assert window.wrist_view._pixmap is not None

    def test_an_analysed_view_renders_without_raising(
        self, window, session, monkeypatch
    ):
        class Result:
            pairs = [DistancePair("ee", 5, "ee_to_cup", subtraj_idx=0)]
            live_distances = {"ee_to_cup": 0.2}
            ref_distances = {"ee_to_cup": 0.18}
            distance_errors = {"ee_to_cup": 0.02}
            entity_centroids = {"ee": np.array([0.0, 0.0, 0.5]), 5: np.zeros(3)}
            current_subtraj_idx = 0
            mta_progress = 0.4
            relevances = {"cup": 0.8}
            perception_overlays = None

        seg = np.zeros((32, 32), dtype=np.int32)
        seg[10:20, 10:20] = 5
        view = LiveView(
            image=np.full((32, 32, 3), 30, dtype=np.uint8),
            step=9,
            timestamp=time.time(),
            guided=True,
            intrinsic=np.array([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]]),
            extrinsic=np.eye(4),
            abstraction=Result(),
            seg=seg,
            names={5: "cup"},
            ee_pos=np.array([0.0, 0.0, 0.5]),
            perception_overlay=np.zeros((32, 32, 3), dtype=np.uint8),
            ref_frame=np.zeros((32, 32, 3), dtype=np.uint8),
            guidance_info={
                "guidance_factor": 0.0,
                "requested_factor": 1.0,
                "gated": True,
            },
            colors={"ee_to_cup": (255, 0, 0)},
        )
        monkeypatch.setattr(session, "live_view", lambda: view)
        session.state = DemoState.RUNNING
        try:
            window._refresh()
        finally:
            session.state = DemoState.IDLE
        assert window.view._pixmap is not None
        assert window.demo_view._pixmap is not None
        assert window.perception_panel.view._pixmap is not None
        assert "ee_to_cup" in window.pairs_panel._body.text()
        assert "withheld" in window.guidance_panel._body.text()

    def test_content_never_grows_the_window(self, window, session, qapp, monkeypatch):
        """The operator's display is ~1400x960: a RUNNING status line, long
        pair labels and a 640x480 frame must all fit inside the window the
        layout was given, not push it off the screen."""
        long_label = "gripper_fingertip_centre_to_red_cable_connector_housing_end"

        class Result:
            pairs = [
                DistancePair("ee", 5, long_label, subtraj_idx=0),
                DistancePair("ee", 6, long_label + "_2", subtraj_idx=0),
            ]
            live_distances = {long_label: 0.2, long_label + "_2": 0.3}
            ref_distances = {long_label: 0.18, long_label + "_2": 0.28}
            distance_errors = {long_label: 0.02, long_label + "_2": 0.02}
            entity_centroids = {}
            current_subtraj_idx = 0
            mta_progress = 0.4
            relevances = {"cup": 0.8}
            perception_overlays = None

        view = LiveView(
            image=np.full((480, 640, 3), 30, dtype=np.uint8),
            step=143,
            timestamp=time.time(),
            guided=False,
            abstraction=Result(),
            ref_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            perception_overlay=np.zeros((480, 640, 3), dtype=np.uint8),
            wrist=np.zeros((480, 640, 3), dtype=np.uint8),
            colors={long_label: (255, 0, 0)},
        )
        frames = [view]
        monkeypatch.setattr(session, "live_view", lambda: frames[-1])
        session._live_step, session._step_limit = 143, 3000
        window.resize(1000, 700)
        qapp.processEvents()
        before = window.size()
        session.state = DemoState.RUNNING
        session.guided = True  # the switch differs from the chunk on the arm
        try:
            window._refresh()
            for _ in range(5):
                frames.append(dataclasses.replace(view, timestamp=time.time()))
                window._tick()
                qapp.processEvents()
            # FakeBackends carry no guidance config, so the switch being on
            # is said as unavailable — and not as a pending switch.
            status = window.status.text()
            assert status.startswith(
                "RUNNING · GUIDANCE UNAVAILABLE · step 143/3000 · "
            )
            assert "replan" not in status
            monkeypatch.setattr(window, "_guidance_unavailable", lambda: False)
            text, _color = window._status_text()
            assert text.startswith("RUNNING · UNGUIDED · step 143/3000 · ")
            assert text.endswith(" s · switching at next replan")
        finally:
            session.state = DemoState.IDLE
            session.guided = False
        assert window.status.wordWrap()
        assert long_label in window.pairs_panel._body.text()
        assert window.sizeHint().width() < 1400
        assert window.minimumSizeHint().width() < 1400
        assert window.minimumSizeHint().width() <= 960
        assert window.minimumSizeHint().height() <= 600
        assert window.size() == before, "content resized the window"
        assert window.side_scroll.width() == 340
        assert 240 <= window.pairs_panel.width() <= 320, "the pairs pane is clamped"
        # The 640x480 frames went to the bottom row's views, which the
        # layout sizes — not the frames.
        assert window.wrist_view._pixmap is not None
        assert window.demo_view._pixmap is not None
        assert window.wrist_view.width() < 640
        assert window.demo_view.height() < 480

    def test_the_side_column_keeps_only_the_info_panels(self, window):
        side = window.side_scroll.widget()
        assert not hasattr(window, "wrist_panel")
        assert not hasattr(window, "ref_panel")
        assert window.perception_panel.parent() is side
        assert window.guidance_panel.parent() is side
        assert window.help_panel.parent() is side
        assert window.wrist_view.window() is window
        # The pairs panel is the trend plot's legend, so it sits beside it.
        assert window.pairs_panel.parent() is window.bottom_row
        assert window.trends.legend is None, "the plots draw no legend of their own"
        assert window.wrist_box.parent() is window.bottom_row
        assert window.demo_box.parent() is window.bottom_row
        assert window.trends.parent() is window.bottom_row
        assert window.bottom_row.count() == 4
        assert window.bottom_row.indexOf(window.pairs_panel) == 3
        assert window.bottom_row.handleWidth() == 12
        assert window.centre_splitter.handleWidth() == 10
        for box in (window.wrist_box, window.demo_box):
            margins = box.layout().contentsMargins()
            assert (margins.left(), margins.right()) == (6, 6)

    def test_the_bottom_row_is_tall_and_shares_its_width(self, window, qapp):
        """The trend plots are what the audience reads: the row starts at
        about 42 % of the height, never under 260 px, and the two views
        beside the trends take about a fifth of the width each."""
        window.resize(1900, 1000)
        qapp.processEvents()
        window._balance_bottom()
        qapp.processEvents()
        assert window.bottom_row.minimumHeight() == 260
        centre, bottom = window.main_splitter.sizes()
        assert bottom >= 260
        assert 0.35 < bottom / (centre + bottom) < 0.5
        wrist, demo, trends, pairs = window.bottom_row.sizes()
        total = wrist + demo + trends + pairs
        assert 0.15 < wrist / total < 0.3
        assert 0.15 < demo / total < 0.3
        assert trends > wrist + demo
        assert 240 <= pairs <= 320
        assert window.pairs_panel.x() > window.trends.x()
        assert window.pairs_panel.width() == pairs
        # Shrinking the window shrinks the views: they never hold their size.
        window.resize(1000, 700)
        qapp.processEvents()
        assert window.bottom_row.height() >= 260
        assert window.wrist_view.height() <= window.bottom_row.height()

    def test_the_layout_scales_with_the_window(self, window, qapp, session):
        window.resize(1000, 700)
        qapp.processEvents()
        _trigger(window, "viewer")  # the placeholder stands in for the scene
        qapp.processEvents()
        small = window.centre_splitter.sizes()
        assert small[1] >= 320 and small[0] >= 160
        window.resize(1400, 960)
        qapp.processEvents()
        large = window.centre_splitter.sizes()
        assert large[2] == small[2] == 340, "the side column keeps its width"
        assert large[0] > small[0] and large[1] > small[1], (
            "the camera and the scene share the extra width"
        )
        assert large[1] > 320, "a panel shown late still gets its share"

    def test_an_unguided_launch_is_said_on_the_frame(self, window, session):
        # FakeBackends' args carry no guidance config, so the switch has
        # nothing to open — the window must say so rather than show a
        # switch that silently does nothing.
        assert not session.guidance_configured
        assert any("unguided" in b for b in window._badges(session.live_view()))


def _bare_view(**overrides) -> LiveView:
    fields = dict(
        image=np.full((32, 32, 3), 30, dtype=np.uint8),
        step=1,
        timestamp=time.time(),
        guided=False,
    )
    fields.update(overrides)
    return LiveView(**fields)


class _Analysed:
    pairs = [DistancePair("ee", 5, "ee_to_cup", subtraj_idx=0)]
    live_distances = {"ee_to_cup": 0.2}
    ref_distances = {"ee_to_cup": 0.18}
    distance_errors = {"ee_to_cup": 0.02}
    entity_centroids = {}
    current_subtraj_idx = 0
    mta_progress = 0.4
    relevances = {"cup": 0.8}
    perception_overlays = None


def _analysed_view(**overrides) -> LiveView:
    return _bare_view(
        abstraction=_Analysed(),
        ref_frame=np.zeros((32, 32, 3), dtype=np.uint8),
        perception_overlay=np.zeros((32, 32, 3), dtype=np.uint8),
        colors={"ee_to_cup": (255, 0, 0)},
        **overrides,
    )


class TestDimming:
    """Whatever only guidance feeds goes dark when the chunk on the arm is
    unguided, so the switch's position reads from across the room."""

    def test_the_guidance_panes_have_dimmers(self, window):
        targets = {d.target for d in window.dimmers}
        assert targets == {
            window.demo_box,
            window.trends,
            window.pairs_panel,
            window.perception_panel,
            window.guidance_panel,
        }
        for dimmer in window.dimmers:
            assert dimmer.parent() is dimmer.target
            assert dimmer.caption == "guidance off"
            assert dimmer.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def test_dimmed_follows_the_executing_chunk(
        self, window, session, qapp, monkeypatch
    ):
        frames = [_analysed_view(guided=False)]
        monkeypatch.setattr(session, "live_view", lambda: frames[-1])
        session.state = DemoState.RUNNING
        try:
            window._refresh()
            assert window.dimmed
            assert all(d.isVisible() for d in window.dimmers)
            for dimmer in window.dimmers:
                assert dimmer.size() == dimmer.target.size()
            frames.append(_analysed_view(guided=True))
            window._tick()
            assert not window.dimmed
            assert all(d.isHidden() for d in window.dimmers)
            # The switch is on, the chunk on the arm still the old kind: the
            # picture is about to change, so no flicker.
            frames.append(_analysed_view(guided=False))
            session.guided = True
            window._tick()
            assert not window.dimmed
            # Flipped off while a guided chunk executes: never dim on a
            # guided chunk.
            frames.append(_analysed_view(guided=True))
            session.guided = False
            window._tick()
            assert not window.dimmed
            frames.append(_analysed_view(guided=False))
            window._tick()
            assert window.dimmed
        finally:
            session.state = DemoState.IDLE
            session.guided = False

    def test_idle_dims_by_the_switch(self, window, session):
        assert session.state is DemoState.IDLE
        assert not session.guided
        window._refresh()
        assert window.dimmed
        assert all(d.isVisible() for d in window.dimmers)
        _trigger(window, "guidance")
        assert session.guided
        assert not window.dimmed
        assert all(d.isHidden() for d in window.dimmers)
        _trigger(window, "guidance")
        assert window.dimmed

    def test_a_dimmer_follows_its_target(self, window, qapp):
        dimmer = window.dimmers[0]
        dimmer.set_active(True)
        window.resize(1300, 900)
        qapp.processEvents()
        assert dimmer.size() == dimmer.target.size()
        assert dimmer.geometry().topLeft().x() == 0


class TestFreshRollout:
    def test_a_new_rollout_starts_with_fresh_panes(
        self, window, session, qapp, monkeypatch
    ):
        """The previous rollout's demo frame, masks and pairs must never be
        on screen at the start of the next one."""
        frames = [_analysed_view()]
        monkeypatch.setattr(session, "live_view", lambda: frames[-1])
        session.state = DemoState.RUNNING
        try:
            window._refresh()
            assert window.demo_view._pixmap is not None
            assert window.perception_panel.view._pixmap is not None
            assert "ee_to_cup" in window.pairs_panel._body.text()
            # The rollout ends; the last analysis stays up while idle.
            session.state = DemoState.IDLE
            window._tick()
            assert window.demo_view._pixmap is not None
            # A new rollout starts on a snapshot the session has blanked,
            # exactly as start_rollout leaves it.
            frames.append(_bare_view())
            session.state = DemoState.RUNNING
            window._tick()
            qapp.processEvents()
            assert window.demo_view._pixmap is None
            assert window.demo_view.pixmap().isNull()
            assert window.demo_view.text() == DEMO_PLACEHOLDER
            assert window.perception_panel.view._pixmap is None
            assert window.perception_panel.view.text() == PERCEPTION_PLACEHOLDER
            assert "ee_to_cup" not in window.pairs_panel._body.text()
            assert window.trends.pair_count == 0
            assert window.trends.sample_count == 0
            # A resize does not bring the old frame back.
            window.resize(1300, 900)
            qapp.processEvents()
            assert window.demo_view.pixmap().isNull()
            # And this rollout's first analysis paints as usual.
            frames.append(_analysed_view())
            window._tick()
            assert window.demo_view._pixmap is not None
            assert "ee_to_cup" in window.pairs_panel._body.text()
        finally:
            session.state = DemoState.IDLE

    def test_starting_for_real_clears_the_panes(
        self, window, session, qapp, monkeypatch
    ):
        real_live_view = session.live_view
        frames = [_analysed_view()]
        monkeypatch.setattr(session, "live_view", lambda: frames[-1])
        session.state = DemoState.RUNNING
        window._refresh()
        session.state = DemoState.IDLE
        window._tick()
        assert window.demo_view._pixmap is not None
        monkeypatch.setattr(session, "live_view", real_live_view)
        _trigger(window, "start")
        _pump(qapp, lambda: session.state is DemoState.RUNNING)
        window._tick()
        assert window.demo_view._pixmap is None
        assert window.perception_panel.view._pixmap is None
        _trigger(window, "stop")
        _pump(qapp, lambda: session.state is DemoState.IDLE)


class TestPalette:
    def test_the_dark_palette_has_the_window_colours(self, qapp):
        palette = app_module.dark_palette()
        role = palette.ColorRole
        assert palette.color(role.Window).name() == "#101014"
        assert palette.color(role.Base).name() == "#101014"
        assert palette.color(role.AlternateBase).name() == "#16161c"
        for r in (role.Text, role.WindowText, role.ButtonText):
            assert palette.color(r).name() == "#d8d8e0"
        assert palette.color(role.Button).name() == "#3a3a44"
        assert palette.color(role.Highlight).name() == "#2b6cb0"
        assert palette.color(role.HighlightedText).name() == "#ffffff"

    def test_install_sets_fusion_and_the_palette(self, qapp):
        before_style, before_palette = qapp.style().objectName(), qapp.palette()
        try:
            app_module.install_dark_palette(qapp)
            assert qapp.style().objectName().lower() == "fusion"
            assert qapp.palette().color(QPalette.ColorRole.Window).name() == "#101014"
            # The placeholder text of a camera view stays legible on it.
            view = window_module.CameraView()
            assert view.palette().color(QPalette.ColorRole.WindowText).lightness() > 128
            view.deleteLater()
        finally:
            qapp.setPalette(before_palette)
            qapp.setStyle(before_style)


class TestClosing:
    def test_closing_a_running_session_asks_then_stops(
        self, window, session, qapp, monkeypatch
    ):
        _trigger(window, "start")
        _pump(qapp, lambda: session.state is DemoState.RUNNING)
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
        )
        window.close()
        assert window.isVisible(), "No keeps the window and the rollout"
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        window.close()
        _pump(qapp, lambda: session.state is DemoState.IDLE)
        assert session.backends.last_runner.stopped_early

    def test_esc_is_the_quit_hotkey(self, window):
        assert window._qactions["quit"].shortcut().toString() == "Esc"
        assert Qt.Key.Key_Escape  # the shortcut string above is what Qt renders it as


class TestHeatmap:
    def test_the_checkbox_is_dead_without_positions(self, window, session):
        assert session.training_heatmap is None
        assert not window.heatmap_box.isEnabled()
        assert not window._qactions["heatmap"].isEnabled()
        assert window._qactions["heatmap"].shortcut().toString() == "H"
        assert "H" in window.help_panel._body.text()

    def test_the_option_boxes_live_in_the_side_column_under_the_keys(self, window):
        side = window.help_panel.parentWidget()
        assert window.options_panel.parentWidget() is side
        layout = side.layout()
        order = [layout.itemAt(i).widget() for i in range(layout.count())]
        order = [w for w in order if w is not None]
        assert order.index(window.options_panel) == order.index(window.help_panel) + 1
        assert window.heatmap_box.parentWidget() is window.options_panel
        assert window.readouts_box.parentWidget() is window.options_panel
        row = window.button_row
        assert all(
            row.itemAt(i).widget() is not window.heatmap_box for i in range(row.count())
        )

    def test_the_readouts_toggle(self, window, session):
        assert session.show_readouts and window.readouts_box.isChecked()
        assert window._qactions["readouts"].shortcut().toString() == "N"
        window._qactions["readouts"].trigger()
        assert not session.show_readouts and not window.readouts_box.isChecked()
        window.readouts_box.setChecked(True)
        assert session.show_readouts
        assert "N" in window.help_panel._body.text()

    def test_the_checkbox_and_h_follow_the_session(self, qapp):
        import json

        from policy_abstraction.demo.heatmap import TrainingHeatmap

        hm = TrainingHeatmap(np.array([[0.0, 0.0, 0.03], [0.05, 0.0, 0.03]]))
        backends = FakeBackends(FakeEnv(wrist=True), runner_kwargs={"steps": 10_000})
        s = DemoSession(backends, guided_warmup=False, training_heatmap=hm)
        s.prepare_env()
        win = DemoWindow(s, refresh_hz=50.0)
        win.show()
        qapp.processEvents()
        try:
            assert win.heatmap_box.isEnabled() and not win.heatmap_box.isChecked()
            win._qactions["heatmap"].trigger()
            assert s.show_heatmap and win.heatmap_box.isChecked()
            win.heatmap_box.setChecked(False)
            assert not s.show_heatmap
            win.heatmap_box.setChecked(True)
            assert s.show_heatmap
            # A frame is rendered with the map without raising (FakeEnv has
            # a calibrated camera; the footprint may or may not project).
            win._refresh()
            assert win.view._pixmap is not None
            json.dumps({"ok": True})  # keep the import honest
        finally:
            win._timer.stop()
            win.close()
            from PyQt6 import sip

            sip.delete(win)
            qapp.processEvents()
            s.shutdown()

    def test_saving_shows_in_the_status(self, window, session):
        session.saving = True
        try:
            text, _ = window._status_text()
            assert "writing the last episode" in text
        finally:
            session.saving = False


class TestPause:
    def test_space_pauses_and_resumes_a_running_rollout(self, window, session, qapp):
        assert "Start" in window._buttons["start"].text()
        _trigger(window, "start")
        _pump(qapp, lambda: session.state is DemoState.RUNNING)
        window._tick()
        assert window._buttons["start"].isEnabled(), "Space stays live while running"
        assert "Pause" in window._buttons["start"].text()
        _trigger(window, "start")
        assert session.paused
        window._tick()
        assert "Resume" in window._buttons["start"].text()
        text, _ = window._status_text()
        assert "PAUSED" in text
        banner, _ = window._frame_banner(session.live_view())
        assert banner == "PAUSED"
        _trigger(window, "start")
        assert not session.paused
        _trigger(window, "stop")
        _pump(qapp, lambda: session.state is DemoState.IDLE)
        window._tick()
        assert "Start" in window._buttons["start"].text()


class TestInstruction:
    def test_the_field_shows_the_task_over_the_picture(self, window, session):
        assert window.task_edit.text() == session.task
        assert window.task_edit.parentWidget() is window.view_box
        assert window.view.parentWidget() is window.view_box
        assert window.task_edit.y() < window.view_box.height() // 4

    def test_enter_applies_and_escape_reverts(self, window, session, qapp):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        edit = window.task_edit
        before = session.task
        edit.setFocus()
        edit.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))
        edit.setText("  Pour the  contents of the cup.  ")
        edit.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
            )
        )
        assert session.task == "Pour the contents of the cup."
        assert edit.text() == "Pour the contents of the cup."
        # Escape reverts to what the field held when editing began — and
        # must not quit the window.
        edit.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))
        edit.setText("garbage")
        edit.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
            )
        )
        assert edit.text() == "Pour the contents of the cup."
        assert session.task == "Pour the contents of the cup."
        assert not window._quitting
        assert before != session.task

    def test_typing_in_the_field_does_not_fire_hotkeys(self, window, session, qapp):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        edit = window.task_edit
        window.show()
        edit.setFocus()
        qapp.processEvents()
        override = QKeyEvent(
            QEvent.Type.ShortcutOverride,
            Qt.Key.Key_S,
            Qt.KeyboardModifier.NoModifier,
            "s",
        )
        if edit.hasFocus():
            assert edit.event(override) is True and override.isAccepted()
        else:
            pytest.skip("no focus offscreen")

    def test_a_running_policy_gets_the_new_prompt_at_the_next_replan(
        self, window, session, qapp
    ):
        _trigger(window, "start")
        _pump(qapp, lambda: session.backends.last_runner is not None)
        runner = session.backends.last_runner
        runner.task_description = session.task
        assert session.set_task("Pour the cup.")
        assert runner.task_description == "Pour the cup."
        _trigger(window, "stop")
        _pump(qapp, lambda: session.state is DemoState.IDLE)

    def test_blank_is_ignored(self, session):
        before = session.task
        assert not session.set_task("   ")
        assert session.task == before

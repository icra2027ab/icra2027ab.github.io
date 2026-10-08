"""The window that is up before the servers are.

The point of the start-up screen is that an operator can tell a three-minute
checkpoint load from a hang, and can cancel without stranding half-launched
servers. Both of those are failure paths, so both are tested here rather than
discovered at the workcell.

Qt runs headless under ``QT_QPA_PLATFORM=offscreen``. The event loop is pumped
with ``processEvents`` rather than ``exec``: these tests need to step through a
hand-off, not sit in a loop waiting to be closed.
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from loguru import logger  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from policy_abstraction.experiments.config import ExperimentConfig  # noqa: E402
from policy_abstraction.experiments.factors import (  # noqa: E402
    GridPositionFactor,
)
from policy_abstraction.experiments.grid import TableGrid  # noqa: E402
from policy_abstraction.experiments.gui.app import (  # noqa: E402
    _begin_deferred,
    _Launch,
)
from policy_abstraction.experiments.gui.startup import StartupWindow  # noqa: E402
from policy_abstraction.experiments.session import ExperimentSession  # noqa: E402
from policy_abstraction.experiments.store import ExperimentStore  # noqa: E402
from tests.experiments.conftest import FakeBackends  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_windows(qapp):
    """Leave no window standing.

    ``QApplication`` quits when its last window closes, so a widget one test
    forgot to close hangs the *next* test's event loop rather than failing its
    own — which is a miserable thing to debug.
    """
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def config():
    return ExperimentConfig(
        name="startup_test",
        task="Put the cable in the box.",
        factors=[
            GridPositionFactor(
                "cable_position",
                TableGrid(
                    origin=(0, 0, 0),
                    u_axis=(0, 1, 0),
                    v_axis=(1, 0, 0),
                    rows=1,
                    cols=2,
                    name="cable",
                ),
            )
        ],
        repeats=1,
        view_scale=1.0,
    )


def pump(app, predicate, timeout: float = 10.0) -> bool:
    """Run the event loop until *predicate* holds, or give up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return predicate()


class TestStartupWindow:
    def test_closing_during_start_up_cancels_instead_of_closing(self, qapp):
        window = StartupWindow("t")
        window.show()
        window.close()
        assert window.cancelled
        assert window.isVisible(), "the window must stay up while cancelling"

        window.mark_finished()
        window.close()
        assert not window.isVisible()
        window.deleteLater()

    def test_the_cancel_callback_fires_once(self, qapp):
        calls = []
        window = StartupWindow("t", on_cancel=lambda: calls.append(1))
        window.close()
        window.close()
        assert calls == [1]
        window.mark_finished()
        window.deleteLater()

    def test_a_failure_is_shown_and_then_the_window_can_close(self, qapp):
        window = StartupWindow("t")
        window.show_failure("RuntimeError: the policy server died")
        assert "the policy server died" in window.log.toPlainText()
        window.close()
        window.deleteLater()

    def test_the_log_is_bounded(self, qapp):
        window = StartupWindow("t")
        for i in range(1200):
            window.append(f"line {i}")
        assert window.log.blockCount() <= window.log.maximumBlockCount() + 1
        window.mark_finished()
        window.deleteLater()


class TestDeferredStartUp:
    """``run_gui`` given a factory rather than a session."""

    def _session(self, tmp_path, config, name="exp"):
        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        store = ExperimentStore.open(tmp_path / name, config)
        session = ExperimentSession(backends, config, store, policy_setting="t")
        session.prepare_env()
        return session

    def test_the_operator_window_replaces_the_start_up_screen(
        self, qapp, tmp_path, config
    ):
        from policy_abstraction.experiments.gui.main_window import ExperimentWindow

        launch = _Launch()
        session = self._session(tmp_path, config)
        _begin_deferred(qapp, lambda: session, config, launch, {"refresh_hz": 30.0})

        assert isinstance(launch.window, StartupWindow), "no window before the servers"
        assert pump(qapp, lambda: launch.session is not None)
        assert isinstance(launch.window, ExperimentWindow)
        assert not launch.failed

        launch.window._timer.stop()
        session.shutdown(timeout=10)

    def test_start_up_log_lines_reach_the_window(self, qapp, tmp_path, config):
        launch = _Launch()
        session = self._session(tmp_path, config)

        def build():
            logger.info("Starting policy and env servers …")
            time.sleep(0.05)
            return session

        _begin_deferred(qapp, build, config, launch, {})
        startup = launch.window
        assert isinstance(startup, StartupWindow)
        assert pump(
            qapp, lambda: "Starting policy and env servers" in startup.log.toPlainText()
        )
        assert pump(qapp, lambda: launch.session is not None)

        launch.window._timer.stop()
        session.shutdown(timeout=10)

    def test_a_failing_start_up_is_reported_in_the_window(self, qapp, config):
        launch = _Launch()

        def build():
            raise RuntimeError("policy server exited during load")

        _begin_deferred(qapp, build, config, launch, {})
        startup = launch.window
        assert isinstance(startup, StartupWindow)
        assert pump(qapp, lambda: launch.failed)
        assert "policy server exited during load" in startup.log.toPlainText()
        # No session was ever built, so there is nothing to shut down.
        assert launch.session is None
        startup.close()
        startup.deleteLater()

    def test_cancelling_still_shuts_the_servers_down(self, qapp, tmp_path, config):
        """Walking away mid-launch must not leave the servers orphaned."""
        launch = _Launch()
        session = self._session(tmp_path, config)
        release = []

        def build():
            while not release:
                time.sleep(0.005)
            return session

        _begin_deferred(qapp, build, config, launch, {})
        startup = launch.window
        assert isinstance(startup, StartupWindow)
        startup.close()  # the operator gives up
        assert startup.cancelled
        assert launch.session is None, "cancel must not race ahead of start-up"

        release.append(True)
        assert pump(qapp, lambda: launch.cancelled and launch.session is session)
        # run_gui's own teardown is what calls shutdown; the session it needs
        # for that is the one cancellation handed it.
        assert launch.session is session
        session.shutdown(timeout=10)

    def test_a_live_session_still_opens_the_operator_window_directly(
        self, qapp, tmp_path, config
    ):
        """The eager path is what the tests and any non-server caller use."""
        from policy_abstraction.experiments.gui.app import _open_operator_window

        session = self._session(tmp_path, config)
        window = _open_operator_window(session, config, {"refresh_hz": 30.0})
        assert window.session is session
        window._timer.stop()
        session.shutdown(timeout=10)
        window.deleteLater()


class TestRunGuiEndToEnd:
    """The real entry point, event loop and teardown included."""

    def test_a_deferred_run_opens_hands_over_and_parks_the_arm(
        self, qapp, tmp_path, config
    ):
        from PyQt6.QtCore import QTimer

        from policy_abstraction.experiments.gui.app import run_gui
        from policy_abstraction.experiments.gui.main_window import ExperimentWindow

        backends = FakeBackends(runner_kwargs={"steps": 10_000})
        store = ExperimentStore.open(tmp_path / "e2e", config)
        built = []

        def build():
            time.sleep(0.05)  # stand-in for the checkpoint load
            session = ExperimentSession(backends, config, store, policy_setting="t")
            session.prepare_env()
            built.append(session)
            return session

        # Close the operator window once it has taken over, so the loop ends.
        def close_when_ready():
            for widget in qapp.topLevelWidgets():
                if isinstance(widget, ExperimentWindow) and widget.isVisible():
                    widget._timer.stop()
                    widget.close()
                    return
            QTimer.singleShot(20, close_when_ready)

        QTimer.singleShot(20, close_when_ready)
        QTimer.singleShot(20_000, qapp.quit)  # a bug must fail, not hang
        code = run_gui(build, config, refresh_hz=30.0)

        assert code == 0
        assert built, "the session factory was never called"
        # run_gui owns teardown: the servers are stopped and the arm parked
        # without the caller doing anything.
        assert backends.shutdown_calls == 1

    def test_a_failing_factory_returns_non_zero_and_builds_nothing(self, qapp, config):
        from PyQt6.QtCore import QTimer

        from policy_abstraction.experiments.gui.app import run_gui
        from policy_abstraction.experiments.gui.startup import StartupWindow

        def build():
            raise RuntimeError("policy server exited during load")

        def close_when_failed():
            for widget in qapp.topLevelWidgets():
                if isinstance(widget, StartupWindow) and widget.isVisible():
                    if "policy server exited" in widget.log.toPlainText():
                        widget.close()
                        return
            QTimer.singleShot(20, close_when_failed)

        QTimer.singleShot(20, close_when_failed)
        QTimer.singleShot(20_000, qapp.quit)
        assert run_gui(build, config, refresh_hz=30.0) == 1

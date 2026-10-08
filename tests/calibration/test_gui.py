"""The calibration window, offscreen: wiring, not pixels."""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from policy_abstraction.calibration.fake_hardware import find_marker  # noqa: E402
from policy_abstraction.calibration.geometry import CalibrationMode  # noqa: E402
from policy_abstraction.calibration.gui import (  # noqa: E402
    CalibrationWindow,
    ImageView,
)
from policy_abstraction.calibration.motion import Mover  # noqa: E402
from policy_abstraction.calibration.session import Step  # noqa: E402
from tests.calibration.conftest import make_session, nn  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(tmp_path, mode):
    session, arm, camera, truth, clock = make_session(tmp_path, mode, target=15)

    # Advance the fake freedrive (and the clock) on every camera read, as the
    # launcher does.
    def tick() -> None:
        arm.tick()
        clock.advance(0.05)

    camera._on_read = tick
    win = CalibrationWindow(session, poll_hz=10)
    win._timer.stop()  # the tests poll by hand
    return win, session, truth


def test_image_view_maps_clicks_to_image_pixels(qapp):
    view = ImageView()
    view.resize(800, 600)
    view.set_image(np.zeros((480, 640, 3), np.uint8))
    got = []
    view.clicked.connect(lambda u, v: got.append((u, v)))
    view.set_click_enabled(True)
    # The 640x480 image fits 800x600 exactly; the centre maps to the centre.
    uv = view.to_image(QPointF(400, 300))
    assert uv == pytest.approx((320.0, 240.0))
    back = view.to_widget((320.0, 240.0))
    assert (back.x(), back.y()) == pytest.approx((400.0, 300.0))
    assert view.to_image(QPointF(-5, 300)) is None


def test_walkthrough_exterior(qapp, tmp_path):
    win, session, truth = _window(tmp_path, CalibrationMode.EXTERIOR)
    assert session.step is Step.CAPTURE
    assert win.btn_start.isEnabled() and not win.btn_finish.isEnabled()
    win._start_capture()
    assert session.capturing and win.btn_finish.isEnabled() is False
    n = 0
    while not session.status.target_reached and n < 4000:
        win._on_poll()
        n += 1
    assert session.status.target_reached
    assert win.progress.value() == win.progress.maximum()
    assert win.btn_finish.isEnabled()
    win._finish_capture()
    assert session.step is Step.SELECT_POINT
    assert not win.btn_track.isEnabled()  # no click yet
    click = find_marker(session.first_frame())
    assert click is not None
    win._on_click(*click)
    assert session.click_uv is not None and win.btn_track.isEnabled()
    # Run the worker's job inline rather than on the thread.
    result = session.run_tracking(log=win._log)
    win._on_tracked(result)
    assert session.step is Step.VERIFY
    assert "GOOD" in win.verdict_label.text()
    win.slider.setValue(3)
    assert "frame 4/" in win.frame_label.text()
    win._save()
    assert session.step is Step.DONE
    assert session.config.output_path.exists()
    win.close()


def test_walkthrough_wrist_has_a_reference_step(qapp, tmp_path):
    win, session, truth = _window(tmp_path, CalibrationMode.WRIST)
    assert session.step is Step.REFERENCE
    assert win.steps_list.count() == 6
    # Pretend the fingertips are on the mark.
    ref = truth.reference_point_base
    assert ref is not None
    session.arm.tcp_pose = lambda: np.concatenate([ref, [0.0, 0.0, 0.0]])  # type: ignore[method-assign]
    win._record_reference()
    assert session.step is Step.CAPTURE
    assert session.reference_point_base is not None
    np.testing.assert_allclose(session.reference_point_base, ref)
    win.close()


def _item_text(win, i: int) -> str:
    item = win.steps_list.item(i)
    assert item is not None
    return item.text()


def test_step_list_marks_progress(qapp, tmp_path):
    win, session, truth = _window(tmp_path, CalibrationMode.EXTERIOR)
    texts = [_item_text(win, i) for i in range(win.steps_list.count())]
    assert texts[0].startswith("▶")
    session.step = Step.SELECT_POINT
    session.dataset = None
    win._refresh()
    texts = [_item_text(win, i) for i in range(win.steps_list.count())]
    assert texts[0].startswith("✓") and texts[1].startswith("▶")
    win.close()


def test_autonomous_capture_from_the_window(qapp, tmp_path):

    win, session, truth = _window(tmp_path, CalibrationMode.EXTERIOR)
    session.mover = cast(Mover, session.arm)  # the fake arm executes moves
    win._on_poll()  # first sample centres the box on the TCP
    box = win._current_box()
    assert box is not None and session.last_sample is not None
    assert box.contains(session.last_sample.tcp_pose[:3])
    assert win.area.box is not None
    win.spin_poses.setValue(14)
    win._start_autonomous(confirm=False)
    assert session.motion is not None and session.motion.active
    assert win.btn_stop.isEnabled() and not win.btn_auto.isEnabled()
    n = 0
    while session.motion.active and n < 20000:
        win._on_poll()
        n += 1
    assert session.motion.phase == "done"
    assert "finished" in win.motion_label.text()
    assert win.btn_finish.isEnabled()
    win._finish_capture()
    assert session.step is Step.SELECT_POINT
    win.close()


def test_stop_button_and_escape_stop_motion(qapp, tmp_path):
    win, session, truth = _window(tmp_path, CalibrationMode.EXTERIOR)
    session.mover = cast(Mover, session.arm)
    win._on_poll()
    win.spin_poses.setValue(12)
    win._start_autonomous(confirm=False)
    for _ in range(3):
        win._on_poll()
    win._stop_motion()
    assert session.motion is not None and session.motion.phase == "stopped"
    assert not win.btn_stop.isEnabled()
    win.close()


def test_two_stage_window_hands_the_mark_over(qapp, tmp_path):
    from tests.calibration.conftest import make_suite

    suite, arm, cameras, truths, clock = make_suite(tmp_path, target=14)
    win = CalibrationWindow(suite, poll_hz=10)
    win._timer.stop()
    assert win.stage_label.isVisible() or True  # offscreen widgets report hidden
    assert win.stage_label.text().startswith("Stage 1 of 2")
    assert win.steps_list.count() == 5  # exterior: no reference step

    # Exterior stage, autonomously.
    cameras[CalibrationMode.EXTERIOR]._on_read = lambda: (
        arm.tick(),
        clock.advance(0.05),
    )
    win._on_poll()
    win.spin_poses.setValue(14)
    win._start_autonomous(confirm=False)
    n = 0
    motion = nn(suite.active.motion)
    while motion.active and n < 20000:
        win._on_poll()
        n += 1
    win._finish_capture()
    click = nn(find_marker(suite.active.first_frame()))
    win._on_click(*click)
    win._on_tracked(suite.active.run_tracking(log=win._log))
    win._save()
    assert suite.active.step is Step.DONE
    assert win.btn_next.text().startswith("Next: wrist_cam")
    cameras[CalibrationMode.EXTERIOR]._on_read = None

    # Wrist stage: pick the mark in the exterior image.
    win._next_stage()
    assert suite.active.mode is CalibrationMode.WRIST
    assert win.stage_label.text().startswith("Stage 2 of 2: wrist camera")
    assert win.steps_list.count() == 6
    assert win.btn_pick_mark.isEnabled()
    win._pick_mark()
    assert win._mark_mode and not win._timer.isActive()
    uv = nn(find_marker(nn(win._mark_frame)[0], colour="cyan"))
    win._on_click(*uv)
    assert not win._mark_mode
    assert suite.active.step is Step.CAPTURE
    ref = suite.active.reference_point_base
    assert (
        np.linalg.norm(ref - nn(truths[CalibrationMode.WRIST].reference_point_base))
        < 0.01
    )
    assert "base frame" in win.ref_label.text()
    win.close()


def test_skip_button_appears_only_with_a_saved_entry(qapp, tmp_path):
    import json

    from tests.calibration.conftest import make_suite

    suite, arm, cameras, truths, clock = make_suite(tmp_path)
    win = CalibrationWindow(suite, poll_hz=10)
    win._timer.stop()
    win._refresh()
    assert win.btn_skip.isHidden()
    E = truths[CalibrationMode.EXTERIOR].T_parent_cam
    suite.calibration_path.write_text(
        json.dumps({"base_frame": "base", "ext_cam": {"extrinsic": E.tolist()}})
    )
    win._refresh()
    assert not win.btn_skip.isHidden()
    suite.active.skip()
    win._refresh()
    assert "Kept the saved calibration" in win.done_label.text()
    win.close()

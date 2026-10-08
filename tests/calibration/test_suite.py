"""Both cameras in one sitting, and the mark handed from one to the other."""

import json

import numpy as np
import pytest

from policy_abstraction.calibration.fake_hardware import find_marker
from policy_abstraction.calibration.geometry import (
    CalibrationMode,
    invert,
    pose_to_matrix,
    project,
)
from policy_abstraction.calibration.motion import MotionConfig, SafeBox
from policy_abstraction.calibration.output import base_link_to_base
from policy_abstraction.calibration.session import Step
from policy_abstraction.calibration.solve import pose_difference
from policy_abstraction.calibration.suite import (
    CalibrationSuite,
    deproject,
    load_saved_extrinsic,
)
from tests.calibration.conftest import make_session, make_suite, nn

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def test_deproject_inverts_project():
    T_base_cam = pose_to_matrix([0.3, -0.8, 0.6, -2.0, 0.3, 0.2])
    P = np.array([[-0.4, -0.1, 0.05]])
    uv, z = project(K, invert(T_base_cam), P)
    np.testing.assert_allclose(deproject(uv[0], z[0], K, T_base_cam), P[0], atol=1e-9)


class TestLoadSavedExtrinsic:
    def test_base_file(self, tmp_path):
        E = pose_to_matrix([1, 2, 3, 0.1, 0.2, 0.3])
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps({"base_frame": "base", "cam": {"extrinsic": E.tolist()}})
        )
        np.testing.assert_allclose(nn(load_saved_extrinsic(path, "cam")), E)
        assert load_saved_extrinsic(path, "other") is None
        assert load_saved_extrinsic(tmp_path / "missing.json", "cam") is None

    def test_base_link_file_is_rotated(self, tmp_path):
        E = pose_to_matrix([1, 2, 3, 0.1, 0.2, 0.3])
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps({"base_frame": "base_link", "cam": {"extrinsic": E.tolist()}})
        )
        np.testing.assert_allclose(
            nn(load_saved_extrinsic(path, "cam")), base_link_to_base() @ E
        )

    def test_tcp_parented_entry_has_no_base_pose(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "base_frame": "base",
                    "w": {"extrinsic": np.eye(4).tolist(), "parent": "tcp"},
                }
            )
        )
        assert load_saved_extrinsic(path, "w") is None


def _run_exterior_stage(suite, arm, clock, cameras):
    ext = suite.active
    assert ext.mode is CalibrationMode.EXTERIOR
    cameras[CalibrationMode.EXTERIOR]._on_read = lambda: (
        arm.tick(),
        clock.advance(0.05),
    )
    box = SafeBox.around(arm.tcp_pose()[:3])
    run = ext.start_autonomous_capture(
        MotionConfig(box=box, n_targets=24, tilt_deg=15, roll_deg=20)
    )
    n = 0
    while run.active and n < 20000:
        ext.poll()
        n += 1
    ext.finish_capture()
    click = find_marker(ext.first_frame())
    assert click is not None
    ext.set_click(*click)
    ext.run_tracking(log=lambda _s: None)
    ext.save()
    cameras[CalibrationMode.EXTERIOR]._on_read = None
    return ext


def test_exterior_then_wrist_with_the_mark_clicked_in_the_exterior_image(tmp_path):
    suite, arm, cameras, truths, clock = make_suite(tmp_path)
    assert suite.label(0).startswith("Stage 1 of 2: exterior camera")
    ext = _run_exterior_stage(suite, arm, clock, cameras)
    assert ext.step is Step.DONE
    np.testing.assert_allclose(
        nn(suite.exterior_extrinsic()), nn(ext.result).solution.T_parent_cam
    )

    wrist = suite.advance()
    assert wrist.mode is CalibrationMode.WRIST and wrist.step is Step.REFERENCE
    # The operator clicks the cyan mark in the exterior image.
    frame = suite.exterior_frame()
    uv = nn(find_marker(frame[0], colour="cyan"))
    point, depth = suite.mark_from_exterior_click(uv[0], uv[1], frame)
    true_mark = nn(truths[CalibrationMode.WRIST].reference_point_base)
    assert np.linalg.norm(point - true_mark) < 0.01, (point, true_mark)
    assert depth > 0.3
    wrist.set_reference_point(point, source="clicked in exterior image")
    assert wrist.step is Step.CAPTURE

    # Then the wrist stage runs like any other, aiming at that mark.
    cameras[CalibrationMode.WRIST]._on_read = lambda: (arm.tick(), clock.advance(0.05))
    tcp = arm.tcp_pose()
    box = SafeBox.around(tcp[:3], half_xy=0.12, above=0.15)
    run = wrist.start_autonomous_capture(
        MotionConfig(box=box, n_targets=20, tilt_deg=12, roll_deg=20)
    )
    n = 0
    while run.active and n < 20000:
        wrist.poll()
        n += 1
    wrist.finish_capture()
    click = find_marker(wrist.first_frame())
    assert click is not None
    wrist.set_click(*click)
    result = wrist.run_tracking(log=lambda _s: None)
    d = pose_difference(
        result.solution.T_parent_cam, truths[CalibrationMode.WRIST].T_parent_cam
    )
    # The mark's error (< 1 cm) propagates into the wrist pose.
    assert d.translation_m < 0.02 and d.rotation_deg < 1.5, d
    wrist.save()
    raw = json.loads(suite.calibration_path.read_text())
    assert set(raw) == {"base_frame", "ext_cam", "wrist_cam"}
    assert raw["wrist_cam"]["parent"] == "tcp"
    assert raw["wrist_cam"]["meta"]["reference_source"] == "clicked in exterior image"
    assert not suite.has_next


def test_skipping_the_exterior_stage_uses_the_saved_file(tmp_path):
    suite, arm, cameras, truths, clock = make_suite(tmp_path)
    # No file, no exterior pose: clicking the mark is refused.
    with pytest.raises(RuntimeError, match="no calibration yet"):
        suite.mark_from_exterior_click(10, 10)
    E = truths[CalibrationMode.EXTERIOR].T_parent_cam
    suite.calibration_path.write_text(
        json.dumps({"base_frame": "base", "ext_cam": {"extrinsic": E.tolist()}})
    )
    suite.active.skip()
    assert suite.active.step is Step.DONE and suite.active.skipped
    np.testing.assert_allclose(nn(suite.exterior_extrinsic()), E)
    suite.advance()
    frame = suite.exterior_frame()
    uv = nn(find_marker(frame[0], colour="cyan"))
    point, _ = suite.mark_from_exterior_click(uv[0], uv[1], frame)
    assert (
        np.linalg.norm(point - truths[CalibrationMode.WRIST].reference_point_base)
        < 0.005
    )


def test_click_without_depth_is_refused(tmp_path):
    suite, arm, cameras, truths, clock = make_suite(tmp_path)
    E = truths[CalibrationMode.EXTERIOR].T_parent_cam
    suite.calibration_path.write_text(
        json.dumps({"base_frame": "base", "ext_cam": {"extrinsic": E.tolist()}})
    )
    rgb, depth = suite.exterior_frame()
    with pytest.raises(RuntimeError, match="no depth"):
        suite.mark_from_exterior_click(5, 5, (rgb, None))
    bad = np.full_like(depth, np.nan)
    with pytest.raises(RuntimeError, match="no valid depth"):
        suite.mark_from_exterior_click(5, 5, (rgb, bad))


def test_close_releases_shared_devices_once(tmp_path):
    suite, arm, cameras, truths, clock = make_suite(tmp_path)
    closes = {"arm": 0}
    arm.close = lambda: closes.__setitem__("arm", closes["arm"] + 1)
    suite.close()
    assert closes["arm"] == 1


def test_single_session_is_a_one_stage_suite(tmp_path):
    session, *_ = make_session(tmp_path, CalibrationMode.WRIST)
    suite = CalibrationSuite([session], calibration_path=session.config.output_path)
    assert not suite.has_next and suite.exterior_camera() is None
    with pytest.raises(RuntimeError):
        suite.advance()

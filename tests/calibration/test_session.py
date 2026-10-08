"""The whole procedure on the fake cell, in both mountings."""

import pathlib

import numpy as np
import pytest

from policy_abstraction.calibration.fake_hardware import find_marker
from policy_abstraction.calibration.geometry import CalibrationMode
from policy_abstraction.calibration.session import Step
from policy_abstraction.calibration.solve import pose_difference
from tests.calibration.conftest import drive_capture, make_session, run_to_verify


def test_recovers_the_true_camera_pose(tmp_path, mode):
    session, arm, camera, truth, clock = make_session(tmp_path, mode)
    result = run_to_verify(session, arm, camera, truth, clock)
    assert session.step is Step.VERIFY
    d = pose_difference(result.solution.T_parent_cam, truth.T_parent_cam)
    assert d.translation_m < 0.005, d
    assert d.rotation_deg < 0.3, d
    assert result.verdict.level == "good", result.verdict.reasons
    assert result.depth is not None and result.depth.median_abs_m < 0.01
    # Only still frames enter the solve.
    settled = np.array([r.settled for r in result.records])
    assert not result.solution.used[~settled].any()
    assert result.solution.n_used >= session.config.target_keyframes * 0.8


def test_capture_takes_still_and_bridge_frames(tmp_path):
    session, arm, camera, truth, clock = make_session(
        tmp_path, CalibrationMode.EXTERIOR
    )
    session.start_capture()
    drive_capture(session, arm, clock)
    st = session.status
    assert st.n_keyframes >= session.config.target_keyframes
    assert st.n_frames > st.n_keyframes  # bridge frames while moving
    assert st.n_frames <= session.config.max_frames
    assert np.all(st.extent_m() > 0.1)  # the fake freedrive covers space
    assert session.dataset is not None
    recs = session.dataset.records()
    assert sum(r.settled for r in recs) == st.n_keyframes
    # Consecutive still poses are at least the minimum distance apart.
    P = np.stack([r.tcp_pose[:3] for r in recs if r.settled and r.tcp_pose is not None])
    assert np.all(
        np.linalg.norm(np.diff(P, axis=0), axis=1)
        >= session.config.keyframe_min_move_m - 1e-9
    )


def test_finish_needs_enough_still_poses(tmp_path):
    session, arm, camera, truth, clock = make_session(
        tmp_path, CalibrationMode.EXTERIOR
    )
    session.start_capture()
    session.poll()
    with pytest.raises(RuntimeError, match="still poses"):
        session.finish_capture()


def test_wrist_needs_the_reference_first(tmp_path):
    session, *_ = make_session(tmp_path, CalibrationMode.WRIST)
    assert session.step is Step.REFERENCE
    with pytest.raises(RuntimeError, match="reference"):
        session.start_capture()
    p = session.record_reference()
    np.testing.assert_allclose(p, session.arm.tcp_pose()[:3])
    assert session.step is Step.CAPTURE


def test_exterior_has_no_reference_step(tmp_path):
    session, *_ = make_session(tmp_path, CalibrationMode.EXTERIOR)
    assert Step.REFERENCE not in session.steps()
    with pytest.raises(RuntimeError):
        session.record_reference()


def test_click_outside_the_image_is_rejected(tmp_path):
    session, arm, camera, truth, clock = make_session(
        tmp_path, CalibrationMode.EXTERIOR
    )
    session.start_capture()
    drive_capture(session, arm, clock)
    session.finish_capture()
    with pytest.raises(ValueError, match="outside"):
        session.set_click(-1, 10)
    with pytest.raises(RuntimeError):
        session.run_tracking(log=lambda _s: None)  # no click yet


def test_save_writes_the_env_layout_with_metadata(tmp_path, mode):
    session, arm, camera, truth, clock = make_session(tmp_path, mode)
    run_to_verify(session, arm, camera, truth, clock)
    path = session.save()
    assert session.step is Step.DONE
    import json

    raw = json.loads(path.read_text())
    assert raw["base_frame"] == "base"
    entry = raw["cam"]
    if mode is CalibrationMode.WRIST:
        assert entry["parent"] == "tcp"
        assert entry["meta"]["reference_point_base"] is not None
    else:
        assert "parent" not in entry
    assert entry["meta"]["tcp_offset"] == session.config.tcp_offset
    assert session.result is not None
    assert entry["meta"]["n_inliers"] == session.result.solution.n_used
    assert entry["meta"]["verdict"] == "good"


def test_reload_a_capture_and_resolve(tmp_path, mode):
    session, arm, camera, truth, clock = make_session(tmp_path, mode)
    first = run_to_verify(session, arm, camera, truth, clock)
    assert session.dataset is not None
    root = session.dataset.root
    session2, *_ = make_session(tmp_path / "second", mode)
    session2.use_existing_dataset(root)
    assert session2.step is Step.SELECT_POINT
    assert session2.status.n_keyframes == session.status.n_keyframes
    click = find_marker(session2.first_frame())
    assert click is not None
    session2.set_click(*click)
    second = session2.run_tracking(log=lambda _s: None)
    d = pose_difference(first.solution.T_parent_cam, second.solution.T_parent_cam)
    assert d.translation_m < 1e-3


def test_reload_refuses_the_other_mode(tmp_path):
    session, arm, camera, truth, clock = make_session(
        tmp_path, CalibrationMode.EXTERIOR
    )
    run_to_verify(session, arm, camera, truth, clock)
    other, *_ = make_session(tmp_path / "w", CalibrationMode.WRIST)
    assert session.dataset is not None
    with pytest.raises(ValueError, match="mode"):
        other.use_existing_dataset(session.dataset.root)


def test_output_must_not_live_in_the_capture_folder(tmp_path):
    from policy_abstraction.calibration.session import SessionConfig

    with pytest.raises(ValueError, match="inside dataset_root"):
        SessionConfig(
            camera_name="c",
            camera_device="fake",
            mode=CalibrationMode.EXTERIOR,
            dataset_root=tmp_path / "cap",
            output_path=tmp_path / "cap" / "calib.json",
        )


def test_paths_are_made_absolute(tmp_path, monkeypatch):
    """The Kalib driver runs with Kalib's checkout as cwd, so a relative
    dataset path would send its output there (it did, once)."""
    from policy_abstraction.calibration.dataset import CaptureMeta, KalibDataset
    from policy_abstraction.calibration.kalib_runner import KalibConfig
    from policy_abstraction.calibration.session import SessionConfig

    monkeypatch.chdir(tmp_path)
    cfg = SessionConfig(
        camera_name="c",
        camera_device="fake",
        mode=CalibrationMode.EXTERIOR,
        dataset_root=pathlib.Path("data/cap"),
        output_path=pathlib.Path("configs/calib.json"),
    )
    assert cfg.dataset_root.is_absolute() and cfg.output_path.is_absolute()
    ds = KalibDataset.create(
        "data/cap",
        CaptureMeta("c", "fake", 64, 48, 30, np.eye(3).tolist(), "exterior", [0] * 6),
    )
    cmd = KalibConfig(root=tmp_path, python=tmp_path / "py").command(
        ds, (1.0, 2.0), ds.kalib_dir / "tracking_result.json"
    )
    for flag in ("--root-dir", "--result"):
        value = cmd[cmd.index(flag) + 1]
        assert value.startswith("/"), (flag, value)

"""Frame conventions: the one place a sign error would poison everything."""

import numpy as np
import pytest

from policy_abstraction.calibration.geometry import (
    CalibrationMode,
    invert,
    is_rigid,
    keypoint_in_parent,
    matrix_to_pose,
    pose_to_matrix,
    project,
    rotation_angle_deg,
    tracked_point_base,
    transform_points,
)


def test_pose_roundtrip():
    pose = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    T = pose_to_matrix(pose)
    assert is_rigid(T)
    np.testing.assert_allclose(matrix_to_pose(T), pose, atol=1e-9)


def test_invert_is_the_inverse():
    T = pose_to_matrix([0.3, 0.2, 0.1, 1.0, -0.3, 0.2])
    np.testing.assert_allclose(invert(T) @ T, np.eye(4), atol=1e-12)


def test_project_puts_the_optical_axis_at_the_principal_point():
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
    uv, z = project(K, np.eye(4), np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]))
    np.testing.assert_allclose(uv[0], [320.0, 240.0])
    np.testing.assert_allclose(uv[1], [320.0 + 60.0, 240.0])
    np.testing.assert_allclose(z, [2.0, 2.0])


def test_project_hides_points_behind_the_camera():
    uv, _ = project(np.eye(3), np.eye(4), np.array([[0.0, 0.0, -1.0]]))
    assert np.all(np.isnan(uv))


def test_exterior_keypoint_is_the_tcp_plus_offset_in_base():
    pose = [0.5, -0.1, 0.3, 0.0, 0.0, np.pi / 2]  # tool yawed 90 deg about z
    P = keypoint_in_parent(CalibrationMode.EXTERIOR, pose, point_offset_tcp=(0.1, 0, 0))
    # An x offset in the tool frame points along +y in base after a 90 deg yaw.
    np.testing.assert_allclose(P, [0.5, 0.0, 0.3], atol=1e-9)
    np.testing.assert_allclose(tracked_point_base(pose), [0.5, -0.1, 0.3])


def test_wrist_keypoint_is_the_reference_mark_in_the_tcp_frame():
    pose = [0.5, -0.1, 0.3, 0.0, 0.0, np.pi / 2]
    ref = np.array([0.5, 0.0, 0.3])  # 10 cm along base +y from the TCP
    P = keypoint_in_parent(CalibrationMode.WRIST, pose, reference_point_base=ref)
    # ... which is +x in the yawed tool frame.
    np.testing.assert_allclose(P, [0.1, 0.0, 0.0], atol=1e-9)
    # And mapping it back lands on the mark.
    np.testing.assert_allclose(transform_points(pose_to_matrix(pose), P[None])[0], ref)


def test_wrist_mode_requires_a_reference():
    with pytest.raises(ValueError, match="reference"):
        keypoint_in_parent(CalibrationMode.WRIST, [0, 0, 0, 0, 0, 0])


def test_parent_frames():
    assert CalibrationMode.EXTERIOR.parent_frame == "base"
    assert CalibrationMode.WRIST.parent_frame == "tcp"


def test_rotation_angle():
    T = pose_to_matrix([0, 0, 0, 0, 0, np.deg2rad(30)])
    assert rotation_angle_deg(T[:3, :3]) == pytest.approx(30.0)

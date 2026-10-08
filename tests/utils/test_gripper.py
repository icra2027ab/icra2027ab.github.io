"""Tests for policy_abstraction.utils.gripper."""

from __future__ import annotations

import numpy as np

from policy_abstraction.utils.gripper import (
    _FINGER_X_REST,
    _FINGER_Z_FROM_GRIP,
    compute_finger_tip_positions,
    compute_finger_tip_positions_single,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pose(T: int, pos=None, axis_angle=None, q1=0.0, q2=0.0):
    """Construct (ee_pos, ee_ori, gripper_states) arrays of length T."""
    ee_pos = np.zeros((T, 3), dtype=np.float64)
    if pos is not None:
        ee_pos[:] = pos
    ee_ori = np.zeros((T, 3), dtype=np.float64)
    if axis_angle is not None:
        ee_ori[:] = axis_angle
    gripper_states = np.column_stack(
        [
            np.full(T, q1, dtype=np.float64),
            np.full(T, q2, dtype=np.float64),
        ]
    )
    return ee_pos, ee_ori, gripper_states


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeFingerTipPositions:
    """Tests for compute_finger_tip_positions."""

    def test_output_shapes(self):
        """Both output arrays should be (T, 3)."""
        T = 7
        ee_pos, ee_ori, gs = _make_pose(T)
        left, right = compute_finger_tip_positions(ee_pos, ee_ori, gs)
        assert left.shape == (T, 3)
        assert right.shape == (T, 3)

    def test_single_timestep(self):
        """T=1 works and produces (1, 3) arrays."""
        ee_pos, ee_ori, gs = _make_pose(1)
        left, right = compute_finger_tip_positions(ee_pos, ee_ori, gs)
        assert left.shape == (1, 3)
        assert right.shape == (1, 3)

    def test_identity_rotation_z_offset(self):
        """With identity rotation, z component of tips equals ee_pos_z + _FINGER_Z_FROM_GRIP."""
        T = 5
        origin = np.array([0.0, 0.0, 1.0])
        ee_pos, ee_ori, gs = _make_pose(T, pos=origin, q1=0.0, q2=0.0)
        left, right = compute_finger_tip_positions(ee_pos, ee_ori, gs)
        expected_z = origin[2] + _FINGER_Z_FROM_GRIP
        np.testing.assert_allclose(left[:, 2], expected_z, atol=1e-9)
        np.testing.assert_allclose(right[:, 2], expected_z, atol=1e-9)

    def test_identity_rotation_closed_y_offset(self):
        """With identity rotation and q1=q2=0, y-offsets are ±_FINGER_X_REST."""
        T = 3
        ee_pos, ee_ori, gs = _make_pose(T, q1=0.0, q2=0.0)
        left, right = compute_finger_tip_positions(ee_pos, ee_ori, gs)
        # left_local y = -(q1 + _FINGER_X_REST)
        np.testing.assert_allclose(left[:, 1], -_FINGER_X_REST, atol=1e-9)
        # right_local y = (-q2 + _FINGER_X_REST) = _FINGER_X_REST (since q2=0)
        np.testing.assert_allclose(right[:, 1], _FINGER_X_REST, atol=1e-9)

    def test_symmetric_tips_closed_gripper(self):
        """Left and right y-offsets are equal and opposite at q1=q2=0."""
        T = 4
        ee_pos, ee_ori, gs = _make_pose(T, q1=0.0, q2=0.0)
        left, right = compute_finger_tip_positions(ee_pos, ee_ori, gs)
        np.testing.assert_allclose(left[:, 1], -right[:, 1], atol=1e-9)

    def test_open_gripper_wider_than_closed(self):
        """Fully-open gripper (q1=0.04, q2=-0.04) produces larger y-separation than closed."""
        T = 3
        ee_pos_c, ee_ori_c, gs_c = _make_pose(T, q1=0.0, q2=0.0)
        ee_pos_o, ee_ori_o, gs_o = _make_pose(T, q1=0.04, q2=-0.04)

        left_c, right_c = compute_finger_tip_positions(ee_pos_c, ee_ori_c, gs_c)
        left_o, right_o = compute_finger_tip_positions(ee_pos_o, ee_ori_o, gs_o)

        gap_closed = right_c[0, 1] - left_c[0, 1]
        gap_open = right_o[0, 1] - left_o[0, 1]
        assert gap_open > gap_closed

    def test_90_deg_z_rotation_maps_y_to_x(self):
        """90° rotation around z maps local y-offset into world x-offset."""
        # Rotation by π/2 around z: y_local → x_world, x_local → -y_world
        axis_angle = np.array([0.0, 0.0, np.pi / 2])
        T = 2
        ee_pos, ee_ori, gs = _make_pose(T, axis_angle=axis_angle, q1=0.0, q2=0.0)
        left, right = compute_finger_tip_positions(ee_pos, ee_ori, gs)

        # After rotation: y_local → x_world
        # left_local y = -_FINGER_X_REST → world x = +_FINGER_X_REST (negated by rotation direction)
        # right_local y = +_FINGER_X_REST → world x = -_FINGER_X_REST
        # The key check: x and y components are swapped relative to identity rotation
        np.testing.assert_allclose(left[:, 1], 0.0, atol=1e-9)  # y_world ≈ 0
        np.testing.assert_allclose(right[:, 1], 0.0, atol=1e-9)  # y_world ≈ 0
        # x components are non-zero (swapped from y)
        assert abs(left[0, 0]) > 1e-4
        assert abs(right[0, 0]) > 1e-4

    def test_batch_consistent_with_single(self):
        """T=5 all-same poses gives same result per row as T=1."""
        pos = np.array([0.1, 0.2, 0.3])
        ori = np.array([0.1, 0.0, 0.0])
        q1, q2 = 0.02, -0.01

        ee_pos_1, ee_ori_1, gs_1 = _make_pose(1, pos=pos, axis_angle=ori, q1=q1, q2=q2)
        ee_pos_5, ee_ori_5, gs_5 = _make_pose(5, pos=pos, axis_angle=ori, q1=q1, q2=q2)

        left_1, right_1 = compute_finger_tip_positions(ee_pos_1, ee_ori_1, gs_1)
        left_5, right_5 = compute_finger_tip_positions(ee_pos_5, ee_ori_5, gs_5)

        np.testing.assert_allclose(left_5, np.tile(left_1, (5, 1)), atol=1e-12)
        np.testing.assert_allclose(right_5, np.tile(right_1, (5, 1)), atol=1e-12)

    def test_ee_pos_offset_propagates(self):
        """Translating ee_pos shifts both tips by the same amount."""
        T = 3
        ee_pos_0, ee_ori, gs = _make_pose(T)
        ee_pos_1 = ee_pos_0 + np.array([1.0, 2.0, 3.0])

        left_0, right_0 = compute_finger_tip_positions(ee_pos_0, ee_ori, gs)
        left_1, right_1 = compute_finger_tip_positions(ee_pos_1, ee_ori, gs)

        expected = np.tile([1.0, 2.0, 3.0], (T, 1))
        np.testing.assert_allclose(left_1 - left_0, expected, atol=1e-9)
        np.testing.assert_allclose(right_1 - right_0, expected, atol=1e-9)


class TestComputeFingerTipPositionsSingle:
    """Tests for the single-timestep convenience wrapper."""

    def test_returns_dict_with_correct_keys(self):
        result = compute_finger_tip_positions_single(
            np.zeros(3), np.zeros(3), np.array([0.0, 0.0])
        )
        assert "gripper_l" in result
        assert "gripper_r" in result
        assert result["gripper_l"].shape == (3,)
        assert result["gripper_r"].shape == (3,)

    def test_consistent_with_batch(self):
        """Single-timestep wrapper matches batch version at index 0."""
        pos = np.array([0.1, 0.2, 0.3])
        ori = np.array([0.05, 0.0, 0.1])
        qpos = np.array([0.02, -0.01])

        left_batch, right_batch = compute_finger_tip_positions(
            pos[None], ori[None], qpos[None]
        )
        single = compute_finger_tip_positions_single(pos, ori, qpos)

        np.testing.assert_allclose(single["gripper_l"], left_batch[0], atol=1e-12)
        np.testing.assert_allclose(single["gripper_r"], right_batch[0], atol=1e-12)

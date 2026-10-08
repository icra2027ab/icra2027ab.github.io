"""Per-robot gripper geometry selection.

These pin down the three things that were silently wrong when the UR5 adapter
was first written: the geometry registry key was ignored (everything used the
Panda), the aperture ranges were mismatched, and a single-DOF gripper crashed
the finger FK outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.envs.ur5.config import ur5_env_spec
from policy_abstraction.utils.gripper import (
    compute_finger_tip_positions,
    compute_finger_tip_positions_single,
    gripper_action_to_metric,
    gripper_qpos_to_metric,
)
from policy_steering.robot_config import load_gripper_geometry

PANDA = "franka_panda"
UR5 = "robotiq_2f140"


class TestGeometryRegistry:
    def test_ur5_spec_selects_2f140(self):
        """The env spec is what picks the geometry — not an import-time default."""
        assert ur5_env_spec().robot.gripper.geometry_id == UR5

    def test_2f140_spans_the_full_stroke(self):
        g = load_gripper_geometry(UR5)
        assert g.metric_open == pytest.approx(0.140)
        assert g.metric_close == pytest.approx(0.0)

    def test_2f140_tcp_sits_at_the_fingertips(self):
        """Standard coupler + TCP at the pad plane leaves no residual offset."""
        g = load_gripper_geometry(UR5)
        assert g.finger_x_rest == pytest.approx(0.0)
        assert g.finger_z_from_grip == pytest.approx(0.0)

    def test_unknown_gripper_names_available_ones(self):
        with pytest.raises(FileNotFoundError, match="robotiq_2f140"):
            load_gripper_geometry("no_such_gripper")


class TestApertureMapping:
    @pytest.mark.parametrize(
        "aperture, expected_m",
        [(0.0, 0.0), (0.5, 0.070), (1.0, 0.140)],
    )
    def test_ur5_action_maps_over_140mm(self, aperture, expected_m):
        got = gripper_action_to_metric(aperture, UR5)
        assert float(got) == pytest.approx(expected_m)

    def test_ur5_closed_is_zero_not_forty_millimetres(self):
        """Regression: the Panda's [-1, 1] bounds mapped a closed UR5 gripper
        (0.0) to 40 mm open, and compressed the stroke into a 40-80 mm band."""
        assert float(gripper_action_to_metric(0.0, UR5)) == pytest.approx(0.0)
        assert float(gripper_action_to_metric(0.0, PANDA)) == pytest.approx(0.04)

    def test_panda_action_mapping_unchanged(self):
        assert float(gripper_action_to_metric(-1.0, PANDA)) == pytest.approx(0.0)
        assert float(gripper_action_to_metric(1.0, PANDA)) == pytest.approx(0.08)

    def test_panda_qpos_is_already_metric(self):
        """Observation and action ranges differ on the Panda; only the Robotiq
        has them coincide. Conflating the two rescales every distance."""
        assert float(gripper_qpos_to_metric(0.08, PANDA)) == pytest.approx(0.08)
        assert float(gripper_action_to_metric(0.08, PANDA)) != pytest.approx(0.08)

    def test_ur5_qpos_scales_by_stroke(self):
        assert float(gripper_qpos_to_metric(1.0, UR5)) == pytest.approx(0.140)
        assert float(gripper_qpos_to_metric(0.25, UR5)) == pytest.approx(0.035)


class TestSingleDofFingerFk:
    def test_scalar_qpos_does_not_raise(self):
        """Regression: the UR5 server publishes ``gripper_qpos`` as (1,) and
        the finger FK indexed [:, 1] unconditionally -> IndexError."""
        out = compute_finger_tip_positions_single(
            np.zeros(3), np.zeros(3), np.array([0.5]), UR5
        )
        assert set(out) == {"gripper_l", "gripper_r"}

    def test_tips_straddle_the_grip_site_by_half_the_stroke(self):
        """Fully open at identity rotation: each tip sits 70 mm out in ∓x --
        the axis the 2F-140 opens along as bolted to the UR5e's flange."""
        out = compute_finger_tip_positions_single(
            np.zeros(3), np.zeros(3), np.array([1.0]), UR5
        )
        assert out["gripper_l"][0] == pytest.approx(-0.070)
        assert out["gripper_r"][0] == pytest.approx(+0.070)
        np.testing.assert_allclose(out["gripper_l"][1], 0.0, atol=1e-12)
        np.testing.assert_allclose(out["gripper_r"][1], 0.0, atol=1e-12)

    def test_the_2f140_opens_along_tool_x_and_the_panda_along_eef_y(self):
        """Measured on the cell: with the Panda's y convention the whole
        gripper (mesh and fingertips) rendered a quarter turn about the tool
        axis from the real one."""
        assert load_gripper_geometry(UR5).finger_axis == "x"
        assert load_gripper_geometry(PANDA).finger_axis == "y"
        assert load_gripper_geometry(UR5).finger_open_direction == (1.0, 0.0, 0.0)
        assert load_gripper_geometry(PANDA).finger_open_direction == (0.0, 1.0, 0.0)

    def test_finger_axis_rotates_with_the_tool(self):
        """A quarter turn of the TCP about z carries x-axis jaws onto world y."""
        out = compute_finger_tip_positions_single(
            np.zeros(3), np.array([0.0, 0.0, np.pi / 2]), np.array([1.0]), UR5
        )
        assert out["gripper_l"][1] == pytest.approx(-0.070)
        assert out["gripper_r"][1] == pytest.approx(+0.070)
        np.testing.assert_allclose(out["gripper_l"][0], 0.0, atol=1e-12)

    def test_unknown_finger_axis_is_rejected(self):
        from dataclasses import replace

        with pytest.raises(ValueError, match="finger_axis"):
            replace(load_gripper_geometry(UR5), finger_axis="z")

    def test_closed_tips_coincide(self):
        out = compute_finger_tip_positions_single(
            np.zeros(3), np.zeros(3), np.array([0.0]), UR5
        )
        np.testing.assert_allclose(out["gripper_l"], out["gripper_r"], atol=1e-12)

    def test_aperture_tracks_the_commanded_opening(self):
        """Tip separation must equal the metric aperture, not a rescaled one."""
        for a in (0.0, 0.3, 0.75, 1.0):
            out = compute_finger_tip_positions_single(
                np.zeros(3), np.zeros(3), np.array([a]), UR5
            )
            gap = np.linalg.norm(out["gripper_l"] - out["gripper_r"])
            assert gap == pytest.approx(a * 0.140)

    def test_two_column_panda_path_is_bit_for_bit_unchanged(self):
        """The (T, 2) branch must use q1/q2 verbatim rather than symmetrising,
        so asymmetric grasps keep their real tip positions."""
        ee_pos = np.zeros((3, 3))
        ee_ori = np.zeros((3, 3))
        qpos = np.array([[0.04, -0.01], [0.02, -0.02], [0.0, 0.0]])
        left, right = compute_finger_tip_positions(ee_pos, ee_ori, qpos)
        rest = load_gripper_geometry(PANDA).finger_x_rest
        np.testing.assert_allclose(left[:, 1], -(qpos[:, 0] + rest))
        np.testing.assert_allclose(right[:, 1], -qpos[:, 1] + rest)

    def test_read_only_inputs_are_accepted(self):
        """Observations arrive over msgpack backed by a read-only buffer, and
        scipy's rotation code needs a writable one."""
        ori = np.zeros(3)
        ori.flags.writeable = False
        out = compute_finger_tip_positions_single(
            np.zeros(3), ori, np.array([0.5]), UR5
        )
        assert set(out) == {"gripper_l", "gripper_r"}

    def test_geometry_actually_changes_the_result(self):
        """Guards against the geometry argument being accepted but ignored."""
        args = (np.zeros(3), np.zeros(3), np.array([1.0]))
        ur5 = compute_finger_tip_positions_single(*args, UR5)
        panda = compute_finger_tip_positions_single(*args, PANDA)
        assert not np.allclose(ur5["gripper_l"], panda["gripper_l"])

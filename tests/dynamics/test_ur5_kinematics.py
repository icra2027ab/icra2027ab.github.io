"""UR5e forward kinematics + the joint-space dynamics model built on it."""

import numpy as np
import pytest

from policy_abstraction.dynamics.ur5_kinematics import (
    Ur5KinematicsDynamicsModel,
    matrix_to_rotvec,
    rotvec_to_matrix,
    tcp_offset_matrix,
    ur5e_flange_pose,
    ur5e_forward_kinematics,
)
from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS

HOME = np.asarray(UR5E_HOME_JOINT_POSITIONS)


class TestRotationHelpers:
    def test_rotvec_matrix_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.0, np.pi - 1e-3)
            rotvec = axis * angle
            back = matrix_to_rotvec(rotvec_to_matrix(rotvec))
            np.testing.assert_allclose(back, rotvec, atol=1e-9)

    def test_identity_maps_to_zero(self):
        np.testing.assert_allclose(matrix_to_rotvec(np.eye(3)), np.zeros(3))
        np.testing.assert_allclose(rotvec_to_matrix(np.zeros(3)), np.eye(3))

    @pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    def test_near_pi_rotations_recover_the_axis(self, axis):
        """sin(theta) -> 0 at 180 degrees; the symmetric-part branch must hold."""
        rotvec = np.asarray(axis, dtype=float) * (np.pi - 1e-8)
        R = rotvec_to_matrix(rotvec)
        recovered = matrix_to_rotvec(R)
        # The sign of a 180-degree axis is arbitrary; compare the rotations.
        assert np.allclose(rotvec_to_matrix(recovered), R, atol=1e-6)


class TestForwardKinematics:
    def test_zero_pose_matches_published_ur5e_geometry(self):
        """At q=0 the arm is fully extended along -x.

        x = a2 + a3, y = -(d4 + d6), z = d1 - d5 — read straight off the
        official UR5e DH table, so this pins the parameters rather than the
        transform chain.
        """
        pos, _ = ur5e_forward_kinematics(np.zeros(6))
        np.testing.assert_allclose(
            pos,
            [-0.425 - 0.3922, -(0.1333 + 0.0996), 0.1625 - 0.0997],
            atol=1e-9,
        )

    def test_home_pose_regression(self):
        """Guard the home-pose flange pose against silent DH edits."""
        pos, rotvec = ur5e_forward_kinematics(HOME)
        np.testing.assert_allclose(pos, [-0.49190, -0.13330, 0.48790], atol=1e-5)
        # Home orientation: tool pointing down, 180 degrees about the x=y diagonal.
        np.testing.assert_allclose(np.linalg.norm(rotvec), np.pi, atol=1e-6)

    def test_shoulder_pan_rotates_about_base_z(self):
        """Joint 0 is a pure rotation about the base z-axis."""
        pos0, _ = ur5e_forward_kinematics(HOME)
        turned = HOME.copy()
        turned[0] += np.pi / 2
        pos1, _ = ur5e_forward_kinematics(turned)
        np.testing.assert_allclose(pos1[2], pos0[2], atol=1e-9)
        np.testing.assert_allclose(
            np.linalg.norm(pos1[:2]), np.linalg.norm(pos0[:2]), atol=1e-9
        )

    def test_flange_pose_is_a_valid_rigid_transform(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            T = ur5e_flange_pose(rng.uniform(-np.pi, np.pi, 6))
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)
            np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)

    def test_tcp_offset_translates_along_the_tool_axis(self):
        pos_flange, _ = ur5e_forward_kinematics(HOME)
        pos_tcp, _ = ur5e_forward_kinematics(HOME, tcp_offset=[0, 0, 0.15, 0, 0, 0])
        assert np.linalg.norm(pos_tcp - pos_flange) == pytest.approx(0.15, abs=1e-9)

    def test_identity_tcp_offset_is_a_no_op(self):
        np.testing.assert_allclose(tcp_offset_matrix(None), np.eye(4))
        np.testing.assert_allclose(tcp_offset_matrix([0] * 6), np.eye(4))

    def test_rejects_short_joint_vectors(self):
        with pytest.raises(ValueError, match="6 joints"):
            ur5e_flange_pose([0.0, 0.0, 0.0])


class TestUr5KinematicsDynamicsModel:
    def _state(self):
        return np.zeros(7, dtype=np.float32)

    def test_maps_joint_chunk_to_eef_states(self):
        model = Ur5KinematicsDynamicsModel()
        actions = np.tile(np.append(HOME, 0.4), (1, 5, 1))  # [1, 5, 7]
        states = model.predict_states(actions, self._state())

        assert states.shape == (1, 5, 7)
        pos, rotvec = ur5e_forward_kinematics(HOME)
        np.testing.assert_allclose(states[0, 0, :3], pos, atol=1e-5)
        np.testing.assert_allclose(states[0, 0, 3:6], rotvec, atol=1e-5)

    def test_gripper_channel_is_absolute_not_integrated(self):
        model = Ur5KinematicsDynamicsModel()
        actions = np.stack([np.append(HOME, g) for g in (0.0, 0.5, 1.0)])[
            None
        ]  # [1, 3, 7]
        states = model.predict_states(actions, self._state())
        np.testing.assert_allclose(states[0, :, 6], [0.0, 0.5, 1.0], atol=1e-6)

    def test_is_stateless_wrt_initial_state(self):
        """Same joint targets -> same poses, whatever the arm was doing before."""
        model = Ur5KinematicsDynamicsModel()
        actions = np.tile(np.append(HOME, 1.0), (1, 3, 1))
        a = model.predict_states(actions, np.zeros(7, dtype=np.float64))
        b = model.predict_states(actions, np.full(7, 99.0))
        np.testing.assert_allclose(a, b)

    def test_tcp_offset_shifts_every_predicted_state(self):
        plain = Ur5KinematicsDynamicsModel()
        offset = Ur5KinematicsDynamicsModel(tcp_offset=[0, 0, 0.2, 0, 0, 0])
        actions = np.tile(np.append(HOME, 1.0), (1, 2, 1))
        d = offset.predict_states(actions, self._state()) - plain.predict_states(
            actions, self._state()
        )
        assert np.linalg.norm(d[0, 0, :3]) == pytest.approx(0.2, abs=1e-5)

    def test_rejects_cartesian_sized_actions(self):
        model = Ur5KinematicsDynamicsModel()
        with pytest.raises(ValueError, match=">=6 action dims"):
            model.predict_states(np.zeros((1, 4, 3)), self._state())


@pytest.mark.slow
def test_fk_agrees_with_mujoco_menagerie():
    """Independent cross-check against ``ur5e_mj_description``.

    Menagerie rounds every link offset to millimetres (e.g. d1 = 0.163 vs the
    official 0.1625), so positions agree only to ~1 mm while the orientation
    chain — which carries no such rounding — must match exactly. Downloads the
    menagerie on first run, hence the ``slow`` marker.
    """
    mujoco = pytest.importorskip("mujoco")
    loader = pytest.importorskip("robot_descriptions.loaders.mujoco")

    model = loader.load_robot_description("ur5e_mj_description")
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    rng = np.random.default_rng(0)
    for q in [np.zeros(6), HOME, *[rng.uniform(-np.pi, np.pi, 6) for _ in range(5)]]:
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        T = ur5e_flange_pose(q)
        np.testing.assert_allclose(data.site_xpos[site_id], T[:3, 3], atol=2e-3)
        rel = data.site_xmat[site_id].reshape(3, 3).T @ T[:3, :3]
        assert np.linalg.norm(matrix_to_rotvec(rel)) < 1e-9

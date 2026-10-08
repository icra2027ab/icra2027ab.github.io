"""The JAX twin of the UR5e forward kinematics, as the guidance loss sees it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from policy_abstraction.dynamics.ur5_kinematics import (
    rotvec_to_matrix,
    ur5e_forward_kinematics,
)
from policy_abstraction.dynamics.ur5_kinematics_jax import (
    jax_matrix_to_rotvec,
    make_ur5e_jax_dynamics,
    ur5e_tcp_pose_np,
)
from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS

HOME = np.asarray(UR5E_HOME_JOINT_POSITIONS, dtype=np.float64)
# The cell's controller TCP (docs/ur5.md): a 223 mm tool along the flange z.
TCP = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]


def _same_rotation(rotvec_a, rotvec_b, atol=1e-5) -> bool:
    """Rotation vectors compared as rotations: at 180° the axis sign is free."""
    return np.allclose(
        rotvec_to_matrix(rotvec_a), rotvec_to_matrix(rotvec_b), atol=atol
    )


class TestMatchesTheNumpyKinematics:
    @pytest.mark.parametrize("tcp_offset", [None, TCP])
    def test_random_configurations_agree(self, tcp_offset):
        rng = np.random.default_rng(0)
        for _ in range(30):
            q = rng.uniform(-np.pi, np.pi, size=6)
            pos_np, rot_np = ur5e_forward_kinematics(q, tcp_offset)
            pos_jx, rot_jx = ur5e_tcp_pose_np(q, tcp_offset)
            np.testing.assert_allclose(pos_jx, pos_np, atol=1e-5)
            assert _same_rotation(rot_jx, rot_np)

    def test_the_home_pose_agrees_including_the_tcp(self):
        """Home is where the arm starts every rollout, and the tool points at
        the table there — a 180° rotation, the case a naive extraction breaks."""
        pos_np, rot_np = ur5e_forward_kinematics(HOME, TCP)
        pos_jx, rot_jx = ur5e_tcp_pose_np(HOME, TCP)
        np.testing.assert_allclose(pos_jx, pos_np, atol=1e-5)
        assert _same_rotation(rot_jx, rot_np)
        assert np.linalg.norm(rot_jx) == pytest.approx(np.pi, abs=1e-4)


class TestRotationVectorExtraction:
    @pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    @pytest.mark.parametrize("angle", [0.0, 1e-7, 0.3, 2.5, np.pi - 1e-6, np.pi])
    def test_round_trips_every_angle(self, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        R = rotvec_to_matrix(axis * angle)
        back = np.asarray(jax_matrix_to_rotvec(jnp.asarray(R)))
        assert _same_rotation(back, axis * angle)
        assert np.linalg.norm(back) <= np.pi + 1e-5

    @pytest.mark.parametrize("angle", [0.0, 0.3, np.pi - 1e-6, np.pi])
    def test_gradient_is_finite_at_the_singular_angles(self, angle):
        """0° and 180° are where acos(trace) has an infinite derivative; the
        UR5e's tool sits at 180° whenever it points at the table."""
        axis = np.array([0.3, -0.5, 0.8])
        axis /= np.linalg.norm(axis)
        R = jnp.asarray(rotvec_to_matrix(axis * angle))
        grad = jax.grad(lambda m: jnp.sum(jax_matrix_to_rotvec(m) ** 2))(R)
        assert bool(jnp.all(jnp.isfinite(grad)))


class TestDynamicsFn:
    def test_a_held_chunk_maps_to_the_current_pose(self):
        """The contract that makes the guided chunk start at the end-effector:
        joint targets equal to the current joints give the current TCP pose."""
        fn = make_ur5e_jax_dynamics(TCP)
        x_t = jnp.asarray(np.tile(np.r_[HOME, 0.5], (25, 1)), dtype=jnp.float32)
        positions, orientations, gripper = fn(x_t, jnp.zeros(3), jnp.zeros(3), 0.02)
        pos_np, rot_np = ur5e_forward_kinematics(HOME, TCP)
        assert positions.shape == (25, 3) and orientations.shape == (25, 3)
        np.testing.assert_allclose(
            np.asarray(positions), np.tile(pos_np, (25, 1)), atol=1e-5
        )
        assert _same_rotation(np.asarray(orientations[0]), rot_np)
        np.testing.assert_allclose(np.asarray(gripper), np.full((25, 1), 0.5))

    def test_ignores_the_integration_seed(self):
        """Joint targets are absolute: the current EE pose must not shift them."""
        fn = make_ur5e_jax_dynamics(TCP)
        x_t = jnp.asarray(np.tile(np.r_[HOME, 0.5], (4, 1)), dtype=jnp.float32)
        a = fn(x_t, jnp.zeros(3), jnp.zeros(3), 0.02)[0]
        b = fn(x_t, jnp.ones(3), jnp.ones(3), 1.0)[0]
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_gradient_reaches_the_joints(self):
        """Pulling the TCP toward a target must produce a finite, non-zero
        gradient on the joint channels and none on the gripper channel."""
        fn = make_ur5e_jax_dynamics(TCP)
        pos0, _ = ur5e_forward_kinematics(HOME, TCP)
        target = jnp.asarray(pos0 + np.array([0.05, 0.0, 0.0]), dtype=jnp.float32)

        def loss(x_t):
            positions, _, _ = fn(x_t, jnp.zeros(3), jnp.zeros(3), 0.02)
            return jnp.sum((positions - target[None, :]) ** 2)

        x_t = jnp.asarray(np.tile(np.r_[HOME, 0.5], (3, 1)), dtype=jnp.float32)
        grad = np.asarray(jax.grad(loss)(x_t))
        assert np.all(np.isfinite(grad))
        assert np.linalg.norm(grad[:, :6]) > 0.0
        np.testing.assert_allclose(grad[:, 6], 0.0)

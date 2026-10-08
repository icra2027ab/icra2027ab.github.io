"""The torch twin of the UR5e forward kinematics, as the LeRobot server's guidance sees it."""

import numpy as np
import pytest
import torch

from policy_abstraction.dynamics.ur5_kinematics import (
    UR5E_DH,
    rotvec_to_matrix,
    ur5e_forward_kinematics,
)
from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS
from policy_steering import ur5e_kinematics_torch as kt

HOME = np.asarray(UR5E_HOME_JOINT_POSITIONS, dtype=np.float64)
# The cell's controller TCP (docs/ur5.md): a 223 mm tool along the flange z.
TCP = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]


def _same_rotation(rotvec_a, rotvec_b, atol=1e-5) -> bool:
    """Rotation vectors compared as rotations: at 180° the axis sign is free."""
    return np.allclose(
        rotvec_to_matrix(np.asarray(rotvec_a)),
        rotvec_to_matrix(np.asarray(rotvec_b)),
        atol=atol,
    )


def test_the_dh_table_is_the_one_policy_abstraction_uses():
    # Duplicated because policy_steering is bridged into the lerobot venv
    # alone; a drifted copy would put the guided tool somewhere else.
    assert kt.UR5E_DH == UR5E_DH


class TestMatchesTheNumpyKinematics:
    @pytest.mark.parametrize("tcp_offset", [None, TCP])
    def test_random_configurations_agree(self, tcp_offset):
        rng = np.random.default_rng(0)
        qs = rng.uniform(-np.pi, np.pi, size=(30, 6))
        pos_t, rot_t = kt.torch_ur5e_tcp_pose(
            torch.as_tensor(qs, dtype=torch.float64), tcp_offset
        )
        for q, p, r in zip(qs, pos_t.numpy(), rot_t.numpy(), strict=True):
            pos_np, rot_np = ur5e_forward_kinematics(q, tcp_offset)
            np.testing.assert_allclose(p, pos_np, atol=1e-6)
            assert _same_rotation(r, rot_np)

    def test_the_home_pose_agrees_including_the_tcp(self):
        """Home is where the arm starts every rollout, and the tool points at
        the table there — a 180° rotation, the case a naive extraction breaks."""
        pos_np, rot_np = ur5e_forward_kinematics(HOME, TCP)
        pos_t, rot_t = kt.torch_ur5e_tcp_pose(torch.as_tensor(HOME)[None], TCP)
        np.testing.assert_allclose(pos_t[0].numpy(), pos_np, atol=1e-6)
        assert _same_rotation(rot_t[0].numpy(), rot_np)
        assert float(torch.linalg.norm(rot_t[0])) == pytest.approx(np.pi, abs=1e-4)

    def test_float32_is_accurate_to_a_tenth_of_a_millimetre(self):
        rng = np.random.default_rng(1)
        qs = rng.uniform(-np.pi, np.pi, size=(50, 6))
        pos_t, _ = kt.torch_ur5e_tcp_pose(torch.as_tensor(qs, dtype=torch.float32), TCP)
        ref = np.stack([ur5e_forward_kinematics(q, TCP)[0] for q in qs])
        assert np.abs(pos_t.numpy() - ref).max() < 1e-4


class TestRotationVectorExtraction:
    @pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    @pytest.mark.parametrize("angle", [0.0, 1e-7, 0.3, 2.5, np.pi - 1e-6, np.pi])
    def test_round_trips_every_angle(self, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        R = torch.as_tensor(rotvec_to_matrix(axis * angle))
        back = kt.torch_matrix_to_rotvec(R).numpy()
        assert _same_rotation(back, axis * angle)
        assert np.linalg.norm(back) <= np.pi + 1e-5

    @pytest.mark.parametrize("angle", [0.0, 0.3, np.pi - 1e-6, np.pi])
    def test_gradient_is_finite_at_the_singular_angles(self, angle):
        """0° and 180° are where acos(trace) has an infinite derivative; the
        UR5e's tool sits at 180° whenever it points at the table."""
        axis = np.array([0.3, -0.5, 0.8])
        axis /= np.linalg.norm(axis)
        R = torch.as_tensor(rotvec_to_matrix(axis * angle), dtype=torch.float32)
        R.requires_grad_(True)
        torch.sum(kt.torch_matrix_to_rotvec(R) ** 2).backward()
        assert R.grad is not None and bool(torch.isfinite(R.grad).all())


class TestDynamicsFn:
    def test_a_held_chunk_maps_to_the_current_pose(self):
        """The contract that makes the guided chunk start at the end-effector:
        joint targets equal to the current joints give the current TCP pose."""
        fn = kt.make_ur5e_torch_dynamics(TCP)
        x_t = torch.as_tensor(
            np.tile(np.r_[HOME, 0.5], (2, 25, 1)), dtype=torch.float32
        )
        positions, orientations, gripper = fn(x_t, torch.zeros(3), torch.zeros(3), 0.02)
        pos_np, rot_np = ur5e_forward_kinematics(HOME, TCP)
        assert positions.shape == (2, 25, 3)
        assert orientations.shape == (2, 25, 3)
        assert gripper.shape == (2, 25, 1)
        np.testing.assert_allclose(
            positions.numpy(), np.broadcast_to(pos_np, (2, 25, 3)), atol=1e-5
        )
        for r in orientations.reshape(-1, 3).numpy():
            assert _same_rotation(r, rot_np, atol=1e-4)
        assert torch.all(gripper == 0.5)

    def test_padded_action_channels_are_ignored(self):
        # pi0/pi0.5 denoise a 32-wide tensor; only [q0..q5, gripper] matter.
        fn = kt.make_ur5e_torch_dynamics(TCP)
        x_t = torch.zeros(1, 4, 32)
        x_t[..., :6] = torch.as_tensor(HOME, dtype=torch.float32)
        positions, _, _ = fn(x_t, torch.zeros(3), torch.zeros(3), 0.02)
        np.testing.assert_allclose(
            positions[0, 0].numpy(), ur5e_forward_kinematics(HOME, TCP)[0], atol=1e-5
        )

    def test_bfloat16_chunks_are_computed_in_float32_and_returned_in_kind(self):
        fn = kt.make_ur5e_torch_dynamics(TCP)
        x_t = torch.as_tensor(np.tile(np.r_[HOME, 0.5], (1, 3, 1))).to(torch.bfloat16)
        positions, orientations, gripper = fn(x_t, torch.zeros(3), torch.zeros(3), 0.02)
        assert positions.dtype == torch.bfloat16
        assert orientations.dtype == torch.bfloat16
        assert gripper.dtype == torch.bfloat16
        # bf16 rounds the *inputs* (HOME survives: its entries are 0 and ±pi/2)
        # but the pose must not carry six chained bf16 products' worth of error.
        pos_np, _ = ur5e_forward_kinematics(HOME, TCP)
        assert np.abs(positions[0, 0].float().numpy() - pos_np).max() < 5e-3

    def test_unnormalize_runs_before_the_kinematics(self):
        # Quantile stats: normalized [-1, 1] spans [q01, q99] per channel.
        q01 = np.r_[HOME - 0.5, 0.0]
        q99 = np.r_[HOME + 0.5, 1.0]
        scale = torch.as_tensor((q99 - q01) / 2.0, dtype=torch.float32)
        offset = torch.as_tensor((q99 + q01) / 2.0, dtype=torch.float32)
        fn = kt.make_ur5e_torch_dynamics(
            TCP, unnormalize=kt.make_affine_unnormalizer(scale, offset)
        )
        # HOME sits at the centre of every window -> normalized 0; gripper
        # openness 1.0 is the top of its window -> normalized +1.
        x_t = torch.zeros(1, 5, 7)
        x_t[..., 6] = 1.0
        positions, _, gripper = fn(x_t, torch.zeros(3), torch.zeros(3), 0.02)
        np.testing.assert_allclose(
            positions[0, 0].numpy(), ur5e_forward_kinematics(HOME, TCP)[0], atol=1e-5
        )
        assert torch.allclose(gripper, torch.ones_like(gripper))

    def test_the_gradient_flows_through_unnormalization_into_the_joints(self):
        scale = torch.full((7,), 0.5)
        offset = torch.as_tensor(np.r_[HOME, 0.5], dtype=torch.float32)
        fn = kt.make_ur5e_torch_dynamics(
            TCP, unnormalize=kt.make_affine_unnormalizer(scale, offset)
        )
        x_t = torch.zeros(1, 3, 7, requires_grad=True)
        positions, _, _ = fn(x_t, torch.zeros(3), torch.zeros(3), 0.02)
        positions[..., 2].sum().backward()  # tool height
        g = x_t.grad
        assert g is not None and bool(torch.isfinite(g).all())
        # Every timestep is its own joint target: no cumsum front-loading.
        per_step = g[0].norm(dim=-1)
        assert torch.allclose(per_step, per_step[0].expand_as(per_step), rtol=1e-4)
        # The shoulder and elbow lift the tool; the gripper channel cannot.
        assert g[0, 0, 1].abs() > 1e-3 and g[0, 0, 2].abs() > 1e-3
        assert g[0, 0, 6] == 0


class TestAffineUnnormalizer:
    def test_maps_only_the_leading_channels(self):
        un = kt.make_affine_unnormalizer(
            torch.full((2,), 2.0), torch.tensor([1.0, -1.0])
        )
        out = un(torch.ones(1, 3, 4))
        assert out.shape == (1, 3, 4)
        assert torch.equal(out[..., 0], torch.full((1, 3), 3.0))
        assert torch.equal(out[..., 1], torch.full((1, 3), 1.0))
        assert torch.equal(out[..., 2:], torch.ones(1, 3, 2))

    def test_rejects_narrower_actions_than_the_stats(self):
        un = kt.make_affine_unnormalizer(torch.ones(7), torch.zeros(7))
        with pytest.raises(ValueError, match="7"):
            un(torch.ones(1, 3, 6))

    def test_rejects_mismatched_stats(self):
        with pytest.raises(ValueError, match="must match"):
            kt.make_affine_unnormalizer(torch.ones(7), torch.zeros(6))

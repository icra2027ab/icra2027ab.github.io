"""The in-process fake, on a joint-absolute arm.

``DummyPolicy`` was written for the simulators, whose actions are Cartesian
deltas: starting from zeros means "don't move" and Euler-integrating the
chunk recovers the tool path. Read as the UR5e's absolute joint targets, a
zero chunk is the fully-extended pose and integration turns radians into
metres per second. These tests pin the joint-space contract: start from the
observed joints, and let guidance see the chunk through forward kinematics.
"""

import numpy as np
import pytest

from policy_abstraction.dynamics.ur5_kinematics import ur5e_forward_kinematics
from policy_abstraction.dynamics.ur5_kinematics_jax import make_ur5e_jax_dynamics
from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS
from policy_abstraction.evaluation.guidance import pack_guidance_params
from policy_abstraction.policies.dummy import DummyPolicy

TCP = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]
HORIZON = 25
HOME = np.asarray(UR5E_HOME_JOINT_POSITIONS, dtype=np.float32)
STATE = np.r_[HOME, 0.5].astype(np.float32)


def _ee_to_waypoint_params(waypoint: np.ndarray, n_steps: int = HORIZON) -> dict:
    """One centroid pair, EE -> ``waypoint``, expected at distance 0 throughout."""
    pos0, ori0 = ur5e_forward_kinematics(HOME, TCP)
    n = 1
    return {
        "__gp_ee_pos_0__": pos0.astype(np.float32),
        "__gp_ee_ori_0__": ori0.astype(np.float32),
        "__gp_gripper_0__": np.array([0.5], np.float32),
        "__gp_dt__": np.float32(0.02),
        "__gp_expected_dists__": np.zeros((n, n_steps), np.float32),
        "__gp_expected_stds__": np.ones((n, n_steps), np.float32),
        "__gp_dist_types__": np.zeros(n, np.uint8),
        "__gp_source_type__": np.zeros(n, np.uint8),
        "__gp_waypoints__": waypoint[None].astype(np.float32),
        "__gp_has_axis__": np.zeros(n, np.uint8),
        "__gp_axes__": np.zeros((n, 3), np.float32),
        "__gp_has_plane__": np.zeros(n, np.uint8),
        "__gp_plane_normals__": np.zeros((n, 3), np.float32),
        "__gp_surface_counts__": np.zeros(n, np.int32),
        "__gp_surface_pts__": np.zeros((n, 1, 3), np.float32),
        "__gp_guidance_weights__": np.ones(n, np.float32),
        "__gp_guidance_factor__": np.float32(1.0),
        "__gp_ee_vectors__": np.tile(np.array([0.0, 0.0, 1.0], np.float32), (n, 1)),
        "__gp_ee_offsets__": np.zeros((n, 3), np.float32),
        "__gp_num_chunks__": np.int32(2),
        "__gp_debug_guidance__": True,
        "__gp_line_search__": True,
        "__gp_normalize__": True,
        "__gp_max_alpha__": np.float32(1.0),
    }


@pytest.fixture(scope="module")
def joint_space_fake() -> DummyPolicy:
    return DummyPolicy(
        action_horizon=HORIZON,
        num_output_actions=7,
        dynamics_fn=make_ur5e_jax_dynamics(TCP),
        init_from_state=True,
    )


class TestHold:
    def test_without_guidance_it_returns_the_observed_state(self, joint_space_fake):
        actions = joint_space_fake.infer({"observation/state": STATE})["actions"]
        assert actions.shape == (2, HORIZON, 7)
        np.testing.assert_allclose(
            actions, np.broadcast_to(STATE, actions.shape), atol=1e-6
        )

    def test_the_held_chunk_starts_at_the_end_effector(self, joint_space_fake):
        """What the operator sees: the predicted trajectory's first point is
        where the tool is, not 22 cm below it or a metre away."""
        actions = joint_space_fake.infer({"observation/state": STATE})["actions"]
        pos0, _ = ur5e_forward_kinematics(HOME, TCP)
        for step in actions[0]:
            np.testing.assert_allclose(
                ur5e_forward_kinematics(step[:6], TCP)[0], pos0, atol=1e-6
            )

    def test_a_missing_state_falls_back_to_zeros(self, joint_space_fake):
        """The warm-up query carries no state; it must not crash, and it must
        not invent one."""
        actions = joint_space_fake.infer({})["actions"]
        assert not np.any(actions)

    def test_the_default_fake_still_emits_zeros(self):
        """The simulator contract is untouched."""
        fake = DummyPolicy(action_horizon=HORIZON, num_output_actions=7)
        actions = fake.infer({"observation/state": STATE})["actions"]
        assert not np.any(actions)


class TestGuidanceThroughForwardKinematics:
    def test_guidance_moves_the_tool_to_the_waypoint(self, joint_space_fake):
        pos0, _ = ur5e_forward_kinematics(HOME, TCP)
        waypoint = pos0 + np.array([0.05, 0.0, 0.0])
        obs = pack_guidance_params(
            {"observation/state": STATE}, _ee_to_waypoint_params(waypoint)
        )
        actions = joint_space_fake.infer(obs)["actions"]
        tool = np.array([ur5e_forward_kinematics(a[:6], TCP)[0] for a in actions[0]])
        # Every step of the guided chunk lands on the target ...
        assert np.linalg.norm(tool - waypoint, axis=1).max() < 0.005
        # ... through a small joint motion, not a jump to the extended pose.
        assert np.abs(actions[0, :, :6] - HOME).max() < 0.3
        # The gripper channel is absolute and this pair does not touch it.
        np.testing.assert_allclose(actions[0, :, 6], 0.5, atol=1e-6)

    def test_an_already_satisfied_pair_leaves_the_hold_alone(self, joint_space_fake):
        pos0, _ = ur5e_forward_kinematics(HOME, TCP)
        obs = pack_guidance_params(
            {"observation/state": STATE}, _ee_to_waypoint_params(pos0)
        )
        actions = joint_space_fake.infer(obs)["actions"]
        np.testing.assert_allclose(
            actions[0], np.broadcast_to(STATE, (HORIZON, 7)), atol=1e-3
        )

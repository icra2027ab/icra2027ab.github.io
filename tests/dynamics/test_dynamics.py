"""
Tests for policy_abstraction.dynamics.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.dynamics.identity import IdentityDynamicsModel
from policy_abstraction.dynamics.integration import IntegrationDynamicsModel

# ---------------------------------------------------------------------------
# IdentityDynamicsModel
# ---------------------------------------------------------------------------


class TestIdentityDynamicsModel:
    def test_same_dim_passthrough(self):
        """When action_dim == state_dim, actions are returned unchanged."""
        actions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (T=2, dim=3)
        initial_state = np.zeros(3)
        model = IdentityDynamicsModel()
        result = model.predict_states(actions, initial_state)
        np.testing.assert_array_equal(result, actions)

    def test_dtype_matches_initial_state(self):
        """Output dtype matches initial_state dtype."""
        actions = np.ones((5, 7))
        initial_state = np.zeros(7, dtype=np.float32)
        result = IdentityDynamicsModel().predict_states(actions, initial_state)
        assert result.dtype == np.float32

    def test_pad_when_action_dim_less_than_state_dim(self):
        """action_dim < state_dim: extra columns are zero-padded."""
        actions = np.ones((3, 4))
        initial_state = np.zeros(7)
        result = IdentityDynamicsModel().predict_states(actions, initial_state)
        assert result.shape == (3, 7)
        np.testing.assert_array_equal(result[:, :4], actions)
        np.testing.assert_array_equal(result[:, 4:], 0.0)

    def test_truncate_when_action_dim_greater_than_state_dim(self):
        """action_dim > state_dim: actions are truncated to state_dim."""
        actions = np.arange(12, dtype=float).reshape(3, 4)
        initial_state = np.zeros(2)
        result = IdentityDynamicsModel().predict_states(actions, initial_state)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, actions[:, :2])

    def test_batched_input(self):
        """Accepts [..., T, action_dim] shaped inputs."""
        actions = np.ones((2, 5, 7))  # batch=2, T=5, dim=7
        initial_state = np.zeros(7)
        result = IdentityDynamicsModel().predict_states(actions, initial_state)
        assert result.shape == (2, 5, 7)


# ---------------------------------------------------------------------------
# IntegrationDynamicsModel
# ---------------------------------------------------------------------------


class TestIntegrationDynamicsModel:
    def test_integration_accumulates_velocity(self):
        """States accumulate via cumulative sum * dt (spatial dims)."""
        dt = 1.0 / 50.0
        T = 5
        dim = 4
        velocity = np.ones((T, dim))
        velocity[:, -1] = 0.5  # gripper: absolute, not integrated

        initial_state = np.zeros(dim)
        model = IntegrationDynamicsModel()
        states = model.predict_states(velocity, initial_state, dt=dt)

        assert states.shape == (T, dim)
        # Non-gripper dims of states[0] equal initial_state (gripper is overridden by actions)
        np.testing.assert_allclose(states[0, :-1], initial_state[:-1], atol=1e-12)
        # states[t, :-1] = t * dt (accumulated velocity 1.0)
        for t in range(1, T):
            np.testing.assert_allclose(states[t, :-1], t * dt, atol=1e-12)

    def test_gripper_dim_is_absolute(self):
        """Last dimension of states always mirrors the action values (not integrated)."""
        T = 6
        actions = np.zeros((T, 7))
        actions[:, -1] = np.linspace(0.0, 1.0, T)  # gripper command varies

        initial_state = np.zeros(7)
        states = IntegrationDynamicsModel().predict_states(actions, initial_state)

        np.testing.assert_allclose(states[:, -1], actions[:, -1], atol=1e-12)

    def test_zero_velocity_holds_initial_state_for_non_gripper_dims(self):
        """Zero velocity → non-gripper dims stay at initial_state[:-1]."""
        T, dim = 5, 4
        actions = np.zeros((T, dim))
        initial_state = np.array([1.0, 2.0, 3.0, 0.5])
        states = IntegrationDynamicsModel().predict_states(actions, initial_state)

        # Non-gripper dims should stay at initial_state values
        for t in range(T):
            np.testing.assert_allclose(states[t, :-1], initial_state[:-1], atol=1e-12)

    def test_first_state_equals_initial(self):
        """states[0, :-1] == initial_state[:-1] (gripper dim is overridden by actions)."""
        actions = np.random.default_rng(7).random((10, 7))
        initial_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        states = IntegrationDynamicsModel().predict_states(actions, initial_state)
        np.testing.assert_allclose(states[0, :-1], initial_state[:-1], atol=1e-12)

    def test_output_shape(self):
        """Output shape is (T, state_dim)."""
        T, dim = 8, 7
        actions = np.ones((T, dim))
        initial_state = np.zeros(dim)
        states = IntegrationDynamicsModel().predict_states(actions, initial_state)
        assert states.shape == (T, dim)

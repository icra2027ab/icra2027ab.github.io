"""
IntegrationDynamicsModel.

Numerically integrates a delta-action (velocity) sequence with a fixed time step
(dt = 1/50 s) to produce an absolute position trajectory.  The last action
dimension (gripper) is treated as an absolute value and not integrated.

Migrated from: libero_evaluation_utils/components/dynamics_model.py
"""

import numpy as np

from policy_abstraction.dynamics.base import BaseDynamicsModel


class IntegrationDynamicsModel(BaseDynamicsModel):
    def predict_states(
        self,
        actions: np.ndarray,
        initial_state: np.ndarray,
        dt: float = 1.0 / 50,
        **kwargs,
    ) -> np.ndarray:

        actions = np.asarray(actions)
        initial_state = np.asarray(initial_state)

        state_dim = initial_state.shape[0]
        action_dim = actions.shape[-1] if actions.ndim > 1 else 1

        # integrate the actions to get predicted states
        T = actions.shape[-2] if actions.ndim > 1 else 1
        states = np.zeros(
            (*actions.shape[:-2], T, state_dim), dtype=initial_state.dtype
        )
        states[..., 0, :] = initial_state
        copy_dim = min(action_dim, state_dim)
        for t in range(1, T):
            if actions.ndim == 1:
                states[..., t, :copy_dim] = states[..., t - 1, :copy_dim] + actions * dt
            else:
                states[..., t, :copy_dim] = (
                    states[..., t - 1, :copy_dim] + actions[..., t, :copy_dim] * dt
                )

        # except the gripper state, which we treat as absolute (not integrated)
        states[..., :, -1:] = actions[..., :, -1:]

        return states

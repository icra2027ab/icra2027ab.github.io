"""
IdentityDynamicsModel.

Treats the action sequence directly as a predicted state trajectory.  Correct
when the policy outputs absolute end-effector poses.  Zero-pads or truncates if
action_dim != state_dim.

Migrated from: libero_evaluation_utils/components/dynamics_model.py
"""

import numpy as np

from policy_abstraction.dynamics.base import BaseDynamicsModel

# ---------------------------------------------------------------------------
# Default: identity – actions == predicted states
# ---------------------------------------------------------------------------


class IdentityDynamicsModel(BaseDynamicsModel):
    """Trivial dynamics model: the action is assumed to be the robot state.

    This is appropriate when actions are absolute end-effector poses and the
    caller simply wants to visualise the commanded waypoints without running a
    full forward-kinematics model.
    """

    def predict_states(
        self,
        actions: np.ndarray,
        initial_state: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return *actions* unchanged as predicted states.

        If ``action_dim != state_dim`` the action array is zero-padded or
        truncated along the last axis to match ``initial_state``'s shape so
        that downstream visualisers always receive consistently sized arrays.

        Args:
            actions:       Array of shape ``[..., T, action_dim]``.
            initial_state: Array of shape ``[state_dim]``.  Only its shape is
                           used by this implementation.

        Returns:
            Array of shape ``[..., T, state_dim]``.
        """
        actions = np.asarray(actions)
        initial_state = np.asarray(initial_state)

        state_dim = initial_state.shape[0]
        action_dim = actions.shape[-1] if actions.ndim > 1 else 1

        if action_dim == state_dim:
            return actions.astype(initial_state.dtype)

        # Pad or truncate
        T = actions.shape[-2] if actions.ndim > 1 else 1
        states = np.zeros(
            (*actions.shape[:-2], T, state_dim), dtype=initial_state.dtype
        )
        copy_dim = min(action_dim, state_dim)
        if actions.ndim == 1:
            states[:, :copy_dim] = actions[:copy_dim]
        else:
            states[..., :copy_dim] = actions[..., :copy_dim]
        return states

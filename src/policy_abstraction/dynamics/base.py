"""Dynamics model components for the EpisodeRunner action loop.

A DynamicsModel maps a sequence of actions (and an initial robot state) to a
sequence of predicted robot states.  This is used to visualise the planned
trajectory before it is executed.

The default ``IdentityDynamicsModel`` treats each action directly as the
resulting robot state, which is a reasonable approximation for many
end-effector–control tasks where the action *is* a delta/absolute pose.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseDynamicsModel(ABC):
    """Abstract base class for dynamics models."""

    @abstractmethod
    def predict_states(
        self,
        actions: np.ndarray,
        initial_state: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict a sequence of robot states from a sequence of actions.

        Args:
            actions:       Array of shape ``[..., T, action_dim]`` – the action
                           sequence to roll out.
            initial_state: Array of shape ``[state_dim]`` – the robot state
                           at the beginning of the sequence.

        Returns:
            Array of shape ``[..., T, state_dim]`` – one predicted state per action
            step.  Index ``i`` is the state *after* executing ``actions[i]``.
        """

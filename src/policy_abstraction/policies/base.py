"""Abstract policy client interface."""

from abc import ABC, abstractmethod


class BasePolicyClient(ABC):
    """Minimal interface exposed to EpisodeRunner."""

    @abstractmethod
    def infer(self, obs_dict: dict) -> dict:
        """Run policy inference.

        Parameters
        ----------
        obs_dict:
            Sensor observations (images, robot state, language, ...).

        Returns
        -------
        dict
            Must contain ``"actions"`` key with shape (H, action_dim).
            Optionally contains ``"action_chunks"`` with shape (N, H, action_dim).
        """

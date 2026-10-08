"""Abstract perception pipeline interface."""

from abc import ABC, abstractmethod


class BasePerceptionPipeline(ABC):
    """Base class for all perception modules."""

    @abstractmethod
    def segment(self, obs: dict) -> dict:
        """Segment the scene.

        Parameters
        ----------
        obs:
            Environment observation dict.

        Returns
        -------
        dict
            Maps object name → binary mask (H, W bool) and/or 3-D point cloud.
        """

    @property
    def object_names(self) -> list:
        """Names of objects currently tracked in the scene."""
        return []

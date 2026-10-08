"""
Ground-truth segmentation + object names from the simulator.

Uses the ``segmentation`` and ``segmentation_name_mapping`` fields injected
into observations by the LIBERO WebSocket env server.  No external models required.
"""

from policy_abstraction.perception.base import BasePerceptionPipeline


class GtSegmentation(BasePerceptionPipeline):
    """Perception pipeline using ground-truth simulator segmentation.

    Parameters
    ----------
    obs:
        The observation dict from the environment.  Must contain
        ``"segmentation"`` (H, W) int array and ``"segmentation_name_mapping"``
        dict (id → name).
    """

    def segment(self, obs: dict) -> dict:
        """Return per-object masks extracted from ground-truth segmentation.

        Returns
        -------
        dict
            Maps object name → binary mask (H, W) bool array.
        """
        seg = obs.get("segmentation")
        name_map = obs.get("segmentation_name_mapping", {})
        result = {}
        if seg is None:
            return result
        for seg_id, name in name_map.items():
            if seg_id == 0:
                continue  # background
            result[name] = seg == seg_id
        return result

    @property
    def object_names(self) -> list:
        """Not stateful — returns empty list; inspect segment() output instead."""
        return []

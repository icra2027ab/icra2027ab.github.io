"""
CentroidDistance feature.

Computes the Euclidean distance between the mean (centroid) positions of two
entities in the shared 3-D point cloud.  Entities are identified either by
the string ``"ee"`` (end-effector) or by an integer segmentation ID.
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.features.base import DistanceFeature


class CentroidDistance(DistanceFeature):
    """Euclidean distance between the centroids of two entities."""

    def compute(
        self,
        from_entity: int | str,
        to_entity: int | str,
        pcd: np.ndarray,
        ee_pos: np.ndarray,
        finger_positions: dict[str, np.ndarray] | None = None,
        *,
        ee_ori: np.ndarray | None = None,
        gripper_state: float | None = None,
    ) -> float | None:
        pts_a = self._resolve(from_entity, pcd, ee_pos, finger_positions)
        pts_b = self._resolve(to_entity, pcd, ee_pos, finger_positions)
        if pts_a is None or pts_b is None:
            return None
        return float(np.linalg.norm(pts_a.mean(axis=0) - pts_b.mean(axis=0)))

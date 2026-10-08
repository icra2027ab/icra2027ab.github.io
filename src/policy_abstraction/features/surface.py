"""
MinSurfaceDistance feature.

Computes the minimum point-to-point distance between the surface point clouds
of two entities using a scipy KD-tree.  More sensitive to proximity than
centroid distance when objects are large or irregularly shaped.
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.features.base import DistanceFeature


class MinSurfaceDistance(DistanceFeature):
    """Minimum distance between any point of entity A and any point of entity B."""

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
        from scipy.spatial import KDTree

        pts_a = self._resolve(from_entity, pcd, ee_pos, finger_positions)
        pts_b = self._resolve(to_entity, pcd, ee_pos, finger_positions)
        if pts_a is None or pts_b is None:
            return None
        dists, _ = KDTree(pts_b).query(pts_a, k=1)
        return float(dists.min())

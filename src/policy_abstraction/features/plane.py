"""
PlaneProjectionDistance feature.

Computes the magnitude of the centroid-to-centroid vector projected onto a
plane defined by a normal vector.  This captures 2-D lateral displacement
while ignoring the component along the normal.

For example, with ``normal=[0,0,1]`` (the XY plane) the result is the
Euclidean distance between the two centroids in the horizontal plane,
ignoring any vertical separation.

The normal is normalised at construction time so callers may pass
unnormalized vectors.
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.features.base import DistanceFeature


class PlaneProjectionDistance(DistanceFeature):
    """Magnitude of (centroid_to - centroid_from) projected onto a plane."""

    def __init__(self, normal: list[float] | np.ndarray) -> None:
        n = np.asarray(normal, dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            raise ValueError(
                f"PlaneProjectionDistance: normal has near-zero norm: {normal}"
            )
        self._normal = n / norm

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
        delta = pts_b.mean(axis=0) - pts_a.mean(axis=0)
        # Remove the component along the normal; keep only the in-plane part
        in_plane = delta - np.dot(delta, self._normal) * self._normal
        return float(np.linalg.norm(in_plane))

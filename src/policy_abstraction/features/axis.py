"""
AxisProjectionDistance feature.

Computes the signed scalar projection of the centroid-to-centroid vector
(from_entity → to_entity) onto a caller-supplied unit axis.  Positive values
mean *to_entity* is further in the axis direction than *from_entity*.

Typical use cases:
- ``axis=[0,0,1]`` — vertical (Z) clearance between two objects
- ``axis=[1,0,0]`` — horizontal (X) separation
- Any normalised world-space direction

The axis is normalised at construction time so callers may pass
unnormalized vectors.
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.features.base import DistanceFeature


class AxisProjectionDistance(DistanceFeature):
    """Signed projection of (centroid_to - centroid_from) onto a given axis."""

    def __init__(self, axis: list[float] | np.ndarray) -> None:
        axis_arr = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis_arr)
        if norm < 1e-9:
            raise ValueError(f"AxisProjectionDistance: axis has near-zero norm: {axis}")
        self._axis = axis_arr / norm

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
        return float(np.dot(delta, self._axis))

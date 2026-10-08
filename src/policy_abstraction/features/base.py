"""Modular distance features for computing pairwise metrics between entities.

An entity is either:
- ``"ee"``         — the robot end-effector (position supplied as ``ee_pos``)
- ``"gripper_l"``  — left gripper finger tip (looked up in ``finger_positions``)
- ``"gripper_r"``  — right gripper finger tip (looked up in ``finger_positions``)
- ``int``          — a segmentation instance ID (looked up in the point cloud)

To add a new metric, subclass :class:`DistanceFeature` and implement
:meth:`compute`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from policy_abstraction.vlm.types import DistancePair

from policy_abstraction.vlm.types import DistanceType


class DistanceFeature(ABC):
    """Computes a scalar distance between two entities at one timestep."""

    @abstractmethod
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
        """Return the distance, or ``None`` if an entity has no points."""

    def _resolve(
        self,
        entity: int | str,
        pcd: np.ndarray,
        ee_pos: np.ndarray,
        finger_positions: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Return the point(s) for *entity* (shape ``[N, 3]`` or ``[1, 3]``)."""
        if entity == "ee":
            return ee_pos[None]  # [1, 3]
        if entity in ("gripper_l", "gripper_r") and finger_positions is not None:
            pos = finger_positions.get(str(entity))
            return pos[None] if pos is not None else None  # [1, 3]
        if isinstance(entity, int):
            mask = pcd[:, 6].astype(np.int32) == entity
            return pcd[mask, :3] if mask.any() else None
        return None


def compute_distances(
    pairs: list[DistancePair],
    feature: DistanceFeature,
    pcd: np.ndarray,
    ee_pos: np.ndarray,
    finger_positions: dict[str, np.ndarray] | None = None,
    *,
    ee_ori: np.ndarray | None = None,
    gripper_state: float | None = None,
) -> dict[str, float]:
    """Compute the appropriate distance for each pair.

    Dispatches to the correct feature based on ``pair.distance_type`` and the
    optional ``pair.axis`` / ``pair.plane`` component selectors:

    - ``ORIENTATION`` — :class:`~policy_abstraction.features.orientation.OrientationProjectionFeature`:
      projection of the gripper's world-frame z-axis onto the supplied
      ``axis`` (signed cosine, in ``[-1, 1]``) or in-plane magnitude when
      ``plane`` is set (in ``[0, 1]``).
    - ``GRIPPER_APERTURE`` — :class:`~policy_abstraction.features.gripper_aperture.GripperApertureFeature`:
      scalar gripper opening value.
    - ``pair.axis`` set — :class:`~policy_abstraction.features.axis.AxisProjectionDistance`:
      signed 1-D projection onto the axis direction.
    - ``pair.plane`` set — :class:`~policy_abstraction.features.plane.PlaneProjectionDistance`:
      2-D magnitude of the in-plane component (perpendicular to the normal).
    - ``SURFACE`` (no axis/plane) — :class:`~policy_abstraction.features.surface.MinSurfaceDistance`.
    - ``CENTROID`` (no axis/plane) — uses the caller-supplied *feature*.

    Pairs with missing points are silently skipped.
    """

    out = {}
    for pair in pairs:
        # gripper_l <-> gripper_r pairs always measure aperture regardless of
        # the VLM-declared distance_type, so the reference/live values match
        # the gripper-scalar semantics used by the JAX guidance loss.
        is_finger_pair = (
            pair.from_entity in ("gripper_l", "gripper_r")
            and pair.to_entity in ("gripper_l", "gripper_r")
            and pair.from_entity != pair.to_entity
        )
        if pair.distance_type == DistanceType.ORIENTATION:
            from policy_abstraction.features.orientation import (
                OrientationProjectionFeature,
            )

            if pair.axis is not None:
                direction = pair.axis
                mode = "axis"
            elif pair.plane is not None:
                direction = pair.plane
                mode = "plane"
            else:
                continue
            f: DistanceFeature = OrientationProjectionFeature(
                direction, mode=mode, ee_vector=pair.ee_vector
            )
        elif pair.distance_type == DistanceType.GRIPPER_APERTURE or is_finger_pair:
            from policy_abstraction.features.gripper_aperture import (
                GripperApertureFeature,
            )

            f = GripperApertureFeature()
        elif pair.axis is not None:
            from policy_abstraction.features.axis import AxisProjectionDistance

            f = AxisProjectionDistance(pair.axis)
        elif pair.plane is not None:
            from policy_abstraction.features.plane import PlaneProjectionDistance

            f = PlaneProjectionDistance(pair.plane)
        elif pair.distance_type == DistanceType.SURFACE:
            from policy_abstraction.features.surface import MinSurfaceDistance

            f = MinSurfaceDistance()
        else:
            f = feature
        val = f.compute(
            pair.from_entity,
            pair.to_entity,
            pcd,
            ee_pos,
            finger_positions,
            ee_ori=ee_ori,
            gripper_state=gripper_state,
        )
        if val is not None:
            out[pair.label] = val
    return out

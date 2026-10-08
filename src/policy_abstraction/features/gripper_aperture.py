"""
GripperApertureFeature.

Returns the current gripper opening value as a scalar "distance".
This feature is used for gripper aperture guidance where the gripper
opening is tracked against a reference trajectory.

The gripper state is the scalar ``q1 - q2`` value from the Franka Panda
gripper, where ``q1 ∈ [0, 0.04]`` (left finger) and ``q2 ∈ [-0.04, 0]``
(right finger).
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.features.base import DistanceFeature


class GripperApertureFeature(DistanceFeature):
    """Return the gripper opening scalar as a distance value."""

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
        return gripper_state

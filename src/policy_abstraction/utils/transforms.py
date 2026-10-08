"""Rotation and rigid-body transform utilities.

  quat2axisangle(quat) -> np.ndarray [3]
      Converts an [x, y, z, w] quaternion to an axis-angle rotation vector.
      Clips w to [-1, 1] to guard against acos domain errors.
      (Identical to the robosuite implementation.)

Migrated from: libero_evaluation_utils/env_utils.py
"""

from __future__ import annotations

import math

import numpy as np


def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.

    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Args:
        quat: Quaternion as numpy array [x, y, z, w]

    Returns:
        Axis-angle representation as numpy array
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

"""Depth buffer conversion utilities.

mujoco_depth_to_metric(depth, near, far) -> np.ndarray
    Converts MuJoCo's OpenGL normalized depth buffer (values in [0, 1]) to
    absolute metric depth in metres using the camera's near and far clip planes.
    Formula: z = near / (1 - d * (1 - near/far))

depth_to_color(depth, max_depth) -> np.ndarray
    Converts a depth map to a color image for visualization.

Extracted from: libero_evaluation_utils/analize_dataset_rollout.py
"""

from __future__ import annotations

import cv2
import numpy as np


def mujoco_depth_to_metric(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Inverse of the MuJoCo perspective depth-buffer mapping.

    MuJoCo renders a normalised depth map ``d ∈ [0, 1]`` where
    0 = near plane and 1 = far plane.  The inverse mapping is:

        real_depth = near / (1 − d · (1 − near/far))
    """
    return near / (1.0 - depth * (1.0 - near / far))


def depth_to_color(depth: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    """Convert a depth map to a color image for visualization."""
    if max_depth is None:
        max_depth = np.max(depth)
    min_depth = np.min(depth)
    depth_normalized = (
        (np.clip(depth, 0, max_depth) - min_depth) / (max_depth - min_depth) * 255
    ).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

"""Bounding-box and crop utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def erode_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Shrink a boolean mask with a square structuring element.

    ``kernel_size <= 1`` is a no-op (returns ``mask`` unchanged).
    """
    if kernel_size <= 1:
        return mask
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    return binary_erosion(mask, structure=structure)


def mask_to_bbox(mask: np.ndarray, pad: int = 0) -> np.ndarray | None:
    """Tight ``[y0, x0, y1, x1]`` bbox around a boolean mask with optional padding."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    H, W = mask.shape[:2]
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, H)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)
    return np.array([y0, x0, y1, x1], dtype=np.int32)


def crop_rgb(
    rgb: np.ndarray,
    bbox: np.ndarray,
    mask: np.ndarray | None = None,
    zero_background: bool = False,
) -> np.ndarray:
    """Crop ``rgb`` to ``bbox``; optionally zero out background outside ``mask``."""
    y0, x0, y1, x1 = (int(v) for v in bbox)
    crop = rgb[y0:y1, x0:x1].copy()
    if zero_background and mask is not None:
        local = mask[y0:y1, x0:x1]
        crop[~local] = 0
    return crop

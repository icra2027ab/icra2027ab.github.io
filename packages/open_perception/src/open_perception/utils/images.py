"""Image-shape validation and numpy<->PIL conversion helpers."""

from __future__ import annotations

import numpy as np


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) uint8 copy of ``img``, casting/clipping if needed."""
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3), got {img.shape}")
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def normalize_name(name: str) -> str:
    """Strip digits and underscores. Mirrors `scripts/extract_objects.py`."""
    import re

    return " ".join(re.sub(r"\d+", "", name).replace("_", " ").split())

"""Stage registry.

Importing this module triggers registration of every built-in stage via the
`@REGISTRY.register` decorators on each submodule.
"""

from __future__ import annotations

# Import submodules for their side effects (stage registration).
from . import (  # noqa: F401
    assign,
    detect_gemini,
    encode,
    fuse_views,
    reconstruct,
    sam2,
    sam3,
    vlm_pre,
)
from .base import REGISTRY, Stage, StageFactory
from .disambiguate import clip, dino, vlm  # noqa: F401
from .filter import erode, overlap, pointcloud_outlier, score, size, top_k  # noqa: F401

__all__ = ["REGISTRY", "Stage", "StageFactory"]

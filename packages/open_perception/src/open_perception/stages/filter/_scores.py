"""Shared per-instance score combination — used by filter.overlap / filter.score.

Both filters need a single 1D score per mask drawn from a weighted mix of
sources stashed on the context: ``ctx.mask_scores`` (the SAM3 segmentation
score, surfaced under the special key ``"segmentation"``) and
``ctx.scores[key]`` (per-query M×P matrices written by disambiguators —
collapsed to 1D via ``score_agg``). Each source is L2-normalised before
summing when ``normalize_sources`` is true so weights mean what callers expect
across CLIP / DINO / VLM scale differences.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from loguru import logger

from ...context import DetectionContext

ScoreAgg = Literal["max", "mean"]
SEGMENTATION_KEY = "segmentation"


def combine_scores(
    ctx: DetectionContext,
    seg_scores: np.ndarray,
    score_weights: dict[str, float],
    score_agg: ScoreAgg = "max",
    normalize_sources: bool = True,
    *,
    log_prefix: str = "[filter]",
) -> np.ndarray:
    """Weighted sum of per-instance score vectors drawn from `ctx.scores`.

    Returns a (N,) float32 array. 2D matrices (M, P) are aggregated to 1D via
    ``score_agg`` (max or mean over queries). The special key
    ``"segmentation"`` pulls from ``seg_scores``. Missing keys log a warning
    and are skipped; if no source matched, ``seg_scores`` is returned
    unchanged so callers behave like the segmentation-only path.
    """
    if ctx.instances is None or ctx.instances.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    N = int(ctx.instances.shape[0])
    out = np.zeros((N,), dtype=np.float32)
    any_active = False
    for key, weight in score_weights.items():
        if key == SEGMENTATION_KEY:
            vec = np.asarray(seg_scores, dtype=np.float32)
        else:
            mat = ctx.scores.get(key)
            if mat is None:
                logger.warning(
                    f"{log_prefix} score source {key!r} not found in "
                    f"ctx.scores (available: {sorted(ctx.scores)}); skipping"
                )
                continue
            if mat.size == 0 or mat.shape[0] != N:
                logger.warning(
                    f"{log_prefix} score source {key!r} has shape {mat.shape}; "
                    f"expected first dim {N} — skipping"
                )
                continue
            if mat.ndim == 1:
                vec = mat.astype(np.float32, copy=False)
            elif score_agg == "max":
                vec = mat.max(axis=1).astype(np.float32, copy=False)
            else:
                vec = mat.mean(axis=1).astype(np.float32, copy=False)
        if normalize_sources:
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
        out += float(weight) * vec
        any_active = True
    if not any_active:
        logger.warning(
            f"{log_prefix} score_weights set but no source matched ctx.scores "
            "— falling back to segmentation score"
        )
        return np.asarray(seg_scores, dtype=np.float32)
    return out

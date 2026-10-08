"""Top-K score filter — keep only the K highest-scoring masks.

By default, ranking uses ``ctx.mask_scores`` (the SAM3 segmentation score).
Pass ``score_weights`` to rank by a weighted combination of sources from
``ctx.scores`` plus the special key ``"segmentation"``, e.g.
``{"dino": 0.5, "segmentation": 0.25, "clip": 0.25}``. Per-query 2D score
matrices are aggregated to 1D via ``score_agg`` (max over queries by
default), and each source is L2-normalised before summing when
``normalize_sources=True``.

Kept masks retain their original ordering in ``ctx.instances`` (this filter
selects which masks survive — it does not re-sort the array). Mirrors
``filter.size`` and ``filter.score``: filters ``ctx.instances``,
``ctx.mask_scores``, ``ctx.bboxes``, ``ctx.crops``, reindexes
``ctx.scores``, and trims ``ctx.detections`` populated upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...context import DetectionContext
from ..base import REGISTRY
from ._scores import ScoreAgg, combine_scores


@dataclass
class TopKFilterStage:
    k: int
    score_weights: dict[str, float] | None = None
    score_agg: ScoreAgg = "max"
    normalize_sources: bool = True
    name: str = "filter.top_k"
    requires: ClassVar[tuple[str, ...]] = ("instances",)
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        if ctx.instances is None or ctx.instances.shape[0] == 0:
            return ctx

        N = int(ctx.instances.shape[0])
        H, W = ctx.image_hw

        seg_scores = ctx.mask_scores
        if seg_scores is None:
            seg_scores = np.ones((N,), dtype=np.float32)

        if self.k <= 0:
            ctx.instances = np.zeros((0, H, W), dtype=bool)
            ctx.mask_scores = np.zeros((0,), dtype=seg_scores.dtype)
            ctx.bboxes = np.zeros((0, 4), dtype=np.int32)
            ctx.crops = []
            for key, mat in list(ctx.scores.items()):
                if mat.ndim >= 1 and mat.shape[0] == N:
                    ctx.scores[key] = mat[:0]
            if ctx.detections:
                ctx.detections = {q: [] for q in ctx.detections}
            return ctx

        if N <= self.k:
            return ctx

        if self.score_weights:
            scores = combine_scores(
                ctx,
                seg_scores,
                self.score_weights,
                score_agg=self.score_agg,
                normalize_sources=self.normalize_sources,
                log_prefix="[filter.top_k]",
            )
        else:
            scores = np.asarray(seg_scores, dtype=np.float32)

        # Top-K by score, then resorted ascending so ctx.instances ordering is
        # preserved (we're selecting, not re-sorting).
        top_k_idx = np.argpartition(-scores, self.k - 1)[: self.k]
        keep_idx = np.sort(top_k_idx)

        ctx.instances = ctx.instances[keep_idx]
        ctx.mask_scores = np.asarray(
            [seg_scores[i] for i in keep_idx], dtype=seg_scores.dtype
        )
        if ctx.bboxes is not None and ctx.bboxes.shape[0] > 0:
            ctx.bboxes = ctx.bboxes[keep_idx]
        if ctx.crops:
            ctx.crops = [ctx.crops[int(i)] for i in keep_idx]

        # Reindex ctx.scores so keys remain aligned with ctx.instances.
        for key, mat in list(ctx.scores.items()):
            if mat.size == 0:
                continue
            if mat.ndim >= 1 and mat.shape[0] == N:
                ctx.scores[key] = mat[keep_idx]

        # Trim detections populated upstream (e.g. sam3 per-query emission).
        if ctx.detections:
            old_to_new = {int(old): new for new, old in enumerate(keep_idx)}
            for q, dets in list(ctx.detections.items()):
                new_list = []
                for det in dets:
                    si = det.source_index
                    if si is None:
                        new_list.append(det)
                        continue
                    if si not in old_to_new:
                        continue
                    det.source_index = old_to_new[si]
                    new_list.append(det)
                ctx.detections[q] = new_list

        return ctx


@REGISTRY.register("filter.top_k")
def _build_filter_top_k(services, **params) -> TopKFilterStage:
    if "k" not in params:
        raise ValueError("filter.top_k requires a 'k' parameter")
    return TopKFilterStage(
        k=int(params["k"]),
        score_weights=params.get("score_weights", None),
        score_agg=params.get("score_agg", "max"),
        normalize_sources=params.get("normalize_sources", True),
    )

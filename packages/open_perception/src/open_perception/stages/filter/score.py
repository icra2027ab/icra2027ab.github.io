"""Score threshold filter — drop masks whose score falls below a cutoff.

By default, the per-instance score is ``ctx.mask_scores`` (the SAM3
segmentation score). Pass ``score_weights`` to rank by a weighted combination
of sources from ``ctx.scores`` plus the special key ``"segmentation"``, e.g.
``{"dino": 0.5, "segmentation": 0.25, "clip": 0.25}``. Per-query 2D score
matrices are aggregated to 1D via ``score_agg`` (max over queries by
default), and each source is L2-normalised before summing when
``normalize_sources=True``.

Drops masks where the (combined) score is strictly less than ``threshold``.
Mirrors ``filter.size``: filters ``ctx.instances``, ``ctx.mask_scores``,
``ctx.bboxes``, ``ctx.crops``, and any ``ctx.detections`` populated upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...context import DetectionContext
from ..base import REGISTRY
from ._scores import ScoreAgg, combine_scores


@dataclass
class ScoreFilterStage:
    threshold: float
    score_weights: dict[str, float] | None = None
    score_agg: ScoreAgg = "max"
    normalize_sources: bool = True
    name: str = "filter.score"
    requires: ClassVar[tuple[str, ...]] = ("instances",)
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        if ctx.instances is None or ctx.instances.shape[0] == 0:
            return ctx

        seg_scores = ctx.mask_scores
        if seg_scores is None:
            seg_scores = np.ones((ctx.instances.shape[0],), dtype=np.float32)

        if self.score_weights:
            scores = combine_scores(
                ctx,
                seg_scores,
                self.score_weights,
                score_agg=self.score_agg,
                normalize_sources=self.normalize_sources,
                log_prefix="[filter.score]",
            )
        else:
            scores = np.asarray(seg_scores, dtype=np.float32)

        keep_mask = scores >= float(self.threshold)
        if bool(keep_mask.all()):
            return ctx

        keep_idx = np.where(keep_mask)[0]
        H, W = ctx.image_hw

        if keep_idx.size == 0:
            ctx.instances = np.zeros((0, H, W), dtype=bool)
            ctx.mask_scores = np.zeros((0,), dtype=seg_scores.dtype)
            ctx.bboxes = np.zeros((0, 4), dtype=np.int32)
            ctx.crops = []
        else:
            ctx.instances = ctx.instances[keep_idx]
            ctx.mask_scores = np.asarray(
                [seg_scores[i] for i in keep_idx], dtype=seg_scores.dtype
            )
            if ctx.bboxes is not None and ctx.bboxes.shape[0] > 0:
                ctx.bboxes = ctx.bboxes[keep_idx]
            if ctx.crops:
                ctx.crops = [ctx.crops[int(i)] for i in keep_idx]

        # Reindex ctx.scores so keys remain aligned with ctx.instances for any
        # downstream stage (filter.overlap with score_weights, assign, …) that
        # reads them.
        for key, mat in list(ctx.scores.items()):
            if mat.size == 0:
                continue
            if mat.ndim >= 1 and mat.shape[0] == keep_mask.shape[0]:
                ctx.scores[key] = mat[keep_idx]

        # Propagate to ctx.detections when populated upstream (e.g. sam3's
        # per-query emission path). Detections point at the pre-filter
        # instances via ``source_index``; drop those whose source was filtered
        # out and rewrite the remaining indices.
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


@REGISTRY.register("filter.score")
def _build_filter_score(services, **params) -> ScoreFilterStage:
    if "threshold" not in params:
        raise ValueError("filter.score requires a 'threshold' parameter")
    return ScoreFilterStage(
        threshold=float(params["threshold"]),
        score_weights=params.get("score_weights", None),
        score_agg=params.get("score_agg", "max"),
        normalize_sources=params.get("normalize_sources", True),
    )

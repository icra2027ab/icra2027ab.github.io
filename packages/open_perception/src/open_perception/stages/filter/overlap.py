"""Mask overlap filter — resolve overlapping masks in favor of higher scores.

Two modes:
    - ``"subtract"`` (default): lower-scoring masks have the intersection with
      every higher-scoring kept mask removed. Nothing is dropped unless the
      remainder falls below ``min_area``.
    - ``"remove"``: greedy NMS — a lower-scoring mask is dropped entirely if
      its IoU with any already-kept mask exceeds ``iou_threshold``.

By default, ranking uses ``ctx.mask_scores`` (the segmentation score). Pass
``score_weights`` to rank by a weighted combination of sources from
``ctx.scores`` plus the special key ``"segmentation"``, e.g.
``{"dino": 0.5, "segmentation": 0.25, "clip": 0.25}``. Per-query 2D score
matrices are aggregated to 1D via ``score_agg`` (max over queries by default),
and each source is L2-normalised before summing when
``normalize_sources=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from ...context import DetectionContext
from ...utils.geometry import mask_to_bbox
from ...utils.iou import nms, subtract_overlaps
from ..base import REGISTRY
from ._scores import ScoreAgg, combine_scores

OverlapMode = Literal["subtract", "remove"]


@dataclass
class OverlapFilterStage:
    iou_threshold: float
    mode: OverlapMode = "subtract"
    min_area: int = 1
    score_weights: dict[str, float] | None = None
    score_agg: ScoreAgg = "max"
    normalize_sources: bool = True
    name: str = "filter.overlap"
    requires: ClassVar[tuple[str, ...]] = ("instances",)
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        # Option-level override beats config-level default.
        enabled = ctx.options.filter_overlap
        if enabled is False:
            ctx.stats.n_after_overlap_filter.append(ctx.num_instances)
            return ctx

        if ctx.instances is None or ctx.instances.shape[0] == 0:
            ctx.stats.n_after_overlap_filter.append(0)
            return ctx

        seg_scores = ctx.mask_scores
        if seg_scores is None:
            seg_scores = np.ones((ctx.instances.shape[0],), dtype=float)

        if self.score_weights:
            scores = combine_scores(
                ctx,
                seg_scores,
                self.score_weights,
                score_agg=self.score_agg,
                normalize_sources=self.normalize_sources,
                log_prefix="[filter.overlap]",
            )
        else:
            scores = seg_scores

        if self.mode == "remove":
            iou_thr = ctx.options.overlap_iou_threshold
            keep_arr = nms(ctx.instances, scores, iou_thr)
            kept_order = [int(i) for i in keep_arr]
            new_masks = {i: ctx.instances[i] for i in kept_order}
        elif self.mode == "subtract":
            kept_order, new_masks = subtract_overlaps(
                ctx.instances, scores, min_area=self.min_area
            )
        else:
            raise ValueError(
                f"OverlapFilterStage: unknown mode {self.mode!r}; "
                "expected 'subtract' or 'remove'"
            )

        N = ctx.instances.shape[0]
        H, W = ctx.image_hw
        if kept_order:
            ctx.instances = np.stack([new_masks[i] for i in kept_order], axis=0)
            # Preserve segmentation scores on writeback (not the combined score
            # used purely for ranking) so downstream consumers see the same
            # semantics as before.
            ctx.mask_scores = np.asarray(
                [seg_scores[i] for i in kept_order], dtype=seg_scores.dtype
            )
            keep_idx = np.asarray(kept_order, dtype=np.int64)
            for key, mat in list(ctx.scores.items()):
                if mat.ndim >= 1 and mat.shape[0] == N:
                    ctx.scores[key] = mat[keep_idx]
        else:
            ctx.instances = np.zeros((0, H, W), dtype=bool)
            ctx.mask_scores = np.zeros((0,), dtype=seg_scores.dtype)
            for key, mat in list(ctx.scores.items()):
                if mat.ndim >= 1 and mat.shape[0] == N:
                    ctx.scores[key] = mat[:0]

        # Propagate to ctx.detections when it was already populated upstream
        # (e.g. sam3 per-query emission). Detections index into the pre-filter
        # instances via ``source_index``; drop or trim accordingly.
        if ctx.detections:
            old_to_new = {old: new for new, old in enumerate(kept_order)}
            for q, dets in list(ctx.detections.items()):
                new_list = []
                for det in dets:
                    si = det.source_index
                    if si is None:
                        new_list.append(det)
                        continue
                    if si not in old_to_new:
                        continue
                    if self.mode == "subtract":
                        det.mask = new_masks[si]
                        det.bbox = mask_to_bbox(det.mask)
                    det.source_index = old_to_new[si]
                    new_list.append(det)
                ctx.detections[q] = new_list

        ctx.stats.n_after_overlap_filter.append(int(ctx.instances.shape[0]))
        return ctx


@REGISTRY.register("filter.overlap")
def _build_filter_overlap(services, **params) -> OverlapFilterStage:
    return OverlapFilterStage(
        iou_threshold=params.get("iou_threshold", 0.7),
        mode=params.get("mode", "subtract"),
        min_area=params.get("min_area", 50),
        score_weights=params.get("score_weights", None),
        score_agg=params.get("score_agg", "max"),
        normalize_sources=params.get("normalize_sources", True),
    )

"""Area + top-K instance cap — filter out masks smaller than ``min_area``.

Port of ``Sam3ClipRerankPredictor._filter_and_crop``. Filters detections
whose mask area falls below a threshold and caps the total at
``max_instances``. Also trims ``ctx.detections`` so lightweight pipelines
(sam3 → filter.*) stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...context import DetectionContext
from ...utils.geometry import crop_rgb, mask_to_bbox
from ..base import REGISTRY


@dataclass
class SizeFilterStage:
    min_area: int
    max_instances: int
    bbox_pad: int
    zero_background: bool
    name: str = "filter.size"
    requires: ClassVar[tuple[str, ...]] = ("instances", "rgb")
    provides: ClassVar[tuple[str, ...]] = ("crops", "bboxes")

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        max_instances = ctx.options.max_instances or self.max_instances
        if ctx.instances is None or ctx.instances.shape[0] == 0:
            ctx.bboxes = np.zeros((0, 4), dtype=np.int32)
            ctx.crops = []
            ctx.stats.n_after_size_filter.append(0)
            return ctx

        keep: list[int] = []
        crops: list[np.ndarray] = []
        bboxes: list[np.ndarray] = []
        for i in range(ctx.instances.shape[0]):
            m = ctx.instances[i]
            if int(m.sum()) < self.min_area:
                continue
            bbox = mask_to_bbox(m, pad=self.bbox_pad)
            if bbox is None:
                continue
            crop = crop_rgb(
                ctx.rgb,
                bbox,
                mask=m if self.zero_background else None,
                zero_background=self.zero_background,
            )
            keep.append(i)
            crops.append(crop)
            bboxes.append(bbox)
            if len(keep) >= max_instances:
                break

        N = ctx.instances.shape[0]
        if not keep:
            ctx.instances = np.zeros((0, *ctx.image_hw), dtype=bool)
            ctx.mask_scores = np.zeros((0,), dtype=np.float32)
            ctx.bboxes = np.zeros((0, 4), dtype=np.int32)
            ctx.crops = []
            for key, mat in list(ctx.scores.items()):
                if mat.ndim >= 1 and mat.shape[0] == N:
                    ctx.scores[key] = mat[:0]
        else:
            idx = np.asarray(keep, dtype=np.int64)
            ctx.instances = ctx.instances[idx]
            ctx.mask_scores = (
                ctx.mask_scores[idx]
                if ctx.mask_scores is not None
                else np.ones((len(keep),), dtype=np.float32)
            )
            ctx.bboxes = np.stack(bboxes, axis=0)
            ctx.crops = crops
            for key, mat in list(ctx.scores.items()):
                if mat.ndim >= 1 and mat.shape[0] == N:
                    ctx.scores[key] = mat[idx]

        # Propagate to ctx.detections when it was populated upstream (e.g. the
        # sam3 1-1 per-query emission path). Detections point at the pre-filter
        # instances via ``source_index``; drop any whose source was filtered
        # out, and rewrite the remaining indices to match the new positions.
        if ctx.detections:
            old_to_new = {old: new for new, old in enumerate(keep)}
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

        ctx.stats.n_after_size_filter.append(int(ctx.instances.shape[0]))
        return ctx


@REGISTRY.register("filter.size")
def _build_filter_size(services, **params) -> SizeFilterStage:
    cfg = services.config
    return SizeFilterStage(
        min_area=params.get("min_area", cfg.default_min_area),
        max_instances=params.get("max_instances", cfg.default_max_instances),
        bbox_pad=params.get("bbox_pad", cfg.default_bbox_pad),
        zero_background=params.get("zero_background", cfg.default_clip_mask_bg),
    )

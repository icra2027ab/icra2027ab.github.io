"""Erode segmentation masks — strip unreliable boundary pixels.

Morphological erosion with a square ``kernel_size``x``kernel_size``
structuring element. Useful before 3D reconstruction where mask
boundaries tend to bleed into background depth and produce spurious
halos around the unprojected pointcloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...context import DetectionContext
from ...utils.geometry import erode_mask, mask_to_bbox
from ..base import REGISTRY


@dataclass
class ErodeFilterStage:
    kernel_size: int = 3
    name: str = "filter.erode"
    runs_on_track: bool = True
    requires: ClassVar[tuple[str, ...]] = ("instances",)
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        if self.kernel_size <= 1:
            return ctx

        if ctx.instances is not None and ctx.instances.shape[0] > 0:
            ctx.instances = np.stack(
                [erode_mask(m, self.kernel_size) for m in ctx.instances], axis=0
            )

        if ctx.detections:
            for dets in ctx.detections.values():
                for det in dets:
                    si = det.source_index
                    if (
                        si is not None
                        and ctx.instances is not None
                        and 0 <= si < ctx.instances.shape[0]
                    ):
                        det.mask = ctx.instances[si].copy()
                    else:
                        det.mask = erode_mask(det.mask, self.kernel_size)
                    bbox = mask_to_bbox(det.mask)
                    if bbox is not None:
                        det.bbox = bbox

        return ctx


@REGISTRY.register("filter.erode")
def _build_filter_erode(services, **params) -> ErodeFilterStage:
    return ErodeFilterStage(kernel_size=params.get("kernel_size", 3))

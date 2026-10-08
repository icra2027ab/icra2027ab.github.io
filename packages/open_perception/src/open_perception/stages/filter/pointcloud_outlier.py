"""Statistical outlier removal on per-detection 3D pointclouds.

Runs after ``reconstruct`` and prunes points whose mean distance to their
``k`` nearest neighbors lies more than ``std_ratio`` standard deviations
above the global mean — the standard SOR algorithm from Open3D / PCL.
Useful for stripping flying pixels and depth-edge halos that survive
mask erosion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...context import DetectionContext
from ...reconstruction.rgbd import remove_statistical_outliers
from ..base import REGISTRY


@dataclass
class PointcloudOutlierFilterStage:
    k: int = 16
    std_ratio: float = 2.0
    name: str = "filter.pointcloud_outlier"
    requires: ClassVar[tuple[str, ...]] = ("detections.pointcloud",)
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        for dets in ctx.detections.values():
            for det in dets:
                pc = det.pointcloud
                if pc is None or pc.shape[0] == 0:
                    continue
                filtered, _ = remove_statistical_outliers(
                    pc, k=self.k, std_ratio=self.std_ratio
                )
                det.pointcloud = filtered
        return ctx


@REGISTRY.register("filter.pointcloud_outlier")
def _build_filter_pointcloud_outlier(
    services, **params
) -> PointcloudOutlierFilterStage:
    return PointcloudOutlierFilterStage(
        k=params.get("k", 16),
        std_ratio=params.get("std_ratio", 2.0),
    )

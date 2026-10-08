"""Unproject per-detection masks into 3D using RGBD + intrinsics/extrinsics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..context import DetectionContext
from ..reconstruction.rgbd import mask_to_pointcloud
from .base import REGISTRY


@dataclass
class ReconstructStage:
    stride: int = 1
    flip_u: bool = False
    name: str = "reconstruct"
    runs_on_track: bool = True
    requires: ClassVar[tuple[str, ...]] = (
        "depth",
        "camera.intrinsics",
        "detections",
    )
    provides: ClassVar[tuple[str, ...]] = ("detections.pointcloud",)

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        if ctx.depth is None or ctx.camera is None:
            return ctx
        for lst in ctx.detections.values():
            for det in lst:
                det.pointcloud = mask_to_pointcloud(
                    ctx.depth,
                    det.mask,
                    ctx.camera.intrinsics,
                    ctx.camera.extrinsics,
                    stride=self.stride,
                    flip_u=self.flip_u,
                )
        return ctx


@REGISTRY.register("reconstruct")
def _build_reconstruct(services, **params) -> ReconstructStage:
    return ReconstructStage(
        stride=params.get("stride", 1),
        flip_u=params.get("flip_u", False),
    )

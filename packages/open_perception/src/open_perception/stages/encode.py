"""Attach per-detection CLIP/DINO embeddings to the final output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..context import DetectionContext
from .base import REGISTRY


@dataclass
class ExportEmbeddingsStage:
    name: str = "encode"
    requires: ClassVar[tuple[str, ...]] = ("detections", "crop_features")
    provides: ClassVar[tuple[str, ...]] = ("detections.embeddings",)

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        if not ctx.options.export_embeddings:
            return ctx
        for lst in ctx.detections.values():
            for det in lst:
                if det.source_index is None:
                    continue
                embeds = {}
                for key, feats in ctx.crop_features.items():
                    if 0 <= det.source_index < feats.shape[0]:
                        embeds[key] = feats[det.source_index].copy()
                det.embeddings = embeds if embeds else None
        return ctx


@REGISTRY.register("encode")
def _build_encode(services, **params) -> ExportEmbeddingsStage:
    return ExportEmbeddingsStage()

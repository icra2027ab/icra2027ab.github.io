"""CLIP-based disambiguator.

Crops are encoded with CLIP image features; queries use reference-bank image
embeddings when available, otherwise fall back to CLIP text embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
from loguru import logger

from ...context import DetectionContext
from ..base import REGISTRY
from ._shared import build_query_vectors, similarity_matrix


@dataclass
class ClipDisambiguator:
    clip_getter: object  # callable returning ClipEncoder
    reference_getter: object  # callable returning ReferenceBank | None (CLIP space)
    agg_mode: Literal["mean", "max"] = "max"
    fallback_to_text: bool = True
    name: str = "disambiguate.clip"
    score_key: str = "clip"
    _warned_no_refs: bool = field(default=False, init=False, repr=False)
    requires: ClassVar[tuple[str, ...]] = ("crops",)
    # Parents (`crop_features`, `scores`) are listed alongside the dotted
    # keys so a downstream stage (`assign`, `encode`) can ask for the
    # generic group rather than naming a specific source.
    provides: ClassVar[tuple[str, ...]] = (
        "crop_features.clip",
        "crop_features",
        "scores.clip",
        "scores",
    )

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        encoder = self.clip_getter()
        bank = self.reference_getter()
        ref_embeddings: dict[str, np.ndarray] = (
            bank.embeddings if bank is not None else {}
        )
        if not ref_embeddings and not self._warned_no_refs:
            fallback = (
                "falling back to CLIP text embeddings"
                if self.fallback_to_text
                else "scores will be zero (fallback_to_text=False)"
            )
            logger.warning(
                "[disambiguate.clip] no reference images registered for CLIP — "
                f"image-based disambiguation is inactive; {fallback}. "
                "Call `perc.register_reference(refs, space='clip')` to provide "
                "reference images."
            )
            self._warned_no_refs = True

        if ctx.num_instances == 0:
            ctx.scores[self.score_key] = np.zeros(
                (0, len(ctx.queries)), dtype=np.float32
            )
            return ctx

        # Reuse crop features if an earlier disambiguator already encoded them.
        if "clip" in ctx.crop_features:
            crop_feats = ctx.crop_features["clip"]
        else:
            crop_feats = encoder.encode_images(ctx.crops)
            ctx.crop_features["clip"] = crop_feats
            ctx.stats.increment_model_call("clip_image")

        missing = [q for q in ctx.queries if q not in ref_embeddings]
        if missing and not self.fallback_to_text:
            logger.info(
                f"[disambiguate.clip] {len(missing)} queries not in reference bank; "
                "using zero vectors (fallback_to_text=False)"
            )
        text_fallback: np.ndarray | None = None
        if missing and self.fallback_to_text:
            text_feats = encoder.encode_texts(ctx.queries)
            ctx.stats.increment_model_call("clip_text")
            text_fallback = text_feats

        per_query_refs, _sources = build_query_vectors(
            ctx.queries, ref_embeddings, text_fallback
        )
        sim = similarity_matrix(crop_feats, per_query_refs, self.agg_mode)
        ctx.scores[self.score_key] = sim
        return ctx


@REGISTRY.register("disambiguate.clip")
def _build_clip(services, **params) -> ClipDisambiguator:
    return ClipDisambiguator(
        clip_getter=services.get_clip,
        reference_getter=lambda: services.get_reference_bank("clip"),
        agg_mode=params.get("agg_mode", "max"),
        fallback_to_text=params.get("fallback_to_text", True),
        score_key=params.get("score_key", "clip"),
    )

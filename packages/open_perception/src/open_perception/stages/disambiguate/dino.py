"""DINO-based disambiguator.

Mirrors the CLIP disambiguator but encodes crops and references with DINO
features. When a query is missing from the DINO bank and ``fallback_to_text``
is on, we reach into the CLIP encoder for text and encode the same crops with
CLIP — the two score scales differ, so that path is labelled in `sources`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from loguru import logger

from ...context import DetectionContext
from ..base import REGISTRY
from ._shared import similarity_matrix


@dataclass
class DinoDisambiguator:
    dino_getter: object
    reference_getter: object
    clip_getter: object | None = None  # required only for text fallback
    agg_mode: Literal["mean", "max"] = "max"
    fallback_to_text: bool = False
    name: str = "disambiguate.dino"
    score_key: str = "dino"
    # When set, dump a per-call PNG panel showing the reference image(s) used
    # for each query alongside each candidate crop and the cosine similarity
    # score between them. One file per (frame_idx, view_idx) call.
    debug_dir: str | None = None
    _warned_no_refs: bool = field(default=False, init=False, repr=False)
    requires: ClassVar[tuple[str, ...]] = ("crops",)
    provides: ClassVar[tuple[str, ...]] = (
        "crop_features.dino",
        "crop_features",
        "scores.dino",
        "scores",
    )

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
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
                "[disambiguate.dino] no reference images registered for DINO — "
                f"image-based disambiguation is inactive; {fallback}. "
                "Call `perc.register_reference(refs, space='dino')` to provide "
                "reference images."
            )
            self._warned_no_refs = True

        if ctx.num_instances == 0:
            ctx.scores[self.score_key] = np.zeros(
                (0, len(ctx.queries)), dtype=np.float32
            )
            return ctx

        if "dino" in ctx.crop_features:
            crop_feats = ctx.crop_features["dino"]
        else:
            encoder = self.dino_getter()
            crop_feats = encoder.encode_images(ctx.crops)
            ctx.crop_features["dino"] = crop_feats
            ctx.stats.increment_model_call("dino_image")

        missing = [q for q in ctx.queries if q not in ref_embeddings]
        text_fallback: np.ndarray | None = None
        if missing and self.fallback_to_text:
            if self.clip_getter is None:
                logger.warning(
                    "[disambiguate.dino] fallback_to_text=True but no CLIP encoder supplied; "
                    "using zero vectors for missing queries"
                )
            else:
                clip = self.clip_getter()
                # In CLIP mode, the text fallback must be scored against CLIP crop
                # features, not DINO. We attach a `clip` source matrix in addition
                # to the DINO matrix so the assigner can weight them separately.
                if "clip" not in ctx.crop_features:
                    ctx.crop_features["clip"] = clip.encode_images(ctx.crops)
                    ctx.stats.increment_model_call("clip_image")
                text_fallback = clip.encode_texts(ctx.queries)
                ctx.stats.increment_model_call("clip_text")

        per_query_refs_dino: list[np.ndarray] = []
        per_query_refs_clip: list[np.ndarray] = []
        has_clip_queries = False
        dim_dino = crop_feats.shape[-1]
        dim_clip = (
            ctx.crop_features.get("clip", np.zeros((0, 1))).shape[-1]
            if "clip" in ctx.crop_features
            else 1
        )
        for i, q in enumerate(ctx.queries):
            if q in ref_embeddings:
                per_query_refs_dino.append(ref_embeddings[q])
                per_query_refs_clip.append(np.zeros((1, dim_clip), dtype=np.float32))
            elif text_fallback is not None:
                per_query_refs_dino.append(np.zeros((1, dim_dino), dtype=np.float32))
                per_query_refs_clip.append(text_fallback[i : i + 1])
                has_clip_queries = True
            else:
                per_query_refs_dino.append(np.zeros((1, dim_dino), dtype=np.float32))
                per_query_refs_clip.append(np.zeros((1, dim_clip), dtype=np.float32))

        ctx.scores[self.score_key] = similarity_matrix(
            crop_feats, per_query_refs_dino, self.agg_mode
        )
        if has_clip_queries and "clip" in ctx.crop_features:
            ctx.scores["clip_text_fallback"] = similarity_matrix(
                ctx.crop_features["clip"], per_query_refs_clip, self.agg_mode
            )

        if self.debug_dir is not None:
            self._dump_panel(ctx, bank, ref_embeddings, text_fallback is not None)
        return ctx

    def _dump_panel(
        self,
        ctx: DetectionContext,
        bank: object | None,
        ref_embeddings: dict[str, np.ndarray],
        text_fallback_used: bool,
    ) -> None:
        """Write a (crops × refs) similarity-score panel for this call."""
        from ...visualization import render_disambiguation_panel

        ref_images: dict[str, list[np.ndarray]] = {}
        if bank is not None:
            bank_images = getattr(bank, "images", {}) or {}
            for q in ctx.queries:
                ref_images[q] = list(bank_images.get(q, []))
        sources = [
            "image"
            if q in ref_embeddings
            else ("text" if text_fallback_used else "missing")
            for q in ctx.queries
        ]
        # Prefer DINO scores; for queries that fell back to text, swap in the
        # CLIP-text-fallback score so the panel reflects what actually drove
        # the assigner downstream.
        scores = ctx.scores[self.score_key].copy()
        if "clip_text_fallback" in ctx.scores:
            cfb = ctx.scores["clip_text_fallback"]
            for j, src in enumerate(sources):
                if src == "text" and j < cfb.shape[1]:
                    scores[:, j] = cfb[:, j]
        out_dir = Path(self.debug_dir)  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir / f"{self.name}_frame{ctx.frame_idx:04d}_view{ctx.view_idx}.png"
        )
        try:
            render_disambiguation_panel(
                crops=list(ctx.crops),
                queries=list(ctx.queries),
                references=ref_images,
                scores=scores,
                path=out_path,
                sources=sources,
                title=(
                    f"{self.name} — frame {ctx.frame_idx} view {ctx.view_idx} "
                    f"(agg={self.agg_mode})"
                ),
            )
        except Exception as exc:
            logger.warning(f"[disambiguate.dino] failed to write debug panel: {exc}")


@REGISTRY.register("disambiguate.dino")
def _build_dino(services, **params) -> DinoDisambiguator:
    return DinoDisambiguator(
        dino_getter=services.get_dino,
        reference_getter=lambda: services.get_reference_bank("dino"),
        clip_getter=services.get_clip,
        agg_mode=params.get("agg_mode", "max"),
        fallback_to_text=params.get("fallback_to_text", False),
        score_key=params.get("score_key", "dino"),
        debug_dir=params.get("debug_dir"),
    )

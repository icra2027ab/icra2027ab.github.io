"""VLM-based candidate reranker (post-segmentation).

Ships a user-extensible `VlmDisambiguator` that takes the SAM3 crops and asks
the VLM for a per-(crop, query) score. A ``vlm_client`` with a
`score_candidates(crops, queries, reference_images) -> (M, P) float array`
contract is expected — the OpenAI implementation sits outside this file so
the optional `openai` dependency stays optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
from loguru import logger

from ...context import DetectionContext
from ..base import REGISTRY


class VlmRerankClient(Protocol):
    def score_candidates(
        self,
        crops: list[np.ndarray],
        queries: list[str],
        reference_images: dict[str, list[np.ndarray]] | None = None,
    ) -> np.ndarray: ...


@dataclass
class VlmDisambiguator:
    client_getter: object  # callable returning VlmRerankClient
    reference_getter: object | None = None
    name: str = "disambiguate.vlm"
    score_key: str = "vlm"
    requires: ClassVar[tuple[str, ...]] = ("crops",)
    provides: ClassVar[tuple[str, ...]] = ("scores.vlm", "scores")

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        if ctx.num_instances == 0:
            ctx.scores[self.score_key] = np.zeros(
                (0, len(ctx.queries)), dtype=np.float32
            )
            return ctx
        client = self.client_getter()
        if client is None:
            logger.warning("[disambiguate.vlm] no VLM client configured; skipping")
            return ctx
        ref_images: dict[str, list[np.ndarray]] | None = None
        if self.reference_getter is not None:
            bank = self.reference_getter()
            if bank is not None:
                ref_images = bank.images or None
        sim = client.score_candidates(ctx.crops, ctx.queries, ref_images)
        ctx.scores[self.score_key] = np.asarray(sim, dtype=np.float32)
        ctx.stats.increment_model_call("vlm_rerank")
        return ctx


@REGISTRY.register("disambiguate.vlm")
def _build_vlm_rerank(services, **params: Any) -> VlmDisambiguator:
    return VlmDisambiguator(
        client_getter=services.get_vlm_client,
        reference_getter=lambda: services.get_reference_bank("clip"),
        score_key=params.get("score_key", "vlm"),
    )

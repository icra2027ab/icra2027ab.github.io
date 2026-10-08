"""Pre-segmentation VLM guidance stage.

Given the full RGB and the user queries, the VLM can (a) prune which queries
actually exist in the scene, and/or (b) propose bounding-box hints that feed
into SAM3's box prompting. Both outputs are optional — the client decides
what to return, and this stage wires them into the context.

The concrete VLM client is pluggable; this stage only knows the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
from loguru import logger

from ..context import DetectionContext
from .base import REGISTRY


class VlmPreClient(Protocol):
    def pre_guide(
        self,
        rgb: np.ndarray,
        queries: list[str],
    ) -> "VlmPreResult": ...


@dataclass
class VlmPreResult:
    present_queries: list[str] | None = None  # None means "no pruning"
    boxes: dict[str, np.ndarray] | None = None  # query -> (K, 4) [y0, x0, y1, x1]


@dataclass
class VlmPreStage:
    client_getter: object
    name: str = "vlm_pre"
    requires: ClassVar[tuple[str, ...]] = ("rgb",)
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        if not ctx.options.vlm_pre_enabled:
            return ctx
        client = self.client_getter()
        if client is None:
            logger.warning("[vlm_pre] no VLM client configured; skipping")
            return ctx
        result = client.pre_guide(ctx.rgb, ctx.queries)
        ctx.stats.increment_model_call("vlm_pre")
        if result.present_queries is not None:
            kept = [q for q in ctx.queries if q in set(result.present_queries)]
            if kept:
                ctx.queries = kept
            else:
                logger.warning(
                    "[vlm_pre] pruning would remove all queries; keeping original set"
                )
        # Box hints are attached to the context; the segment stage may consult
        # them in a future extension. For now they live on stats so downstream
        # tooling can see what the VLM proposed. The intended future sink is
        # ``ctx.box_prompts`` (consumed by sam2 bbox mode, as detect.gemini
        # does) — wiring it here needs VlmPreResult's yxyx/units convention
        # pinned down first.
        if result.boxes is not None:
            ctx.stats.stages_run.append(
                f"vlm_pre:boxes[{','.join(sorted(result.boxes.keys()))}]"
            )
        return ctx


@REGISTRY.register("vlm_pre")
def _build_vlm_pre(services, **params: Any) -> VlmPreStage:
    return VlmPreStage(client_getter=services.get_vlm_client)

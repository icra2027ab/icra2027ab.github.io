"""Gemini box-proposal stage (tiptop-style detection).

Given the full RGB and (optionally) a natural-language task, a VLM proposes
the objects in the scene and returns one or more bounding boxes per object.
The boxes are written to ``ctx.box_prompts`` (xyxy pixel) so a downstream
``sam2`` stage in ``bbox`` mode can segment each one.

Two modes:

- ``"open_vocab"`` : the VLM enumerates objects freely (inventing unique
  labels). The proposed labels become ``ctx.queries`` so the bbox segmenter
  has something to iterate. This mirrors tiptop's ``detect_and_segment``.
- ``"grounded"``   : the VLM is asked to locate only the queries the caller
  already passed (``ctx.queries`` are forwarded as a constraint).

The concrete proposer is pluggable — this stage only knows the
``BoxDetectClient`` protocol. The default implementation is
``open_perception.models.gemini.GeminiBoxDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
from loguru import logger

from ..context import DetectionContext
from .base import REGISTRY


class BoxDetectClient(Protocol):
    def detect(
        self,
        rgb: np.ndarray,
        queries: list[str] | None,
        task: str | None,
        proposal: str = "box",
    ) -> "BoxDetectResult":
        """Propose objects in ``rgb``.

        ``queries`` is ``None`` (or empty) for open-vocabulary enumeration,
        or the list of names to ground when the caller has a fixed query set.
        ``task`` is an optional natural-language instruction that biases which
        objects matter. ``proposal`` selects the geometry the detector returns:
        ``"box"`` (2D boxes) or ``"point"`` (one point per instance).
        """
        ...


@dataclass
class BoxDetectResult:
    # label -> (N, 4) xyxy *pixel* boxes
    boxes: dict[str, np.ndarray] = field(default_factory=dict)
    # label -> (N, 2) xy *pixel* points (one row per proposed instance)
    points: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class GeminiDetectStage:
    detector_getter: object  # callable returning BoxDetectClient | None
    stage: Literal["open_vocab", "grounded"] = "open_vocab"
    # Proposal geometry: "box" -> ctx.box_prompts (sam2/sam3 bbox mode),
    # "point" -> ctx.point_prompts (sam2/sam3 point mode).
    proposal: Literal["box", "point"] = "box"
    task: str | None = None
    # open_vocab only: push the proposed labels into ctx.queries so the
    # downstream bbox segmenter iterates them.
    set_queries: bool = True
    name: str = "detect.gemini"
    requires: ClassVar[tuple[str, ...]] = ("rgb",)
    provides: ClassVar[tuple[str, ...]] = ("box_prompts", "point_prompts")

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        detector = self.detector_getter()
        if detector is None:
            logger.warning("[detect.gemini] no box detector configured; skipping")
            return ctx
        queries = None if self.stage == "open_vocab" else list(ctx.queries)
        result = detector.detect(ctx.rgb, queries, self.task, proposal=self.proposal)
        ctx.stats.increment_model_call("detect.gemini")
        prompts = result.points if self.proposal == "point" else result.boxes
        if self.stage == "grounded":
            # The VLM is told to echo the requested names, but detectors
            # sanitize labels (spaces -> underscores), so a query like
            # "alphabet soup" would come back keyed "alphabet_soup" and the
            # downstream segmenter's prompt lookup (keyed by ctx.queries
            # verbatim) would miss it. Re-key to the exact requested queries.
            lookup = {_label_key(q): q for q in ctx.queries}
            prompts = {lookup.get(_label_key(k), k): v for k, v in prompts.items()}
        if self.proposal == "point":
            ctx.point_prompts = prompts
        else:
            ctx.box_prompts = prompts
        if self.stage == "open_vocab" and self.set_queries:
            labels = list(prompts.keys())
            if labels:
                ctx.queries = labels
            else:
                logger.warning("[detect.gemini] proposed no objects; queries unchanged")
        return ctx


def _label_key(name: str) -> str:
    """Case/space/underscore-insensitive label key for grounded re-keying."""
    return name.replace(" ", "_").lower()


@REGISTRY.register("detect.gemini")
def _build_detect_gemini(services, **params: Any) -> GeminiDetectStage:
    return GeminiDetectStage(
        detector_getter=services.get_box_detector,
        stage=params.get("stage", "open_vocab"),
        proposal=params.get("proposal", "box"),
        task=params.get("task"),
        set_queries=params.get("set_queries", True),
    )

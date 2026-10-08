"""SAM3 segmentation stage.

Three modes:

- ``"text"`` (default): runs ``segment_per_prompt`` so each returned mask is
  attributed to the specific text prompt that produced it. ``ctx.detections``
  is keyed by the *resolved prompt string* — expanding ``<object_names>``
  against the caller's queries (see ``Sam3Wrapper.resolve_prompts``). Literal
  prompts like ``"object"`` appear in the output as their own key alongside
  the expanded queries, which is useful for a union-style pipeline that skips
  the disambiguate/assign stages.
- ``"bbox"`` / ``"point"``: geometric prompting via SAM3's tracker head
  (``Sam3Wrapper.tracker`` — same API as SAM2). Per-query prompts come from
  the static ``prompts`` dict, falling back to ``ctx.box_prompts`` /
  ``ctx.point_prompts`` proposed upstream (e.g. by ``detect.gemini``) —
  identical semantics to the sam2 stage's bbox/point modes.

``ctx.instances`` is populated with the flat candidate pool so downstream
disambiguate/assign can still re-attribute masks to ``ctx.queries`` when
the full pipeline runs — in that case ``ctx.detections`` is overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from ..context import DetectionContext
from ..types import Detection
from ..utils.geometry import mask_to_bbox
from .base import REGISTRY
from .sam2 import populate_geometric_ctx, segment_bbox_prompts, segment_point_prompts


@dataclass
class Sam3SegmentStage:
    sam3_getter: object  # callable returning Sam3Wrapper
    prompt_templates: list[str]
    score_threshold: float
    mask_threshold: float
    mode: Literal["text", "bbox", "point"] = "text"
    # ``bbox`` / ``point`` mode: per-query prompts, same format as the sam2
    # stage. Falls back to ctx.box_prompts / ctx.point_prompts when empty.
    prompts: dict[str, Any] = field(default_factory=dict)
    multimask_output: bool = False
    name: str = "sam3"
    requires: ClassVar[tuple[str, ...]] = ("rgb",)
    provides: ClassVar[tuple[str, ...]] = ("instances", "mask_scores", "detections")

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        sam3 = self.sam3_getter()
        H, W = ctx.image_hw

        if self.mode in ("bbox", "point"):
            tracker = sam3.tracker
            if self.mode == "bbox":
                prompts = self.prompts if self.prompts else (ctx.box_prompts or {})
                per_prompt = segment_bbox_prompts(
                    tracker, ctx, prompts, self.multimask_output
                )
            else:
                prompts = self.prompts if self.prompts else (ctx.point_prompts or {})
                per_prompt = segment_point_prompts(
                    tracker, ctx, prompts, self.multimask_output
                )
            populate_geometric_ctx(ctx, per_prompt, H, W)
            ctx.stats.increment_model_call("sam3")
            return ctx

        text_prompts = sam3.resolve_prompts(self.prompt_templates, ctx.queries)
        if not text_prompts:
            ctx.instances = np.zeros((0, H, W), dtype=bool)
            ctx.mask_scores = np.zeros((0,), dtype=np.float32)
            ctx.detections = {}
            ctx.stats.n_candidates_per_view.append(0)
            ctx.stats.increment_model_call("sam3")
            return ctx

        per_prompt = sam3.segment_per_prompt(
            ctx.rgb,
            text_prompts=text_prompts,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
        )

        out: dict[str, list[Detection]] = {p: [] for p in text_prompts}
        flat_masks: list[np.ndarray] = []
        flat_scores: list[float] = []
        for prompt, (masks, scores) in zip(text_prompts, per_prompt, strict=True):
            for i in range(masks.shape[0]):
                m = masks[i].astype(bool)
                s = float(scores[i])
                out[prompt].append(
                    Detection(
                        query=prompt,
                        mask=m,
                        score=s,
                        view_idx=ctx.view_idx,
                        bbox=mask_to_bbox(m),
                        source_index=len(flat_masks),
                    )
                )
                flat_masks.append(m)
                flat_scores.append(s)

        if flat_masks:
            ctx.instances = np.stack(flat_masks).astype(bool)
            ctx.mask_scores = np.asarray(flat_scores, dtype=np.float32)
        else:
            ctx.instances = np.zeros((0, H, W), dtype=bool)
            ctx.mask_scores = np.zeros((0,), dtype=np.float32)
        ctx.detections = out
        ctx.stats.n_candidates_per_view.append(int(ctx.instances.shape[0]))
        ctx.stats.increment_model_call("sam3")
        return ctx


@REGISTRY.register("sam3")
def _build_sam3(services, **params) -> Sam3SegmentStage:
    cfg = services.config
    mode = params.get("mode", "text")
    # ``prompts`` is overloaded for backwards compatibility: in text mode it's
    # the template list (e.g. ``["<object_names>"]``); in bbox/point mode it's
    # the per-query geometric prompt dict (same as the sam2 stage).
    prompts_param = params.get("prompts")
    return Sam3SegmentStage(
        sam3_getter=services.get_sam3,
        prompt_templates=(
            prompts_param
            if mode == "text" and prompts_param is not None
            else cfg.sam3_prompts
        ),
        score_threshold=params.get("score_threshold", cfg.sam3_score_threshold),
        mask_threshold=params.get("mask_threshold", cfg.sam3_mask_threshold),
        mode=mode,
        prompts=(prompts_param if mode != "text" else None) or {},
        multimask_output=params.get("multimask_output", False),
    )

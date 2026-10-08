"""SAM2 segmentation stage.

Drop-in alternative to ``Sam3SegmentStage`` for pipelines that want
SAM2's class-agnostic mask proposals (or point/bbox-prompted masks).
SAM2 has no text-conditioning, so the stage runs in one of three modes:

- ``"auto"``  : run SAM2's automatic mask generator (grid of points →
  NMS), bucketing every kept mask under a single literal key
  (``auto_prompt``, default ``"object"``). Pair with a downstream
  ``disambiguate.*`` + ``assign`` to attribute masks to ``ctx.queries``.
- ``"bbox"``  : caller supplies ``prompts={query: [(x1, y1, x2, y2), ...]}``;
  the stage runs SAM2 with those boxes and emits one detection per box
  under its query. When ``prompts`` is empty it falls back to boxes proposed
  upstream in ``ctx.box_prompts`` (e.g. by the ``detect.gemini`` stage).
- ``"point"`` : caller supplies ``prompts={query: [{"points": [(x, y), ...],
  "labels": [1, 0, ...]}]}``; SAM2 returns one mask per ``{points,
  labels}`` group under its query.

In ``bbox`` / ``point`` mode the stage skips queries that have no entry
in ``prompts`` rather than failing — useful for partially-prompted runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from ..context import DetectionContext
from ..types import Detection
from ..utils.geometry import mask_to_bbox
from .base import REGISTRY


@dataclass
class Sam2SegmentStage:
    sam2_getter: object  # callable returning Sam2Wrapper
    mode: Literal["auto", "bbox", "point"] = "auto"
    # ``auto`` mode: name used as the dict key for every proposed mask.
    # Pick a literal like ``"object"`` so a downstream disambiguator can
    # rebucket the candidates against ``ctx.queries``.
    auto_prompt: str = "object"
    # ``auto`` mode params (forwarded to the HF mask-generation pipeline).
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92
    crops_nms_thresh: float = 0.7
    # ``bbox`` / ``point`` mode: per-query prompts. Resolved against
    # ``ctx.queries`` at call time so callers can keep the stage instance
    # around and just swap the prompts dict between calls.
    prompts: dict[str, Any] = field(default_factory=dict)
    multimask_output: bool = False
    name: str = "sam2"
    requires: ClassVar[tuple[str, ...]] = ("rgb",)
    provides: ClassVar[tuple[str, ...]] = ("instances", "mask_scores", "detections")

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        sam2 = self.sam2_getter()
        H, W = ctx.image_hw

        if self.mode == "auto":
            masks, scores = sam2.segment_auto(
                ctx.rgb,
                points_per_side=self.points_per_side,
                points_per_batch=self.points_per_batch,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crops_nms_thresh=self.crops_nms_thresh,
            )
            self._populate_ctx(
                ctx,
                {self.auto_prompt: (masks, scores)},
                H,
                W,
            )
        elif self.mode == "bbox":
            self._populate_ctx(ctx, self._segment_bbox(sam2, ctx), H, W)
        elif self.mode == "point":
            self._populate_ctx(ctx, self._segment_point(sam2, ctx), H, W)
        else:
            raise ValueError(
                f"unknown sam2 mode {self.mode!r}; expected 'auto', 'bbox', or 'point'"
            )

        ctx.stats.increment_model_call("sam2")
        return ctx

    # ------------------------------------------------------------------

    def _segment_bbox(
        self, sam2: Any, ctx: DetectionContext
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        # Static per-call prompts win; otherwise fall back to boxes proposed
        # upstream (e.g. by detect.gemini), already in xyxy pixel coords.
        prompts = self.prompts if self.prompts else (ctx.box_prompts or {})
        return segment_bbox_prompts(sam2, ctx, prompts, self.multimask_output)

    def _segment_point(
        self, sam2: Any, ctx: DetectionContext
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        # Static per-call prompts win; otherwise fall back to points proposed
        # upstream (e.g. by detect.gemini proposal="point"), in xy pixel coords.
        prompts = self.prompts if self.prompts else (ctx.point_prompts or {})
        return segment_point_prompts(sam2, ctx, prompts, self.multimask_output)

    @staticmethod
    def _populate_ctx(
        ctx: DetectionContext,
        per_prompt: dict[str, tuple[np.ndarray, np.ndarray]],
        H: int,
        W: int,
    ) -> None:
        populate_geometric_ctx(ctx, per_prompt, H, W)


# ---------------------------------------------------------------------------
# Shared geometric-prompt helpers — used by both the sam2 stage and the sam3
# stage's bbox/point modes (SAM3's tracker head exposes the same
# ``segment_with_boxes`` / ``segment_with_points`` API).
# ---------------------------------------------------------------------------


def segment_bbox_prompts(
    segmenter: Any,
    ctx: DetectionContext,
    prompts: dict[str, Any],
    multimask_output: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One ``segment_with_boxes`` call per query with a box-prompt entry."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for query in ctx.queries:
        spec = prompts.get(query)
        # ``spec`` may be a list (static prompts) or an ndarray
        # (ctx.box_prompts) — avoid truthiness on arrays.
        if spec is None:
            continue
        boxes = np.asarray(spec, dtype=np.float32).reshape(-1, 4)
        if boxes.shape[0] == 0:
            continue
        masks, scores = segmenter.segment_with_boxes(
            ctx.rgb, boxes, multimask_output=multimask_output
        )
        out[query] = (masks, scores)
    return out


def segment_point_prompts(
    segmenter: Any,
    ctx: DetectionContext,
    prompts: dict[str, Any],
    multimask_output: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One ``segment_with_points`` call per query with a point-prompt entry."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for query in ctx.queries:
        spec = prompts.get(query)
        # ``spec`` may be a dict/list (static prompts) or an (N, 2) ndarray
        # (ctx.point_prompts) — avoid truthiness on arrays.
        if spec is None or (not isinstance(spec, np.ndarray) and not spec):
            continue
        points, labels = _normalize_point_spec(spec)
        if not points:
            continue
        masks, scores = segmenter.segment_with_points(
            ctx.rgb,
            points,
            labels,
            multimask_output=multimask_output,
        )
        out[query] = (masks, scores)
    return out


def populate_geometric_ctx(
    ctx: DetectionContext,
    per_prompt: dict[str, tuple[np.ndarray, np.ndarray]],
    H: int,
    W: int,
) -> None:
    """Write per-prompt (masks, scores) into ctx instances/detections."""
    out: dict[str, list[Detection]] = {p: [] for p in per_prompt}
    flat_masks: list[np.ndarray] = []
    flat_scores: list[float] = []
    for prompt, (masks, scores) in per_prompt.items():
        for i in range(masks.shape[0]):
            m = masks[i].astype(bool)
            if not m.any():
                continue
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


def _normalize_point_spec(
    spec: Any,
) -> tuple[list[list[tuple[float, float]]], list[list[int]]]:
    """Coerce a flexible per-query point spec into Sam2Wrapper format.

    Accepts either:
      - ``{"points": [...], "labels": [...]}`` for a single object,
      - ``[{"points": [...], "labels": [...]}, ...]`` for multiple objects, or
      - an ``(N, 2)`` xy array (e.g. ``ctx.point_prompts``) — one
        single-foreground-point object per row.

    ``labels`` defaults to all-foreground when omitted.
    """
    if isinstance(spec, np.ndarray):
        spec = [{"points": [(float(x), float(y))]} for x, y in spec.reshape(-1, 2)]
    if isinstance(spec, dict):
        spec = [spec]
    points: list[list[tuple[float, float]]] = []
    labels: list[list[int]] = []
    for entry in spec:
        pts = [(float(x), float(y)) for x, y in entry["points"]]
        if "labels" in entry and entry["labels"] is not None:
            lbls = [int(v) for v in entry["labels"]]
            if len(lbls) != len(pts):
                raise ValueError(
                    f"point spec has {len(pts)} points but {len(lbls)} labels"
                )
        else:
            lbls = [1] * len(pts)
        points.append(pts)
        labels.append(lbls)
    return points, labels


@REGISTRY.register("sam2")
def _build_sam2(services, **params) -> Sam2SegmentStage:
    cfg = services.config
    return Sam2SegmentStage(
        sam2_getter=services.get_sam2,
        mode=params.get("mode", "auto"),
        auto_prompt=params.get("auto_prompt", "object"),
        points_per_side=params.get("points_per_side", cfg.sam2_points_per_side),
        points_per_batch=params.get("points_per_batch", cfg.sam2_points_per_batch),
        pred_iou_thresh=params.get("pred_iou_thresh", cfg.sam2_pred_iou_thresh),
        stability_score_thresh=params.get(
            "stability_score_thresh", cfg.sam2_stability_score_thresh
        ),
        crops_nms_thresh=params.get("crops_nms_thresh", cfg.sam2_crops_nms_thresh),
        prompts=params.get("prompts", {}),
        multimask_output=params.get("multimask_output", False),
    )

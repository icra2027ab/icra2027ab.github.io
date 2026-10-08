"""Assign SAM3 candidates to queries — threshold-multi-match or top-1 Hungarian.

Combines per-source score matrices (from CLIP / DINO / VLM disambiguators)
into a single similarity matrix, then assigns based on ``top_k``:

1. ``top_k=1`` — exact bipartite matching via Hungarian (legacy 1-mask-per-query),
   mirrors `Sam3ClipRerankPredictor.predict`.
2. ``top_k=-1`` (default) — return all matches above threshold. By default
   (``one_per_instance=True``) each instance is claimed by at most one
   query — its argmax over the score row, provided that score clears
   ``match_threshold``. Set ``one_per_instance=False`` to revert to
   per-query matches, where the same mask can be emitted under multiple
   queries if it scores high for several.
3. ``top_k>1`` — threshold matching as in (2), but additionally capped to
   the top ``top_k`` matches per query (sorted by score).

Each source matrix is L2-normalised per-query before summing so CLIP/DINO/VLM
scale differences don't let one source dominate assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from loguru import logger

from ..context import DetectionContext
from ..types import Detection
from ..utils.geometry import mask_to_bbox
from .base import REGISTRY


def _combine_scores(
    score_matrices: dict[str, np.ndarray],
    weights: dict[str, float],
    normalize: bool = True,
) -> np.ndarray:
    active = [(k, v) for k, v in score_matrices.items() if v.size > 0]
    if not active:
        return np.zeros((0, 0), dtype=np.float32)
    M, P = active[0][1].shape
    out = np.zeros((M, P), dtype=np.float32)
    for key, mat in active:
        if mat.shape != (M, P):
            logger.warning(
                f"[assign] score source {key!r} has shape {mat.shape} != {(M, P)}; skipping"
            )
            continue
        w = weights.get(key, 1.0)
        if normalize:
            norm = np.linalg.norm(mat, axis=0, keepdims=True)
            norm = np.where(norm > 1e-8, norm, 1.0)
            out += w * (mat / norm)
        else:
            out += w * mat
    return out


@dataclass
class AssignStage:
    top_k: int = -1
    one_per_instance: bool = True
    weights: dict[str, float] = field(default_factory=dict)
    normalize_sources: bool = True
    name: str = "assign"
    requires: ClassVar[tuple[str, ...]] = ("instances", "scores")
    provides: ClassVar[tuple[str, ...]] = ("detections",)

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        queries = ctx.queries
        H, W = ctx.image_hw
        out: dict[str, list[Detection]] = {q: [] for q in queries}

        if ctx.instances is None or ctx.instances.shape[0] == 0 or not queries:
            ctx.detections = out
            return ctx

        sim = _combine_scores(
            ctx.scores, self.weights, normalize=self.normalize_sources
        )
        if sim.size == 0:
            ctx.detections = out
            return ctx

        if self.top_k == 1:
            from scipy.optimize import linear_sum_assignment

            row_idx, col_idx = linear_sum_assignment(-sim)
            for r, c in zip(row_idx, col_idx, strict=True):
                q = queries[c]
                mask = ctx.instances[r]
                bbox = mask_to_bbox(mask)
                out[q].append(
                    Detection(
                        query=q,
                        mask=mask.copy(),
                        score=float(sim[r, c]),
                        view_idx=ctx.view_idx,
                        bbox=bbox,
                        source_index=int(r),
                    )
                )
        else:
            threshold = ctx.options.match_threshold
            top_k = None if self.top_k == -1 else self.top_k
            if self.one_per_instance:
                # For each instance, pick its single best query (argmax over
                # the score row). Drop instances whose best score doesn't
                # clear the threshold. Then per-query, sort by score and
                # apply the top_k cap as usual.
                per_query: dict[str, list[tuple[int, float]]] = {q: [] for q in queries}
                best_cols = np.argmax(sim, axis=1)
                best_scores = sim[np.arange(sim.shape[0]), best_cols]
                for r in range(sim.shape[0]):
                    s = float(best_scores[r])
                    if s < threshold:
                        continue
                    per_query[queries[int(best_cols[r])]].append((int(r), s))
                for q, items in per_query.items():
                    items.sort(key=lambda x: -x[1])
                    if top_k is not None:
                        items = items[:top_k]
                    for r, s in items:
                        mask = ctx.instances[r]
                        bbox = mask_to_bbox(mask)
                        out[q].append(
                            Detection(
                                query=q,
                                mask=mask.copy(),
                                score=s,
                                view_idx=ctx.view_idx,
                                bbox=bbox,
                                source_index=r,
                            )
                        )
            else:
                for c, q in enumerate(queries):
                    col = sim[:, c]
                    above = np.where(col >= threshold)[0]
                    if above.size == 0:
                        continue
                    order = above[np.argsort(-col[above])]
                    if top_k is not None:
                        order = order[:top_k]
                    for r in order:
                        mask = ctx.instances[int(r)]
                        bbox = mask_to_bbox(mask)
                        out[q].append(
                            Detection(
                                query=q,
                                mask=mask.copy(),
                                score=float(col[int(r)]),
                                view_idx=ctx.view_idx,
                                bbox=bbox,
                                source_index=int(r),
                            )
                        )

        ctx.detections = out
        return ctx


@REGISTRY.register("assign")
def _build_assign(services, **params) -> AssignStage:
    return AssignStage(
        top_k=params.get("top_k", -1),
        one_per_instance=params.get("one_per_instance", True),
        weights=params.get("weights", {}),
        normalize_sources=params.get("normalize_sources", True),
    )

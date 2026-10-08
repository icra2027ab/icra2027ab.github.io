"""Shared helpers for CLIP and DINO disambiguators.

Reproduces the per-reference cosine similarity + aggregation behaviour from
`Sam3ClipRerankPredictor._clip_similarity` — operating on numpy, not torch,
because encoders in this package already return numpy arrays.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def aggregate(sim_per_ref: np.ndarray, mode: Literal["mean", "max"]) -> np.ndarray:
    """Collapse ``(M, K)`` per-reference similarities to ``(M,)`` per prompt."""
    if sim_per_ref.size == 0:
        return np.zeros((sim_per_ref.shape[0],), dtype=np.float32)
    if mode == "max":
        return sim_per_ref.max(axis=1)
    return sim_per_ref.mean(axis=1)


def build_query_vectors(
    queries: list[str],
    reference_embeddings: dict[str, np.ndarray],
    text_fallback: np.ndarray | None,
) -> tuple[list[np.ndarray], list[str]]:
    """Return per-query (K_i, D) reference vectors + source tags.

    ``text_fallback[i]`` (shape (1, D)) is used when query ``i`` isn't in the
    reference bank. If ``text_fallback`` is None for a missing query, a zero
    vector is returned and the source is tagged ``missing``.
    """
    per_query: list[np.ndarray] = []
    sources: list[str] = []
    for i, q in enumerate(queries):
        if q in reference_embeddings:
            per_query.append(reference_embeddings[q])
            sources.append("image")
        elif text_fallback is not None:
            per_query.append(text_fallback[i : i + 1])
            sources.append("text")
        else:
            dim = (
                next(iter(reference_embeddings.values())).shape[-1]
                if reference_embeddings
                else 1
            )
            per_query.append(np.zeros((1, dim), dtype=np.float32))
            sources.append("missing")
    return per_query, sources


def similarity_matrix(
    crop_feats: np.ndarray,  # (M, D)
    per_query_refs: list[np.ndarray],
    agg_mode: Literal["mean", "max"],
) -> np.ndarray:
    """Return (M, P) cosine-similarity matrix aggregated across references."""
    M = crop_feats.shape[0]
    P = len(per_query_refs)
    out = np.zeros((M, P), dtype=np.float32)
    if M == 0 or P == 0:
        return out
    for i, refs in enumerate(per_query_refs):
        sim_per_ref = crop_feats @ refs.T  # (M, K_i)
        out[:, i] = aggregate(sim_per_ref, agg_mode)
    return out

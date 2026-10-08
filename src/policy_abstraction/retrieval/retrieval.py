"""EmbeddingRetriever: top-k demo retrieval via weighted cosine similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_abstraction.core.retrieval import DemoRetriever
from policy_abstraction.retrieval.encoder import DemoEncoder
from policy_abstraction.retrieval.index import DatasetIndex


class EmbeddingRetriever(DemoRetriever):
    """Retrieve the k most similar demos via combined visual + text cosine similarity.

    Both the query and the index embeddings are L2-normalised, so cosine
    similarity is computed as a plain dot product (BLAS GEMV).

    Parameters
    ----------
    index:
        Pre-built :class:`DatasetIndex` containing all candidate demo embeddings.
    encoder:
        :class:`DemoEncoder` used to embed the query at retrieval time.
    visual_weight:
        Relative weight for the DINOv2 visual similarity term.
    text_weight:
        Relative weight for the CLIP text similarity term.

    Query format
    ------------
    The ``query`` argument passed to :meth:`retrieve` must be a dict with keys:

    ``"frames"``
        ``list[np.ndarray]`` — sequence of ``(H, W, 3)`` uint8 RGB frames,
        **already flipped** (LIBERO images need ``[::-1, ::-1, :]`` applied
        by the caller before passing here).
    ``"task"``
        ``str`` — human-readable task description.

    Return format
    -------------
    Each returned dict has keys:
    ``source_kind``, ``source_path``, ``demo_key``, ``task``, ``score``,
    ``visual_score``, ``text_score``.
    """

    def __init__(
        self,
        index: DatasetIndex,
        encoder: DemoEncoder,
        visual_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> None:
        self.index = index
        self.encoder = encoder
        total = visual_weight + text_weight
        self._alpha = visual_weight / total  # normalised visual weight
        self._beta = text_weight / total  # normalised text weight

    def retrieve(self, query: Any, k: int = 5) -> list[dict]:
        """Return the k most similar demonstrations for the given query.

        Parameters
        ----------
        query:
            Dict with keys ``"frames"`` (list of RGB arrays) and ``"task"`` (str).
        k:
            Number of demonstrations to retrieve.

        Returns
        -------
        list[dict]
            Sorted by descending similarity. Each dict contains:
            ``source_kind``, ``source_path``, ``demo_key``, ``task``,
            ``score``, ``visual_score``, ``text_score``.
        """
        frames: list[np.ndarray] = query["frames"]
        task: str = query["task"]

        visual_q, text_q = self.encoder.encode_demo(frames, task)

        visual_scores = self.index.visual_matrix @ visual_q  # (N,)
        text_scores = self.index.text_matrix @ text_q  # (N,)
        scores = self._alpha * visual_scores + self._beta * text_scores  # (N,)

        n = self.index.size
        k_clamped = min(k, n)

        # O(N) partial sort — only fully sort the top-k candidates
        top_k_idx = np.argpartition(scores, -k_clamped)[-k_clamped:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        results = []
        for idx in top_k_idx:
            meta = self.index.metadata[idx]
            results.append(
                {
                    "source_kind": meta["source_kind"],
                    "source_path": meta["source_path"],
                    "demo_key": meta["demo_key"],
                    "task": meta["task"],
                    "score": float(scores[idx]),
                    "visual_score": float(visual_scores[idx]),
                    "text_score": float(text_scores[idx]),
                }
            )
        return results

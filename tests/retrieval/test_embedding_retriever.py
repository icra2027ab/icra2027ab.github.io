"""Tests for policy_abstraction.retrieval.retrieval.EmbeddingRetriever."""

from __future__ import annotations

import numpy as np

from policy_abstraction.retrieval.index import DatasetIndex
from policy_abstraction.retrieval.retrieval import EmbeddingRetriever

# ---------------------------------------------------------------------------
# Fake encoder — no GPU, no model
# ---------------------------------------------------------------------------


class _FakeEncoder:
    """Encoder that returns pre-set query vectors (pure numpy)."""

    def __init__(
        self,
        visual_q: np.ndarray | None = None,
        text_q: np.ndarray | None = None,
    ) -> None:
        self._vq = visual_q if visual_q is not None else np.zeros(768, dtype=np.float32)
        self._tq = text_q if text_q is not None else np.zeros(512, dtype=np.float32)

    def encode_demo(self, frames, task_text, batch_size=32):
        return self._vq, self._tq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_index_with_data(
    tmp_path,
    n_demos: int = 5,
    visual_dim: int = 768,
    text_dim: int = 512,
) -> tuple[DatasetIndex, np.ndarray, np.ndarray]:
    """Write a synthetic npz and return (index, visual_embs, text_embs)."""
    rng = np.random.default_rng(42)
    visual_embs = rng.standard_normal((n_demos, visual_dim)).astype(np.float32)
    text_embs = rng.standard_normal((n_demos, text_dim)).astype(np.float32)

    # L2-normalise so dot-product == cosine similarity
    visual_embs /= np.linalg.norm(visual_embs, axis=1, keepdims=True)
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)

    np.savez(
        tmp_path / "task_embeddings.npz",
        visual_embeddings=visual_embs,
        text_embeddings=text_embs,
        demo_keys=np.array([f"demo_{i}" for i in range(n_demos)]),
        source_kind=np.array("libero"),
        source_path=np.array("fake.hdf5"),
        task=np.array("pick up the mug"),
    )
    index = DatasetIndex([tmp_path])
    return index, visual_embs, text_embs


def _dummy_query() -> dict:
    return {"frames": [np.zeros((4, 4, 3), dtype=np.uint8)], "task": "pick up the mug"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingRetriever:
    """Tests for EmbeddingRetriever.retrieve."""

    def test_top1_is_highest_scoring(self, tmp_path):
        """When the query exactly matches demo_2, it is returned first."""
        index, visual_embs, text_embs = _build_index_with_data(tmp_path, n_demos=5)
        target = 2
        encoder = _FakeEncoder(visual_q=visual_embs[target], text_q=text_embs[target])
        retriever = EmbeddingRetriever(index=index, encoder=encoder)  # ty: ignore[invalid-argument-type]
        results = retriever.retrieve(_dummy_query(), k=1)
        assert len(results) == 1
        assert results[0]["demo_key"] == f"demo_{target}"

    def test_k_clamped_to_index_size(self, tmp_path):
        """Requesting k=100 when only 5 demos exist returns 5."""
        index, _, _ = _build_index_with_data(tmp_path, n_demos=5)
        encoder = _FakeEncoder()
        retriever = EmbeddingRetriever(index=index, encoder=encoder)  # ty: ignore[invalid-argument-type]
        results = retriever.retrieve(_dummy_query(), k=100)
        assert len(results) == 5

    def test_result_keys_present(self, tmp_path):
        """Each result dict has the expected keys."""
        index, _, _ = _build_index_with_data(tmp_path, n_demos=3)
        encoder = _FakeEncoder()
        retriever = EmbeddingRetriever(index=index, encoder=encoder)  # ty: ignore[invalid-argument-type]
        results = retriever.retrieve(_dummy_query(), k=1)
        for key in (
            "source_kind",
            "source_path",
            "demo_key",
            "task",
            "score",
            "visual_score",
            "text_score",
        ):
            assert key in results[0], f"Missing result key: {key}"

    def test_results_sorted_descending(self, tmp_path):
        """Scores in the result list are monotonically non-increasing."""
        index, _, _ = _build_index_with_data(tmp_path, n_demos=5)
        encoder = _FakeEncoder()
        retriever = EmbeddingRetriever(index=index, encoder=encoder)  # ty: ignore[invalid-argument-type]
        results = retriever.retrieve(_dummy_query(), k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_visual_weight_only(self, tmp_path):
        """With visual_weight=1.0, text_weight=0, scores equal visual_scores."""
        index, visual_embs, _ = _build_index_with_data(tmp_path, n_demos=4)
        encoder = _FakeEncoder(visual_q=visual_embs[0])
        retriever = EmbeddingRetriever(
            index=index,
            encoder=encoder,  # ty: ignore[invalid-argument-type]
            visual_weight=1.0,
            text_weight=0.0,
        )
        results = retriever.retrieve(_dummy_query(), k=4)
        for r in results:
            np.testing.assert_allclose(r["score"], r["visual_score"], atol=1e-5)

    def test_text_weight_only(self, tmp_path):
        """With visual_weight=0.0, text_weight=1.0, scores equal text_scores."""
        index, _, text_embs = _build_index_with_data(tmp_path, n_demos=4)
        encoder = _FakeEncoder(text_q=text_embs[0])
        retriever = EmbeddingRetriever(
            index=index,
            encoder=encoder,  # ty: ignore[invalid-argument-type]
            visual_weight=0.0,
            text_weight=1.0,
        )
        results = retriever.retrieve(_dummy_query(), k=4)
        for r in results:
            np.testing.assert_allclose(r["score"], r["text_score"], atol=1e-5)

    def test_equal_weights_normalized(self, tmp_path):
        """visual_weight=2, text_weight=2 behaves the same as 1,1 (normalised)."""
        index, visual_embs, text_embs = _build_index_with_data(tmp_path, n_demos=4)
        enc = _FakeEncoder(visual_q=visual_embs[0], text_q=text_embs[0])

        ret_11 = EmbeddingRetriever(
            index=index,
            encoder=enc,  # ty: ignore[invalid-argument-type]
            visual_weight=1.0,
            text_weight=1.0,
        )
        ret_22 = EmbeddingRetriever(
            index=index,
            encoder=enc,  # ty: ignore[invalid-argument-type]
            visual_weight=2.0,
            text_weight=2.0,
        )

        r11 = ret_11.retrieve(_dummy_query(), k=4)
        r22 = ret_22.retrieve(_dummy_query(), k=4)

        for a, b in zip(r11, r22, strict=False):
            np.testing.assert_allclose(a["score"], b["score"], atol=1e-5)

    def test_retrieve_returns_list(self, tmp_path):
        """Return type is always a list."""
        index, _, _ = _build_index_with_data(tmp_path, n_demos=2)
        encoder = _FakeEncoder()
        retriever = EmbeddingRetriever(index=index, encoder=encoder)  # ty: ignore[invalid-argument-type]
        results = retriever.retrieve(_dummy_query(), k=1)
        assert isinstance(results, list)

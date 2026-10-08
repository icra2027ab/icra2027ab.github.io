from __future__ import annotations

import numpy as np
from open_perception.context import DetectionContext
from open_perception.stages.filter.top_k import TopKFilterStage
from open_perception.types import Detection, InferenceOptions, InferenceStats


def _ctx(masks: np.ndarray, scores: np.ndarray) -> DetectionContext:
    return DetectionContext(
        queries=["x"],
        rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        depth=None,
        camera=None,
        options=InferenceOptions(),
        instances=masks.copy(),
        mask_scores=scores.copy(),
        stats=InferenceStats(),
    )


def _four_disjoint_masks() -> np.ndarray:
    m = np.zeros((4, 20, 20), dtype=bool)
    m[0, 0:5, 0:5] = True
    m[1, 0:5, 10:15] = True
    m[2, 10:15, 0:5] = True
    m[3, 10:15, 10:15] = True
    return m


def test_top_k_keeps_highest_scoring():
    masks = _four_disjoint_masks()
    scores = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float32)
    ctx = _ctx(masks, scores)
    TopKFilterStage(k=2)(ctx)
    assert ctx.instances.shape[0] == 2
    # Ordering preserved (ascending original index): 0.9 (idx 1) and 0.7 (idx 3).
    assert np.array_equal(ctx.instances[0], masks[1])
    assert np.array_equal(ctx.instances[1], masks[3])
    np.testing.assert_allclose(ctx.mask_scores, [0.9, 0.7], atol=1e-6)


def test_top_k_passthrough_when_n_le_k():
    masks = _four_disjoint_masks()[:2]
    scores = np.array([0.1, 0.9], dtype=np.float32)
    ctx = _ctx(masks, scores)
    TopKFilterStage(k=5)(ctx)
    assert ctx.instances.shape[0] == 2
    np.testing.assert_allclose(ctx.mask_scores, [0.1, 0.9], atol=1e-6)


def test_top_k_zero_drops_all():
    masks = _four_disjoint_masks()
    scores = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float32)
    ctx = _ctx(masks, scores)
    TopKFilterStage(k=0)(ctx)
    assert ctx.instances.shape[0] == 0
    assert ctx.mask_scores.shape[0] == 0


def test_top_k_combines_dino_and_segmentation():
    # Mask 0 wins on segmentation; mask 2 wins on dino. With dino-dominant
    # weights, the dino winner should be in the top-1.
    masks = _four_disjoint_masks()
    seg = np.array([0.9, 0.1, 0.1, 0.1], dtype=np.float32)
    dino = np.array([[0.1], [0.2], [0.95], [0.3]], dtype=np.float32)
    ctx = _ctx(masks, seg)
    ctx.scores["dino"] = dino
    TopKFilterStage(
        k=1,
        score_weights={"dino": 1.0, "segmentation": 0.0},
    )(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[2])
    # Segmentation score is preserved on writeback (not the combined ranking).
    np.testing.assert_allclose(ctx.mask_scores, [0.1], atol=1e-6)


def test_top_k_updates_detections_and_scores():
    masks = _four_disjoint_masks()
    seg = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float32)
    dino = np.array([[0.5], [0.6], [0.7], [0.8]], dtype=np.float32)
    ctx = _ctx(masks, seg)
    ctx.scores["dino"] = dino
    ctx.detections = {
        "a": [Detection(query="a", mask=masks[1].copy(), score=0.9, source_index=1)],
        "b": [Detection(query="b", mask=masks[2].copy(), score=0.4, source_index=2)],
    }
    TopKFilterStage(k=2)(ctx)
    # Top-2 by seg: idx 1 (0.9) and idx 3 (0.7). Idx 2 detection is dropped.
    assert ctx.instances.shape[0] == 2
    np.testing.assert_allclose(ctx.scores["dino"], [[0.6], [0.8]], atol=1e-6)
    assert len(ctx.detections["a"]) == 1
    assert ctx.detections["a"][0].source_index == 0
    assert len(ctx.detections["b"]) == 0


def test_top_k_missing_key_falls_back_to_segmentation():
    masks = _four_disjoint_masks()
    seg = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float32)
    ctx = _ctx(masks, seg)
    # No "dino" entry in ctx.scores -> falls back to segmentation order.
    TopKFilterStage(k=2, score_weights={"dino": 1.0})(ctx)
    assert ctx.instances.shape[0] == 2
    assert np.array_equal(ctx.instances[0], masks[1])  # 0.9
    assert np.array_equal(ctx.instances[1], masks[3])  # 0.7


def test_top_k_empty_ctx_passthrough():
    masks = np.zeros((0, 20, 20), dtype=bool)
    scores = np.zeros((0,), dtype=np.float32)
    ctx = _ctx(masks, scores)
    TopKFilterStage(k=3)(ctx)
    assert ctx.instances.shape == (0, 20, 20)
    assert ctx.mask_scores.shape == (0,)


def test_top_k_factory_requires_k():
    from open_perception.stages.base import REGISTRY

    factory = REGISTRY.get("filter.top_k")
    services = type("S", (), {"config": None})()
    try:
        factory(services)
    except ValueError as e:
        assert "k" in str(e)
    else:
        raise AssertionError("expected ValueError when k is missing")

from __future__ import annotations

import numpy as np
from open_perception.context import DetectionContext
from open_perception.stages.filter.score import ScoreFilterStage
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


def _three_disjoint_masks() -> np.ndarray:
    m0 = np.zeros((20, 20), dtype=bool)
    m0[0:5, 0:5] = True
    m1 = np.zeros((20, 20), dtype=bool)
    m1[0:5, 10:15] = True
    m2 = np.zeros((20, 20), dtype=bool)
    m2[10:15, 0:5] = True
    return np.stack([m0, m1, m2], axis=0)


def test_score_filter_drops_below_threshold():
    masks = _three_disjoint_masks()
    scores = np.array([0.9, 0.4, 0.1], dtype=np.float32)
    ctx = _ctx(masks, scores)
    ScoreFilterStage(threshold=0.5)(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[0])
    np.testing.assert_allclose(ctx.mask_scores, [0.9], atol=1e-6)


def test_score_filter_keeps_all_when_above_threshold():
    masks = _three_disjoint_masks()
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    ctx = _ctx(masks, scores)
    ScoreFilterStage(threshold=0.5)(ctx)
    assert ctx.instances.shape[0] == 3
    np.testing.assert_allclose(ctx.mask_scores, [0.9, 0.8, 0.7], atol=1e-6)


def test_score_filter_drops_all_when_threshold_too_high():
    masks = _three_disjoint_masks()
    scores = np.array([0.9, 0.4, 0.1], dtype=np.float32)
    ctx = _ctx(masks, scores)
    ScoreFilterStage(threshold=1.5)(ctx)
    assert ctx.instances.shape[0] == 0
    assert ctx.mask_scores.shape[0] == 0


def test_score_filter_combines_dino_and_segmentation():
    # Mask 0: high seg, low dino. Mask 1: low seg, high dino.
    # With dino weight dominating + L2 normalisation, mask 1 survives.
    masks = _three_disjoint_masks()[:2]
    seg = np.array([0.9, 0.1], dtype=np.float32)
    dino = np.array([[0.05], [0.95]], dtype=np.float32)
    ctx = _ctx(masks, seg)
    ctx.scores["dino"] = dino
    ScoreFilterStage(
        threshold=0.6,
        score_weights={"dino": 1.0, "segmentation": 0.0},
    )(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[1])
    # Segmentation score is preserved on writeback (not the combined ranking).
    np.testing.assert_allclose(ctx.mask_scores, [0.1], atol=1e-6)


def test_score_filter_updates_detections_and_scores():
    masks = _three_disjoint_masks()
    seg = np.array([0.9, 0.4, 0.1], dtype=np.float32)
    dino = np.array([[0.7], [0.5], [0.3]], dtype=np.float32)
    ctx = _ctx(masks, seg)
    ctx.scores["dino"] = dino
    ctx.detections = {
        "a": [Detection(query="a", mask=masks[0].copy(), score=0.9, source_index=0)],
        "b": [Detection(query="b", mask=masks[2].copy(), score=0.1, source_index=2)],
    }
    ScoreFilterStage(threshold=0.3)(ctx)
    assert ctx.instances.shape[0] == 2
    # ctx.scores entries are reindexed to match the kept instances.
    np.testing.assert_allclose(ctx.scores["dino"], [[0.7], [0.5]], atol=1e-6)
    # Surviving detection points at the new index; dropped one is gone.
    assert len(ctx.detections["a"]) == 1
    assert ctx.detections["a"][0].source_index == 0
    assert len(ctx.detections["b"]) == 0


def test_score_filter_missing_key_falls_back_to_segmentation():
    masks = _three_disjoint_masks()
    seg = np.array([0.9, 0.4, 0.1], dtype=np.float32)
    ctx = _ctx(masks, seg)
    # No "dino" entry in ctx.scores -> falls back to seg scores.
    ScoreFilterStage(threshold=0.5, score_weights={"dino": 1.0})(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[0])


def test_score_filter_empty_ctx_passthrough():
    masks = np.zeros((0, 20, 20), dtype=bool)
    scores = np.zeros((0,), dtype=np.float32)
    ctx = _ctx(masks, scores)
    ScoreFilterStage(threshold=0.5)(ctx)
    assert ctx.instances.shape == (0, 20, 20)
    assert ctx.mask_scores.shape == (0,)


def test_score_filter_factory_requires_threshold():
    from open_perception.stages.base import REGISTRY

    factory = REGISTRY.get("filter.score")
    services = type("S", (), {"config": None})()
    try:
        factory(services)
    except ValueError as e:
        assert "threshold" in str(e)
    else:
        raise AssertionError("expected ValueError when threshold is missing")

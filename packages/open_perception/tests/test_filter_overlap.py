from __future__ import annotations

import numpy as np
from open_perception.context import DetectionContext
from open_perception.stages.filter.overlap import OverlapFilterStage
from open_perception.types import Detection, InferenceOptions, InferenceStats


def _make_masks(overlaps: float) -> np.ndarray:
    """Two 10x10 masks. overlaps=0 -> disjoint; overlaps=0.9 -> near-identical."""
    m1 = np.zeros((20, 20), dtype=bool)
    m1[0:10, 0:10] = True  # area 100
    m2 = np.zeros((20, 20), dtype=bool)
    if overlaps >= 0.99:
        m2 = m1.copy()
    elif overlaps > 0:
        # IoU ~ overlaps: slide m2 by `shift` pixels in x so intersection ~ 100*(1-shift/10)
        # IoU = inter / (2*100 - inter).
        shift = int(round(10 * (1 - overlaps) / (1 + overlaps)))
        m2[0:10, shift : shift + 10] = True
    else:
        m2[15:, 15:] = True
    return np.stack([m1, m2], axis=0)


def _ctx(masks: np.ndarray, scores: np.ndarray, **opt_overrides) -> DetectionContext:
    return DetectionContext(
        queries=["x"],
        rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        depth=None,
        camera=None,
        options=InferenceOptions(**opt_overrides),
        instances=masks.copy(),
        mask_scores=scores.copy(),
        stats=InferenceStats(),
    )


def test_remove_mode_drops_overlapping():
    masks = _make_masks(0.9)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, scores, overlap_iou_threshold=0.5)
    OverlapFilterStage(iou_threshold=0.5, mode="remove")(ctx)
    assert ctx.instances.shape[0] == 1
    # The higher-scoring mask (first) survives.
    assert np.array_equal(ctx.instances[0], masks[0])


def test_remove_mode_keeps_disjoint():
    masks = _make_masks(0.0)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, scores, overlap_iou_threshold=0.5)
    OverlapFilterStage(iou_threshold=0.5, mode="remove")(ctx)
    assert ctx.instances.shape[0] == 2


def test_filter_disabled_via_options():
    masks = _make_masks(0.99)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, scores, filter_overlap=False, overlap_iou_threshold=0.5)
    OverlapFilterStage(iou_threshold=0.5)(ctx)
    assert ctx.instances.shape[0] == 2
    assert ctx.stats.n_after_overlap_filter == [2]


def test_subtract_mode_trims_lower_score_mask():
    # Two 10x10 masks with a 5-pixel-wide overlap band.
    m1 = np.zeros((20, 20), dtype=bool)
    m1[0:10, 0:10] = True  # area 100
    m2 = np.zeros((20, 20), dtype=bool)
    m2[0:10, 5:15] = True  # area 100, overlap region is 10x5=50 pixels
    masks = np.stack([m1, m2], axis=0)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, scores)
    OverlapFilterStage(iou_threshold=0.5, mode="subtract")(ctx)
    assert ctx.instances.shape[0] == 2
    # Higher-score mask is untouched.
    assert np.array_equal(ctx.instances[0], m1)
    # Lower-score mask has the intersection removed.
    assert int(ctx.instances[1].sum()) == 50
    assert not np.any(ctx.instances[1] & m1)
    # After filtering, no two kept masks share a pixel.
    assert not np.any(ctx.instances[0] & ctx.instances[1])


def test_subtract_mode_drops_fully_covered_mask():
    # Identical masks — lower-score one gets fully subtracted away.
    masks = _make_masks(0.99)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, scores)
    OverlapFilterStage(iou_threshold=0.5, mode="subtract", min_area=1)(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[0])


def test_subtract_mode_updates_detections():
    # Simulate the sam3 per-query emission path: ctx.detections is populated
    # with source_index pointing into ctx.instances.
    m1 = np.zeros((20, 20), dtype=bool)
    m1[0:10, 0:10] = True
    m2 = np.zeros((20, 20), dtype=bool)
    m2[0:10, 5:15] = True
    masks = np.stack([m1, m2], axis=0)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, scores)
    ctx.detections = {
        "a": [Detection(query="a", mask=m1.copy(), score=0.9, source_index=0)],
        "b": [Detection(query="b", mask=m2.copy(), score=0.5, source_index=1)],
    }
    OverlapFilterStage(iou_threshold=0.5, mode="subtract")(ctx)
    # Higher-score detection unchanged.
    assert int(ctx.detections["a"][0].mask.sum()) == 100
    # Lower-score detection trimmed to just its non-overlapping pixels.
    assert int(ctx.detections["b"][0].mask.sum()) == 50
    # No pixel overlap between the two remaining detection masks.
    assert not np.any(ctx.detections["a"][0].mask & ctx.detections["b"][0].mask)


def test_remove_mode_drops_detections():
    masks = _make_masks(0.99)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, scores, overlap_iou_threshold=0.5)
    ctx.detections = {
        "a": [Detection(query="a", mask=masks[0].copy(), score=0.9, source_index=0)],
        "b": [Detection(query="b", mask=masks[1].copy(), score=0.5, source_index=1)],
    }
    OverlapFilterStage(iou_threshold=0.5, mode="remove")(ctx)
    assert len(ctx.detections["a"]) == 1
    assert len(ctx.detections["b"]) == 0


def test_score_weights_dino_overrides_segmentation_winner():
    # Identical masks — segmentation prefers index 0, dino prefers index 1.
    # With weights tilted to dino, the dino winner survives.
    masks = _make_masks(0.99)
    seg = np.array([0.9, 0.5], dtype=np.float32)
    dino = np.array([[0.1], [0.9]], dtype=np.float32)  # (M=2, P=1)
    ctx = _ctx(masks, seg, overlap_iou_threshold=0.5)
    ctx.scores["dino"] = dino
    OverlapFilterStage(
        iou_threshold=0.5,
        mode="remove",
        score_weights={"dino": 1.0, "segmentation": 0.0},
    )(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[1])
    # Segmentation score is preserved on writeback (not the combined ranking).
    assert float(ctx.mask_scores[0]) == 0.5


def test_score_weights_combine_multiple_sources():
    # Three masks with very different per-source rankings.
    m0 = np.zeros((20, 20), dtype=bool)
    m0[0:10, 0:10] = True
    m1 = m0.copy()
    m2 = m0.copy()
    masks = np.stack([m0, m1, m2], axis=0)
    seg = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    dino = np.array([[0.1], [0.9], [0.1]], dtype=np.float32)
    clip = np.array([[0.1], [0.1], [0.9]], dtype=np.float32)
    ctx = _ctx(masks, seg, overlap_iou_threshold=0.5)
    ctx.scores["dino"] = dino
    ctx.scores["clip"] = clip
    # Dino weighted highest -> mask 1 should win.
    OverlapFilterStage(
        iou_threshold=0.5,
        mode="remove",
        score_weights={"dino": 0.5, "segmentation": 0.25, "clip": 0.25},
    )(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[1])


def test_score_weights_missing_key_falls_back_to_segmentation():
    # No keys in score_weights match ctx.scores -> behave like segmentation.
    masks = _make_masks(0.99)
    seg = np.array([0.9, 0.5], dtype=np.float32)
    ctx = _ctx(masks, seg, overlap_iou_threshold=0.5)
    OverlapFilterStage(
        iou_threshold=0.5,
        mode="remove",
        score_weights={"dino": 1.0},  # not present in ctx.scores
    )(ctx)
    assert ctx.instances.shape[0] == 1
    assert np.array_equal(ctx.instances[0], masks[0])


def test_overlap_reindexes_ctx_scores_on_drop():
    # Stale ctx.scores would break a downstream filter.score / filter.top_k
    # using score_weights — overlap must keep ctx.scores aligned with instances.
    masks = _make_masks(0.99)
    seg = np.array([0.9, 0.5], dtype=np.float32)
    dino = np.array([[0.7, 0.1], [0.2, 0.8]], dtype=np.float32)
    ctx = _ctx(masks, seg, overlap_iou_threshold=0.5)
    ctx.scores["dino"] = dino
    OverlapFilterStage(iou_threshold=0.5, mode="remove")(ctx)
    assert ctx.instances.shape[0] == 1
    np.testing.assert_allclose(ctx.scores["dino"], [[0.7, 0.1]], atol=1e-6)


def test_score_weights_max_aggregation_over_queries():
    # Mask 0 wins on query A only, mask 1 wins on query B only. With max
    # aggregation both have the same per-instance score (0.9), so the tiebreak
    # falls to segmentation via the mixed weights.
    masks = _make_masks(0.99)
    seg = np.array([0.4, 0.9], dtype=np.float32)
    dino = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)  # (M=2, P=2)
    ctx = _ctx(masks, seg, overlap_iou_threshold=0.5)
    ctx.scores["dino"] = dino
    OverlapFilterStage(
        iou_threshold=0.5,
        mode="remove",
        score_weights={"dino": 0.5, "segmentation": 0.5},
        score_agg="max",
    )(ctx)
    assert ctx.instances.shape[0] == 1
    # Mask 1 wins on segmentation tiebreak.
    assert np.array_equal(ctx.instances[0], masks[1])

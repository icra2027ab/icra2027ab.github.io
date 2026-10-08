from __future__ import annotations

import numpy as np
from open_perception.context import DetectionContext
from open_perception.stages.filter.size import SizeFilterStage, _build_filter_size
from open_perception.types import Detection, InferenceOptions, InferenceStats


def test_size_filter_drops_small_and_crops(synthetic_rgb, synthetic_sam3_output):
    masks, scores = synthetic_sam3_output
    ctx = DetectionContext(
        queries=["mug", "plate"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        instances=masks,
        mask_scores=scores,
        stats=InferenceStats(),
    )
    SizeFilterStage(min_area=10, max_instances=5, bbox_pad=0, zero_background=False)(
        ctx
    )
    # The third mask is 36 px > 10, so all three survive.
    assert ctx.instances.shape[0] == 3
    assert len(ctx.crops) == 3
    assert ctx.bboxes.shape == (3, 4)


def test_size_filter_caps_at_max_instances(synthetic_rgb, synthetic_sam3_output):
    masks, scores = synthetic_sam3_output
    ctx = DetectionContext(
        queries=["mug"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        instances=masks,
        mask_scores=scores,
        stats=InferenceStats(),
    )
    SizeFilterStage(min_area=1, max_instances=2, bbox_pad=0, zero_background=False)(ctx)
    assert ctx.instances.shape[0] == 2


def test_size_filter_excludes_below_min_area(synthetic_rgb):
    tiny = np.zeros((32, 32), dtype=bool)
    tiny[0, 0] = True  # 1 px — below default min area of 10
    masks = np.stack([tiny], axis=0)
    scores = np.array([0.9], dtype=np.float32)
    ctx = DetectionContext(
        queries=["mug"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        instances=masks,
        mask_scores=scores,
        stats=InferenceStats(),
    )
    SizeFilterStage(min_area=10, max_instances=5, bbox_pad=0, zero_background=False)(
        ctx
    )
    assert ctx.instances.shape[0] == 0


def test_size_filter_trims_detections(synthetic_rgb, synthetic_sam3_output):
    masks, scores = synthetic_sam3_output  # areas 64, 64, 36
    ctx = DetectionContext(
        queries=["mug", "plate", "tiny"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        instances=masks.copy(),
        mask_scores=scores.copy(),
        stats=InferenceStats(),
    )
    ctx.detections = {
        "mug": [
            Detection(query="mug", mask=masks[0].copy(), score=0.9, source_index=0)
        ],
        "plate": [
            Detection(query="plate", mask=masks[1].copy(), score=0.8, source_index=1)
        ],
        "tiny": [
            Detection(query="tiny", mask=masks[2].copy(), score=0.3, source_index=2)
        ],
    }
    SizeFilterStage(min_area=50, max_instances=5, bbox_pad=0, zero_background=False)(
        ctx
    )
    assert ctx.instances.shape[0] == 2
    # The 36-px mask is filtered out from detections too.
    assert len(ctx.detections["tiny"]) == 0
    assert len(ctx.detections["mug"]) == 1
    assert len(ctx.detections["plate"]) == 1
    # source_index is rewritten to the new position in ctx.instances.
    assert ctx.detections["mug"][0].source_index == 0
    assert ctx.detections["plate"][0].source_index == 1


def test_size_filter_reindexes_ctx_scores(synthetic_rgb):
    # When size drops instances, ctx.scores entries with the same first dim
    # must be reindexed so a downstream filter (filter.score, filter.top_k)
    # using score_weights doesn't see stale shapes.
    tiny = np.zeros((32, 32), dtype=bool)
    tiny[0, 0] = True  # area 1 — gets dropped by min_area=10
    big = np.zeros((32, 32), dtype=bool)
    big[0:10, 0:10] = True  # area 100 — kept
    masks = np.stack([tiny, big], axis=0)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    dino = np.array([[0.1, 0.2], [0.7, 0.8]], dtype=np.float32)
    ctx = DetectionContext(
        queries=["mug", "plate"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        instances=masks,
        mask_scores=scores,
        stats=InferenceStats(),
    )
    ctx.scores["dino"] = dino
    SizeFilterStage(min_area=10, max_instances=5, bbox_pad=0, zero_background=False)(
        ctx
    )
    assert ctx.instances.shape[0] == 1
    np.testing.assert_allclose(ctx.scores["dino"], [[0.7, 0.8]], atol=1e-6)


def test_size_filter_factory_reads_min_area():
    class _Cfg:
        default_min_area = 5
        default_max_instances = 10
        default_bbox_pad = 0
        default_clip_mask_bg = False

    class _Services:
        config = _Cfg()

    services = _Services()
    stage = _build_filter_size(services, min_area=10000)
    assert stage.min_area == 10000
    # No param → config default.
    stage_default = _build_filter_size(services)
    assert stage_default.min_area == 5

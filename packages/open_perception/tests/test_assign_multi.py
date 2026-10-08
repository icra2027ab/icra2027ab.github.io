from __future__ import annotations

import numpy as np
from open_perception.context import DetectionContext
from open_perception.stages.assign import AssignStage
from open_perception.types import InferenceOptions, InferenceStats


def _ctx_with_three_instances(rgb, scores_matrix, options):
    masks = np.stack(
        [
            np.zeros_like(rgb[..., 0], dtype=bool),
            np.zeros_like(rgb[..., 0], dtype=bool),
            np.zeros_like(rgb[..., 0], dtype=bool),
        ],
        axis=0,
    )
    masks[0, 0:4, 0:4] = True
    masks[1, 0:4, 4:8] = True
    masks[2, 4:8, 0:4] = True
    return DetectionContext(
        queries=["mug", "plate"],
        rgb=rgb,
        depth=None,
        camera=None,
        options=options,
        instances=masks,
        mask_scores=np.ones((3,), dtype=np.float32),
        crops=[rgb[0:4, 0:4], rgb[0:4, 4:8], rgb[4:8, 0:4]],
        scores={"clip": scores_matrix},
        stats=InferenceStats(),
    )


def test_multi_match_above_threshold(synthetic_rgb):
    # Two strong matches for 'mug' (rows 0 and 2) and one strong for 'plate' (row 1).
    sim = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.95],
            [0.7, 0.1],
        ],
        dtype=np.float32,
    )
    ctx = _ctx_with_three_instances(
        synthetic_rgb,
        sim,
        InferenceOptions(match_threshold=0.5),
    )
    AssignStage(top_k=-1)(ctx)
    assert len(ctx.detections["mug"]) == 2
    assert len(ctx.detections["plate"]) == 1
    # Detections are sorted by descending score.
    scores_mug = [d.score for d in ctx.detections["mug"]]
    assert scores_mug == sorted(scores_mug, reverse=True)


def test_top_k_1_hungarian_mode(synthetic_rgb):
    sim = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.95],
            [0.7, 0.1],
        ],
        dtype=np.float32,
    )
    ctx = _ctx_with_three_instances(
        synthetic_rgb, sim, InferenceOptions(match_threshold=0.0)
    )
    AssignStage(top_k=1)(ctx)
    # Hungarian assigns exactly one crop per query.
    assert len(ctx.detections["mug"]) == 1
    assert len(ctx.detections["plate"]) == 1
    # Optimal assignment: (row 0 -> mug), (row 1 -> plate) — row 2 is unused.
    assert ctx.detections["mug"][0].source_index == 0
    assert ctx.detections["plate"][0].source_index == 1


def test_top_k_per_query(synthetic_rgb):
    sim = np.array(
        [
            [0.9, 0.0],
            [0.8, 0.0],
            [0.7, 0.0],
        ],
        dtype=np.float32,
    )
    ctx = _ctx_with_three_instances(
        synthetic_rgb, sim, InferenceOptions(match_threshold=0.5)
    )
    AssignStage(top_k=2)(ctx)
    assert len(ctx.detections["mug"]) == 2
    assert len(ctx.detections["plate"]) == 0

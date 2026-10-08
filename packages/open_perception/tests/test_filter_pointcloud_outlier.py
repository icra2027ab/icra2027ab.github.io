from __future__ import annotations

import numpy as np
from open_perception.context import DetectionContext
from open_perception.stages.filter.pointcloud_outlier import (
    PointcloudOutlierFilterStage,
    _build_filter_pointcloud_outlier,
)
from open_perception.types import Detection, InferenceOptions, InferenceStats


def _make_ctx(pointcloud: np.ndarray, synthetic_rgb: np.ndarray) -> DetectionContext:
    ctx = DetectionContext(
        queries=["mug"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        stats=InferenceStats(),
    )
    mask = np.zeros((32, 32), dtype=bool)
    ctx.detections = {
        "mug": [Detection(query="mug", mask=mask, score=0.9, pointcloud=pointcloud)]
    }
    return ctx


def test_pointcloud_outlier_stage_drops_flyers(synthetic_rgb):
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(200, 3)).astype(np.float32)
    flyers = np.array([[5.0, 5.0, 5.0], [-4.0, -4.0, -4.0]], dtype=np.float32)
    pc = np.vstack([cluster, flyers])
    ctx = _make_ctx(pc, synthetic_rgb)

    PointcloudOutlierFilterStage(k=16, std_ratio=2.0)(ctx)

    out_pc = ctx.detections["mug"][0].pointcloud
    assert out_pc.shape[0] < pc.shape[0]
    # The flyers should be gone — none of the surviving points should match.
    assert not np.any(np.all(out_pc == flyers[0], axis=1))
    assert not np.any(np.all(out_pc == flyers[1], axis=1))


def test_pointcloud_outlier_stage_handles_empty_pc(synthetic_rgb):
    ctx = _make_ctx(np.zeros((0, 3), dtype=np.float32), synthetic_rgb)
    PointcloudOutlierFilterStage()(ctx)
    assert ctx.detections["mug"][0].pointcloud.shape == (0, 3)


def test_pointcloud_outlier_stage_skips_none(synthetic_rgb):
    ctx = _make_ctx(np.zeros((0, 3), dtype=np.float32), synthetic_rgb)
    ctx.detections["mug"][0].pointcloud = None
    PointcloudOutlierFilterStage()(ctx)
    assert ctx.detections["mug"][0].pointcloud is None


def test_pointcloud_outlier_factory_reads_params():
    stage = _build_filter_pointcloud_outlier(services=None, k=8, std_ratio=1.5)
    assert stage.k == 8
    assert stage.std_ratio == 1.5
    default = _build_filter_pointcloud_outlier(services=None)
    assert default.k == 16
    assert default.std_ratio == 2.0

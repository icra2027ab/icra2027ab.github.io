"""Tests for the cross-view fusion stage."""

from __future__ import annotations

import numpy as np
from open_perception.context import DetectionContext
from open_perception.stages.fuse_views import FuseViewsStage
from open_perception.types import (
    CameraInfo,
    Detection,
    InferenceOptions,
    InferenceStats,
)


def _camera(image_hw: tuple[int, int] = (32, 32)) -> CameraInfo:
    H, W = image_hw
    K = np.array(
        [[20.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return CameraInfo(intrinsics=K, extrinsics=np.eye(4, dtype=np.float32))


def _make_ctx(view_idx: int, camera: CameraInfo, image_hw=(32, 32)) -> DetectionContext:
    H, W = image_hw
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 1.0, dtype=np.float32)
    return DetectionContext(
        queries=["mug"],
        rgb=rgb,
        depth=depth,
        camera=camera,
        options=InferenceOptions(),
        view_idx=view_idx,
        stats=InferenceStats(),
    )


def _square_mask(image_hw: tuple[int, int], y0: int, x0: int, side: int) -> np.ndarray:
    m = np.zeros(image_hw, dtype=bool)
    m[y0 : y0 + side, x0 : x0 + side] = True
    return m


def test_independent_mode_is_noop() -> None:
    cam = _camera()
    ctxs = [_make_ctx(0, cam), _make_ctx(1, cam)]
    ctxs[0].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 10, 10, 6),
                score=0.9,
                view_idx=0,
                instance_id=42,
                pointcloud=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            )
        ]
    }
    ctxs[1].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 10, 10, 6),
                score=0.9,
                view_idx=1,
                instance_id=99,
            )
        ]
    }
    stage = FuseViewsStage(mode="independent")
    out = stage(ctxs)
    assert out is ctxs
    assert out[1].detections["mug"][0].instance_id == 99


def _projection_grid() -> np.ndarray:
    """Pointcloud that, with the test camera, lands at pixels (14..18, 14..18)."""
    return np.array(
        [
            [(x - 16) * 1.0 / 20.0, (y - 16) * 1.0 / 20.0, 1.0]
            for y in range(14, 19)
            for x in range(14, 19)
        ],
        dtype=np.float32,
    )


def test_primary_guided_skips_when_same_id_already_overlaps() -> None:
    """If the secondary already has main's id covering the footprint, do nothing."""
    cam = _camera()
    ctxs = [_make_ctx(0, cam), _make_ctx(1, cam)]
    ctxs[0].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.9,
                view_idx=0,
                instance_id=42,
                pointcloud=_projection_grid(),
            )
        ]
    }
    # Secondary already has a same-id detection that overlaps where the
    # footprint will land — nothing to condition.
    ctxs[1].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.8,
                view_idx=1,
                instance_id=42,
            )
        ]
    }

    seed_calls: list[tuple] = []

    class _Tracker:
        is_initialized = True

        def add_object_with_points(self, *_a, **_k):
            seed_calls.append(_a)
            raise AssertionError("should not seed when same id is already aligned")

    def attach(*_a, **_k):
        raise AssertionError("should not attach when same id is already aligned")

    stage = FuseViewsStage(
        mode="primary_guided",
        main_camera=0,
        get_tracker_fn=lambda v: _Tracker(),
        attach_id_fn=attach,
    )
    stage(ctxs)
    assert seed_calls == []
    # Still record a ProjectedReference so the visualization can show
    # the points landed correctly on the same-id mask.
    refs = ctxs[1].projected_refs
    assert len(refs) == 1
    assert refs[0].matched is True


def test_primary_guided_reindexes_when_overlap_has_different_id() -> None:
    """High IoU but a *different* id in the secondary is case (b): re-index — no seeding."""
    cam = _camera()
    ctxs = [_make_ctx(0, cam), _make_ctx(1, cam)]
    ctxs[0].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.9,
                view_idx=0,
                instance_id=42,
                pointcloud=_projection_grid(),
            )
        ]
    }
    ctxs[1].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.8,
                view_idx=1,
                instance_id=99,
            )
        ]
    }

    class _Tracker:
        is_initialized = True

        def add_object_with_points(self, *_a, **_k):
            raise AssertionError("should not seed when re-indexing is sufficient")

    attach_calls: list[tuple] = []

    def attach(view_idx, query, obj_id, mask):
        attach_calls.append((view_idx, query, obj_id))

    stage = FuseViewsStage(
        mode="primary_guided",
        main_camera=0,
        get_tracker_fn=lambda v: _Tracker(),
        attach_id_fn=attach,
    )
    stage(ctxs)
    # The existing detection's id is rewritten in place — no new sec_det added.
    assert len(ctxs[1].detections["mug"]) == 1
    assert ctxs[1].detections["mug"][0].instance_id == 42
    assert (1, "mug", 42) in attach_calls


def test_primary_guided_reindexes_across_queries() -> None:
    """Secondary segmented the object under the wrong query → re-index *and* move it."""
    cam = _camera()
    ctxs = [_make_ctx(0, cam), _make_ctx(1, cam)]
    ctxs[0].detections = {
        "plate": [
            Detection(
                query="plate",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.9,
                view_idx=0,
                instance_id=4,
                pointcloud=_projection_grid(),
            )
        ]
    }
    # Secondary mis-classified the plate as a mug under id=6.
    ctxs[1].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.8,
                view_idx=1,
                instance_id=6,
            )
        ]
    }

    class _Tracker:
        is_initialized = True

        def add_object_with_points(self, *_a, **_k):
            raise AssertionError("should not seed when re-indexing across queries")

    attach_calls: list[tuple] = []

    stage = FuseViewsStage(
        mode="primary_guided",
        main_camera=0,
        get_tracker_fn=lambda v: _Tracker(),
        attach_id_fn=lambda v, q, oid, m: attach_calls.append((v, q, oid)),
    )
    stage(ctxs)
    # The wrong-query bucket is now empty; the detection moved to "plate".
    assert ctxs[1].detections.get("mug", []) == []
    assert len(ctxs[1].detections.get("plate", [])) == 1
    moved = ctxs[1].detections["plate"][0]
    assert moved.instance_id == 4
    assert moved.query == "plate"
    assert (1, "plate", 4) in attach_calls


def test_primary_guided_skips_when_no_pointcloud() -> None:
    cam = _camera()
    ctxs = [_make_ctx(0, cam), _make_ctx(1, cam)]
    ctxs[0].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.9,
                view_idx=0,
                instance_id=42,
                pointcloud=None,
            )
        ]
    }
    ctxs[1].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.8,
                view_idx=1,
                instance_id=99,
            )
        ]
    }
    stage = FuseViewsStage(mode="primary_guided", main_camera=0)
    stage(ctxs)
    # Without a pointcloud, fusion can't project — secondary id stays put.
    assert ctxs[1].detections["mug"][0].instance_id == 99


def test_primary_guided_no_seed_at_detect_time() -> None:
    """When the secondary tracker isn't initialized yet, no point-prompt seeding fires."""
    cam = _camera()
    ctxs = [_make_ctx(0, cam), _make_ctx(1, cam)]
    grid = np.array(
        [
            [(x - 16) * 1.0 / 20.0, (y - 16) * 1.0 / 20.0, 1.0]
            for y in range(14, 19)
            for x in range(14, 19)
        ],
        dtype=np.float32,
    )
    ctxs[0].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.9,
                view_idx=0,
                instance_id=42,
                pointcloud=grid,
            )
        ]
    }
    # Secondary has no detections at all — would normally trigger seeding.
    ctxs[1].detections = {"mug": []}

    class _UninitTracker:
        is_initialized = False

        def add_object_with_points(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("should not be called when tracker is uninitialized")

    def get_tracker(view_idx):
        return _UninitTracker()

    def attach(*_a, **_k):
        pass

    stage = FuseViewsStage(
        mode="primary_guided",
        main_camera=0,
        get_tracker_fn=get_tracker,
        attach_id_fn=attach,
    )
    stage(ctxs)
    assert ctxs[1].detections["mug"] == []


def test_primary_guided_seeds_via_points_when_tracker_initialized() -> None:
    """At track time (tracker initialized) a no-IoU-match triggers point-prompt seeding."""
    cam = _camera()
    ctxs = [_make_ctx(0, cam), _make_ctx(1, cam)]
    grid = np.array(
        [
            [(x - 16) * 1.0 / 20.0, (y - 16) * 1.0 / 20.0, 1.0]
            for y in range(14, 19)
            for x in range(14, 19)
        ],
        dtype=np.float32,
    )
    ctxs[0].detections = {
        "mug": [
            Detection(
                query="mug",
                mask=_square_mask((32, 32), 14, 14, 5),
                score=0.9,
                view_idx=0,
                instance_id=42,
                pointcloud=grid,
            )
        ]
    }
    ctxs[1].detections = {"mug": []}

    seed_calls: list[tuple] = []
    fake_mask = _square_mask((32, 32), 14, 14, 5)

    class _Tracker:
        is_initialized = True

        def add_object_with_points(self, frame_rgb, obj_id, points_uv, point_labels):
            seed_calls.append((obj_id, points_uv.shape[0]))
            return fake_mask

    def get_tracker(view_idx):
        return _Tracker()

    attach_calls: list[tuple] = []

    def attach(view_idx, query, obj_id, mask):
        attach_calls.append((view_idx, query, obj_id))

    stage = FuseViewsStage(
        mode="primary_guided",
        main_camera=0,
        get_tracker_fn=get_tracker,
        attach_id_fn=attach,
    )
    stage(ctxs)
    assert seed_calls and seed_calls[0][0] == 42
    assert ctxs[1].detections["mug"]
    assert ctxs[1].detections["mug"][0].instance_id == 42
    assert (1, "mug", 42) in attach_calls

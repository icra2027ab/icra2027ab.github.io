"""Tests for the CoTracker streaming wrapper.

These tests do **not** require the optional `[cotracker]` extras — they
inject a deterministic fake into ``CoTrackerStream(load_fn=...)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _cotracker_fakes import fake_loader_factory
from open_perception.tracking.cotracker_stream import (
    CoTrackerStream,
    PointTracks,
    mask_to_depth_queries,
)
from open_perception.types import Detection

# ---------------------------------------------------------------------------
# mask_to_depth_queries
# ---------------------------------------------------------------------------


def _disc_mask(h: int = 32, w: int = 32, cy: int = 16, cx: int = 16, r: int = 8):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def test_mask_to_depth_queries_requires_depth():
    mask = _disc_mask()
    with pytest.raises(ValueError, match="requires depth"):
        mask_to_depth_queries(mask, None)  # type: ignore[arg-type]


def test_mask_to_depth_queries_filters_invalid_depth():
    mask = _disc_mask(r=10)
    depth = np.full((32, 32), 1.0, dtype=np.float32)
    # Zero out depth in half of the mask area; those pixels must not be
    # selected as queries.
    depth[:, :16] = 0.0
    rng = np.random.default_rng(0)
    queries = mask_to_depth_queries(
        mask,
        depth,
        max_points=200,
        min_points=4,
        erode_kernel_size=0,
        rng=rng,
    )
    assert queries is not None
    xs = queries[:, 1].astype(int)
    assert (xs >= 16).all(), "query (x) must avoid the depth==0 half"


def test_mask_to_depth_queries_below_min_returns_none():
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, 0] = True  # exactly 1 valid pixel
    depth = np.full((32, 32), 1.0, dtype=np.float32)
    out = mask_to_depth_queries(
        mask, depth, max_points=50, min_points=8, erode_kernel_size=0
    )
    assert out is None


# ---------------------------------------------------------------------------
# CoTrackerStream warm-up + step
# ---------------------------------------------------------------------------


def _make_detection(
    instance_id: int, mask: np.ndarray, query: str = "obj"
) -> Detection:
    return Detection(
        query=query,
        mask=mask,
        score=0.9,
        view_idx=0,
        instance_id=instance_id,
    )


def _identity_intrinsics() -> np.ndarray:
    return np.array(
        [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def _seeded_stream(step: int = 4):
    """Build a CoTrackerStream + seed it with one disc detection.

    Returns ``(stream, rgbs)`` where ``rgbs`` is a list of synthetic frames
    we'll feed in via :meth:`step`.
    """
    stream = CoTrackerStream(
        device="cpu",
        model_name="cotracker3_online",
        load_fn=fake_loader_factory(step=step),
    )
    H, W = 32, 32
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 1.0, dtype=np.float32)
    mask = _disc_mask(h=H, w=W, r=8)
    det = _make_detection(instance_id=1, mask=mask, query="disc")
    instance_slices = stream.seed(
        view_idx=0,
        frame_idx=100,  # absolute frame this view started at
        rgb=rgb,
        detections=[det],
        depth=depth,
        intrinsics=_identity_intrinsics(),
        extrinsics=None,
        max_points_per_det=16,
        min_points_per_det=4,
        erode_kernel_size=0,
    )
    rgbs = [np.full((H, W, 3), i, dtype=np.uint8) for i in range(1, 4 * step + 1)]
    return stream, rgbs, instance_slices


def test_warm_up_no_tracks_until_two_steps():
    step = 4
    stream, rgbs, _slices = _seeded_stream(step=step)

    assert not stream.is_warm
    # Steps 1..2*step - 1: no tracks yet.
    for k in range(1, 2 * step):
        snap = stream.step(rgbs[k - 1])
        assert snap is None, f"expected warm-up at step {k}, got {snap!r}"
        assert not stream.is_warm

    # Step 2*step: first real cotracker output.
    snap = stream.step(rgbs[2 * step - 1])
    assert snap is not None, "expected first real snapshot at step 2*S"
    assert stream.is_warm
    assert isinstance(snap, PointTracks)
    # Tracks shape: (T_so_far, N, 2). T_so_far comes from the fake; what
    # matters is N and the (..., 2) dim.
    assert snap.tracks_uv.ndim == 3
    assert snap.tracks_uv.shape[2] == 2
    assert snap.visibility.shape[:2] == snap.tracks_uv.shape[:2]
    assert snap.tracks_uv.shape[1] == snap.point_ids.shape[0]


def test_instance_slice_for_single_detection():
    stream, rgbs, slices = _seeded_stream(step=4)
    for rgb in rgbs[: 2 * 4]:
        stream.step(rgb)
    snap = stream._latest_tracks
    assert snap is not None
    # Single detection → slice covers all columns.
    assert list(slices.keys()) == [1]
    sl = snap.instance_slices[1]
    assert sl.start == 0
    assert sl.stop == snap.tracks_uv.shape[1]
    tracks_inst, vis_inst = snap.for_instance(1)
    assert tracks_inst.shape == snap.tracks_uv.shape
    assert vis_inst.shape == snap.visibility.shape


def test_two_detections_share_snapshot_via_slices():
    step = 4
    stream = CoTrackerStream(
        device="cpu",
        model_name="cotracker3_online",
        load_fn=fake_loader_factory(step=step),
    )
    H, W = 32, 32
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 1.0, dtype=np.float32)
    mask_a = _disc_mask(h=H, w=W, cy=10, cx=10, r=5)
    mask_b = _disc_mask(h=H, w=W, cy=22, cx=22, r=5)
    dets = [
        _make_detection(instance_id=1, mask=mask_a, query="a"),
        _make_detection(instance_id=2, mask=mask_b, query="b"),
    ]
    slices = stream.seed(
        view_idx=0,
        frame_idx=0,
        rgb=rgb,
        detections=dets,
        depth=depth,
        intrinsics=_identity_intrinsics(),
        max_points_per_det=12,
        min_points_per_det=4,
        erode_kernel_size=0,
    )
    assert set(slices.keys()) == {1, 2}
    assert slices[1].stop == slices[2].start, "slices must be contiguous"

    for k in range(1, 2 * step + 1):
        stream.step(np.full((H, W, 3), k, dtype=np.uint8))
    snap = stream._latest_tracks
    assert snap is not None
    a_tracks, _ = snap.for_instance(1)
    b_tracks, _ = snap.for_instance(2)
    assert a_tracks.shape[1] + b_tracks.shape[1] == snap.tracks_uv.shape[1]


def test_reset_returns_to_warm_up():
    stream, rgbs, _ = _seeded_stream(step=4)
    for rgb in rgbs[:8]:
        stream.step(rgb)
    assert stream.is_warm
    stream.reset()
    assert not stream.is_warm
    assert not stream.is_seeded
    with pytest.raises(RuntimeError, match="before seed"):
        stream.step(rgbs[0])


def test_seed_without_depth_raises():
    stream = CoTrackerStream(device="cpu", load_fn=fake_loader_factory(step=4))
    H, W = 16, 16
    det = _make_detection(instance_id=1, mask=_disc_mask(h=H, w=W, r=4))
    with pytest.raises(RuntimeError, match="requires depth"):
        stream.seed(
            view_idx=0,
            frame_idx=0,
            rgb=np.zeros((H, W, 3), dtype=np.uint8),
            detections=[det],
            depth=None,  # type: ignore[arg-type]
            intrinsics=_identity_intrinsics(),
        )


def test_seed_with_no_valid_detections_raises():
    stream = CoTrackerStream(device="cpu", load_fn=fake_loader_factory(step=4))
    H, W = 16, 16
    # Mask has 0 valid pixels → mask_to_depth_queries returns None for it,
    # leaving zero queries overall.
    empty_det = _make_detection(instance_id=1, mask=np.zeros((H, W), dtype=bool))
    with pytest.raises(RuntimeError, match="produced no queries"):
        stream.seed(
            view_idx=0,
            frame_idx=0,
            rgb=np.zeros((H, W, 3), dtype=np.uint8),
            detections=[empty_det],
            depth=np.full((H, W), 1.0, dtype=np.float32),
            intrinsics=_identity_intrinsics(),
            min_points_per_det=4,
        )

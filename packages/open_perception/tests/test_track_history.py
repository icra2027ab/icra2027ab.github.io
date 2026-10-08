"""TrackHistory accumulation across detect/track and the visualization accessors."""

from __future__ import annotations

import numpy as np
import pytest
from open_perception.api import OpenPerception
from open_perception.config import PerceptionConfig
from open_perception.types import (
    CameraInfo,
    Detection,
    InferenceOptions,
    InferenceStats,
    PerceptionInput,
    TrackedFrame,
    TrackHistory,
)


def _frame(
    *,
    frame_idx: int = 0,
    queries: list[str] | None = None,
    rgbs: list[np.ndarray] | None = None,
    detections: dict[str, list[Detection]] | None = None,
    cameras: list[CameraInfo | None] | None = None,
) -> TrackedFrame:
    rgbs = rgbs if rgbs is not None else [np.zeros((4, 4, 3), dtype=np.uint8)]
    return TrackedFrame(
        frame_idx=frame_idx,
        queries=queries or ["mug"],
        rgbs=rgbs,
        depths=[None] * len(rgbs),
        cameras=cameras if cameras is not None else [None] * len(rgbs),
        detections=detections or {},
        stage_history=[],
        stats=InferenceStats(),
    )


def test_track_history_grows_and_resets() -> None:
    h = TrackHistory()
    assert len(h) == 0
    h.append(_frame(frame_idx=0))
    h.append(_frame(frame_idx=1))
    assert len(h) == 2
    assert h[0].frame_idx == 0
    assert h[1].frame_idx == 1
    h.reset()
    assert len(h) == 0
    assert h.queries == []


def test_track_history_unions_queries_across_frames() -> None:
    h = TrackHistory()
    h.append(_frame(queries=["mug"]))
    h.append(_frame(queries=["mug", "plate"]))
    assert h.queries == ["mug", "plate"]


def test_pointclouds_by_query_view_shape_and_concatenation() -> None:
    h = TrackHistory()
    pc_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    pc_b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    mug_v0 = Detection(
        query="mug",
        mask=np.zeros((4, 4), dtype=bool),
        score=0.9,
        view_idx=0,
        pointcloud=pc_a,
    )
    mug_v1 = Detection(
        query="mug",
        mask=np.zeros((4, 4), dtype=bool),
        score=0.8,
        view_idx=1,
        pointcloud=pc_b,
    )
    h.append(
        _frame(
            queries=["mug", "plate"],
            rgbs=[
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
            ],
            detections={"mug": [mug_v0, mug_v1], "plate": []},
        )
    )
    pcs = h.pointclouds_by_query_view()
    assert len(pcs) == 1
    # 2 queries × 2 views = 4 slots
    assert len(pcs[0]) == 4
    np.testing.assert_array_equal(pcs[0][0], pc_a)  # mug, view 0
    np.testing.assert_array_equal(pcs[0][1], pc_b)  # mug, view 1
    assert pcs[0][2].shape == (0, 3)  # plate, view 0
    assert pcs[0][3].shape == (0, 3)  # plate, view 1


def test_labels_skip_camera_suffix_for_single_view() -> None:
    h = TrackHistory()
    h.append(_frame(queries=["mug", "plate"]))
    assert h.labels_by_query_view() == ["mug", "plate"]


def test_labels_include_camera_suffix_for_multi_view() -> None:
    h = TrackHistory()
    h.append(
        _frame(
            queries=["mug"],
            rgbs=[
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
            ],
        )
    )
    assert h.labels_by_query_view() == ["mug (cam0)", "mug (cam1)"]


def _patch_services(perc, sam3_stub, clip_stub, bank) -> None:
    perc._sam3._instance = sam3_stub
    perc._clip._instance = clip_stub
    if bank is not None:
        perc._reference_banks["clip"] = bank


def test_open_perception_appends_history_on_detect(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
) -> None:
    masks, scores = synthetic_sam3_output
    from _stubs import StubSam3

    perc = OpenPerception(
        PerceptionConfig(eager_load=False).order(
            ["sam3", "filter.size", "disambiguate.clip", "assign"]
        )
    )
    clip, bank = stub_clip_and_bank
    _patch_services(perc, StubSam3(masks, scores), clip, bank)

    assert len(perc.history) == 0
    perc.detect(
        PerceptionInput(
            queries=["mug", "plate"],
            rgb=synthetic_rgb,
            options=InferenceOptions(match_threshold=0.5),
        )
    )
    assert len(perc.history) == 1
    frame = perc.history[0]
    assert frame.frame_idx == 0
    assert frame.queries == ["mug", "plate"]
    assert "mug" in frame.detections


def test_record_track_history_off_keeps_history_empty(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
) -> None:
    masks, scores = synthetic_sam3_output
    from _stubs import StubSam3

    perc = OpenPerception(
        PerceptionConfig(eager_load=False).order(
            ["sam3", "filter.size", "disambiguate.clip", "assign"]
        )
    )
    clip, bank = stub_clip_and_bank
    _patch_services(perc, StubSam3(masks, scores), clip, bank)
    perc.detect(
        PerceptionInput(
            queries=["mug", "plate"],
            rgb=synthetic_rgb,
            options=InferenceOptions(match_threshold=0.5, record_track_history=False),
        )
    )
    assert len(perc.history) == 0


def test_reset_clears_history(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
) -> None:
    masks, scores = synthetic_sam3_output
    from _stubs import StubSam3

    perc = OpenPerception(
        PerceptionConfig(eager_load=False).order(
            ["sam3", "filter.size", "disambiguate.clip", "assign"]
        )
    )
    clip, bank = stub_clip_and_bank
    _patch_services(perc, StubSam3(masks, scores), clip, bank)
    perc.detect(
        PerceptionInput(
            queries=["mug", "plate"],
            rgb=synthetic_rgb,
            options=InferenceOptions(match_threshold=0.5),
        )
    )
    assert len(perc.history) == 1
    perc.reset()
    assert len(perc.history) == 0


def test_render_segmentation_video_empty_raises() -> None:
    from open_perception import render_segmentation_video

    with pytest.raises(ValueError, match="empty"):
        render_segmentation_video(TrackHistory(), "/tmp/never_written.mp4")

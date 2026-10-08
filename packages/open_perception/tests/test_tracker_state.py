"""Tracker-state + drift-policy behavior tests (no real SAM3 required).

Uses a stub `Sam3VideoTracker` that records every step call and returns a
user-controlled mask per object_id.
"""

from __future__ import annotations

import numpy as np
from open_perception.api import OpenPerception
from open_perception.config import PerceptionConfig
from open_perception.models.lazy import LazyModel
from open_perception.tracking.sam3_tracker import drift_ok
from open_perception.types import InferenceOptions


def _inject_tracker(perc: OpenPerception, view_idx: int, tracker: object) -> None:
    """Force ``perc._trackers[view_idx]`` to surface ``tracker`` without loading weights."""
    while len(perc._trackers) <= view_idx:
        perc._trackers.append(LazyModel(lambda: tracker, "stub_tracker"))
    perc._trackers[view_idx] = LazyModel(lambda: tracker, "stub_tracker")
    perc._trackers[view_idx]._instance = tracker


class _StubVideoTracker:
    """In-memory replacement for `Sam3VideoTracker`."""

    def __init__(self, next_masks_per_step: list[dict[int, np.ndarray]] | None = None):
        self._queue = list(next_masks_per_step or [])
        self._obj_masks: dict[int, np.ndarray] = {}
        self._initialized = False
        self.step_calls = 0
        self.init_calls = 0
        self.reset_calls = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, first_frame, masks_by_obj_id):
        self.init_calls += 1
        self._obj_masks = {k: v.copy() for k, v in masks_by_obj_id.items()}
        self._initialized = True
        return dict(self._obj_masks)

    def step(self, frame):
        self.step_calls += 1
        if self._queue:
            return self._queue.pop(0)
        return dict(self._obj_masks)

    def reset(self):
        self.reset_calls += 1
        self._initialized = False
        self._obj_masks = {}


def test_drift_ok_uses_anchor_iou():
    anchor = np.zeros((8, 8), dtype=bool)
    anchor[0:4, 0:4] = True
    matching = anchor.copy()
    disjoint = np.zeros_like(anchor)
    disjoint[4:, 4:] = True
    assert drift_ok(matching, anchor, 0.5) is True
    assert drift_ok(disjoint, anchor, 0.5) is False


def test_detect_seeds_state_and_initializes_video_tracker(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
):
    from _stubs import StubSam3

    masks, scores = synthetic_sam3_output
    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank

    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=1)
    perc = OpenPerception(cfg)
    perc._sam3._instance = sam3
    perc._clip._instance = clip
    perc._reference_banks["clip"] = bank
    tracker = _StubVideoTracker()
    _inject_tracker(perc, 0, tracker)

    perc.detect(["mug", "plate"], synthetic_rgb)

    assert tracker.init_calls == 1
    assert tracker.is_initialized
    # Single view → one entry in the per-view list. Hungarian + 2 queries
    # → two seeded obj_ids.
    assert len(perc._obj_id_to_query) == 1
    assert len(perc._obj_id_to_query[0]) == 2


def test_track_steps_video_tracker_once_per_frame(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
):
    from _stubs import StubSam3

    masks, scores = synthetic_sam3_output
    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank

    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=1)
    perc = OpenPerception(cfg)
    perc._sam3._instance = sam3
    perc._clip._instance = clip
    perc._reference_banks["clip"] = bank
    tracker = _StubVideoTracker()
    _inject_tracker(perc, 0, tracker)

    perc.detect(["mug", "plate"], synthetic_rgb)
    out = perc.track(synthetic_rgb)
    assert tracker.step_calls == 1
    # Each tracked query gets a detection (from the tracker stub return).
    assert len(out.detections["mug"]) == 1
    assert len(out.detections["plate"]) == 1


def test_track_auto_recovery_redetects_drifted_queries(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
):
    from _stubs import StubSam3

    masks, scores = synthetic_sam3_output
    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=1)
    perc = OpenPerception(cfg)
    perc._sam3._instance = sam3
    perc._clip._instance = clip
    perc._reference_banks["clip"] = bank

    tracker = _StubVideoTracker()
    _inject_tracker(perc, 0, tracker)
    perc.detect(["mug", "plate"], synthetic_rgb)
    # Build the drift queue using whatever obj_ids detect() actually issued.
    H, W = synthetic_rgb.shape[:2]
    drifted = {oid: np.zeros((H, W), dtype=bool) for oid in perc._obj_id_to_query[0]}
    tracker._queue = [drifted]
    out = perc.track(
        synthetic_rgb,
        options=InferenceOptions(drift_policy="auto", drift_iou_threshold=0.3),
    )
    assert sorted(out.stats.recoveries) == ["mug", "plate"]
    # Auto-recovery refreshed both queries from SAM3, so detections are non-empty.
    assert len(out.detections["mug"]) == 1
    assert len(out.detections["plate"]) == 1


def test_track_surface_policy_flags_drifted_without_recovery(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
):
    from _stubs import StubSam3

    masks, scores = synthetic_sam3_output
    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=1)
    perc = OpenPerception(cfg)
    perc._sam3._instance = sam3
    perc._clip._instance = clip
    perc._reference_banks["clip"] = bank

    tracker = _StubVideoTracker()
    _inject_tracker(perc, 0, tracker)
    perc.detect(["mug"], synthetic_rgb)
    H, W = synthetic_rgb.shape[:2]
    drifted = {oid: np.zeros((H, W), dtype=bool) for oid in perc._obj_id_to_query[0]}
    tracker._queue = [drifted]
    out = perc.track(
        synthetic_rgb,
        options=InferenceOptions(drift_policy="surface", drift_iou_threshold=0.3),
    )
    assert out.stats.recoveries == []
    assert len(out.detections["mug"]) == 1
    assert out.detections["mug"][0].drifted is True


def test_tracker_backend_switch_picks_sam2_class():
    """`tracker_backend="sam2"` routes to Sam2VideoTracker without loading weights."""
    from open_perception.tracking.sam2_tracker import Sam2VideoTracker
    from open_perception.tracking.sam3_tracker import Sam3VideoTracker

    cfg_sam3 = PerceptionConfig(tracker_backend="sam3", device="cpu", eager_load=False)
    cfg_sam2 = PerceptionConfig(tracker_backend="sam2", device="cpu", eager_load=False)
    perc_sam3 = OpenPerception(cfg_sam3)
    perc_sam2 = OpenPerception(cfg_sam2)
    # `_build_tracker` is the factory we want to route without instantiating HF weights.
    # Monkey-patch each concrete subclass so we can observe which one is picked.
    called: list[str] = []

    def make_fake(name: str):
        def fake_init(self, model_id, device="cuda"):
            called.append(name)
            self.model_id = model_id
            self.device = device
            self.model = None
            self.processor = None
            self._session = None
            self._frame_idx = -1
            self._obj_ids = []
            self._log_tag = name

        return fake_init

    orig_sam3_init = Sam3VideoTracker.__init__
    orig_sam2_init = Sam2VideoTracker.__init__
    Sam3VideoTracker.__init__ = make_fake("sam3")  # type: ignore[assignment]
    Sam2VideoTracker.__init__ = make_fake("sam2")  # type: ignore[assignment]
    try:
        t3 = perc_sam3._build_tracker()
        t2 = perc_sam2._build_tracker()
    finally:
        Sam3VideoTracker.__init__ = orig_sam3_init  # type: ignore[assignment]
        Sam2VideoTracker.__init__ = orig_sam2_init  # type: ignore[assignment]

    assert isinstance(t3, Sam3VideoTracker)
    assert isinstance(t2, Sam2VideoTracker)
    assert called == ["sam3", "sam2"]
    assert t3.model_id == cfg_sam3.sam3_model
    assert t2.model_id == cfg_sam2.sam2_model


def test_stream_first_frame_is_detect_rest_track(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
):
    from _stubs import StubSam3

    masks, scores = synthetic_sam3_output
    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=1)
    perc = OpenPerception(cfg)
    perc._sam3._instance = sam3
    perc._clip._instance = clip
    perc._reference_banks["clip"] = bank
    tracker = _StubVideoTracker()
    _inject_tracker(perc, 0, tracker)

    frames = [synthetic_rgb, synthetic_rgb, synthetic_rgb]
    outputs = list(perc.stream(["mug", "plate"], frames))
    assert len(outputs) == 3
    assert tracker.init_calls == 1  # only the first frame initializes
    assert tracker.step_calls == 2  # the remaining two frames step

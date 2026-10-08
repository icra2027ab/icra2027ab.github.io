"""End-to-end smoke test through OpenPerception with stubbed models."""

from __future__ import annotations

from open_perception.api import OpenPerception
from open_perception.config import PerceptionConfig
from open_perception.types import InferenceOptions, PerceptionInput


def _patch_services(perc: OpenPerception, sam3_stub, clip_stub, reference_bank):
    """Redirect the lazy-model holders to already-built stubs."""
    perc._sam3._instance = sam3_stub
    perc._clip._instance = clip_stub
    if reference_bank is not None:
        perc._reference_banks["clip"] = reference_bank


def test_detect_multi_instance_above_threshold(
    synthetic_rgb,
    synthetic_sam3_output,
    stub_clip_and_bank,
):
    masks, scores = synthetic_sam3_output
    from _stubs import StubSam3

    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank

    cfg = PerceptionConfig(eager_load=False)
    # Skip overlap filter so all candidates reach assign.
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=-1)
    perc = OpenPerception(cfg)
    _patch_services(perc, sam3, clip, bank)

    out = perc.detect(
        PerceptionInput(
            queries=["mug", "plate"],
            rgb=synthetic_rgb,
            options=InferenceOptions(match_threshold=0.5),
        )
    )
    assert "mug" in out.detections
    assert "plate" in out.detections
    # mug and plate each match exactly one crop above threshold (third crop
    # has zero similarity to both).
    assert len(out.detections["mug"]) == 1
    assert len(out.detections["plate"]) == 1
    assert out.stats.stages_run[0] == "sam3"
    assert "assign" in out.stats.stages_run
    assert out.stats.n_detections_per_query == {"mug": 1, "plate": 1}


def test_detect_hungarian_compat_mode(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
):
    """Top-1 Hungarian reproduces the original eval's 1-mask-per-query behaviour."""
    masks, scores = synthetic_sam3_output
    from _stubs import StubSam3

    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank

    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=1)
    perc = OpenPerception(cfg)
    _patch_services(perc, sam3, clip, bank)
    out = perc.detect(PerceptionInput(queries=["mug", "plate"], rgb=synthetic_rgb))
    assert len(out.detections["mug"]) == 1
    assert len(out.detections["plate"]) == 1
    # Tracker state seeded for both queries.
    assert "mug" in perc._tracker_state
    assert "plate" in perc._tracker_state


def test_reset_clears_tracker_state(
    synthetic_rgb, synthetic_sam3_output, stub_clip_and_bank
):
    masks, scores = synthetic_sam3_output
    from _stubs import StubSam3

    sam3 = StubSam3(masks, scores)
    clip, bank = stub_clip_and_bank
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam3", "filter.size", "disambiguate.clip", "assign"])
    cfg.update("assign", top_k=1)
    perc = OpenPerception(cfg)
    _patch_services(perc, sam3, clip, bank)
    perc.detect(PerceptionInput(queries=["mug", "plate"], rgb=synthetic_rgb))
    assert "mug" in perc._tracker_state
    perc.reset(queries=["mug"])
    assert "mug" not in perc._tracker_state
    assert "plate" in perc._tracker_state
    perc.reset()
    assert "plate" not in perc._tracker_state

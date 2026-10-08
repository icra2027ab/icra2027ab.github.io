from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from open_perception.config import PerceptionConfig
from open_perception.context import DetectionContext
from open_perception.pipeline import Pipeline
from open_perception.stages.base import REGISTRY
from open_perception.types import InferenceOptions, InferenceStats


@dataclass
class _RecorderStage:
    token: str
    name: str = "recorder"

    def __call__(self, ctx: DetectionContext) -> DetectionContext:
        ctx.crop_features.setdefault("trace", np.array([]))
        trace = ctx.crop_features.get("trace_list", [])
        trace.append(self.token)
        ctx.crop_features["trace_list"] = trace
        return ctx


def _register_recorder(token: str, name: str = "recorder"):
    def _factory(services, **params):
        return _RecorderStage(token=token, name=name)

    REGISTRY.add(name, _factory)


def test_stage_order_honors_config(synthetic_rgb):
    _register_recorder("A", "recorder_a")
    _register_recorder("B", "recorder_b")
    _register_recorder("C", "recorder_c")
    cfg = PerceptionConfig().order(["recorder_b", "recorder_a", "recorder_c"])
    pipe = Pipeline(cfg, services=object())
    ctx = DetectionContext(
        queries=["x"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        stats=InferenceStats(),
    )
    pipe.run(ctx)
    assert ctx.crop_features["trace_list"] == ["B", "A", "C"]
    assert ctx.stats.stages_run == ["recorder_b", "recorder_a", "recorder_c"]


def test_omitted_stage_is_skipped(synthetic_rgb):
    _register_recorder("X", "recorder_x")
    _register_recorder("Y", "recorder_y")
    # recorder_x is not in the order, so it must not run.
    cfg = PerceptionConfig().order(["recorder_y"])
    pipe = Pipeline(cfg, services=object())
    ctx = DetectionContext(
        queries=[],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        stats=InferenceStats(),
    )
    pipe.run(ctx)
    assert ctx.crop_features["trace_list"] == ["Y"]
    assert ctx.stats.stages_run == ["recorder_y"]


def test_duplicate_stage_names_get_index_suffix(synthetic_rgb):
    _register_recorder("R", "recorder_r")
    cfg = PerceptionConfig().order(["recorder_r", "recorder_r"])
    pipe = Pipeline(cfg, services=object())
    ctx = DetectionContext(
        queries=[],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(),
        stats=InferenceStats(),
    )
    pipe.run(ctx)
    assert ctx.stats.stages_run == ["recorder_r#0", "recorder_r#1"]


def test_config_order_and_update_chain():
    cfg = PerceptionConfig()
    cfg.order(["sam3", "filter.overlap"])
    assert cfg._order == ["sam3", "filter.overlap"]
    cfg.update("sam3", prompts=["mug"])
    assert cfg._params["sam3"]["prompts"] == ["mug"]


def test_config_rejects_unknown_stage():
    import pytest

    cfg = PerceptionConfig().order(["not_a_real_stage"])
    with pytest.raises(ValueError, match="unknown stage"):
        cfg.validate(REGISTRY.names())


def test_config_summary_shows_pipeline_and_params():
    cfg = PerceptionConfig().order(["sam3", "filter.overlap", "assign"])
    cfg.update("assign", top_k=1)
    text = cfg.summary
    assert "sam3" in text
    assert "assign" in text
    assert "top_k" in text

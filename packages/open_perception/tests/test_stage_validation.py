"""PerceptionConfig.validate() — warn on unmet stage dependencies."""

from __future__ import annotations

import logging

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger
from open_perception.config import PerceptionConfig
from open_perception.stages.base import REGISTRY


@pytest.fixture
def capture_loguru(caplog: LogCaptureFixture):
    """Bridge loguru emissions into pytest's caplog records."""

    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level="DEBUG",
    )
    yield caplog
    logger.remove(handler_id)


def test_assign_before_disambiguate_warns_about_missing_scores(
    capture_loguru: LogCaptureFixture,
) -> None:
    cfg = PerceptionConfig().order(["sam3", "filter.size", "assign"])
    with capture_loguru.at_level(logging.WARNING):
        cfg.validate(REGISTRY.names())
    text = capture_loguru.text
    assert "assign" in text
    assert "scores" in text


def test_reconstruct_without_depth_input_warns(
    capture_loguru: LogCaptureFixture,
) -> None:
    cfg = PerceptionConfig().order(["sam3", "reconstruct"])
    with capture_loguru.at_level(logging.WARNING):
        cfg.validate(REGISTRY.names(), input_fields={"rgb"})
    text = capture_loguru.text
    assert "reconstruct" in text
    assert "depth" in text


def test_default_fuse_views_independent_does_not_warn(
    capture_loguru: LogCaptureFixture,
) -> None:
    cfg = PerceptionConfig().order(["sam3", "fuse_views"])
    with capture_loguru.at_level(logging.WARNING):
        cfg.validate(
            REGISTRY.names(),
            input_fields={"rgb", "camera.intrinsics"},
        )
    # Default mode is "independent" — no requires beyond what's available.
    assert "fuse_views" not in capture_loguru.text


def test_primary_guided_without_reconstruct_warns(
    capture_loguru: LogCaptureFixture,
) -> None:
    cfg = PerceptionConfig().order(["sam3", "fuse_views"])
    cfg.update("fuse_views", mode="primary_guided")
    with capture_loguru.at_level(logging.WARNING):
        cfg.validate(
            REGISTRY.names(),
            input_fields={"rgb", "camera.intrinsics"},
        )
    text = capture_loguru.text
    assert "fuse_views" in text
    assert "detections.pointcloud" in text


def test_primary_guided_with_reconstruct_does_not_warn(
    capture_loguru: LogCaptureFixture,
) -> None:
    cfg = PerceptionConfig().order(["sam3", "filter.size", "reconstruct", "fuse_views"])
    cfg.update("fuse_views", mode="primary_guided")
    with capture_loguru.at_level(logging.WARNING):
        cfg.validate(
            REGISTRY.names(),
            input_fields={"rgb", "depth", "camera.intrinsics"},
        )
    assert "fuse_views" not in capture_loguru.text


def test_unknown_stage_still_raises() -> None:
    cfg = PerceptionConfig().order(["not_a_real_stage"])
    with pytest.raises(ValueError, match="unknown stage"):
        cfg.validate(REGISTRY.names())

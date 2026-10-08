"""Open-vocabulary multi-object segmentation and tracking."""

from __future__ import annotations

from .api import OpenPerception
from .async_api import AsyncOpenPerception
from .config import PerceptionConfig
from .context import DetectionContext
from .encoders.reference import ReferenceBank
from .stages.base import Stage
from .tracking.cotracker_stream import PointTracks
from .types import (
    CameraInfo,
    Detection,
    InferenceOptions,
    InferenceStats,
    PerceptionInput,
    PerceptionOutput,
    PipelineSnapshot,
    TrackedFrame,
    TrackHistory,
)
from .visualization import (
    DEFAULT_COLORS,
    draw_label,
    overlay_detections,
    overlay_masks,
    render_debug_image,
    render_disambiguation_panel,
    render_history_gif,
    render_pointcloud_orbit_gif,
    render_segmentation_video,
)

__all__ = [
    "DEFAULT_COLORS",
    "AsyncOpenPerception",
    "CameraInfo",
    "Detection",
    "DetectionContext",
    "InferenceOptions",
    "InferenceStats",
    "OpenPerception",
    "PerceptionConfig",
    "PerceptionInput",
    "PerceptionOutput",
    "PipelineSnapshot",
    "PointTracks",
    "ReferenceBank",
    "Stage",
    "TrackHistory",
    "TrackedFrame",
    "draw_label",
    "overlay_detections",
    "overlay_masks",
    "render_debug_image",
    "render_disambiguation_panel",
    "render_history_gif",
    "render_pointcloud_orbit_gif",
    "render_segmentation_video",
]

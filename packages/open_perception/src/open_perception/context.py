"""Mutable per-view state threaded through pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import (
    CameraInfo,
    Detection,
    InferenceOptions,
    InferenceStats,
    PipelineSnapshot,
    ProjectedReference,
)


@dataclass
class DetectionContext:
    queries: list[str]
    rgb: np.ndarray
    depth: np.ndarray | None
    camera: CameraInfo | None
    options: InferenceOptions
    view_idx: int = 0
    frame_idx: int = 0
    instances: np.ndarray | None = None
    mask_scores: np.ndarray | None = None
    crops: list[np.ndarray] = field(default_factory=list)
    bboxes: np.ndarray | None = None
    # label/query -> (N, 4) xyxy *pixel* boxes proposed by a detector stage
    # (e.g. detect.gemini). Consumed by sam2/sam3 bbox mode when their static
    # ``prompts`` dict is empty. Pixel xyxy to match Sam2Wrapper.segment_with_boxes.
    box_prompts: dict[str, np.ndarray] | None = None
    # label/query -> (N, 2) xy *pixel* points proposed by a detector stage
    # (e.g. detect.gemini with proposal="point"). One row per instance; consumed
    # by sam2/sam3 point mode when their static ``prompts`` dict is empty.
    point_prompts: dict[str, np.ndarray] | None = None
    crop_features: dict[str, np.ndarray] = field(default_factory=dict)
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    detections: dict[str, list[Detection]] = field(default_factory=dict)
    stats: InferenceStats = field(default_factory=InferenceStats)
    history: list[PipelineSnapshot] = field(default_factory=list)
    projected_refs: list[ProjectedReference] = field(default_factory=list)

    @property
    def image_hw(self) -> tuple[int, int]:
        return int(self.rgb.shape[0]), int(self.rgb.shape[1])

    @property
    def num_instances(self) -> int:
        return 0 if self.instances is None else int(self.instances.shape[0])

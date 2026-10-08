"""Input / output / stats dataclasses for open_perception."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .encoders.reference import ReferenceBank
    from .tracking.cotracker_stream import PointTracks


@dataclass(frozen=True)
class CameraInfo:
    intrinsics: np.ndarray
    extrinsics: np.ndarray | None = None
    name: str = "camera_0"


@dataclass(frozen=True)
class InferenceOptions:
    export_embeddings: bool = False
    filter_overlap: bool | None = None
    overlap_iou_threshold: float = 0.5
    match_threshold: float = 0.25
    # Default to the single best match per query — most callers expect
    # one instance per name. Set to None for unlimited multi-instance.
    top_k_per_query: int | None = 1
    max_instances: int | None = None
    streaming: bool = False
    drift_policy: Literal["auto", "surface", "off"] = "surface"
    drift_iou_threshold: float = 0.2
    return_candidates: bool = False
    vlm_pre_enabled: bool = False
    # When True, the pipeline records a `PipelineSnapshot` after every stage
    # runs so callers can inspect intermediate mask state (or render a GIF
    # via `render_history_gif`). Snapshots shallow-copy Detections + copy the
    # instances array — cheap relative to inference itself. Set to False to
    # skip capture if you want to trim the per-call memory overhead.
    record_history: bool = True
    # When True, OpenPerception accumulates a `TrackedFrame` on its `history`
    # attribute after every detect/track call — gives downstream visualization
    # helpers (`render_pointcloud_orbit_gif`, `render_segmentation_video`)
    # the full per-frame state without manual aggregation. Disable for long
    # video runs where retaining per-frame RGB+depth would blow memory.
    record_track_history: bool = True


@dataclass(frozen=True)
class PerceptionInput:
    queries: str | list[str]
    rgb: np.ndarray | list[np.ndarray]
    depth: np.ndarray | list[np.ndarray] | None = None
    cameras: CameraInfo | list[CameraInfo] | None = None
    reference: str | dict[str, Any] | "ReferenceBank" | None = None
    options: InferenceOptions | None = None

    def normalized(self) -> "NormalizedInput":
        """Expand scalars to lists and validate shapes."""
        queries = (
            [self.queries] if isinstance(self.queries, str) else list(self.queries)
        )

        rgbs: list[np.ndarray]
        if isinstance(self.rgb, np.ndarray):
            if self.rgb.ndim == 4:
                rgbs = [self.rgb[i] for i in range(self.rgb.shape[0])]
            elif self.rgb.ndim == 3:
                rgbs = [self.rgb]
            else:
                raise ValueError(
                    f"rgb must be (H,W,3) or (V,H,W,3), got shape {self.rgb.shape}"
                )
        else:
            rgbs = list(self.rgb)
        for i, r in enumerate(rgbs):
            if r.ndim != 3 or r.shape[-1] != 3:
                raise ValueError(f"rgb[{i}] must be (H,W,3), got {r.shape}")

        depths: list[np.ndarray | None]
        if self.depth is None:
            depths = [None] * len(rgbs)
        elif isinstance(self.depth, np.ndarray):
            d = self.depth
            if d.ndim == 4 and d.shape[-1] == 1:
                d = d[..., 0]
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            if d.ndim == 3 and d.shape[0] == len(rgbs):
                depths = [d[i] for i in range(d.shape[0])]
            elif d.ndim == 2:
                depths = [d]
            else:
                raise ValueError(
                    f"depth shape {self.depth.shape} incompatible with {len(rgbs)} views"
                )
        else:
            depths = [
                dd[..., 0]
                if isinstance(dd, np.ndarray) and dd.ndim == 3 and dd.shape[-1] == 1
                else dd
                for dd in self.depth
            ]
        if len(depths) != len(rgbs):
            raise ValueError(f"depth has {len(depths)} views but rgb has {len(rgbs)}")

        cameras: list[CameraInfo | None]
        if self.cameras is None:
            cameras = [None] * len(rgbs)
        elif isinstance(self.cameras, CameraInfo):
            cameras = [self.cameras]
        else:
            cameras = list(self.cameras)
        if len(cameras) != len(rgbs):
            raise ValueError(
                f"cameras has {len(cameras)} views but rgb has {len(rgbs)}"
            )

        opts = self.options if self.options is not None else InferenceOptions()
        return NormalizedInput(
            queries=queries,
            rgbs=rgbs,
            depths=depths,
            cameras=cameras,
            reference=self.reference,
            options=opts,
        )


@dataclass(frozen=True)
class NormalizedInput:
    queries: list[str]
    rgbs: list[np.ndarray]
    depths: list[np.ndarray | None]
    cameras: list[CameraInfo | None]
    reference: str | dict[str, Any] | "ReferenceBank" | None
    options: InferenceOptions


@dataclass
class Detection:
    query: str
    mask: np.ndarray
    score: float
    view_idx: int = 0
    bbox: np.ndarray | None = None
    instance_id: int | None = None
    drifted: bool = False
    pointcloud: np.ndarray | None = None
    embeddings: dict[str, np.ndarray] | None = None
    # Per-detection 2D point trajectories from CoTracker, when point tracking
    # is enabled. None during the cotracker warm-up window (the first
    # ``2*model.step - 1`` frames after ``detect()``) and on views without
    # point tracking. The same ``PointTracks`` object is shared across all
    # detections in one view; per-instance columns are accessible via
    # ``det.point_tracks.for_instance(det.instance_id)``.
    point_tracks: "PointTracks | None" = None
    # Internal index into DetectionContext.instances / crops — used by the
    # encode stage to look up embeddings. Not part of the public contract.
    source_index: int | None = None


@dataclass
class InferenceStats:
    timings_ms: dict[str, float] = field(default_factory=dict)
    n_candidates_per_view: list[int] = field(default_factory=list)
    n_after_overlap_filter: list[int] = field(default_factory=list)
    n_after_size_filter: list[int] = field(default_factory=list)
    n_detections_per_query: dict[str, int] = field(default_factory=dict)
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    recoveries: list[str] = field(default_factory=list)
    model_calls: dict[str, int] = field(default_factory=dict)

    def add_timing(self, stage: str, ms: float) -> None:
        self.timings_ms[stage] = self.timings_ms.get(stage, 0.0) + ms

    def increment_model_call(self, model: str) -> None:
        self.model_calls[model] = self.model_calls.get(model, 0) + 1


@dataclass
class ProjectedReference:
    """A main-camera 3D detection reprojected into a secondary view.

    Emitted by ``fuse_views`` in ``primary_guided`` mode so callers can
    visualize *which pixels in this view's frame* were used to align (or
    seed) the secondary's tracker. Carries the dilated footprint mask
    used for IoU matching plus the matching outcome (``iou`` and
    ``matched``) so renderers can colour aligned vs. seeded references
    differently.
    """

    query: str
    instance_id: int
    source_view_idx: int
    uv: np.ndarray  # (M, 2) int32 — pixels in the *target* view's frame
    footprint: np.ndarray  # (H, W) bool — dilated footprint used for IoU
    iou: float = 0.0
    matched: bool = False


@dataclass
class PipelineSnapshot:
    """A snapshot of mask state captured after a pipeline stage runs.

    Only populated when ``InferenceOptions.record_history`` is True. The
    ``detections`` dict and ``instances`` array are deep-copied so later
    stages can mutate their own state without affecting past snapshots.
    """

    stage: str
    view_idx: int = 0
    frame_idx: int = 0
    queries: list[str] = field(default_factory=list)
    detections: dict[str, list[Detection]] = field(default_factory=dict)
    instances: np.ndarray | None = None
    mask_scores: np.ndarray | None = None
    # Per-source score matrices keyed like ``"dino"``, ``"clip"``,
    # ``"clip_text_fallback"``. Each matrix is shape (N_instances,
    # N_queries) with columns aligned to ``queries``. Captured so debug
    # tooling can show *why* a detection got its label.
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    projected_refs: list[ProjectedReference] = field(default_factory=list)
    # Geometric prompts proposed by a detector stage (e.g. detect.gemini) as of
    # this snapshot: query -> (N, 4) xyxy pixel boxes / (N, 2) xy pixel points.
    # Captured so history renderers can draw the intermediate box/point
    # proposals on the stage tile (they're not masks, so ``instances`` /
    # ``detections`` don't cover them).
    box_prompts: dict[str, np.ndarray] = field(default_factory=dict)
    point_prompts: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PerceptionOutput:
    detections: dict[str, list[Detection]]
    stats: InferenceStats
    history: list[PipelineSnapshot] = field(default_factory=list)


@dataclass
class TrackedFrame:
    """One frame's worth of multi-view state, captured at the end of detect/track.

    Stores everything needed to reconstruct what happened on this frame
    without keeping the original `OpenPerception` call's locals around:
    per-view RGB/depth/cameras, the merged detections (each tagged with
    `view_idx`), the per-stage snapshots from the pipeline, and the
    timing/model-call stats.
    """

    frame_idx: int
    queries: list[str]
    rgbs: list[np.ndarray]
    depths: list[np.ndarray | None]
    cameras: list[CameraInfo | None]
    detections: dict[str, list[Detection]]
    stage_history: list[PipelineSnapshot] = field(default_factory=list)
    stats: InferenceStats = field(default_factory=InferenceStats)
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def n_views(self) -> int:
        return len(self.rgbs)

    def detections_per_view(self) -> list[dict[str, list[Detection]]]:
        """Bucket merged detections back into per-view dicts."""
        per_view: list[dict[str, list[Detection]]] = [{} for _ in range(self.n_views)]
        for q, dets in self.detections.items():
            for det in dets:
                if 0 <= det.view_idx < self.n_views:
                    per_view[det.view_idx].setdefault(q, []).append(det)
        return per_view


@dataclass
class TrackHistory:
    """Accumulator owned by `OpenPerception` — grows on every detect/track call.

    The container that visualization helpers consume directly:
    `render_pointcloud_orbit_gif(perc.history, ...)` reads point clouds,
    cameras, and labels off this object instead of asking the caller to
    hand-build them. `reset()` is called automatically by
    `OpenPerception.reset()`.
    """

    queries: list[str] = field(default_factory=list)
    n_views: int = 0
    frames: list[TrackedFrame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[TrackedFrame]:
        return iter(self.frames)

    def __getitem__(self, i: int) -> TrackedFrame:
        return self.frames[i]

    def append(self, frame: TrackedFrame) -> None:
        self.frames.append(frame)
        self.n_views = max(self.n_views, frame.n_views)
        for q in frame.queries:
            if q not in self.queries:
                self.queries.append(q)

    def reset(self) -> None:
        self.frames.clear()
        self.queries.clear()
        self.n_views = 0

    @property
    def n_views_effective(self) -> int:
        """Number of view slots used by accessor outputs (always at least 1)."""
        return max(self.n_views, 1)

    def cameras_per_frame(self) -> list[list[CameraInfo | None]]:
        return [list(f.cameras) for f in self.frames]

    def labels_by_query_view(self) -> list[str]:
        """Per-(query, view) label string used by visualization legends.

        Layout matches `pointclouds_by_query_view`: the outer axis is
        queries, the inner is views. With one view, the camera suffix is
        omitted to keep single-camera examples clean.
        """
        n_v = self.n_views_effective
        if n_v == 1:
            return list(self.queries)
        return [f"{q} (cam{v})" for q in self.queries for v in range(n_v)]

    def pointclouds_by_query_view(self) -> list[list[np.ndarray]]:
        """Per-frame point clouds bucketed by (query, view).

        Returns a list of length ``len(self)``. Each inner list has length
        ``len(queries) * n_views_effective`` — matching `labels_by_query_view`
        — with point clouds concatenated within each (query, view) slot.
        Empty slots return zero-length ``(0, 3)`` arrays.
        """
        n_v = self.n_views_effective
        out: list[list[np.ndarray]] = []
        for f in self.frames:
            step: list[np.ndarray] = []
            for q in self.queries:
                for v in range(n_v):
                    pcs = [
                        det.pointcloud
                        for det in f.detections.get(q, [])
                        if det.view_idx == v
                        and det.pointcloud is not None
                        and det.pointcloud.size > 0
                    ]
                    step.append(
                        np.concatenate(pcs, axis=0)
                        if pcs
                        else np.zeros((0, 3), dtype=np.float32)
                    )
            out.append(step)
        return out

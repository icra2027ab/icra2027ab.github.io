"""Modular CoTracker wrapper for tracking segmented object points through time.

Supports both offline (batch) and online (streaming) modes.

Usage (offline)::

    tracker = OfflineObjectTracker(device="cuda")
    tracker.initialize(video)           # video: (T, C, H, W) torch float
    pred_tracks, pred_vis = tracker.track(
        queries=queries,                 # (N, 3) [t, x, y]
    )

Usage (online)::

    tracker = OnlineObjectTracker(device="cuda")
    tracker.initialize(first_chunk, queries=queries)
    for chunk in chunks:
        tracks, vis = tracker.step(chunk)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TrackedPoints:
    """Result of a tracking step.

    Attributes:
        tracks:     ``(T, N, 2)`` float32 array of (x, y) pixel coordinates.
        visibility: ``(T, N)`` bool array; True = visible.
        point_ids:  ``(N,)`` int array of unique point identifiers.
    """

    tracks: np.ndarray  # (T, N, 2)
    visibility: np.ndarray  # (T, N)
    point_ids: np.ndarray  # (N,)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ObjectTracker(ABC):
    """Abstract base for offline and online CoTracker wrappers."""

    def __init__(self, device: str = "cuda", grid_size: int = 0) -> None:
        self.device = device
        self.grid_size = grid_size
        self._model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(
        self,
        video: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
    ) -> None:
        """Set up the tracker for a new sequence.

        Args:
            video:   ``(1, T, C, H, W)`` or ``(T, C, H, W)`` float tensor.
            queries: ``(N, 3)`` float tensor of [t, x, y] seed points, or
                     ``None`` to rely on *grid_size* for auto-seeding.
        """

    @abstractmethod
    def get_all_tracks(self) -> TrackedPoints:
        """Return tracks for the full sequence processed so far."""


# ---------------------------------------------------------------------------
# Helper: segmentation mask → query points
# ---------------------------------------------------------------------------


def mask_to_queries(
    mask: np.ndarray,
    frame_index: int = 0,
    max_points: int = 200,
    min_points: int = 10,
    erode_kernel_size: int = 5,
    rng: np.random.Generator | None = None,
) -> Optional[np.ndarray]:
    """Sample (x, y) positions from a binary mask and return CoTracker queries.

    Args:
        mask:        ``(H, W)`` bool array where True = object pixels.
        frame_index: The frame index ``t`` to embed in the query (CoTracker
                     convention: column 0 = frame index).
        max_points:  Upper bound on sampled points (random sub-sample if
                     the mask has more pixels).
        min_points:  If fewer pixels are found, return ``None``.
        erode_kernel_size: If > 0, erode the mask by this kernel size to avoid
                           selecting boundary pixels.
        rng:         Scoped numpy random generator.  Falls back to a
                     default ``default_rng(0)`` when ``None``.

    Returns:
        ``(N, 3)`` float32 array ``[t, x, y]``, or ``None`` if the mask
        has fewer than *min_points* pixels.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if erode_kernel_size > 0:
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    ys, xs = np.where(mask)
    if len(xs) < min_points:
        return None

    if len(xs) > max_points:
        idx = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]

    t_col = np.full(len(xs), frame_index, dtype=np.float32)
    queries = np.stack([t_col, xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    return queries


# ---------------------------------------------------------------------------
# Offline tracker
# ---------------------------------------------------------------------------


class OfflineObjectTracker(ObjectTracker):
    """Offline CoTracker3 wrapper – processes the full video in one pass."""

    def __init__(self, device: str = "cuda", grid_size: int = 0) -> None:
        super().__init__(device=device, grid_size=grid_size)
        print("[OfflineObjectTracker] Loading CoTracker3 offline model…")
        self._model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline"
        ).to(device)

    def initialize(
        self,
        video: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
    ) -> None:
        """Run offline tracking on the full video.

        Args:
            video:   ``(1, T, C, H, W)`` or ``(T, C, H, W)`` float tensor.
            queries: ``(1, N, 3)`` or ``(N, 3)`` [t, x, y] float tensor.
        """
        # Ensure batch dim
        if video.ndim == 4:
            video = video.unsqueeze(0)  # (1, T, C, H, W)
        video = video.to(self.device)

        if queries is not None:
            if queries.ndim == 2:
                queries = queries.unsqueeze(0)  # (1, N, 3)
            queries = queries.to(self.device)

        assert self._model is not None
        with torch.no_grad():
            if queries is not None:
                pred_tracks, pred_vis = self._model(video, queries=queries)
            else:
                pred_tracks, pred_vis = self._model(video, grid_size=self.grid_size)

        # Store as numpy for easy downstream use
        # pred_tracks: (1, T, N, 2)
        # pred_vis:    (1, T, N, 1) or (1, T, N)
        self._tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
        vis = pred_vis[0].cpu().numpy()
        if vis.ndim == 3:
            vis = vis[..., 0]
        self._visibility = vis.astype(bool)  # (T, N)
        N = self._tracks.shape[1]
        self._point_ids = np.arange(N)

    def get_all_tracks(self) -> TrackedPoints:
        return TrackedPoints(
            tracks=self._tracks,
            visibility=self._visibility,
            point_ids=self._point_ids,
        )


# ---------------------------------------------------------------------------
# Online tracker
# ---------------------------------------------------------------------------


class OnlineObjectTracker(ObjectTracker):
    """Online CoTracker3 wrapper – processes the video in streaming chunks.

    Points can be added incrementally (e.g., when new segmentation pixels
    become visible).  This matches the API that will be used in the rollout
    analysis pipeline.
    """

    def __init__(self, device: str = "cuda", grid_size: int = 0) -> None:
        super().__init__(device=device, grid_size=grid_size)
        print("[OnlineObjectTracker] Loading CoTracker3 online model…")
        self._model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)

        self._all_tracks: list[np.ndarray] = []  # list of (step, N, 2)
        self._all_visibility: list[np.ndarray] = []  # list of (step, N)
        self._point_ids: Optional[np.ndarray] = None
        self._n_points: int = 0
        self._initialized: bool = False
        self._video_buffer: Optional[torch.Tensor] = None  # (1, T, C, H, W)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(
        self,
        video: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the online tracker with the first video chunk.

        This sends the first chunk to the model with ``is_first_step=True``.

        Args:
            video:   ``(1, T_total, C, H, W)`` or ``(T_total, C, H, W)`` tensor
                     representing the **entire** video (or at least enough frames
                     to contain all query start frames).
            queries: ``(N, 3)`` or ``(1, N, 3)`` [t, x, y] float tensor.
        """
        if video.ndim == 4:
            video = video.unsqueeze(0)
        video = video.to(self.device)
        self._video_buffer = video

        if queries is not None:
            if queries.ndim == 2:
                queries = queries.unsqueeze(0)
            queries = queries.to(self.device)
            self._n_points = queries.shape[1]
            self._point_ids = np.arange(self._n_points)

        # First step: feed the first window
        assert self._model is not None
        self._model(
            video_chunk=video,
            is_first_step=True,
            grid_size=self.grid_size if queries is None else 0,
            queries=queries,
        )
        self._initialized = True
        self._all_tracks = []
        self._all_visibility = []

    def step(self, video_chunk: torch.Tensor) -> TrackedPoints:
        """Process a video chunk and return tracks for the processed frames.

        Args:
            video_chunk: ``(1, step*2, C, H, W)`` or ``(step*2, C, H, W)``
                         float tensor (two-step sliding window).

        Returns:
            :class:`TrackedPoints` for the frames processed so far.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before step()")

        if video_chunk.ndim == 4:
            video_chunk = video_chunk.unsqueeze(0)
        video_chunk = video_chunk.to(self.device)

        assert self._model is not None
        with torch.no_grad():
            pred_tracks, pred_vis = self._model(video_chunk=video_chunk)

        # pred_tracks: (1, step, N, 2) or (1, T_so_far, N, 2)
        t = pred_tracks[0].cpu().numpy()  # (T_chunk, N, 2)
        v = pred_vis[0].cpu().numpy()
        if v.ndim == 3:
            v = v[..., 0]
        v = v.astype(bool)

        self._all_tracks.append(t)
        self._all_visibility.append(v)

        if self._point_ids is None:
            self._n_points = t.shape[1]
            self._point_ids = np.arange(self._n_points)

        return self.get_all_tracks()

    def run_full_video(self, video: torch.Tensor) -> TrackedPoints:
        """Convenience: initialize + step over an entire pre-loaded video.

        Mirrors the loop pattern in the CoTracker online demo.

        Args:
            video:    ``(1, T, C, H, W)`` float tensor.

        Returns:
            :class:`TrackedPoints` for the full video.
        """
        self.initialize(video)
        assert self._model is not None
        step: int = int(self._model.step)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        T = video.shape[1]
        for ind in range(0, T - step, step):
            chunk = video[:, ind : ind + step * 2]
            self.step(chunk)
        return self.get_all_tracks()

    def get_all_tracks(self) -> TrackedPoints:
        if not self._all_tracks:
            return TrackedPoints(
                tracks=np.empty((0, self._n_points, 2), dtype=np.float32),
                visibility=np.empty((0, self._n_points), dtype=bool),
                point_ids=self._point_ids
                if self._point_ids is not None
                else np.array([]),
            )
        tracks = np.concatenate(self._all_tracks, axis=0)  # (T, N, 2)
        visibility = np.concatenate(self._all_visibility, axis=0)
        return TrackedPoints(
            tracks=tracks,
            visibility=visibility,
            point_ids=self._point_ids
            if self._point_ids is not None
            else np.arange(tracks.shape[1]),
        )


# ---------------------------------------------------------------------------
# Incremental tracker (online with refresh)
# ---------------------------------------------------------------------------


@dataclass
class TrackingState:
    """Mutable state for an incremental online tracker with periodic refresh."""

    tracked_pixel_mask: np.ndarray  # (H, W) bool, pixels currently being tracked
    point_ids: list[int] = field(default_factory=list)
    next_id: int = 0


class IncrementalObjectTracker:
    """Tracks segmented object points through time with periodic refresh.

    At regular intervals (*refresh_every*) the segmentation mask is checked
    for points not yet being tracked, and new queries are added for those.

    This class wraps :class:`OnlineObjectTracker` and accumulates results
    across refreshes.  Each *camera stream* is tracked independently.
    """

    def __init__(
        self,
        device: str = "cuda",
        refresh_every: int = 20,
        max_new_points_per_refresh: int = 100,
        min_mask_points: int = 10,
        erode_kernel_size: int = 3,
        seed: int = 0,
    ) -> None:
        self.device = device
        self.refresh_every = refresh_every
        self.max_new_points_per_refresh = max_new_points_per_refresh
        self.min_mask_points = min_mask_points
        self.erode_kernel_size = erode_kernel_size
        self._rng = np.random.default_rng(seed)

        # Per-camera accumulation: camera_key → list of (frame_idx, TrackedPoints)
        self._camera_results: dict[str, list[tuple[int, TrackedPoints]]] = {}
        # Per-camera pending query state
        self._camera_queries: dict[str, list[np.ndarray]] = {}  # camera → [(N,3)]

    def process_demo_offline(
        self,
        camera_videos: dict[str, np.ndarray],
        camera_seg_masks: dict[str, np.ndarray],
        object_name: str,
        seg_id: int,
    ) -> dict[str, TrackedPoints]:
        """Track segmented object pixels through time using offline CoTracker.

        Processes each camera independently.  Uses offline mode (full video
        batch), but simulates incremental refresh by re-running tracking when
        new queries are added.

        Args:
            camera_videos:    camera_key → ``(T, H, W, 3)`` uint8 RGB frames.
            camera_seg_masks: camera_key → ``(T, H, W)`` int32 segmentation masks.
            object_name:      Human-readable name (used only for logging).
            seg_id:           Segmentation ID for the target object.

        Returns:
            Mapping camera_key → :class:`TrackedPoints` for all cameras.
        """
        print(
            f"[IncrementalObjectTracker] Offline tracking '{object_name}' "
            f"(seg_id={seg_id}) across {list(camera_videos.keys())}"
        )

        offline_tracker = OfflineObjectTracker(device=self.device)
        results: dict[str, TrackedPoints] = {}

        for cam_key, frames in camera_videos.items():
            seg_masks = camera_seg_masks[cam_key]  # (T, H, W)
            T, H, W, _ = frames.shape

            # Build tensor video: (1, T, C, H, W)
            video_tensor = (
                torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float()
            )  # (1, T, 3, H, W)

            # Collect queries across refresh intervals
            all_queries: list[np.ndarray] = []
            tracked_px_mask = np.zeros((H, W), dtype=bool)

            for t in range(0, T, self.refresh_every):
                obj_mask = seg_masks[t] == seg_id  # (H, W)
                new_mask = obj_mask & ~tracked_px_mask
                queries = mask_to_queries(
                    new_mask,
                    frame_index=t,
                    max_points=self.max_new_points_per_refresh,
                    min_points=self.min_mask_points,
                    erode_kernel_size=self.erode_kernel_size,
                    rng=self._rng,
                )
                if queries is not None:
                    all_queries.append(queries)
                    tracked_px_mask |= new_mask
                    print(
                        f"  [{cam_key}] t={t}: added {len(queries)} new queries "
                        f"(total tracked px: {tracked_px_mask.sum()})"
                    )

            if not all_queries:
                print(f"  [{cam_key}] No valid queries found – skipping.")
                continue

            combined_queries = np.concatenate(all_queries, axis=0)  # (N_total, 3)
            queries_tensor = (
                torch.from_numpy(combined_queries).unsqueeze(0).float()
            )  # (1, N_total, 3)

            print(
                f"  [{cam_key}] Running offline tracker with "
                f"{combined_queries.shape[0]} total query points…"
            )
            offline_tracker.initialize(video_tensor, queries=queries_tensor)
            results[cam_key] = offline_tracker.get_all_tracks()
            print(f"  [{cam_key}] Done. tracks shape: {results[cam_key].tracks.shape}")

        return results

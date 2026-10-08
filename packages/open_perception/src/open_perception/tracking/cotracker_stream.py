"""Streaming CoTracker wrapper for tracking reconstructed-point pixels.

CoTracker tracks 2D pixel queries through time. This module wraps the
`cotracker3_online` model from torch.hub into a per-view streaming session
that fits the live ``OpenPerception.detect()/track()`` loop.

Live-streaming protocol (mirrors facebookresearch/co-tracker's online_demo)::

    seed(rgb_t0, detections, depth, ...)         # push frame 0, build queries
    step(rgb_t1) → None                          # warm-up
    step(rgb_t2) → None
    ...
    step(rgb_t_{2*S - 1}) → None                 # 2*S - 1 step() calls during warm-up
    step(rgb_t_{2*S}) → {instance_id: PointTracks}  # first real output

where ``S = model.step`` (the cotracker stride, typically 4–8).

CoTracker is an *optional* dependency: this module imports cleanly without
``cotracker`` installed; ``torch.hub.load`` is deferred until the first
cotracker call and raises with a clear install hint if it fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from loguru import logger

from ..reconstruction.rgbd import mask_to_pointcloud


@dataclass
class PointTracks:
    """A snapshot of CoTracker output for one view, shared across instances.

    ``instance_slices`` maps each detection's ``instance_id`` to the column
    range in ``tracks_uv`` / ``visibility`` / ``point_ids`` carrying that
    detection's queries. Use :meth:`for_instance` to slice ergonomically.
    """

    point_ids: np.ndarray  # (N,) int
    tracks_uv: np.ndarray  # (T_so_far, N, 2) float32
    visibility: np.ndarray  # (T_so_far, N) bool
    seed_frame_idx: int
    seed_view_idx: int
    seed_uv: np.ndarray  # (N, 2) int32
    seed_xyz: np.ndarray  # (N, 3) float32
    instance_slices: dict[int, slice] = field(default_factory=dict)

    def for_instance(self, instance_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(tracks, visibility)`` sliced to one detection's columns."""
        sl = self.instance_slices[instance_id]
        return self.tracks_uv[:, sl], self.visibility[:, sl]


# ---------------------------------------------------------------------------
# Query sampling
# ---------------------------------------------------------------------------


def mask_to_depth_queries(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    frame_index: int = 0,
    max_points: int = 64,
    min_points: int = 8,
    erode_kernel_size: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Sample ``[t, x, y]`` queries from mask pixels with valid depth.

    Identical contract to the reference ``mask_to_queries`` plus a depth
    filter: only pixels with ``depth > 0`` are kept, enforcing the
    "reconstructed-points only" constraint.

    Returns ``(N, 3)`` float32 ``[t, x, y]`` or ``None`` if fewer than
    ``min_points`` pixels survive.
    """
    if depth is None:
        raise ValueError("mask_to_depth_queries requires depth; got None")
    if mask.shape[:2] != depth.shape[:2]:
        raise ValueError(f"mask {mask.shape} and depth {depth.shape} shapes differ")
    if rng is None:
        rng = np.random.default_rng(0)

    valid = mask.astype(bool)
    if erode_kernel_size > 0:
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    valid &= depth > 0

    ys, xs = np.where(valid)
    if len(xs) < min_points:
        return None
    if len(xs) > max_points:
        idx = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]

    t_col = np.full(len(xs), frame_index, dtype=np.float32)
    return np.stack([t_col, xs.astype(np.float32), ys.astype(np.float32)], axis=1)


# ---------------------------------------------------------------------------
# CoTracker streaming wrapper
# ---------------------------------------------------------------------------


_INSTALL_HINT = (
    "open_perception point tracking requires the cotracker model "
    "(loaded via torch.hub from 'facebookresearch/co-tracker'). "
    "Install with: pip install 'open-perception[cotracker]' and ensure "
    "the cotracker repo is reachable via torch.hub (network + git)."
)


def _load_online_cotracker(model_name: str, device: str) -> Any:
    """Load cotracker via torch.hub. Raises a clear RuntimeError on failure."""
    try:
        import torch

        model: Any = torch.hub.load("facebookresearch/co-tracker", model_name)
        return model.to(device)
    except Exception as exc:
        raise RuntimeError(f"{_INSTALL_HINT} Original error: {exc}") from exc


class CoTrackerStream:
    """Per-view streaming CoTracker session.

    Buffers RGB frames; defers cotracker initialization until the buffer
    holds ``S = model.step`` frames; produces tracks from frame ``2*S``
    onwards. Use one instance per camera view.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "cotracker3_online",
        grid_size: int = 0,
        load_fn: Any = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.grid_size = grid_size
        # Injectable loader for tests; defaults to the real torch.hub path.
        self._load_fn = load_fn if load_fn is not None else _load_online_cotracker

        self._model: Any | None = None
        self._model_step: int | None = None
        self._buffer: list[np.ndarray] = []
        self._n_appended: int = 0
        self._initialized: bool = False
        self._closed: bool = False

        self._pending_queries: np.ndarray | None = None
        self._latest_tracks: PointTracks | None = None
        self._instance_slices: dict[int, slice] = {}
        self._point_ids: np.ndarray | None = None
        self._seed_uv: np.ndarray | None = None
        self._seed_xyz: np.ndarray | None = None
        self._seed_frame_idx: int = 0
        self._seed_view_idx: int = 0

    # ------------------------------------------------------------------

    @property
    def is_warm(self) -> bool:
        """True once at least one real (non-first-step) cotracker call has run."""
        return self._latest_tracks is not None

    @property
    def is_seeded(self) -> bool:
        """True once :meth:`seed` has been called and not yet :meth:`reset`."""
        return self._pending_queries is not None

    @property
    def model_step(self) -> int:
        """Cotracker stride ``S``. Available after the first model load."""
        if self._model_step is None:
            self._ensure_model()
        assert self._model_step is not None
        return self._model_step

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model: Any = self._load_fn(self.model_name, self.device)
        try:
            step = int(model.step)
        except Exception as exc:
            raise RuntimeError(
                f"loaded cotracker model has no integer `step` attribute: {exc}"
            ) from exc
        if step <= 0:
            raise RuntimeError(f"cotracker model.step must be positive, got {step}")
        self._model = model
        self._model_step = step

    # ------------------------------------------------------------------
    # Public API

    def seed(
        self,
        view_idx: int,
        frame_idx: int,
        rgb: np.ndarray,
        detections: list,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray | None = None,
        *,
        max_points_per_det: int = 64,
        min_points_per_det: int = 8,
        erode_kernel_size: int = 3,
        rng: np.random.Generator | None = None,
    ) -> dict[int, slice]:
        """Build queries from each detection's mask + depth and start the buffer.

        Returns the ``{instance_id: column_slice}`` map that will be embedded
        in every produced :class:`PointTracks`.

        Raises ``RuntimeError`` if no detection has a valid ``instance_id``,
        and ``ValueError`` if any detection lacks a usable mask or its
        ``mask_to_depth_queries`` returns fewer than ``min_points_per_det``.
        """
        if depth is None:
            raise RuntimeError(
                f"point tracking requires depth for view {view_idx}; "
                "pass depth=... to detect() or set enable_point_tracking=False"
            )
        if rng is None:
            rng = np.random.default_rng(0)

        per_det_queries: list[np.ndarray] = []
        per_det_xyz: list[np.ndarray] = []
        instance_slices: dict[int, slice] = {}
        cursor = 0
        for det in detections:
            if det.instance_id is None:
                continue
            queries = mask_to_depth_queries(
                det.mask,
                depth,
                frame_index=0,
                max_points=max_points_per_det,
                min_points=min_points_per_det,
                erode_kernel_size=erode_kernel_size,
                rng=rng,
            )
            if queries is None:
                logger.warning(
                    f"[cotracker] view {view_idx} det "
                    f"{det.query!r}#{det.instance_id} produced no valid queries "
                    "(mask + valid-depth pixels < min_points_per_det); skipping"
                )
                continue
            n = queries.shape[0]
            uv = queries[:, 1:3].astype(np.int32)
            # Build a per-detection mask of just these sampled pixels so we can
            # back-project them with the same convention used elsewhere in the
            # package (mask_to_pointcloud handles flip_u, extrinsics, etc.).
            seed_mask = np.zeros_like(det.mask, dtype=bool)
            seed_mask[uv[:, 1], uv[:, 0]] = True
            xyz = mask_to_pointcloud(
                depth, seed_mask, intrinsics, extrinsics, stride=1, flip_u=False
            )
            # Pad/truncate xyz to N — mask_to_pointcloud may drop pixels with
            # non-positive depth (already filtered above, but be defensive).
            if xyz.shape[0] != n:
                xyz_full = np.zeros((n, 3), dtype=np.float32)
                xyz_full[: min(n, xyz.shape[0])] = xyz[: min(n, xyz.shape[0])]
                xyz = xyz_full
            per_det_queries.append(queries)
            per_det_xyz.append(xyz)
            instance_slices[int(det.instance_id)] = slice(cursor, cursor + n)
            cursor += n

        if not per_det_queries:
            raise RuntimeError(
                f"point tracking enabled but view {view_idx} produced no "
                "queries from any detection (no instance_id with valid "
                "depth-gated mask pixels)"
            )

        self._pending_queries = np.concatenate(per_det_queries, axis=0)  # (N,3)
        self._seed_uv = self._pending_queries[:, 1:3].astype(np.int32)
        self._seed_xyz = np.concatenate(per_det_xyz, axis=0).astype(np.float32)
        self._point_ids = np.arange(self._pending_queries.shape[0], dtype=np.int64)
        self._instance_slices = instance_slices
        self._seed_frame_idx = frame_idx
        self._seed_view_idx = view_idx

        self._buffer = [rgb.copy()]
        self._n_appended = 1
        self._initialized = False
        self._latest_tracks = None
        self._closed = False
        return dict(instance_slices)

    def step(self, rgb: np.ndarray) -> PointTracks | None:
        """Process one new frame. Returns the latest snapshot or ``None`` during warm-up."""
        if self._closed:
            raise RuntimeError("CoTrackerStream is closed; call seed() to restart")
        if self._pending_queries is None:
            raise RuntimeError("CoTrackerStream.step() called before seed()")
        # Lazy model load on the first step() so seed() stays cheap and
        # callers without cotracker installed get the import error here, not
        # on the detect() that populated the queries.
        self._ensure_model()
        S = self.model_step
        i = self._n_appended  # frames already in buffer (>= 1)
        if i % S == 0 and i != 0:
            is_first = i == S
            chunk_frames = self._buffer[-S * 2 :]
            self._run_model(chunk_frames, is_first_step=is_first)
            if is_first:
                self._initialized = True
        self._buffer.append(rgb.copy())
        self._n_appended += 1
        return self._latest_tracks

    def reset(self) -> None:
        """Drop all session state. Next call must be ``seed()``."""
        self._buffer = []
        self._n_appended = 0
        self._initialized = False
        self._latest_tracks = None
        self._pending_queries = None
        self._instance_slices = {}
        self._point_ids = None
        self._seed_uv = None
        self._seed_xyz = None
        self._closed = False

    # ------------------------------------------------------------------
    # Internals

    def _run_model(self, frames: list[np.ndarray], *, is_first_step: bool) -> None:
        import torch

        assert self._model is not None, "_run_model called before _ensure_model"
        # (T, H, W, 3) uint8 → (1, T, 3, H, W) float on device
        arr = np.stack(frames, axis=0)
        chunk = (
            torch.from_numpy(arr)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        kwargs: dict[str, Any] = {"video_chunk": chunk, "is_first_step": is_first_step}
        if is_first_step:
            queries_t = (
                torch.from_numpy(self._pending_queries)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            kwargs["queries"] = queries_t
            kwargs["grid_size"] = self.grid_size if self._pending_queries is None else 0
        with torch.no_grad():
            output = self._model(**kwargs)
        # is_first_step calls return placeholder/None tracks — only absorb on
        # subsequent calls.
        if is_first_step:
            return
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                f"cotracker returned unexpected output shape: {type(output)!r}"
            )
        pred_tracks, pred_vis = output
        # pred_tracks: (1, T_so_far, N, 2) — cumulative per online model.
        tracks_np = pred_tracks[0].detach().cpu().numpy().astype(np.float32)
        vis = pred_vis[0].detach().cpu().numpy()
        if vis.ndim == 3:
            vis = vis[..., 0]
        vis_np = vis.astype(bool)
        assert self._point_ids is not None
        assert self._seed_uv is not None
        assert self._seed_xyz is not None
        self._latest_tracks = PointTracks(
            point_ids=self._point_ids,
            tracks_uv=tracks_np,
            visibility=vis_np,
            seed_frame_idx=self._seed_frame_idx,
            seed_view_idx=self._seed_view_idx,
            seed_uv=self._seed_uv,
            seed_xyz=self._seed_xyz,
            instance_slices=dict(self._instance_slices),
        )

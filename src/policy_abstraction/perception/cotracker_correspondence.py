"""CoTracker + DINOv3 pixel-level correspondence matching.

Tracks segmented object points through a retrieved interaction using CoTracker,
associates each tracked point with its average DINOv3 patch feature (fused across
both cameras), then finds pixel-level correspondences between a query interaction
(tracked up to *t_query*) and the reference tracked points via cosine similarity.

Usage::

    tracker = InteractionPointTracker(device="cuda")
    tracked = tracker.process_interaction(
        source_file="path/to/demo.hdf5",
        demo_key="demo_0",
        start_timestep=5,
        end_timestep=45,
        object_name="alphabet_soup_1",
    )
    correspondences = tracker.find_correspondences(
        source_file="path/to/demo.hdf5",
        demo_key_query="demo_1",
        object_name="alphabet_soup_1",
        t_query=10,
        tracked_features=tracked,
        top_k=10,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from policy_abstraction.perception.cotracker_tracker import (
    IncrementalObjectTracker,
    TrackedPoints,
)
from policy_abstraction.perception.sd_dino_extractor import DINOv3PatchExtractor

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TrackedPointFeatures:
    """Tracked object points with associated average DINOv3 features.

    Attributes:
        avg_features:       ``(N, feat_dim)`` float32, L2-normalized average DINO
                            feature per tracked point across its visible frames and
                            cameras.
        ref_pixel_xy:       ``(N, 2)`` float32, ``[x, y]`` pixel position of each
                            point in its last visible frame (in the point's camera).
        ref_frame_idx:      ``(N,)`` int, absolute (within-interaction) frame index
                            of each point's last visible frame.
        visibility_counts:  ``(N,)`` int, number of frames each point was visible.
        point_ids:          ``(N,)`` int, unique identifiers (scoped per camera).
        image_hw:           ``(H, W)`` of the source frames.
        camera_ids:         ``(N,)`` str array, camera name for each tracked point.
    """

    avg_features: np.ndarray  # (N, feat_dim)
    ref_pixel_xy: np.ndarray  # (N, 2) float32
    ref_frame_idx: np.ndarray  # (N,) int
    visibility_counts: np.ndarray  # (N,) int
    point_ids: np.ndarray  # (N,) int
    image_hw: tuple[int, int]  # (H, W)
    camera_ids: np.ndarray  # (N,) str


@dataclass
class PixelCorrespondence:
    """A pixel-level correspondence between a tracked query point and a reference point.

    Attributes:
        obs_xy:        ``(2,)`` float32, ``[x, y]`` in query frame pixel space.
        ref_xy:        ``(2,)`` float32, ``[x, y]`` in reference frame pixel space.
        score:         Cosine similarity in ``[-1, 1]``; higher is better.
        point_id:      Identifier of the matched reference tracked point.
        ref_frame_idx: Frame index (within the interaction) where ``ref_xy`` was last seen.
        obs_camera:    Camera that produced ``obs_xy``.
        ref_camera:    Camera that produced ``ref_xy``.
        spatial_score: Spatial consistency with other top matches; 1.0 = consistent.
    """

    obs_xy: np.ndarray  # (2,) float32
    ref_xy: np.ndarray  # (2,) float32
    score: float
    point_id: int
    ref_frame_idx: int
    obs_camera: str
    ref_camera: str
    spatial_score: float = 1.0


@dataclass
class TemporalDescriptors:
    """Per-timestep global DINOv3 descriptors for a tracked interaction.

    Each descriptor is the L2-normalized average of DINOv3 patch features
    sampled at all *visible* tracked object points across all cameras at
    that timestep.  Timesteps where no tracked point is visible are marked
    as invalid (``valid[t] == False``) and their descriptor is zero.

    Attributes:
        descriptors: ``(T, D)`` float32, one L2-normalized descriptor per frame.
        valid:       ``(T,)`` bool, ``True`` when at least one point was visible.
        image_hw:    ``(H, W)`` of the source frames.
    """

    descriptors: np.ndarray  # (T, D) float32
    valid: np.ndarray  # (T,) bool
    image_hw: tuple[int, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_seg_id(seg_name_map: dict[str, str], object_name: str) -> int:
    """Return the integer segmentation ID for *object_name*.

    Args:
        seg_name_map: Mapping ``{str(seg_id): object_name}`` from HDF5 attrs.
        object_name:  Human-readable object name to look up.

    Returns:
        Integer segmentation ID.

    Raises:
        KeyError: If no entry matching *object_name* is found.
    """
    for id_str, name in seg_name_map.items():
        if name == object_name:
            return int(id_str)
    raise KeyError(
        f"Object '{object_name}' not found in segmentation map. "
        f"Available: {list(seg_name_map.values())}"
    )


def _sample_features_at_tracks(
    feat_4d: torch.Tensor,  # (T, D, H_p, W_p) on device
    tracks: np.ndarray,  # (T, N, 2) float32, [x, y] in image pixel space
    img_h: int,
    img_w: int,
) -> torch.Tensor:  # (T, N, D) on CPU
    """Sample feature maps at tracked pixel positions via bilinear interpolation.

    Uses ``F.grid_sample`` so that only N sparse points are sampled per frame,
    avoiding materializing a full ``(T, H_img, W_img, D)`` dense tensor.

    Args:
        feat_4d: Feature maps ``(T, D, H_p, W_p)``.
        tracks:  Tracked pixel coordinates ``(T, N, 2)`` where ``[:, :, 0]`` is x (col)
                 and ``[:, :, 1]`` is y (row), in original image pixel space.
        img_h:   Original image height (used to normalize coordinates).
        img_w:   Original image width.

    Returns:
        Sampled features ``(T, N, D)`` on CPU.
    """
    T, D, H_p, W_p = feat_4d.shape
    device = feat_4d.device

    tracks_t = torch.from_numpy(tracks).to(device)  # (T, N, 2)

    # Normalize pixel coords [0, W] × [0, H] → [-1, 1] for grid_sample
    norm_x = (tracks_t[:, :, 0] / img_w) * 2.0 - 1.0  # (T, N)
    norm_y = (tracks_t[:, :, 1] / img_h) * 2.0 - 1.0  # (T, N)

    # grid_sample expects (B, H_out, W_out, 2); treat T as batch, N as H_out, 1 as W_out
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)  # (T, N, 1, 2)

    # (T, D, N, 1) → squeeze → (T, D, N)
    sampled = F.grid_sample(
        feat_4d, grid, mode="bilinear", align_corners=False, padding_mode="border"
    ).squeeze(-1)  # (T, D, N)

    return sampled.permute(0, 2, 1).cpu()  # (T, N, D)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InteractionPointTracker:
    """Tracks object points in a retrieved interaction and matches them to query observations.

    Workflow:
    1. ``process_interaction`` — load HDF5 frames for both cameras, run CoTracker,
       extract DINOv3 features, average features per tracked point across visible
       frames and cameras.
    2. ``find_correspondences`` — track the query object from the start of the demo
       up to *t_query* (using both cameras), average DINO features, then match
       against reference tracked-point features via cosine similarity.
    """

    def __init__(
        self,
        device: str = "cuda",
        max_track_points: int = 100,
        refresh_every: int = 20,
        dino_img_size: int = 448,
        dino_model: str = "dinov3_vits16",
    ) -> None:
        self.device = device
        self.max_track_points = max_track_points
        self.refresh_every = refresh_every
        self.dino_img_size = dino_img_size
        self.dino_model = dino_model

        # Lazy-loaded models
        self._dino: DINOv3PatchExtractor | None = None
        self._cotracker: IncrementalObjectTracker | None = None

        # Cached per-camera data from the most recent process_interaction call
        self.last_tracked_points: TrackedPoints | None = None  # backward compat
        self.last_ref_tracked_by_camera: dict[str, TrackedPoints] = {}
        self.last_ref_camera_frames: dict[str, np.ndarray] = {}

        # Cached per-camera data from the most recent find_correspondences call
        self.last_query_tracked_by_camera: dict[str, TrackedPoints] = {}
        self.last_query_camera_frames: dict[str, np.ndarray] = {}

        # Cached outputs from temporal alignment methods
        self.last_ref_temporal: TemporalDescriptors | None = None
        self.last_query_descriptor: np.ndarray | None = (
            None  # (D,) after find_temporal_alignment
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def dino(self) -> DINOv3PatchExtractor:
        if self._dino is None:
            self._dino = DINOv3PatchExtractor(
                model_name=self.dino_model, device=self.device
            )
        return self._dino

    @property
    def cotracker(self) -> IncrementalObjectTracker:
        if self._cotracker is None:
            self._cotracker = IncrementalObjectTracker(
                device=self.device,
                refresh_every=self.refresh_every,
                max_new_points_per_refresh=self.max_track_points,
            )
        return self._cotracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_camera_data(
        self,
        source_file: Path,
        demo_key: str,
        start_timestep: int,
        end_timestep: int,
        cameras: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
        """Load RGB frames and segmentation masks for all cameras.

        Returns:
            (camera_frames, camera_segs, seg_name_map)
        """
        with h5py.File(source_file, "r") as f:
            seg_name_map: dict[str, str] = json.loads(
                f["data"].attrs["segmentation_name_mapping"]
            )

            camera_frames: dict[str, np.ndarray] = {}
            camera_segs: dict[str, np.ndarray] = {}

            for cam in cameras:
                rgb_raw = np.asarray(
                    f["data"][demo_key]["obs"][f"{cam}_rgb"][
                        start_timestep:end_timestep
                    ]
                )  # (T, H, W, 3) uint8
                seg_raw = np.asarray(
                    f["data"][demo_key]["obs"][f"{cam}_segmentation"][
                        start_timestep:end_timestep
                    ]
                )  # (T, H, W, 1) int32
                camera_frames[cam] = np.ascontiguousarray(rgb_raw[:, ::-1, ::-1, :])
                camera_segs[cam] = np.ascontiguousarray(seg_raw[:, ::-1, ::-1, 0])

        return camera_frames, camera_segs, seg_name_map

    def _extract_tracked_features(
        self,
        source_file: str | Path,
        demo_key: str,
        start_timestep: int,
        end_timestep: int,
        object_name: str,
        cameras: list[str],
        tracked_cache: Optional[dict[str, TrackedPoints]] = None,
        frames_cache: Optional[dict[str, np.ndarray]] = None,
    ) -> TrackedPointFeatures:
        """Core logic: load frames, track, extract DINO, average, fuse cameras.

        Args:
            source_file:      Path to source HDF5.
            demo_key:         Demo key inside HDF5.
            start_timestep:   First frame (inclusive).
            end_timestep:     Last frame (exclusive).
            object_name:      Human-readable object name.
            cameras:          List of camera names to process.
            tracked_cache:    If provided, populated with per-camera TrackedPoints.
            frames_cache:     If provided, populated with per-camera RGB frames.

        Returns:
            :class:`TrackedPointFeatures` with avg DINO features from all cameras.
        """
        source_file = Path(source_file)

        # ── 1. Load frames and segmentation for all cameras ────────────────────
        print(
            f"[InteractionPointTracker] Loading frames from {source_file.name} "
            f"({demo_key}, t={start_timestep}:{end_timestep}, cameras={cameras})"
        )
        with h5py.File(source_file, "r") as f:
            seg_name_map: dict[str, str] = json.loads(
                f["data"].attrs["segmentation_name_mapping"]
            )
            seg_id = _resolve_seg_id(seg_name_map, object_name)

            camera_frames: dict[str, np.ndarray] = {}
            camera_segs: dict[str, np.ndarray] = {}
            for cam in cameras:
                rgb_raw = np.asarray(
                    f["data"][demo_key]["obs"][f"{cam}_rgb"][
                        start_timestep:end_timestep
                    ]
                )
                seg_raw = np.asarray(
                    f["data"][demo_key]["obs"][f"{cam}_segmentation"][
                        start_timestep:end_timestep
                    ]
                )
                camera_frames[cam] = np.ascontiguousarray(rgb_raw[:, ::-1, ::-1, :])
                camera_segs[cam] = np.ascontiguousarray(seg_raw[:, ::-1, ::-1, 0])

        T, H, W, _ = next(iter(camera_frames.values())).shape
        print(f"  Loaded {T} frames at {H}×{W}, seg_id={seg_id}")

        if frames_cache is not None:
            frames_cache.clear()
            frames_cache.update(camera_frames)

        # ── 2. Run CoTracker for all cameras ───────────────────────────────────
        print("  Running CoTracker…")
        cotracker_results = self.cotracker.process_demo_offline(
            camera_videos=camera_frames,
            camera_seg_masks=camera_segs,
            object_name=object_name,
            seg_id=seg_id,
        )

        if tracked_cache is not None:
            tracked_cache.clear()
            tracked_cache.update(cotracker_results)

        # ── 3. Per-camera: extract DINO, sample, average ───────────────────────
        all_avg_features: list[np.ndarray] = []
        all_ref_pixel_xy: list[np.ndarray] = []
        all_ref_frame_idx: list[np.ndarray] = []
        all_vis_counts: list[np.ndarray] = []
        all_point_ids: list[np.ndarray] = []
        all_camera_ids: list[str] = []

        point_id_offset = 0  # ensure unique IDs across cameras

        for cam in cameras:
            if cam not in cotracker_results:
                print(f"  [{cam}] No valid queries – skipping.")
                continue

            frames = camera_frames[cam]
            tracked = cotracker_results[cam]
            tracks = tracked.tracks  # (T, N, 2)
            visibility = tracked.visibility  # (T, N)
            N = tracks.shape[1]

            # Extract DINO features per frame for this camera
            print(f"  [{cam}] Extracting DINOv3 features…")
            feat_list: list[torch.Tensor] = []
            for t in range(T):
                frame_pil = Image.fromarray(frames[t])
                feat = self.dino.extract(frame_pil, img_size=self.dino_img_size)
                feat_list.append(feat)

            all_feats = torch.stack(feat_list, dim=0)  # (T, H_p, W_p, D)
            feat_4d = all_feats.permute(0, 3, 1, 2).to(self.device)  # (T, D, H_p, W_p)

            print(f"  [{cam}] Sampling features at tracked positions…")
            sampled = _sample_features_at_tracks(feat_4d, tracks, H, W)  # (T, N, D)

            # Average across visible frames
            vis_t = torch.from_numpy(visibility.astype(np.float32))  # (T, N)
            vis_sum = vis_t.sum(dim=0).clamp(min=1.0)  # (N,)
            avg_feat = (sampled * vis_t.unsqueeze(-1)).sum(dim=0) / vis_sum.unsqueeze(
                -1
            )
            avg_feat = F.normalize(avg_feat, dim=-1)  # (N, D)

            # Reference pixel: last visible frame per point
            vis_np = visibility  # (T, N)
            never_vis = ~vis_np.any(axis=0)
            reversed_vis = vis_np[::-1, :]
            last_vis_idx = (T - 1) - np.argmax(reversed_vis, axis=0)
            last_vis_idx[never_vis] = 0

            ref_pixel_xy = tracks[last_vis_idx, np.arange(N)]
            vis_counts = vis_np.sum(axis=0).astype(int)

            valid = ~never_vis
            if not valid.any():
                print(f"  [{cam}] All tracked points invisible – skipping.")
                continue

            n_valid = int(valid.sum())
            print(f"  [{cam}] {n_valid} / {N} points have at least one visible frame.")

            all_avg_features.append(avg_feat[valid].numpy().astype(np.float32))
            all_ref_pixel_xy.append(ref_pixel_xy[valid].astype(np.float32))
            all_ref_frame_idx.append(last_vis_idx[valid].astype(int))
            all_vis_counts.append(vis_counts[valid])
            all_point_ids.append(tracked.point_ids[valid] + point_id_offset)
            all_camera_ids.extend([cam] * n_valid)

            point_id_offset += N

        if not all_avg_features:
            raise RuntimeError(
                f"No valid tracked points found for '{object_name}' in any camera."
            )

        return TrackedPointFeatures(
            avg_features=np.concatenate(all_avg_features, axis=0),
            ref_pixel_xy=np.concatenate(all_ref_pixel_xy, axis=0),
            ref_frame_idx=np.concatenate(all_ref_frame_idx, axis=0),
            visibility_counts=np.concatenate(all_vis_counts, axis=0),
            point_ids=np.concatenate(all_point_ids, axis=0),
            image_hw=(H, W),
            camera_ids=np.array(all_camera_ids),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_interaction(
        self,
        source_file: str | Path,
        demo_key: str,
        start_timestep: int,
        end_timestep: int,
        object_name: str,
        cameras: list[str] | None = None,
        camera: str = "eye_in_hand",  # kept for backward compatibility
    ) -> TrackedPointFeatures:
        """Track object pixels through the interaction and compute average DINO features.

        Tracks the object in all *cameras* using CoTracker and averages DINOv3 patch
        features across visible frames.  Each camera contributes independent tracked
        points to the returned feature set.

        Args:
            source_file:     Path to the source HDF5 demo file.
            demo_key:        Demo key inside HDF5 (e.g. ``"demo_0"``).
            start_timestep:  First frame index (inclusive) of the interaction.
            end_timestep:    Last frame index (exclusive) of the interaction.
            object_name:     Human-readable object name matching the segmentation map.
            cameras:         List of camera streams to use.  Defaults to
                             ``["eye_in_hand", "agentview"]``.  Pass a single-element
                             list to use one camera only.
            camera:          Ignored when *cameras* is provided; kept for backward
                             compatibility.

        Returns:
            :class:`TrackedPointFeatures` with averaged DINO features per tracked point.
        """
        if cameras is None:
            cameras = ["eye_in_hand", "agentview"]

        result = self._extract_tracked_features(
            source_file=source_file,
            demo_key=demo_key,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            object_name=object_name,
            cameras=cameras,
            tracked_cache=self.last_ref_tracked_by_camera,
            frames_cache=self.last_ref_camera_frames,
        )

        # Backward compatibility: expose eye_in_hand result as last_tracked_points
        self.last_tracked_points = self.last_ref_tracked_by_camera.get("eye_in_hand")

        total = len(result.point_ids)
        print(f"\nTracked {total} valid points across {cameras}.")
        return result

    def find_correspondences(
        self,
        source_file: str | Path,
        demo_key_query: str,
        object_name: str,
        t_query: int,
        tracked_features: TrackedPointFeatures,
        cameras: list[str] | None = None,
        top_k: int = 10,
        mutual: bool = True,
    ) -> list[PixelCorrespondence]:
        """Find correspondences between a query interaction and the reference.

        Tracks the target object in the query demo from t=0 up to *t_query*
        using CoTracker on all *cameras*, averages the DINOv3 patch features of
        each tracked query point, then matches them against the pre-computed
        reference features via cosine similarity.

        Args:
            source_file:      Path to the source HDF5 demo file.
            demo_key_query:   Demo key for the query demo (e.g. ``"demo_1"``).
            object_name:      Human-readable object name matching the segmentation map.
            t_query:          Last frame index (inclusive) for query tracking.
                              Frames ``[0, t_query]`` are used.
            tracked_features: Reference features from :meth:`process_interaction`.
            cameras:          Camera streams to use (default: ``["eye_in_hand", "agentview"]``).
            top_k:            Maximum number of correspondences to return.
            mutual:           If ``True``, apply mutual nearest-neighbour filtering.

        Returns:
            List of :class:`PixelCorrespondence`, sorted by score descending.
        """
        if cameras is None:
            cameras = ["eye_in_hand", "agentview"]

        # ── 1. Track query object up to t_query ───────────────────────────────
        print(f"\n[find_correspondences] Tracking query object up to t={t_query}…")
        query_features = self._extract_tracked_features(
            source_file=source_file,
            demo_key=demo_key_query,
            start_timestep=0,
            end_timestep=t_query + 1,
            object_name=object_name,
            cameras=cameras,
            tracked_cache=self.last_query_tracked_by_camera,
            frames_cache=self.last_query_camera_frames,
        )

        N_q = len(query_features.point_ids)
        print(f"  Query: {N_q} tracked points across {cameras}.")

        # ── 2. Cosine similarity: (N_ref, N_q) ────────────────────────────────
        ref_feat = torch.from_numpy(tracked_features.avg_features)  # (N_ref, D)
        q_feat = torch.from_numpy(query_features.avg_features)  # (N_q, D)

        sim = ref_feat @ q_feat.T  # (N_ref, N_q)
        N_ref = ref_feat.shape[0]

        # ── 3. Forward: for each ref point, best matching query point ──────────
        best_q = sim.argmax(dim=1)  # (N_ref,)
        best_scores = sim[torch.arange(N_ref), best_q]  # (N_ref,)

        # ── 4. Mutual nearest-neighbour filter ─────────────────────────────────
        if mutual:
            best_ref = sim.argmax(dim=0)  # (N_q,)
            is_mutual = best_ref[best_q] == torch.arange(N_ref)  # (N_ref,)
        else:
            is_mutual = torch.ones(N_ref, dtype=torch.bool)

        # ── 5. Build correspondences ───────────────────────────────────────────
        candidates: list[PixelCorrespondence] = []
        for n in range(N_ref):
            if not is_mutual[n]:
                continue
            q_idx = int(best_q[n].item())
            score = float(best_scores[n].item())
            candidates.append(
                PixelCorrespondence(
                    obs_xy=query_features.ref_pixel_xy[q_idx].copy(),
                    ref_xy=tracked_features.ref_pixel_xy[n].copy(),
                    score=score,
                    point_id=int(tracked_features.point_ids[n]),
                    ref_frame_idx=int(tracked_features.ref_frame_idx[n]),
                    obs_camera=str(query_features.camera_ids[q_idx]),
                    ref_camera=str(tracked_features.camera_ids[n]),
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        result = candidates[:top_k]

        # ── 6. Spatial consistency scoring ─────────────────────────────────────
        if len(result) >= 3:
            from policy_abstraction.utils.spatial_consistency import (
                score_spatial_consistency,
            )

            ref_arr = np.stack([c.ref_xy for c in result])
            obs_arr = np.stack([c.obs_xy for c in result])
            sp_scores = score_spatial_consistency(ref_arr, obs_arr)
            for c, sp in zip(result, sp_scores, strict=False):
                c.spatial_score = float(sp)

        return result

    def compute_ref_temporal_descriptors(
        self,
        source_file: str | Path,
        demo_key: str,
        start_timestep: int,
        end_timestep: int,
        object_name: str,
        cameras: list[str] | None = None,
    ) -> TemporalDescriptors:
        """Compute per-timestep global DINOv3 descriptors for a reference interaction.

        Tracks the object through the interaction with CoTracker on all *cameras*,
        extracts DINOv3 features, then for each timestep **averages** the sampled
        features across all *visible* tracked points from all cameras.  The result
        is one L2-normalized global descriptor per frame that represents the
        appearance of the tracked object at that moment.

        The computed descriptors are cached in :attr:`last_ref_temporal`.

        Args:
            source_file:    Path to the source HDF5 demo file.
            demo_key:       Demo key inside HDF5 (e.g. ``"demo_0"``).
            start_timestep: First frame index (inclusive).
            end_timestep:   Last frame index (exclusive).
            object_name:    Human-readable object name matching the segmentation map.
            cameras:        Camera streams to use (default: ``["eye_in_hand", "agentview"]``).

        Returns:
            :class:`TemporalDescriptors` with ``(T, D)`` per-timestep descriptors.
        """
        if cameras is None:
            cameras = ["eye_in_hand", "agentview"]

        source_file = Path(source_file)

        print(
            f"[compute_ref_temporal_descriptors] {source_file.name} "
            f"({demo_key}, t={start_timestep}:{end_timestep}, cameras={cameras})"
        )

        # ── 1. Load frames and segmentation ───────────────────────────────────
        with h5py.File(source_file, "r") as f:
            seg_name_map: dict[str, str] = json.loads(
                f["data"].attrs["segmentation_name_mapping"]
            )
            seg_id = _resolve_seg_id(seg_name_map, object_name)

            camera_frames: dict[str, np.ndarray] = {}
            camera_segs: dict[str, np.ndarray] = {}
            for cam in cameras:
                rgb_raw = np.asarray(
                    f["data"][demo_key]["obs"][f"{cam}_rgb"][
                        start_timestep:end_timestep
                    ]
                )
                seg_raw = np.asarray(
                    f["data"][demo_key]["obs"][f"{cam}_segmentation"][
                        start_timestep:end_timestep
                    ]
                )
                camera_frames[cam] = np.ascontiguousarray(rgb_raw[:, ::-1, ::-1, :])
                camera_segs[cam] = np.ascontiguousarray(seg_raw[:, ::-1, ::-1, 0])

        T, H, W, _ = next(iter(camera_frames.values())).shape
        print(f"  Loaded {T} frames at {H}×{W}, seg_id={seg_id}")

        # ── 2. Run CoTracker for all cameras ───────────────────────────────────
        print("  Running CoTracker…")
        cotracker_results = self.cotracker.process_demo_offline(
            camera_videos=camera_frames,
            camera_seg_masks=camera_segs,
            object_name=object_name,
            seg_id=seg_id,
        )

        # ── 3. Per-camera: extract DINO and sample at tracked positions ────────
        cam_sampled: list[torch.Tensor] = []  # (T, N_cam, D) per camera
        cam_visibility: list[np.ndarray] = []  # (T, N_cam) bool per camera

        for cam in cameras:
            if cam not in cotracker_results:
                print(f"  [{cam}] No valid queries – skipping.")
                continue

            frames = camera_frames[cam]
            tracked = cotracker_results[cam]
            tracks = tracked.tracks  # (T, N, 2)
            visibility = tracked.visibility  # (T, N)

            print(f"  [{cam}] Extracting DINOv3 features…")
            feat_list: list[torch.Tensor] = []
            for t in range(T):
                frame_pil = Image.fromarray(frames[t])
                feat = self.dino.extract(frame_pil, img_size=self.dino_img_size)
                feat_list.append(feat)

            all_feats = torch.stack(feat_list, dim=0)  # (T, H_p, W_p, D)
            feat_4d = all_feats.permute(0, 3, 1, 2).to(self.device)  # (T, D, H_p, W_p)

            print(f"  [{cam}] Sampling features at tracked positions…")
            sampled = _sample_features_at_tracks(
                feat_4d, tracks, H, W
            )  # (T, N, D) on CPU
            cam_sampled.append(sampled)
            cam_visibility.append(visibility)

        if not cam_sampled:
            raise RuntimeError(
                f"No valid tracked points found for '{object_name}' in any camera."
            )

        # ── 4. Build per-timestep global descriptors ───────────────────────────
        D = cam_sampled[0].shape[2]
        descriptors = np.zeros((T, D), dtype=np.float32)
        valid = np.zeros(T, dtype=bool)

        for t in range(T):
            vis_feats_at_t: list[np.ndarray] = []
            for sampled, visibility in zip(cam_sampled, cam_visibility, strict=False):
                vis_mask = visibility[t]  # (N_cam,)
                if vis_mask.any():
                    vis_feats_at_t.append(sampled[t, vis_mask].numpy())  # (n_vis, D)

            if vis_feats_at_t:
                combined = np.concatenate(vis_feats_at_t, axis=0)  # (n_total, D)
                desc = combined.mean(axis=0)  # (D,)
                norm = float(np.linalg.norm(desc))
                if norm > 1e-8:
                    desc /= norm
                descriptors[t] = desc
                valid[t] = True

        n_valid = int(valid.sum())
        print(f"  {n_valid}/{T} timesteps with valid descriptors.")

        result = TemporalDescriptors(
            descriptors=descriptors, valid=valid, image_hw=(H, W)
        )
        self.last_ref_temporal = result
        return result

    def find_temporal_alignment(
        self,
        source_file: str | Path,
        demo_key_query: str,
        object_name: str,
        t_query: int,
        ref_temporal: TemporalDescriptors,
        cameras: list[str] | None = None,
    ) -> tuple[int, float, np.ndarray]:
        """Find the reference timestep that best matches the query object state.

        Tracks the target object in the query demo from t=0 up to *t_query* using
        CoTracker on all *cameras*, extracts DINOv3 features per tracked point per
        frame, then averages across **all tracked points, all frames, and all
        cameras** into a single global descriptor.  That descriptor is matched
        against the per-timestep reference descriptors in *ref_temporal* via cosine
        similarity to identify the closest reference moment.

        The query descriptor is cached in :attr:`last_query_descriptor`.
        Tracked points and frames are cached in :attr:`last_query_tracked_by_camera`
        and :attr:`last_query_camera_frames` (shared with :meth:`find_correspondences`).

        Args:
            source_file:      Path to the source HDF5 demo file.
            demo_key_query:   Query demo key (e.g. ``"demo_1"``).
            object_name:      Human-readable object name matching the segmentation map.
            t_query:          Last frame index (inclusive) for query tracking.
                              Frames ``[0, t_query]`` are used.
            ref_temporal:     Output of :meth:`compute_ref_temporal_descriptors`.
            cameras:          Camera streams (default: ``["eye_in_hand", "agentview"]``).

        Returns:
            ``(best_t_ref, similarity, similarities)`` where:

            * ``best_t_ref`` — index into the reference interaction (0-based) of the
              best-matching frame,
            * ``similarity`` — cosine similarity at ``best_t_ref`` in ``[-1, 1]``,
            * ``similarities`` — ``(T_ref,)`` float32 cosine similarity for every
              reference frame (invalid frames have value ``-inf``).
        """
        if cameras is None:
            cameras = ["eye_in_hand", "agentview"]

        print(f"\n[find_temporal_alignment] Tracking query object t=0..{t_query}…")

        # ── 1. Track query object and extract per-point avg features ──────────
        query_features = self._extract_tracked_features(
            source_file=source_file,
            demo_key=demo_key_query,
            start_timestep=0,
            end_timestep=t_query + 1,
            object_name=object_name,
            cameras=cameras,
            tracked_cache=self.last_query_tracked_by_camera,
            frames_cache=self.last_query_camera_frames,
        )

        N_q = len(query_features.point_ids)
        print(f"  Query: {N_q} tracked points across {cameras}.")

        # ── 2. Average all per-point features → single global descriptor ──────
        # avg_features is already per-point average across visible frames;
        # averaging again across points gives the global object descriptor.
        avg_per_point = query_features.avg_features  # (N_q, D)
        query_desc = avg_per_point.mean(axis=0)  # (D,)
        norm = float(np.linalg.norm(query_desc))
        if norm > 1e-8:
            query_desc = (query_desc / norm).astype(np.float32)
        self.last_query_descriptor = query_desc

        # ── 3. Cosine similarity against per-timestep reference descriptors ────
        ref_desc = ref_temporal.descriptors  # (T_ref, D)
        sims = (ref_desc @ query_desc).astype(np.float32)  # (T_ref,)

        # Mask invalid reference timesteps
        masked = np.where(ref_temporal.valid, sims, -np.inf)
        best_t = int(np.argmax(masked))
        best_score = float(masked[best_t])

        print(f"  Best match: t_ref={best_t}, similarity={best_score:.4f}")
        return best_t, best_score, sims

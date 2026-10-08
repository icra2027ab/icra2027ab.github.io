"""Track per-detection 2D point trajectories with CoTracker, lifted to 3D.

Runs SAM3 detection + reconstruction on a multi-frame video with the optional
CoTracker streaming wrapper enabled. Each detection's reconstructed mask
pixels are seeded as cotracker queries, then tracked through the video.

For visualization, each tracked 2D point ``(u, v)`` is lifted to a 3D world
position by sampling the corresponding frame's depth and unprojecting through
that frame's camera. The result is rendered as an orbiting 3D GIF showing
per-instance point *trails* growing through time alongside the SAM3 mask
point cloud for context.

Requires the ``[cotracker]`` extras::

    uv pip install -e 'packages/open_perception[cotracker]'
    uv run python packages/open_perception/examples/12_cotracker_points.py

Note: CoTracker has a warm-up period of ``2 * model.step - 1`` frames
(typically 7 for ``cotracker3_online``) where ``det.point_tracks is None``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from open_perception import (
    CameraInfo,
    OpenPerception,
    PerceptionConfig,
    PointTracks,
    render_pointcloud_orbit_gif,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _unproject_uv(
    uv: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray | None,
    *,
    flip_u: bool = True,
) -> np.ndarray:
    """Lift ``(N, 2)`` pixel coords to ``(M, 3)`` world-frame points.

    Drops out-of-bounds pixels and points with non-positive depth. Uses the
    same ``flip_u`` convention as ``mask_to_pointcloud`` so the result lines
    up with ``det.pointcloud``.
    """
    H, W = depth.shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return np.zeros((0, 3), dtype=np.float32)
    u, v = u[in_bounds], v[in_bounds]
    z = depth[v, u].astype(np.float32)
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u_K = (W - 1 - u) if flip_u else u
    X = (u_K.astype(np.float32) - cx) * z / fx
    Y = (v.astype(np.float32) - cy) * z / fy
    pts = np.stack([X, Y, z], axis=1)
    if extrinsics is not None:
        R = extrinsics[:3, :3].astype(np.float32)
        t = extrinsics[:3, 3].astype(np.float32)
        pts = pts @ R.T + t
    return pts


def _final_snapshot_per_instance(
    history,
) -> dict[int, PointTracks]:
    """Walk ``perc.history`` and pick the latest snapshot per instance.

    The cotracker snapshot from the last frame contains the cumulative
    trajectories for every prior frame, so one lookup per instance is enough.
    """
    out: dict[int, PointTracks] = {}
    for frame in reversed(history.frames):
        for dets in frame.detections.values():
            for det in dets:
                if (
                    det.instance_id is not None
                    and det.point_tracks is not None
                    and det.instance_id not in out
                    and det.instance_id in det.point_tracks.instance_slices
                ):
                    out[det.instance_id] = det.point_tracks
        if out:
            break
    return out


def main() -> None:
    multi = np.load(ASSETS / "demo" / "demo_multi_camera.npz")
    rgbs = multi["rgb_cam0"]
    depths = multi["depth_cam0"]
    K = multi["intrinsics_cam0"]
    Es = multi["extrinsics_cam0"]

    cfg = PerceptionConfig().order(
        ["sam3", "filter.overlap", "filter.erode", "reconstruct"]
    )
    cfg.update("reconstruct", flip_u=True)
    cfg.enable_point_tracking = True
    cfg.cotracker_max_points_per_detection = 32
    cfg.cotracker_min_points_per_detection = 4

    perc = OpenPerception(cfg)

    queries = ["mug", "plate"]
    for t, rgb in enumerate(rgbs):
        cam = CameraInfo(intrinsics=K, extrinsics=Es[t])
        if t == 0:
            perc.detect(queries, rgb, depth=depths[t], camera=cam)
        else:
            perc.track(rgb, depth=depths[t], camera=cam)

    print(f"tracked {len(perc.history)} frames")

    snapshots = _final_snapshot_per_instance(perc.history)
    if not snapshots:
        print("no cotracker snapshots in history (warm-up never completed)")
        return

    # Resolve a stable instance order + per-instance label and color slot.
    instance_ids = sorted(snapshots.keys())
    instance_labels: dict[int, str] = {}
    for frame in perc.history.frames:
        for q, dets in frame.detections.items():
            for det in dets:
                if (
                    det.instance_id in instance_ids
                    and det.instance_id not in instance_labels
                ):
                    instance_labels[det.instance_id] = f"{q}#{det.instance_id}"
    label_trail = [f"{instance_labels[i]} (trail)" for i in instance_ids]
    label_mask = [f"{instance_labels[i]} (sam3)" for i in instance_ids]
    labels = label_trail + label_mask

    # Per-frame point clouds. Object slot layout:
    #   slots 0..N-1            → cotracker 3D trail per instance (growing each frame)
    #   slots N..2N-1           → SAM3 reconstructed point cloud per instance (current)
    timesteps: list[list[np.ndarray]] = []
    cameras_per_ts: list[list[CameraInfo | None]] = []

    for t, frame in enumerate(perc.history.frames):
        cam_t = CameraInfo(intrinsics=K, extrinsics=Es[t])

        # SAM3 reconstructed cloud per instance on this frame.
        sam3_per_instance: dict[int, list[np.ndarray]] = {i: [] for i in instance_ids}
        for dets in frame.detections.values():
            for det in dets:
                if (
                    det.instance_id in sam3_per_instance
                    and det.pointcloud is not None
                    and det.pointcloud.size > 0
                ):
                    sam3_per_instance[det.instance_id].append(det.pointcloud)

        # CoTracker 3D trail per instance: positions for frames 0..t.
        trail_per_instance: dict[int, np.ndarray] = {}
        for inst_id in instance_ids:
            snap = snapshots[inst_id]
            sl = snap.instance_slices[inst_id]
            T_so_far = snap.tracks_uv.shape[0]
            t_clamped = min(t + 1, T_so_far)  # number of frames up to and including t
            if t_clamped <= 0:
                trail_per_instance[inst_id] = np.zeros((0, 3), dtype=np.float32)
                continue
            trail_pts: list[np.ndarray] = []
            for s in range(t_clamped):
                if s >= len(perc.history):
                    break
                uv_s = snap.tracks_uv[s, sl, :]
                vis_s = snap.visibility[s, sl] if snap.visibility.size else None
                if vis_s is not None and vis_s.size:
                    uv_s = uv_s[vis_s]
                if uv_s.size == 0:
                    continue
                pts = _unproject_uv(uv_s, depths[s], K, Es[s], flip_u=True)
                if pts.size:
                    trail_pts.append(pts)
            trail_per_instance[inst_id] = (
                np.concatenate(trail_pts, axis=0)
                if trail_pts
                else np.zeros((0, 3), dtype=np.float32)
            )

        slots: list[np.ndarray] = []
        slots.extend(trail_per_instance[i] for i in instance_ids)
        slots.extend(
            np.concatenate(sam3_per_instance[i], axis=0)
            if sam3_per_instance[i]
            else np.zeros((0, 3), dtype=np.float32)
            for i in instance_ids
        )
        timesteps.append(slots)
        cameras_per_ts.append([cam_t])

    out_path = Path("tmp/12/cotracker_3d_trails.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_pointcloud_orbit_gif(
        timesteps,
        out_path,
        cameras=cameras_per_ts,
        labels=labels,
        point_size=3.0,
        orbit_turns=0.6,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

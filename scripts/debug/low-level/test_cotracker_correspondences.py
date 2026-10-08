"""Test script for InteractionPointTracker: CoTracker + DINOv3 correspondence matching
and temporal alignment.

Loads interaction metadata from the extracted interactions dataset, tracks
segmented object points through BOTH cameras (eye_in_hand + agentview) using
CoTracker, then:

1. Point-level correspondences: finds pixel-level correspondences between the
   tracked reference points and a query interaction (tracked from t=0 to t_query,
   both cameras) via per-point average DINOv3 feature similarity.

2. Temporal alignment: tracks the query object from t=0 to t_query (both cameras),
   averages DINOv3 features across ALL tracked points and ALL frames into a single
   global descriptor, then finds the reference frame with the highest cosine
   similarity against per-timestep reference descriptors.

Visualizes results in Rerun:
  - Left:  Reference interaction frames for each camera (time-stepped with tracked
           points), with per-timestep similarity curve for temporal alignment.
  - Right: Query frames at t_query for each camera with matched correspondence
           points, a DINOv3 similarity heatmap, and side-by-side correspondence video.

Usage::

    uv run python tmp/test_cotracker_correspondence.py [options]

    Options:
      --interactions   Path to interactions HDF5 (default: datasets/interactions/libero_object_interactions.hdf5)
      --interaction-id Index of the interaction to use (default: 0)
      --demo-query     Demo index for the query observation (default: 1)
      --t-query        Last frame index (inclusive) for query tracking (default: 55)
      --device         cuda or cpu (default: cuda)
      --top-k          Number of correspondences to return (default: 10)
      --skip-correspondences  Skip point-level correspondence matching (faster)
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib.cm as cm
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Make the project src/ importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from policy_abstraction.perception.cotracker_correspondence import (  # noqa: E402
    InteractionPointTracker,
    PixelCorrespondence,
    TrackedPointFeatures,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INTERACTIONS_PATH = (
    _REPO_ROOT / "datasets" / "interactions" / "libero_object_interactions.hdf5"
)

_REMOTE_PREFIX = os.environ.get(
    "POLICY_ABSTRACTION_LEGACY_PREFIX", "/scratch/<user>/policy_abstraction/"
)
_LOCAL_PREFIX = str(_REPO_ROOT) + "/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def remap_source_path(source_file: str) -> Path:
    local = source_file.replace(_REMOTE_PREFIX, _LOCAL_PREFIX)
    p = Path(local)
    if not p.exists():
        name = Path(source_file).name
        candidates = list(_REPO_ROOT.glob(f"datasets/**/{name}"))
        if candidates:
            return candidates[0]
    return p


def load_interaction_metadata(interactions_path: Path, interaction_id: int) -> dict:
    with h5py.File(interactions_path, "r") as f:
        attrs = dict(f["interactions"][str(interaction_id)].attrs)
    attrs["source_file"] = remap_source_path(attrs["source_file"])
    return attrs


def point_colors_hsv(n: int) -> np.ndarray:
    colors = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.95)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255), 255]
    return colors


def rgba_to_bgr(colors: np.ndarray) -> np.ndarray:
    return colors[:, :3][:, ::-1].copy()


def dino_similarity_heatmap(
    obs_image: np.ndarray,
    tracked_features: TrackedPointFeatures,
    dino_extractor,
    point_idx: int = 0,
    dino_img_size: int = 448,
) -> np.ndarray:
    """Cosine-similarity heatmap between a reference point feature and all query patches."""
    H_obs, W_obs = obs_image.shape[:2]
    obs_feat = dino_extractor.extract(
        Image.fromarray(obs_image), img_size=dino_img_size
    )
    H_p, W_p, D = obs_feat.shape
    obs_flat = F.normalize(obs_feat.reshape(H_p * W_p, D), dim=-1)
    ref_vec = torch.from_numpy(tracked_features.avg_features[point_idx])
    sim = (obs_flat @ ref_vec).reshape(H_p, W_p).numpy()
    sim_up = cv2.resize(sim, (W_obs, H_obs), interpolation=cv2.INTER_LINEAR)
    sim_norm = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-8)
    return (cm.jet(sim_norm)[:, :, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Rerun visualization
# ---------------------------------------------------------------------------


def log_to_rerun(
    ref_frames_by_camera: dict[str, np.ndarray],  # cam → (T, H, W, 3)
    ref_tracked_by_camera: dict,  # cam → TrackedPoints
    query_frames_by_camera: dict[str, np.ndarray],  # cam → (T_q, H, W, 3)
    query_tracked_by_camera: dict,  # cam → TrackedPoints
    tracked_features: TrackedPointFeatures,
    correspondences: list[PixelCorrespondence],
    heatmap: np.ndarray,
    meta: dict,
    t_query: int,
    cameras: list[str],
    temporal_similarities: np.ndarray | None = None,  # (T_ref,) cosine sims
    temporal_best_t: int | None = None,
) -> None:
    T_ref = next(iter(ref_frames_by_camera.values())).shape[0]
    T_q = next(iter(query_frames_by_camera.values())).shape[0]

    N_ref = len(tracked_features.point_ids)
    ref_colors = point_colors_hsv(N_ref)

    # Build per-camera query color tables (per-camera TrackedPoints has its own point_ids)
    query_colors_by_camera: dict[str, np.ndarray] = {}
    for cam, qt in query_tracked_by_camera.items():
        query_colors_by_camera[cam] = point_colors_hsv(len(qt.point_ids))

    ref_views = [
        rrb.Spatial2DView(name=f"Ref – {cam}", origin=f"ref/{cam}") for cam in cameras
    ]
    query_views = [
        rrb.Spatial2DView(name=f"Query – {cam}", origin=f"query/{cam}")
        for cam in cameras
    ]

    extra_views = [
        rrb.Spatial2DView(name="DINO Heatmap (top-1 ref pt, wrist)", origin="heatmap"),
        rrb.Spatial2DView(name="Correspondence Video", origin="correspondence_video"),
    ]
    if temporal_similarities is not None:
        extra_views.append(
            rrb.TimeSeriesView(
                name="Temporal Alignment Similarity", origin="temporal_sim"
            )
        )

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*ref_views),
            rrb.Vertical(*query_views, *extra_views),
        )
    )
    rr.send_blueprint(blueprint)

    rr.log(
        "info/interaction",
        rr.TextDocument(
            f"object: {meta['object_name']}\n"
            f"demo: {meta['demo_key']}\n"
            f"t: [{meta['start_timestep']}, {meta['end_timestep']})\n"
            f"chain: {meta.get('interaction_chain', '')}"
        ),
        static=True,
    )

    # ── Reference frames + tracks (time-stepped) ───────────────────────────────
    for t in range(T_ref):
        rr.set_time("frame", sequence=t)
        for cam, frames in ref_frames_by_camera.items():
            rr.log(f"ref/{cam}/image", rr.Image(frames[t]))
        for cam, tracked in ref_tracked_by_camera.items():
            if t >= tracked.visibility.shape[0]:
                continue
            vis_mask = tracked.visibility[t]  # (N_cam,)
            if not vis_mask.any():
                continue
            # Map local cam indices → global ref_colors index via point_ids
            cam_mask = tracked_features.camera_ids == cam
            global_indices = np.where(cam_mask)[0]  # positions in tracked_features
            local_valid = np.where(vis_mask)[0]
            pts = tracked.tracks[t][local_valid].astype(np.float32)
            # local_valid[i] → global index if within bounds
            colors_for_pts = (
                ref_colors[
                    global_indices[local_valid[local_valid < len(global_indices)]]
                ]
                if len(global_indices)
                else ref_colors[: len(pts)]
            )
            if len(pts):
                rr.log(
                    f"ref/{cam}/tracks",
                    rr.Points2D(pts, colors=colors_for_pts, radii=3.0),
                )

        # Log per-timestep temporal similarity score
        if temporal_similarities is not None and t < len(temporal_similarities):
            sim_val = float(temporal_similarities[t])
            if np.isfinite(sim_val):
                rr.log("temporal_sim/similarity", rr.Scalars(sim_val))
            # Highlight best-match frame
            if temporal_best_t is not None and t == temporal_best_t:
                rr.log("temporal_sim/best_match", rr.Scalars(sim_val))

    # ── Query frames + tracks (time-stepped) ──────────────────────────────────
    for t in range(T_q):
        rr.set_time("frame", sequence=t)
        for cam, frames in query_frames_by_camera.items():
            rr.log(f"query/{cam}/image", rr.Image(frames[t]))
        for cam, tracked in query_tracked_by_camera.items():
            if t >= tracked.visibility.shape[0]:
                continue
            vis_mask = tracked.visibility[t]
            if not vis_mask.any():
                continue
            pts = tracked.tracks[t][vis_mask].astype(np.float32)
            colors_q = query_colors_by_camera[cam]
            rr.log(
                f"query/{cam}/tracks",
                rr.Points2D(pts, colors=colors_q[vis_mask], radii=3.0),
            )

    rr.log("heatmap/image", rr.Image(heatmap), static=True)

    if not correspondences:
        print("  [Rerun] No correspondences to log.")
        return

    # ── Correspondences at t_query ─────────────────────────────────────────────
    rr.set_time("frame", sequence=t_query)
    for cam in cameras:
        obs_pts = np.array(
            [c.obs_xy for c in correspondences if c.obs_camera == cam], dtype=np.float32
        )
        ref_pts = np.array(
            [c.ref_xy for c in correspondences if c.ref_camera == cam], dtype=np.float32
        )
        if len(obs_pts):
            rr.log(
                f"query/{cam}/matched_points",
                rr.Points2D(obs_pts, radii=5.0),
                static=True,
            )
        if len(ref_pts):
            rr.log(
                f"ref/{cam}/matched_ref_points",
                rr.Points2D(ref_pts, radii=5.0),
                static=True,
            )

    print(f"\n  Top-{len(correspondences)} correspondences:")
    print(
        f"  {'pt_id':>6}  {'obs_x':>7}  {'obs_y':>7}  {'ref_x':>7}  {'ref_y':>7}  {'score':>7}  obs_cam   ref_cam"
    )
    for c in correspondences:
        print(
            f"  {c.point_id:>6}  {c.obs_xy[0]:>7.1f}  {c.obs_xy[1]:>7.1f}  "
            f"{c.ref_xy[0]:>7.1f}  {c.ref_xy[1]:>7.1f}  {c.score:>7.4f}  "
            f"{c.obs_camera:<9} {c.ref_camera}"
        )


def render_correspondence_video(
    ref_frames_by_camera: dict[str, np.ndarray],
    query_frames_by_camera: dict[str, np.ndarray],
    correspondences: list[PixelCorrespondence],
    ref_tracked_by_camera: dict,
    tracked_features: TrackedPointFeatures,
    t_query: int,
    video_path: Path | None = None,
    fps: float = 12.0,
    display: bool = False,
    gap: int = 40,
    log_to_rr: bool = True,
    camera: str = "eye_in_hand",
) -> None:
    """Side-by-side video: reference (scrolling) vs query (fixed at t_query), wrist cam."""
    if not (display or video_path is not None or log_to_rr):
        return
    if not correspondences:
        print("Skipped correspondence video: no correspondences found.")
        return

    ref_frames = ref_frames_by_camera.get(camera)
    query_frames = query_frames_by_camera.get(camera)
    if ref_frames is None or query_frames is None:
        print(f"Skipped correspondence video: '{camera}' not available.")
        return

    tracked = ref_tracked_by_camera.get(camera)
    if tracked is None:
        return

    # Filter to correspondences on the reference camera side
    cam_corrs = [c for c in correspondences if c.ref_camera == camera]
    if not cam_corrs:
        print(f"Skipped correspondence video: no '{camera}' ref correspondences.")
        return

    # Local index inside tracked.tracks for each reference point_id
    cam_mask = tracked_features.camera_ids == camera
    cam_point_ids = tracked_features.point_ids[cam_mask]
    pid_to_local = {int(pid): i for i, pid in enumerate(cam_point_ids)}

    colors_bgr = rgba_to_bgr(point_colors_hsv(len(cam_point_ids)))

    query_frame = query_frames[min(t_query, len(query_frames) - 1)]
    H_q, W_q = query_frame.shape[:2]
    H_r, W_r = ref_frames.shape[1:3]
    canvas_h, canvas_w = max(H_q, H_r), W_q + gap + W_r
    ref_offset_x = W_q + gap

    fps = max(fps, 1.0)
    query_bgr = cv2.cvtColor(query_frame, cv2.COLOR_RGB2BGR)

    writer = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (canvas_w, canvas_h))
        if not writer.isOpened():
            print(f"WARNING: could not open video writer at {video_path}")
            writer = None

    if display:
        cv2.namedWindow("Correspondence Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Correspondence Video", canvas_w, canvas_h)

    for t, frame_rgb in enumerate(ref_frames):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:H_q, :W_q] = query_bgr
        canvas[:H_r, ref_offset_x : ref_offset_x + W_r] = cv2.cvtColor(
            frame_rgb, cv2.COLOR_RGB2BGR
        )
        cv2.putText(
            canvas,
            f"Frame {t}",
            (ref_offset_x + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        for corr in cam_corrs:
            local_idx = pid_to_local.get(int(corr.point_id))
            if local_idx is None or t >= tracked.visibility.shape[0]:
                continue
            if not tracked.visibility[t, local_idx]:
                continue
            ref_xy = tracked.tracks[t, local_idx]
            if np.any(np.isnan(ref_xy)):
                continue
            color = tuple(int(v) for v in colors_bgr[local_idx])
            obs_pt = (int(round(corr.obs_xy[0])), int(round(corr.obs_xy[1])))
            ref_pt = (int(round(ref_xy[0] + ref_offset_x)), int(round(ref_xy[1])))
            cv2.circle(canvas, obs_pt, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, ref_pt, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.line(canvas, obs_pt, ref_pt, color, 2, lineType=cv2.LINE_AA)

        if writer is not None:
            writer.write(canvas)
        if display:
            cv2.imshow("Correspondence Video", canvas)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break
        if log_to_rr:
            rr.set_time("frame", sequence=t)
            rr.log(
                "correspondence_video/image",
                rr.Image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)),
            )

    if writer is not None:
        writer.release()
        print(f"Saved correspondence video to {video_path}")
    if display:
        cv2.destroyWindow("Correspondence Video")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS_PATH)
    parser.add_argument("--interaction-id", type=int, default=0)
    parser.add_argument("--demo-query", type=int, default=1)
    parser.add_argument(
        "--t-query",
        type=int,
        default=55,
        help="Last frame index (inclusive) for query tracking",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--video-out", type=Path, default=None)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument(
        "--skip-correspondences",
        action="store_true",
        help="Skip point-level correspondence matching (still runs temporal alignment)",
    )
    args = parser.parse_args()

    if not args.interactions.exists():
        print(f"ERROR: Interactions file not found: {args.interactions}")
        sys.exit(1)

    meta = load_interaction_metadata(args.interactions, args.interaction_id)
    source_file: Path = meta["source_file"]
    demo_key_ref: str = meta["demo_key"]
    start: int = int(meta["start_timestep"])
    end: int = int(meta["end_timestep"])
    object_name: str = meta["object_name"]
    demo_key_query = f"demo_{args.demo_query}"

    print(f"Interactions: {args.interactions.name}")
    print(
        f"Interaction #{args.interaction_id}: {meta.get('interaction_chain', object_name)}"
    )
    print(f"  Source:  {source_file.name}")
    print(f"  Demo:    {demo_key_ref}  [{start}:{end}]  ({end - start} frames)")
    print(f"  Object:  {object_name}")
    print(f"Query:     {demo_key_query}  t=0..{args.t_query} (both cameras)")
    print()

    if not source_file.exists():
        print(f"ERROR: Source demo not found: {source_file}")
        sys.exit(1)

    rr.init("cotracker_correspondence", spawn=True)

    cameras = ["eye_in_hand", "agentview"]
    tracker = InteractionPointTracker(device=args.device)

    # ── Process reference interaction (both cameras) ─────────────────────────
    tracked_features = tracker.process_interaction(
        source_file=source_file,
        demo_key=demo_key_ref,
        start_timestep=start,
        end_timestep=end,
        object_name=object_name,
        cameras=cameras,
    )

    # ── Temporal alignment: per-timestep reference descriptors ───────────────
    print("\nComputing per-timestep reference descriptors for temporal alignment…")
    ref_temporal = tracker.compute_ref_temporal_descriptors(
        source_file=source_file,
        demo_key=demo_key_ref,
        start_timestep=start,
        end_timestep=end,
        object_name=object_name,
        cameras=cameras,
    )

    print(f"\nRunning temporal alignment (query t=0..{args.t_query}, both cameras)…")
    best_t_ref, best_sim, temporal_similarities = tracker.find_temporal_alignment(
        source_file=source_file,
        demo_key_query=demo_key_query,
        object_name=object_name,
        t_query=args.t_query,
        ref_temporal=ref_temporal,
        cameras=cameras,
    )
    print(
        f"Temporal alignment result: ref_t={best_t_ref} / {ref_temporal.descriptors.shape[0] - 1} "
        f"(similarity={best_sim:.4f})"
    )

    # ── Point-level correspondences (optional) ───────────────────────────────
    correspondences: list[PixelCorrespondence] = []
    if not args.skip_correspondences:
        print(f"\nFinding top-{args.top_k} correspondences…")
        correspondences = tracker.find_correspondences(
            source_file=source_file,
            demo_key_query=demo_key_query,
            object_name=object_name,
            t_query=args.t_query,
            tracked_features=tracked_features,
            cameras=cameras,
            top_k=args.top_k,
            mutual=True,
        )
        print(f"Found {len(correspondences)} mutual nearest-neighbour correspondences.")
    else:
        # Ensure query tracking cache is populated for visualization
        # (find_temporal_alignment already populated last_query_tracked_by_camera)
        print("Skipping point-level correspondences (--skip-correspondences).")

    # ── Retrieve cached frames and tracked points for visualization ───────────
    ref_frames_by_camera = tracker.last_ref_camera_frames
    ref_tracked_by_camera = tracker.last_ref_tracked_by_camera
    query_frames_by_camera = tracker.last_query_camera_frames
    query_tracked_by_camera = tracker.last_query_tracked_by_camera

    # ── DINO heatmap: top-1 reference point vs wrist query frame at t_query ───
    print("\nComputing DINO similarity heatmap…")
    wrist_q_frames = query_frames_by_camera.get("eye_in_hand")
    wrist_q_frame = wrist_q_frames[min(args.t_query, len(wrist_q_frames) - 1)]

    pid_to_ref_idx = {int(pid): i for i, pid in enumerate(tracked_features.point_ids)}
    top_ref_idx = (
        pid_to_ref_idx.get(correspondences[0].point_id, 0) if correspondences else 0
    )
    heatmap = dino_similarity_heatmap(
        wrist_q_frame, tracked_features, tracker.dino, point_idx=top_ref_idx
    )

    # ── Visualize in Rerun ───────────────────────────────────────────────────
    print("\nLogging to Rerun…")
    log_to_rerun(
        ref_frames_by_camera=ref_frames_by_camera,
        ref_tracked_by_camera=ref_tracked_by_camera,
        query_frames_by_camera=query_frames_by_camera,
        query_tracked_by_camera=query_tracked_by_camera,
        tracked_features=tracked_features,
        correspondences=correspondences,
        heatmap=heatmap,
        meta=meta,
        t_query=args.t_query,
        cameras=cameras,
        temporal_similarities=temporal_similarities,
        temporal_best_t=best_t_ref,
    )

    render_correspondence_video(
        ref_frames_by_camera=ref_frames_by_camera,
        query_frames_by_camera=query_frames_by_camera,
        correspondences=correspondences,
        ref_tracked_by_camera=ref_tracked_by_camera,
        tracked_features=tracked_features,
        t_query=args.t_query,
        video_path=args.video_out,
        fps=args.video_fps,
        display=args.show_video,
    )

    print("\nDone. Rerun viewer should be open.")


if __name__ == "__main__":
    main()

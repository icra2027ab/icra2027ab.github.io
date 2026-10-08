"""Save videos of camera views from a LIBERO dataset demo file.

When ``--annotate-fingers`` is passed (the default), the video also overlays
the projected 3-D positions of both Panda finger tips on every frame:
  • left  finger → green dot
  • right finger → red   dot

Pass ``--rerun`` to visualise trajectories and camera feeds live in Rerun
instead of (or in addition to) saving a video.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio
import numpy as np
import rerun as rr

from policy_abstraction.envs.libero.config import LIBERO_DEPTH_FAR, LIBERO_DEPTH_NEAR
from policy_abstraction.features.base import DistanceFeature, compute_distances
from policy_abstraction.features.centroid import CentroidDistance
from policy_abstraction.perception.dino_projector import DinoPCAProjector
from policy_abstraction.utils.depth import depth_to_color, mujoco_depth_to_metric
from policy_abstraction.utils.gripper import compute_finger_tip_positions
from policy_abstraction.utils.pcd import pixels_to_3d, project_points_to_pixels
from policy_abstraction.utils.segmentation import (
    _make_palette,
    build_segmented_pointcloud,
    colorize_dual_segmentation,
    colorize_segmentation,
    draw_dual_interaction_legend,
    draw_interaction_legend,
    find_interacting_instance_ids,
    merge_pointclouds,
)
from policy_abstraction.visualization.drawing import (
    draw_pairs_on_image,
)
from policy_abstraction.vlm.types import DistancePair

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helper
# ─────────────────────────────────────────────────────────────────────────────


def _draw_finger_dots(
    img: np.ndarray,
    pts_2d: np.ndarray,
    valid: np.ndarray,
    colors: list[tuple[int, int, int]],
    radius: int = 6,
) -> np.ndarray:
    """Draw filled circles for each valid projected point (in-place copy)."""
    out = img.copy()
    H, W = out.shape[:2]
    for i, (color, is_valid) in enumerate(zip(colors, valid, strict=False)):
        if not is_valid:
            continue
        u, v = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(out, (u, v), radius, color, thickness=-1)
            cv2.circle(out, (u, v), radius + 1, (0, 0, 0), thickness=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Proximity helper
# ─────────────────────────────────────────────────────────────────────────────


def _gripper_near_vlm_ids(
    seg_mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    finger_tips: np.ndarray,
    vlm_ids: set[int],
    thresh: float = 0.10,
) -> bool:
    """Return True if any finger tip is within *thresh* metres of any VLM-selected segment.

    Back-projects the agentview pixels belonging to *vlm_ids* into 3-D world
    coordinates and queries the nearest-neighbour distance to each finger tip.
    """
    from scipy.spatial import KDTree

    seg_2d = seg_mask[..., 0] if seg_mask.ndim == 3 else seg_mask
    dep_2d = depth[..., 0] if depth.ndim == 3 else depth

    mask = np.zeros(seg_2d.shape, dtype=bool)
    for sid in vlm_ids:
        mask |= seg_2d == sid
    mask &= dep_2d > 1e-3
    if not mask.any():
        return False

    H, W = dep_2d.shape
    ys, xs = np.where(mask)
    pixels = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=-1)
    depths = dep_2d[mask]
    xyz = pixels_to_3d(pixels, depths, K, E, image_width=W, preprocessed=True)
    if xyz.shape[0] == 0:
        return False

    tree = KDTree(xyz)
    for tip in finger_tips:
        d, _ = tree.query(tip)
        if d < thresh:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Video saver
# ─────────────────────────────────────────────────────────────────────────────


def save_demo_video(
    demo_data,
    output_path: Path,
    fps: int = 20,
    annotate_fingers: bool = True,
    interacting_ids: set[int] | None = None,
    vlm_selected_ids: set[int] | None = None,
    distance_pairs: list[DistancePair] | None = None,
    seg_name_map: dict[str, str] | None = None,
    depth_near: float = LIBERO_DEPTH_NEAR,
    depth_far: float = LIBERO_DEPTH_FAR,
    dino_projector: DinoPCAProjector | None = None,
) -> None:
    """Save a side-by-side video for one demo.

    Layout per frame (left → right):
      agentview RGB | agentview depth | eye-in-hand RGB | eye-in-hand depth
      | segmentation overlay (agentview, interacting objects highlighted)

    If *annotate_fingers* is True the projected 3-D finger-tip positions are
    overlaid on both camera views.

    If *interacting_ids* is provided a segmentation overlay panel is added.
    When *vlm_selected_ids* is also provided the overlay shows both sets with
    a dual colour scheme and legend.
    """
    obs = demo_data["obs"]
    agentview = obs["agentview_rgb"][:][:, ::-1, ::-1, :]  # (T, H, W, 3)
    eye_in_hand = obs["eye_in_hand_rgb"][:][:, ::-1, ::-1, :]  # (T, H, W, 3)

    agentview_depth = obs["agentview_depth"][:][:, ::-1, ::-1]  # (T, H, W, 1)
    eye_in_hand_depth = obs["eye_in_hand_depth"][:][:, ::-1, ::-1]  # (T, H, W, 1)

    # Convert normalised MuJoCo depth [0, 1] → metric depth (metres) if needed.
    if agentview_depth.max() <= 1.0:
        agentview_depth = mujoco_depth_to_metric(agentview_depth, depth_near, depth_far)
        eye_in_hand_depth = mujoco_depth_to_metric(
            eye_in_hand_depth, depth_near, depth_far
        )

    agentview_seg = obs["agentview_segmentation"][:][:, ::-1, ::-1]  # (T, H, W, 1)
    eye_in_hand_seg = obs["eye_in_hand_segmentation"][:][:, ::-1, ::-1]  # (T, H, W, 1)

    T, H, W, _ = agentview.shape

    ee_pos = obs["ee_pos"][:]  # (T, 3)
    ee_ori = obs["ee_ori"][:]  # (T, 3)  axis-angle
    gripper_states = obs["gripper_states"][:]  # (T, 2)

    left_tips, right_tips = compute_finger_tip_positions(ee_pos, ee_ori, gripper_states)

    av_K = obs["agentview_intrinsic"][:]  # (T, 3, 3)
    av_E = obs["agentview_extrinsic"][:]  # (T, 4, 4)
    eih_K = obs["eye_in_hand_intrinsic"][:]  # (T, 3, 3)
    eih_E = obs["eye_in_hand_extrinsic"][:]  # (T, 4, 4)

    # BGR colors (cv2 convention)
    LEFT_COLOR = (0, 255, 0)  # green
    RIGHT_COLOR = (0, 0, 255)  # red

    # ------------------------------------------------------------------
    # Pre-compute per-frame DINO wrist panels using batched inference.
    # We first identify every timestep where the gripper is near a
    # VLM-selected segment, then run all those frames through DINOv2
    # in a single forward pass instead of one-at-a-time.
    # ------------------------------------------------------------------
    near_mask: list[bool] = [False] * T
    dino_wrist_cache: dict[int, np.ndarray] = {}

    if interacting_ids is not None and vlm_selected_ids:
        for t in range(T):
            near_mask[t] = _gripper_near_vlm_ids(
                agentview_seg[t],
                agentview_depth[t],
                av_K[t],
                av_E[t],
                np.stack([left_tips[t], right_tips[t]]),
                vlm_selected_ids,
                thresh=0.10,
            )

        near_timesteps = [t for t in range(T) if near_mask[t]]
        if near_timesteps and dino_projector is not None:
            batch_rgb = [eye_in_hand[t] for t in near_timesteps]
            batch_seg = [eye_in_hand_seg[t] for t in near_timesteps]
            batch_vis = dino_projector.process_frames_batch(
                batch_rgb, batch_seg, vlm_selected_ids
            )
            dino_wrist_cache = dict(zip(near_timesteps, batch_vis, strict=False))

    with imageio.get_writer(str(output_path), fps=fps, macro_block_size=1) as writer:
        for t in range(T):
            av_frame = agentview[t].copy()
            eih_frame = eye_in_hand[t].copy()

            av_depth = agentview_depth[t].copy()
            eih_depth = eye_in_hand_depth[t].copy()

            tips = np.stack([left_tips[t], right_tips[t]])  # (2, 3)
            tip_colors = [LEFT_COLOR, RIGHT_COLOR]

            if annotate_fingers:
                # Agentview projection
                pts_av, valid_av = project_points_to_pixels(
                    tips, av_K[t], av_E[t], H, W, preprocessed=True
                )
                av_frame = _draw_finger_dots(av_frame, pts_av, valid_av, tip_colors)

                # Eye-in-hand projection
                pts_eih, valid_eih = project_points_to_pixels(
                    tips, eih_K[t], eih_E[t], H, W, preprocessed=True
                )
                eih_frame = _draw_finger_dots(eih_frame, pts_eih, valid_eih, tip_colors)

            panels = [av_frame, eih_frame]
            panels2 = [depth_to_color(av_depth), depth_to_color(eih_depth)]

            # Segmentation overlay panel
            if interacting_ids is not None:
                seg_t = agentview_seg[t]  # (H, W, 1)
                id_to_name = (
                    {int(k): v for k, v in seg_name_map.items()}
                    if seg_name_map
                    else None
                )
                if vlm_selected_ids is not None:
                    seg_panel = colorize_dual_segmentation(
                        seg_t, interacting_ids, vlm_selected_ids
                    )
                    seg_panel = draw_dual_interaction_legend(
                        seg_panel,
                        interacting_ids,
                        vlm_selected_ids,
                        id_to_name=id_to_name,
                    )
                else:
                    seg_panel = colorize_segmentation(seg_t, interacting_ids)  # BGR
                    seg_panel = draw_interaction_legend(
                        seg_panel, interacting_ids, id_to_name=id_to_name
                    )

                # Use the pre-computed batched DINO result when the gripper is
                # near a VLM-selected object, otherwise fall back to a segmentation
                # overlay or leave the panel blank.
                wrist_panel: np.ndarray | None = None
                if vlm_selected_ids:
                    if t in dino_wrist_cache:
                        wrist_panel = dino_wrist_cache[t]
                    elif near_mask[t]:
                        # Fallback: segmentation overlay (dino_projector unavailable).
                        eih_seg_t = eye_in_hand_seg[t]
                        wrist_seg_bgr = colorize_dual_segmentation(
                            eih_seg_t, interacting_ids, vlm_selected_ids
                        )
                        eih_bgr = cv2.cvtColor(eih_frame, cv2.COLOR_RGB2BGR)
                        wrist_blended_bgr = cv2.addWeighted(
                            eih_bgr, 0.4, wrist_seg_bgr, 0.6, 0
                        )
                        wrist_panel = cv2.cvtColor(wrist_blended_bgr, cv2.COLOR_BGR2RGB)

                # Draw lines connecting VLM-selected distance pair entities.
                if distance_pairs:
                    seg_mask_2d = seg_t[..., 0] if seg_t.ndim == 3 else seg_t
                    K3 = av_K[t][:3, :3] if av_K[t].shape == (4, 4) else av_K[t]
                    draw_pairs_on_image(
                        seg_panel,
                        distance_pairs,
                        seg_mask_2d,
                        K3,
                        av_E[t],
                        ee_pos[t],
                        H,
                        W,
                        distance_errors={},
                        live_distances=None,
                        entity_3d=None,
                    )

                # Convert BGR→RGB for imageio
                seg_panel_rgb = cv2.cvtColor(seg_panel, cv2.COLOR_BGR2RGB)
                panels.append(seg_panel_rgb)
                panels2.append(
                    wrist_panel
                    if wrist_panel is not None
                    else np.zeros_like(seg_panel_rgb)
                )

            frame = np.concatenate(
                (np.concatenate(panels, axis=1), np.concatenate(panels2, axis=1)),
                axis=0,
            )

            writer.append_data(frame)  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Rerun visualizer
# ─────────────────────────────────────────────────────────────────────────────


def _pcd_colors_for_rerun(
    pcd: np.ndarray,
    interacting_ids: set[int],
) -> np.ndarray:
    """Map seg_id to RGBA colours: interacting ids → distinct hues, rest → grey.

    Args:
        pcd:              ``[M, 7]`` array ``[X, Y, Z, R, G, B, seg_id]``.
        interacting_ids:  Set of IDs to highlight.

    Returns:
        ``[M, 4]`` uint8 RGBA colour array.
    """
    seg_ids = pcd[:, 6].astype(np.int32)
    palette = _make_palette(len(interacting_ids))
    id_to_color: dict[int, tuple[int, int, int]] = {
        inst_id: bgr
        for inst_id, bgr in zip(sorted(interacting_ids), palette, strict=False)
    }

    colors = np.full((len(seg_ids), 4), [80, 80, 80, 180], dtype=np.uint8)
    for inst_id, bgr in id_to_color.items():
        mask = seg_ids == inst_id
        # palette gives BGR; Rerun expects RGB
        colors[mask] = [bgr[2], bgr[1], bgr[0], 255]

    return colors


def visualize_in_rerun(
    demo_data,
    demo_key: str,
    interacting_ids: set[int] | None = None,
    vlm_selected_ids: set[int] | None = None,
    seg_annotated_image: np.ndarray | None = None,
    distance_pairs: list[DistancePair] | None = None,
    distance_feature: DistanceFeature | None = None,
    depth_near: float = LIBERO_DEPTH_NEAR,
    depth_far: float = LIBERO_DEPTH_FAR,
) -> None:
    """Visualise a demo rollout live in the Rerun viewer.

    Logs per-timestep RGB images, depth, finger-tip positions, the combined
    point cloud (agentview + eye-in-hand) coloured by interaction status, and
    optional distance scalar plots derived from VLM-selected pairs.

    Args:
        demo_data:          HDF5 group for one demo (contains ``"obs"``).
        demo_key:           Human-readable key used as the Rerun application ID.
        interacting_ids:    Pre-computed gripper-contact instance IDs.  When
                            ``None`` they are recomputed from the rollout.
        vlm_selected_ids:   VLM-selected instance IDs (for highlighting).
        seg_annotated_image: RGB image labelling all segment IDs (logged static).
        distance_pairs:     VLM-selected entity pairs to plot distances for.
        distance_feature:   Distance metric; defaults to :class:`CentroidDistance`.
        depth_near:         Near-clip depth for MuJoCo depth conversion.
        depth_far:          Far-clip depth for MuJoCo depth conversion.
    """
    from policy_abstraction.visualization.rerun import RerunVisualizer

    obs = demo_data["obs"]
    T = obs["agentview_rgb"].shape[0]

    ee_pos = obs["ee_pos"][:]  # (T, 3)
    ee_ori = obs["ee_ori"][:]  # (T, 3)
    gripper_states = obs["gripper_states"][:]  # (T, 2)
    left_tips, right_tips = compute_finger_tip_positions(ee_pos, ee_ori, gripper_states)

    av_K = obs["agentview_intrinsic"][:]  # (T, 3, 3)
    av_E = obs["agentview_extrinsic"][:]  # (T, 4, 4)
    eih_K = obs["eye_in_hand_intrinsic"][:]  # (T, 3, 3)
    eih_E = obs["eye_in_hand_extrinsic"][:]  # (T, 4, 4)

    # Load all image arrays into numpy first (h5py doesn't support negative strides)
    agentview_rgb = obs["agentview_rgb"][:]
    eye_in_hand_rgb = obs["eye_in_hand_rgb"][:]
    agentview_depth = obs["agentview_depth"][:]
    eye_in_hand_depth = obs["eye_in_hand_depth"][:]
    agentview_seg = obs["agentview_segmentation"][:]
    eye_in_hand_seg = obs["eye_in_hand_segmentation"][:]

    # Convert normalised MuJoCo depth [0, 1] → metric depth (metres) if needed.
    if agentview_depth.max() <= 1.0:
        agentview_depth = mujoco_depth_to_metric(agentview_depth, depth_near, depth_far)
        eye_in_hand_depth = mujoco_depth_to_metric(
            eye_in_hand_depth, depth_near, depth_far
        )

    # Preprocessed (180°-flipped) versions
    av_rgb_pp = agentview_rgb[:, ::-1, ::-1, :]
    eih_rgb_pp = eye_in_hand_rgb[:, ::-1, ::-1, :]
    av_dep_pp = agentview_depth[:, ::-1, ::-1]
    eih_dep_pp = eye_in_hand_depth[:, ::-1, ::-1]
    av_seg_pp = agentview_seg[:, ::-1, ::-1]
    eih_seg_pp = eye_in_hand_seg[:, ::-1, ::-1]

    # Compute interacting IDs once from the pre-loaded arrays (or reuse provided ones).
    if interacting_ids is None:
        print(f"  Computing gripper interactions for {demo_key}…")
        all_interacting: set[int] = set()
        for t in range(T):
            finger_tips = np.stack([left_tips[t], right_tips[t]])
            av_pcd = build_segmented_pointcloud(
                av_rgb_pp[t], av_dep_pp[t], av_seg_pp[t], av_K[t], av_E[t]
            )
            eih_pcd = build_segmented_pointcloud(
                eih_rgb_pp[t], eih_dep_pp[t], eih_seg_pp[t], eih_K[t], eih_E[t]
            )
            pcd = merge_pointclouds(av_pcd, eih_pcd)
            all_interacting |= find_interacting_instance_ids(pcd, finger_tips)
    else:
        all_interacting = interacting_ids

    print(f"  Interacting instance IDs: {sorted(all_interacting)}")

    viz = RerunVisualizer(application_id=f"libero/{demo_key}", spawn=True)
    viz.setup(task_description=demo_key)

    # Static full-episode trajectories
    rr.log(
        "trajectories/ee_pos",
        rr.LineStrips3D([ee_pos], colors=[[255, 255, 0]]),
        static=True,
    )
    rr.log(
        "trajectories/left_tip",
        rr.LineStrips3D([left_tips], colors=[[0, 255, 0]]),
        static=True,
    )
    rr.log(
        "trajectories/right_tip",
        rr.LineStrips3D([right_tips], colors=[[255, 0, 0]]),
        static=True,
    )

    # Log the VLM-annotated segmentation image (seg IDs labelled) statically.
    if seg_annotated_image is not None:
        rr.log(
            "vlm/seg_annotated_scene",
            rr.Image(seg_annotated_image),
            static=True,
        )

    # Log the VLM-selected distance pairs as a text document.
    if distance_pairs:
        pair_lines = [
            f"• {p.label}: {p.from_entity} ↔ {p.to_entity}" for p in distance_pairs
        ]
        rr.log(
            "vlm/selected_pairs",
            rr.TextDocument("\n".join(pair_lines), media_type="text/plain"),
            static=True,
        )

    for t in range(T):
        rr.set_time("it", sequence=t)

        # W-flip only; RerunVisualizer.update will H-flip for correct display
        adapted_obs = {
            "agentview_image": agentview_rgb[t, :, ::-1],
            "robot0_eye_in_hand_image": eye_in_hand_rgb[t, :, ::-1],
        }
        camera_params = {
            "agentview": {"intrinsic": av_K[t], "extrinsic": av_E[t]},
            "robot0_eye_in_hand": {"intrinsic": eih_K[t], "extrinsic": eih_E[t]},
        }
        viz.update(
            timestep=t,
            obs=adapted_obs,
            robot_state=np.concatenate([ee_pos[t], ee_ori[t]]),
            camera_params=camera_params,
        )

        # Finger tips
        rr.log(
            "robot/left_tip",
            rr.Points3D([left_tips[t]], colors=[[0, 255, 0]], radii=[0.005]),
        )
        rr.log(
            "robot/right_tip",
            rr.Points3D([right_tips[t]], colors=[[255, 0, 0]], radii=[0.005]),
        )

        # Combined point cloud (both cameras) with segmentation coloring
        av_pcd = build_segmented_pointcloud(
            av_rgb_pp[t], av_dep_pp[t], av_seg_pp[t], av_K[t], av_E[t]
        )
        eih_pcd = build_segmented_pointcloud(
            eih_rgb_pp[t], eih_dep_pp[t], eih_seg_pp[t], eih_K[t], eih_E[t]
        )
        pcd = merge_pointclouds(av_pcd, eih_pcd)

        if pcd.shape[0] > 0:
            pcd_xyz = pcd[:, :3]
            pcd_colors = _pcd_colors_for_rerun(pcd, all_interacting)
            rr.log(
                "scene/pointcloud",
                rr.Points3D(pcd_xyz, colors=pcd_colors, radii=0.002),
            )

        if distance_pairs and pcd.shape[0] > 0:
            feat = distance_feature or CentroidDistance()
            for label, val in compute_distances(
                distance_pairs, feat, pcd, ee_pos[t]
            ).items():
                rr.log(f"distances/{label}", rr.Scalars(val))

    viz.close()

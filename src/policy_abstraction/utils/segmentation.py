"""Utilities for gripper-object interaction analysis via segmentation masks.

Workflow
--------
1. Build a labelled point cloud ``[X, Y, Z, R, G, B, seg_id]`` from one or
   more RGBD+segmentation camera views (:func:`build_segmented_pointcloud`,
   :func:`merge_pointclouds`).
2. Determine which instance IDs the gripper was within *threshold_m* of
   at each timestep (:func:`find_interacting_instance_ids`).
3. Accumulate these IDs over a full rollout
   (:func:`accumulate_interacting_ids`).
4. Produce a colour overlay image that highlights just those instances
   (:func:`colorize_segmentation`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from policy_abstraction.utils.gripper import compute_finger_tip_positions
from policy_abstraction.utils.pcd import pixels_to_3d

# ---------------------------------------------------------------------------
# Robot / gripper segment filtering
# ---------------------------------------------------------------------------

# Substrings that identify robot or gripper segments. Covers LIBERO (Panda*,
# NullMount, OnTheGroundPanda, Gripper) and RoboCasa (PandaMobile/PandaOmron
# base, mount, mobile platform).
ROBOT_SEGMENT_KEYWORDS = {
    "Panda",
    "NullMount",
    "OnTheGroundPanda",
    "Gripper",
    "PandaMobile",
    "PandaOmron",
    "OmronBase",
    "MobileBase",
    "Mount",
}


def get_robot_segment_ids(
    id_to_name: dict[int, str],
    keywords: set[str] = ROBOT_SEGMENT_KEYWORDS,
    all_seg_ids: set[int] | None = None,
) -> set[int]:
    """Return segment IDs belonging to the robot or gripper.

    A segment is considered a robot part if:
    1. Its name contains any of the *keywords*, OR
    2. It is visible in the segmentation (present in *all_seg_ids*) but
       absent from *id_to_name* — the LIBERO env wrapper intentionally
       excludes robot parts (``Panda0``, ``PandaGripper0``, etc.) from
       the name mapping.

    Args:
        id_to_name:  Mapping from segment ID to human-readable name.
        keywords:    Substrings that identify robot/gripper parts.
        all_seg_ids: All segment IDs visible in the current mask (excluding
                     background 0).  When provided, IDs not in *id_to_name*
                     are treated as robot parts.

    Returns:
        Set of integer segment IDs belonging to the robot or gripper.
    """
    robot = {
        sid for sid, name in id_to_name.items() if any(kw in name for kw in keywords)
    }
    if all_seg_ids is not None:
        mapped_ids = set(id_to_name.keys())
        robot |= {sid for sid in all_seg_ids if sid not in mapped_ids}
    return robot


# ---------------------------------------------------------------------------
# Point-cloud construction
# ---------------------------------------------------------------------------


def build_segmented_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    segmentation: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    depth_min: float = 1e-3,
    depth_max: float = np.inf,
    preprocessed: bool = True,
) -> np.ndarray:
    """Build a coloured, labelled point cloud from one RGBD+segmentation view.

    Args:
        rgb:          ``[H, W, 3]`` uint8 image (preprocessed, 180-rotated).
        depth:        ``[H, W]`` or ``[H, W, 1]`` float depth map
                      (world-metric, preprocessed).
        segmentation: ``[H, W]`` or ``[H, W, 1]`` int32 segmentation mask
                      (preprocessed).
        intrinsic:    ``[3, 3]`` camera intrinsic K.
        extrinsic:    ``[4, 4]`` camera-to-world pose.
        depth_min:    Minimum valid depth (metres).
        depth_max:    Maximum valid depth (metres).
        preprocessed: Whether images have already been 180°-flipped
                      (u-axis mirror correction applied to K).

    Returns:
        ``[M, 7]`` float32 array with columns
        ``[X, Y, Z, R, G, B, seg_id]``.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float64)
    segmentation = np.asarray(segmentation, dtype=np.int32)

    # Squeeze trailing channel dims if present.
    if depth.ndim == 3:
        depth = depth[..., 0]
    if segmentation.ndim == 3:
        segmentation = segmentation[..., 0]

    H, W = depth.shape
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    mask = (depth > depth_min) & (depth < depth_max)

    u_flat = cols[mask].astype(np.float64)
    v_flat = rows[mask].astype(np.float64)
    d_flat = depth[mask]

    pixels_flat = np.stack([u_flat, v_flat], axis=-1)
    xyz = pixels_to_3d(
        pixels_flat,
        d_flat,
        intrinsic,
        extrinsic,
        image_width=W,
        preprocessed=preprocessed,
    )  # [M, 3]

    colors = rgb[mask].astype(np.float32)  # [M, 3]
    seg_ids = segmentation[mask].astype(np.float32)  # [M]

    return np.concatenate(
        [xyz.astype(np.float32), colors, seg_ids[:, np.newaxis]], axis=-1
    )  # [M, 7]


def merge_pointclouds(*pcds: np.ndarray) -> np.ndarray:
    """Concatenate multiple ``[M_i, 7]`` point clouds into one ``[M, 7]`` array."""
    non_empty = [p for p in pcds if p.shape[0] > 0]
    if not non_empty:
        return np.empty((0, 7), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


def erode_segmentation(seg: np.ndarray, erosion_px: int = 3) -> np.ndarray:
    """Erode each segment in a segmentation mask independently.

    Each instance region is shrunk inward by *erosion_px* pixels using an
    elliptical structuring element.  Boundary pixels that become ambiguous
    (i.e. belong to no eroded region) are set to 0 (background).  This
    reduces the influence of depth-discontinuity artefacts at object borders
    when building point clouds for distance computation.

    Args:
        seg:         ``[H, W]`` or ``[H, W, 1]`` int32 segmentation mask.
        erosion_px:  Erosion radius in pixels (default 3).

    Returns:
        ``[H, W]`` int32 eroded segmentation mask (same shape as squeezed input).
    """
    import cv2

    seg = np.asarray(seg, dtype=np.int32)
    if seg.ndim == 3:
        seg = seg[..., 0]

    if erosion_px <= 0:
        return seg.copy()

    diameter = 2 * erosion_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

    result = np.zeros_like(seg)
    for uid in np.unique(seg):
        if uid == 0:
            continue
        binary = (seg == uid).astype(np.uint8)
        eroded = cv2.erode(binary, kernel)
        result[eroded > 0] = uid

    return result


# ---------------------------------------------------------------------------
# Interaction detection
# ---------------------------------------------------------------------------


def remove_statistical_outliers(
    pcd: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Remove statistical outliers from a point cloud.

    For each point, computes the mean distance to its *nb_neighbors* nearest
    neighbours.  Points whose mean distance exceeds
    ``global_mean + std_ratio * global_std`` are discarded.

    Args:
        pcd:          ``[M, 7]`` array ``[X, Y, Z, R, G, B, seg_id]``.
        nb_neighbors: Number of nearest neighbours to consider.
        std_ratio:    Multiplier on the global standard deviation.

    Returns:
        Filtered ``[M', 7]`` array with outliers removed.
    """
    if pcd.shape[0] <= nb_neighbors:
        return pcd

    xyz = pcd[:, :3]
    tree = KDTree(xyz)
    # k+1 because the query includes the point itself (distance = 0)
    dists, _ = tree.query(xyz, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return pcd[mean_dists <= threshold]


def find_interacting_instance_ids(
    pcd_with_seg: np.ndarray,
    finger_tips: np.ndarray,
    threshold_m: float = 0.05,
    min_points: int = 30,
) -> set[int]:
    """Return instance IDs within *threshold_m* of any finger tip.

    Only instances with at least *min_points* unique points near any finger tip
    are returned.  This guards against isolated projection artefacts without
    requiring an expensive full-cloud outlier removal pass.

    Args:
        pcd_with_seg: ``[M, 7]`` array ``[X, Y, Z, R, G, B, seg_id]``.
        finger_tips:  ``[N, 3]`` world-frame finger-tip positions.
        threshold_m:  Distance threshold in metres (default 5 cm).
        min_points:   Minimum number of distinct points of an instance that
                      must fall within *threshold_m* (default 30).

    Returns:
        Set of integer segment IDs (background id=0 excluded).
    """
    if pcd_with_seg.shape[0] == 0:
        return set()

    xyz = pcd_with_seg[:, :3]
    seg_ids = pcd_with_seg[:, 6]

    tree = KDTree(xyz)
    index_lists = tree.query_ball_point(np.asarray(finger_tips), r=threshold_m)

    # Collect unique point indices near any finger tip, then count per instance.
    nearby: set[int] = set()
    for idx_list in index_lists:
        nearby.update(idx_list)

    counts: dict[int, int] = {}
    for idx in nearby:
        sid = int(seg_ids[idx])
        counts[sid] = counts.get(sid, 0) + 1

    interacting = {sid for sid, cnt in counts.items() if cnt >= min_points}
    interacting.discard(0)  # background
    return interacting


def accumulate_interacting_ids(
    obs,
    *,
    threshold_m: float = 0.03,
    min_points: int = 30,
) -> set[int]:
    """Accumulate all instance IDs the gripper interacts with over a rollout.

    Builds a merged point cloud (agentview + eye-in-hand) at every timestep
    and finds instance IDs within *threshold_m* of either finger tip.

    Args:
        obs:         HDF5 group or dict containing observation arrays.
        threshold_m: Distance threshold in metres.
        min_points:  Minimum distinct points per instance to count as an
                     interaction (forwarded to
                     :func:`find_interacting_instance_ids`).

    Returns:
        Set of instance IDs touched by the gripper at any point.
    """
    ee_pos = obs["ee_pos"][:]
    ee_ori = obs["ee_ori"][:]
    gripper_states = obs["gripper_states"][:]
    T_steps = ee_pos.shape[0]

    av_rgb = obs["agentview_rgb"][:][:, ::-1, ::-1, :]
    av_dep = obs["agentview_depth"][:][:, ::-1, ::-1]
    av_seg = obs["agentview_segmentation"][:][:, ::-1, ::-1]
    av_K = obs["agentview_intrinsic"][:]
    av_E = obs["agentview_extrinsic"][:]

    eih_rgb = obs["eye_in_hand_rgb"][:][:, ::-1, ::-1, :]
    eih_dep = obs["eye_in_hand_depth"][:][:, ::-1, ::-1]
    eih_seg = obs["eye_in_hand_segmentation"][:][:, ::-1, ::-1]
    eih_K = obs["eye_in_hand_intrinsic"][:]
    eih_E = obs["eye_in_hand_extrinsic"][:]

    left_tips, right_tips = compute_finger_tip_positions(ee_pos, ee_ori, gripper_states)

    all_ids: set[int] = set()
    for t in range(T_steps):
        finger_tips = np.stack([left_tips[t], right_tips[t]])  # (2, 3)

        av_pcd = build_segmented_pointcloud(
            av_rgb[t], av_dep[t], av_seg[t], av_K[t], av_E[t]
        )
        eih_pcd = build_segmented_pointcloud(
            eih_rgb[t], eih_dep[t], eih_seg[t], eih_K[t], eih_E[t]
        )
        pcd = merge_pointclouds(av_pcd, eih_pcd)
        all_ids |= find_interacting_instance_ids(
            pcd,
            finger_tips,
            threshold_m=threshold_m,
            min_points=min_points,
        )

    return all_ids


# ---------------------------------------------------------------------------
# Segmentation visualisation helpers
# ---------------------------------------------------------------------------


_PALETTE_HSVALS = [
    (0, 220, 230),  # red-ish
    (60, 220, 230),  # yellow
    (120, 220, 230),  # green
    (150, 220, 230),  # cyan
    (100, 220, 230),  # teal
    (20, 220, 230),  # orange
    (170, 220, 230),  # magenta
    (80, 220, 230),  # lime
]

# Cool-hue palette used for VLM-selected IDs (blues / purples).
_PALETTE_COOL_HSVALS = [
    (110, 200, 240),  # sky-blue
    (130, 210, 235),  # blue-green
    (145, 215, 245),  # cornflower blue
    (160, 200, 230),  # cobalt
    (175, 210, 240),  # violet-blue
    (190, 205, 235),  # purple
]


def _make_palette(n: int) -> list[tuple[int, int, int]]:
    """Return *n* distinct BGR colours (wraps if n > palette size)."""
    import cv2

    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        hs, s, v = _PALETTE_HSVALS[i % len(_PALETTE_HSVALS)]
        hsv = np.array([[[hs, s, v]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append(tuple(bgr))
    return colors


def _make_secondary_palette(n: int) -> list[tuple[int, int, int]]:
    """Return *n* distinct cool-hue BGR colours for VLM-selected IDs."""
    import cv2

    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        hs, s, v = _PALETTE_COOL_HSVALS[i % len(_PALETTE_COOL_HSVALS)]
        hsv = np.array([[[hs, s, v]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append(tuple(bgr))
    return colors


def colorize_segmentation(
    seg_mask: np.ndarray,
    interacting_ids: set[int],
    *,
    bg_gray: int = 30,
) -> np.ndarray:
    """Render a segmentation mask highlighting interacting instances.

    Args:
        seg_mask:        ``[H, W]`` or ``[H, W, 1]`` int32 mask.
        interacting_ids: Instance IDs to highlight with distinct colours.
        bg_gray:         Intensity of background pixels.

    Returns:
        ``[H, W, 3]`` uint8 BGR image.
    """
    seg_mask = np.asarray(seg_mask)
    if seg_mask.ndim == 3:
        seg_mask = seg_mask[..., 0]

    H, W = seg_mask.shape
    out = np.full((H, W, 3), bg_gray, dtype=np.uint8)

    palette = _make_palette(len(interacting_ids))
    for color, inst_id in zip(palette, sorted(interacting_ids), strict=False):
        out[seg_mask == inst_id] = color

    return out


def draw_interaction_legend(
    image: np.ndarray,
    interacting_ids: set[int],
    *,
    id_to_name: dict[int, str] | None = None,
    font_scale: float = 0.55,
    thickness: int = 1,
) -> np.ndarray:
    """Overlay a small legend box listing active instance IDs.

    Args:
        image:           ``[H, W, 3]`` uint8 BGR canvas to draw on (copy made).
        interacting_ids: Set of IDs printed in their respective colours.
        id_to_name:      Optional mapping from instance ID to human-readable name.
        font_scale:      cv2 font scale.
        thickness:       cv2 line thickness.

    Returns:
        Annotated copy of *image*.
    """
    import cv2

    out = image.copy()
    palette = _make_palette(len(interacting_ids))
    x, y0 = 8, 20
    for color, inst_id in zip(palette, sorted(interacting_ids), strict=False):
        if id_to_name and inst_id in id_to_name:
            label = f"ID {inst_id} ({id_to_name[inst_id]})"
        else:
            label = f"ID {inst_id}"
        cv2.rectangle(out, (x, y0 - 11), (x + 14, y0 + 2), color, -1)
        cv2.putText(
            out,
            label,
            (x + 18, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y0 += 20

    return out


def draw_all_segment_ids(
    seg_mask: np.ndarray,
    rgb: np.ndarray,
    *,
    exclude_ids: set[int] | None = None,
    alpha: float = 0.45,  # noqa: ARG001  (kept for backwards-compatible signature)
    font_scale: float = 0.8,
    thickness: int = 2,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Render a labeled segmentation overlay showing every instance's ID.

    Each unique segment is outlined with a coloured contour (no fill) so the
    underlying RGB texture remains visible, and labelled at the centroid with
    its integer ID drawn on a filled black rectangle for legibility.

    Args:
        seg_mask:          ``[H, W]`` or ``[H, W, 1]`` int32 mask.
        rgb:               ``[H, W, 3]`` uint8 RGB image to overlay on.
        exclude_ids:       Segment IDs to skip (e.g. robot/gripper parts).
        alpha:             Unused, kept for backwards compatibility.
        font_scale:        cv2 font scale for ID labels.
        thickness:         cv2 line thickness for ID labels.
        contour_thickness: cv2 line thickness for the segment contour.

    Returns:
        ``[H, W, 3]`` uint8 BGR image with contour outlines and text labels.
    """
    import cv2

    seg_mask = np.asarray(seg_mask)
    if seg_mask.ndim == 3:
        seg_mask = seg_mask[..., 0]

    _exclude = exclude_ids or set()
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    unique_ids = sorted(
        {int(v) for v in np.unique(seg_mask) if v != 0 and v not in _exclude}
    )
    if not unique_ids:
        return bgr.copy()

    palette = _make_palette(len(unique_ids))
    out = bgr.copy()

    for color, inst_id in zip(palette, unique_ids, strict=False):
        binary = (seg_mask == inst_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cv2.drawContours(
            out,
            contours,
            -1,
            (int(color[0]), int(color[1]), int(color[2])),
            contour_thickness,
            cv2.LINE_AA,
        )

    for inst_id in unique_ids:
        ys, xs = np.where(seg_mask == inst_id)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(inst_id)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        pad = 3
        cv2.rectangle(
            out,
            (cx - pad, cy - th - pad),
            (cx + tw + pad, cy + baseline + pad),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            out,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return out


def colorize_dual_segmentation(
    seg_mask: np.ndarray,
    interacting_ids: set[int],
    vlm_selected_ids: set[int],
    *,
    bg_gray: int = 30,
) -> np.ndarray:
    """Render a segmentation mask highlighting two sets of instances.

    *interacting_ids* (gripper contact) use the primary warm palette;
    *vlm_selected_ids* (task-relevant, VLM-chosen) use a secondary cool palette.
    IDs present in both sets are coloured with the primary palette.

    Args:
        seg_mask:         ``[H, W]`` or ``[H, W, 1]`` int32 mask.
        interacting_ids:  IDs to highlight with the primary palette.
        vlm_selected_ids: IDs to highlight with the secondary (cool) palette.
        bg_gray:          Background pixel intensity.

    Returns:
        ``[H, W, 3]`` uint8 BGR image.
    """
    seg_mask = np.asarray(seg_mask)
    if seg_mask.ndim == 3:
        seg_mask = seg_mask[..., 0]

    H, W = seg_mask.shape
    out = np.full((H, W, 3), bg_gray, dtype=np.uint8)

    palette_interact = _make_palette(len(interacting_ids))
    for color, inst_id in zip(palette_interact, sorted(interacting_ids), strict=False):
        out[seg_mask == inst_id] = color

    vlm_only = vlm_selected_ids - interacting_ids
    palette_vlm = _make_secondary_palette(len(vlm_only))
    for color, inst_id in zip(palette_vlm, sorted(vlm_only), strict=False):
        out[seg_mask == inst_id] = color

    return out


def draw_dual_interaction_legend(
    image: np.ndarray,
    interacting_ids: set[int],
    vlm_selected_ids: set[int],
    *,
    id_to_name: dict[int, str] | None = None,
    font_scale: float = 0.50,
    thickness: int = 1,
) -> np.ndarray:
    """Overlay a legend listing both gripper-contact and VLM-selected IDs.

    Args:
        image:            ``[H, W, 3]`` uint8 BGR canvas to draw on (copy made).
        interacting_ids:  IDs coloured with the primary palette.
        vlm_selected_ids: IDs coloured with the secondary palette.
        id_to_name:       Optional mapping from instance ID to human-readable name.
        font_scale:       cv2 font scale.
        thickness:        cv2 line thickness.

    Returns:
        Annotated copy of *image*.
    """
    import cv2

    out = image.copy()
    x, y0 = 8, 20

    # --- Primary section: gripper interaction IDs ---
    if interacting_ids:
        cv2.putText(
            out,
            "Gripper contact:",
            (x, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (200, 200, 200),
            thickness,
            cv2.LINE_AA,
        )
        y0 += 18
        palette = _make_palette(len(interacting_ids))
        for color, inst_id in zip(palette, sorted(interacting_ids), strict=False):
            label = (
                f"ID {inst_id} ({id_to_name[inst_id]})"
                if id_to_name and inst_id in id_to_name
                else f"ID {inst_id}"
            )
            cv2.rectangle(out, (x, y0 - 10), (x + 12, y0 + 2), color, -1)
            cv2.putText(
                out,
                label,
                (x + 16, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y0 += 18

    # --- Secondary section: VLM-selected IDs ---
    vlm_only = vlm_selected_ids - interacting_ids
    if vlm_only:
        y0 += 6
        cv2.putText(
            out,
            "VLM-selected:",
            (x, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (200, 200, 200),
            thickness,
            cv2.LINE_AA,
        )
        y0 += 18
        palette_vlm = _make_secondary_palette(len(vlm_only))
        for color, inst_id in zip(palette_vlm, sorted(vlm_only), strict=False):
            label = (
                f"ID {inst_id} ({id_to_name[inst_id]})"
                if id_to_name and inst_id in id_to_name
                else f"ID {inst_id}"
            )
            cv2.rectangle(out, (x, y0 - 10), (x + 12, y0 + 2), color, -1)
            cv2.putText(
                out,
                label,
                (x + 16, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y0 += 18

    return out

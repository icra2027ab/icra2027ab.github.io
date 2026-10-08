"""Extract gripper-object interaction sub-trajectories from LIBERO HDF5 datasets.

For each demo in a dataset file, this script identifies every timestep window
during which at least one gripper finger is within 3 cm of a segmented object
(interaction onset), and ends when the distance exceeds 5 cm (interaction end).
Multiple simultaneous interactions (gripper near several objects) are each
recorded independently.

Second-order interactions are also detected: when an object is within 1 cm of a
gripper (primary object), any other object within start_thresh of that primary
object is recorded as a second-order interaction.  The same hysteresis thresholds
(start_thresh / stop_thresh) govern onset and offset of second-order proximity.

Output is an HDF5 file whose structure mirrors the source files:

  interactions.hdf5
    /interactions/
      /0/                          <- interaction index (sequential across all demos)
        attrs:
          source_file              <- absolute path to the source HDF5
          demo_key                 <- e.g. "demo_3"
          interaction_order        <- 1 (direct) or 2 (via another object)
          object_id                <- segmentation integer ID
          object_name              <- human-readable name (if mapping available)
          via_object_id            <- [2nd-order only] primary object seg ID
          via_object_name          <- [2nd-order only] primary object name
          interaction_chain        <- e.g. "gripper → basket → bowl"
          start_timestep           <- inclusive
          end_timestep             <- exclusive
        datasets (1st-order):
          timesteps                <- [N] int32
          left_finger_pos          <- [N, 3] float32
          right_finger_pos         <- [N, 3] float32
          left_finger_dist         <- [N] float32
          right_finger_dist        <- [N] float32
          min_dist                 <- [N] float32
          left_closest_point       <- [N, 3] float32
          right_closest_point      <- [N, 3] float32
        datasets (2nd-order):
          timesteps                <- [N] int32
          left_finger_pos          <- [N, 3] float32
          right_finger_pos         <- [N, 3] float32
          gripper_to_primary_dist  <- [N] float32 - min(left,right) → primary
          primary_to_secondary_dist <- [N] float32 - primary → secondary surface
          closest_on_primary       <- [N, 3] float32
          closest_on_secondary     <- [N, 3] float32

Usage
-----
  uv run scripts/ablations/extraction/extract_interactions.py \
      --input datasets/libero_object_new/ \
      --output datasets/interactions/libero_object_interactions.hdf5

  # Or process a single file:
  uv run scripts/ablations/extraction/extract_interactions.py \
      --input datasets/libero_object_new/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
      --output datasets/interactions/alphabet_soup_interactions.hdf5
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from scipy.spatial import KDTree

from policy_abstraction.utils.gripper import compute_finger_tip_positions
from policy_abstraction.utils.seeding import seed_everything
from policy_abstraction.utils.segmentation import (
    build_segmented_pointcloud,
    draw_all_segment_ids,
    erode_segmentation,
    get_robot_segment_ids,
    merge_pointclouds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERACTION_START_M: float = 0.05  # 5 cm  – onset threshold
INTERACTION_STOP_M: float = 0.10  # 10 cm  – hysteresis off-threshold
MIN_INTERACTION_STEPS: int = 20  # discard interactions shorter than this

# Second-order: gripper must be this close to an object for it to become "primary"
SECOND_ORDER_PRIMARY_M: float = 0.01  # 1 cm

_DEBUG_MAX_PCD_POINTS: int = 15_000  # subsample large point clouds for rerun


# ---------------------------------------------------------------------------
# Debug / Rerun helpers
# ---------------------------------------------------------------------------


def _seg_id_to_rgb(seg_ids: np.ndarray) -> np.ndarray:
    """Map integer seg IDs to distinct RGB colours using a golden-ratio hash."""
    out = np.zeros((len(seg_ids), 3), dtype=np.uint8)
    for uid in np.unique(seg_ids):
        mask = seg_ids == uid
        if uid == 0:
            out[mask] = [60, 60, 60]
        else:
            hue = (int(uid) * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.92)
            out[mask] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


def _debug_init_rerun(
    app_id: str = "extract_interactions_debug",
    connect_addr: str | None = None,
) -> None:
    import rerun as rr

    rr.init(app_id, spawn=False)
    if connect_addr:
        rr.connect_grpc(connect_addr)
    else:
        rr.spawn()


def _debug_send_blueprint() -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            # Left: 3D scene + distances
            rrb.Vertical(
                rrb.Spatial3DView(name="3D Scene", origin="/"),
                rrb.TimeSeriesView(
                    name="Object Distances",
                    contents=["distances/**"],
                ),
                row_shares=[3, 2],
            ),
            # Right: images, segmentation, config, interactions
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Agentview RGB", origin="camera/agentview"),
                    rrb.Spatial2DView(name="Wrist RGB", origin="camera/eye_in_hand"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="Agentview Seg", origin="segmentation/agentview"
                    ),
                    rrb.Spatial2DView(
                        name="Wrist Seg", origin="segmentation/eye_in_hand"
                    ),
                ),
                rrb.TextDocumentView(name="Config", origin="info/config"),
                rrb.TimeSeriesView(
                    name="Interaction Windows",
                    contents=["interactions/**"],
                ),
                row_shares=[3, 3, 1, 2],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="timestep"),
    )
    rr.send_blueprint(blueprint)


def _debug_log_static(
    seg_name_map: dict[str, str],
    start_thresh: float,
    stop_thresh: float,
    demo_key: str,
    source_file: str,
) -> None:
    import rerun as rr

    rr.log(
        "info/config",
        rr.TextDocument(
            json.dumps(
                {
                    "source_file": source_file,
                    "demo_key": demo_key,
                    "start_thresh_m": start_thresh,
                    "stop_thresh_m": stop_thresh,
                    "min_interaction_steps": MIN_INTERACTION_STEPS,
                    "seg_name_map": seg_name_map,
                },
                indent=2,
            ),
            media_type="application/json",
        ),
        static=True,
    )


def _debug_log_timestep(
    t: int,
    av_rgb: np.ndarray,
    av_seg: np.ndarray,
    eih_rgb: np.ndarray,
    eih_seg: np.ndarray,
    pcd: np.ndarray,
    left_tip: np.ndarray,
    right_tip: np.ndarray,
    obj_dists: dict[int, dict],
    id_to_name: dict[int, str],
    start_thresh: float,
    stop_thresh: float,
) -> None:
    import cv2
    import rerun as rr

    rr.set_time("timestep", sequence=t)

    # Camera images (RGB)
    rr.log("camera/agentview", rr.Image(av_rgb))
    rr.log("camera/eye_in_hand", rr.Image(eih_rgb))

    # Segmentation overlays with labelled IDs (exclude robot/gripper)
    av_seg_2d = av_seg[..., 0] if av_seg.ndim == 3 else av_seg
    eih_seg_2d = eih_seg[..., 0] if eih_seg.ndim == 3 else eih_seg
    all_vis = {int(v) for v in np.unique(av_seg_2d) if v != 0} | {
        int(v) for v in np.unique(eih_seg_2d) if v != 0
    }
    robot_ids = get_robot_segment_ids(id_to_name, all_seg_ids=all_vis)

    av_seg_vis = draw_all_segment_ids(av_seg_2d, av_rgb, exclude_ids=robot_ids)  # BGR
    eih_seg_vis = draw_all_segment_ids(
        eih_seg_2d, eih_rgb, exclude_ids=robot_ids
    )  # BGR
    rr.log(
        "segmentation/agentview", rr.Image(cv2.cvtColor(av_seg_vis, cv2.COLOR_BGR2RGB))
    )
    rr.log(
        "segmentation/eye_in_hand",
        rr.Image(cv2.cvtColor(eih_seg_vis, cv2.COLOR_BGR2RGB)),
    )

    # Point cloud coloured by seg_id (subsampled for performance)
    if pcd.shape[0] > 0:
        pts = pcd
        if pts.shape[0] > _DEBUG_MAX_PCD_POINTS:
            idx = np.random.choice(pts.shape[0], _DEBUG_MAX_PCD_POINTS, replace=False)
            pts = pts[idx]
        xyz = pts[:, :3]
        seg_rgb = _seg_id_to_rgb(pts[:, 6].astype(np.int32))
        rr.log("scene/pointcloud", rr.Points3D(xyz, colors=seg_rgb, radii=0.003))

    # Finger tip positions and span line
    rr.log(
        "gripper/left_tip",
        rr.Points3D([left_tip], radii=0.012, colors=[[255, 100, 100, 255]]),
    )
    rr.log(
        "gripper/right_tip",
        rr.Points3D([right_tip], radii=0.012, colors=[[100, 100, 255, 255]]),
    )
    rr.log(
        "gripper/span",
        rr.LineStrips3D(
            [[left_tip.tolist(), right_tip.tolist()]], colors=[[200, 200, 200, 180]]
        ),
    )

    # Per-object distances and threshold reference lines
    rr.log("distances/_start_thresh", rr.Scalars(start_thresh))
    rr.log("distances/_stop_thresh", rr.Scalars(stop_thresh))
    for sid, info in obj_dists.items():
        name = id_to_name.get(sid, str(sid))
        safe = name.replace(" ", "_")
        l_d = info["left_dist"]
        r_d = info["right_dist"]
        rr.log(f"distances/{safe}/left_dist", rr.Scalars(l_d))
        rr.log(f"distances/{safe}/right_dist", rr.Scalars(r_d))
        rr.log(f"distances/{safe}/min_dist", rr.Scalars(min(l_d, r_d)))
        # Closest points on object surface
        rr.log(
            f"scene/objects/{safe}/closest_points",
            rr.Points3D(
                [info["left_closest"], info["right_closest"]],
                radii=0.008,
                colors=[[255, 200, 0, 255], [0, 200, 255, 255]],
            ),
        )


def _debug_log_interactions(interactions: list[dict], T: int) -> None:
    """Log a 0/1 active-window scalar per interaction on the timestep timeline."""
    import rerun as rr

    # Build a lookup: object_id → list of (start, end) windows
    windows: dict[str, list[tuple[int, int]]] = {}
    for iact in interactions:
        name = iact["object_name"].replace(" ", "_")
        key = f"{name}_id{iact['object_id']}"
        windows.setdefault(key, []).append(
            (iact["start_timestep"], iact["end_timestep"])
        )

    for key, segs in windows.items():
        for t in range(T):
            rr.set_time("timestep", sequence=t)
            active = any(s <= t < e for s, e in segs)
            rr.log(f"interactions/{key}", rr.Scalars(1.0 if active else 0.0))


# ---------------------------------------------------------------------------
# Core per-timestep distance computation
# ---------------------------------------------------------------------------


def _compute_object_distances(
    pcd: np.ndarray,
    left_tip: np.ndarray,
    right_tip: np.ndarray,
) -> dict[int, dict]:
    """For each object in *pcd*, return distance and closest point to each finger.

    Args:
        pcd:       ``[M, 7]`` point cloud with columns [X, Y, Z, R, G, B, seg_id].
        left_tip:  ``[3]`` world-frame left finger tip position.
        right_tip: ``[3]`` world-frame right finger tip position.

    Returns:
        Mapping from seg_id → {
            "left_dist": float,
            "right_dist": float,
            "left_closest": np.ndarray[3],
            "right_closest": np.ndarray[3],
        }
        Background (seg_id == 0) is excluded.
    """
    if pcd.shape[0] == 0:
        return {}

    seg_ids = pcd[:, 6].astype(np.int32)
    unique_ids = np.unique(seg_ids)
    unique_ids = unique_ids[unique_ids != 0]

    results: dict[int, dict] = {}
    for sid in unique_ids:
        mask = seg_ids == sid
        obj_pts = pcd[mask, :3]
        if obj_pts.shape[0] < 5:
            continue

        tree = KDTree(obj_pts)

        l_dist, l_idx = tree.query(left_tip)
        r_dist, r_idx = tree.query(right_tip)

        results[int(sid)] = {
            "left_dist": float(l_dist),
            "right_dist": float(r_dist),
            "left_closest": obj_pts[l_idx].astype(np.float32),
            "right_closest": obj_pts[r_idx].astype(np.float32),
        }

    return results


# ---------------------------------------------------------------------------
# Object-to-object distance (for second-order interaction detection)
# ---------------------------------------------------------------------------


def _compute_object_to_object_distances(
    pcd: np.ndarray,
    primary_sid: int,
    allowed_ids: set[int] | None = None,
) -> dict[int, dict]:
    """Compute minimum distance from *primary_sid* object to every other object.

    Args:
        pcd:         ``[M, 7]`` point cloud with columns [X, Y, Z, R, G, B, seg_id].
        primary_sid: Segment ID of the primary (gripper-close) object.
        allowed_ids: If provided, restrict secondary objects to this set.

    Returns:
        Mapping from secondary_seg_id → {
            "dist": float,
            "closest_on_primary": np.ndarray[3],
            "closest_on_secondary": np.ndarray[3],
        }
    """
    if pcd.shape[0] == 0:
        return {}

    seg_ids = pcd[:, 6].astype(np.int32)
    primary_pts = pcd[seg_ids == primary_sid, :3]
    if primary_pts.shape[0] < 5:
        return {}

    primary_tree = KDTree(primary_pts)

    unique_ids = np.unique(seg_ids)
    unique_ids = unique_ids[(unique_ids != 0) & (unique_ids != primary_sid)]
    if allowed_ids is not None:
        unique_ids = unique_ids[np.isin(unique_ids, list(allowed_ids))]

    results: dict[int, dict] = {}
    for sid in unique_ids:
        sec_pts = pcd[seg_ids == sid, :3]
        if sec_pts.shape[0] < 5:
            continue
        # Minimum distance between the two point sets.
        dists, prim_idxs = primary_tree.query(sec_pts)
        min_idx = int(np.argmin(dists))
        results[int(sid)] = {
            "dist": float(dists[min_idx]),
            "closest_on_primary": primary_pts[prim_idxs[min_idx]].astype(np.float32),
            "closest_on_secondary": sec_pts[min_idx].astype(np.float32),
        }

    return results


# ---------------------------------------------------------------------------
# Per-demo interaction extraction
# ---------------------------------------------------------------------------


def extract_demo_interactions(
    obs,
    seg_name_map: dict[str, str],
    start_thresh: float = INTERACTION_START_M,
    stop_thresh: float = INTERACTION_STOP_M,
    *,
    debug: bool = False,
) -> list[dict]:
    """Extract interaction windows from a single demo observation group.

    Args:
        obs:          HDF5 group (or dict-like) with standard LIBERO obs keys.
        seg_name_map: Mapping from seg ID (str) to object name.

    Returns:
        List of interaction dicts, each containing:
          - object_id, object_name, start_timestep, end_timestep
          - per-timestep arrays: timesteps, left_finger_pos, right_finger_pos,
            left_finger_dist, right_finger_dist, min_dist,
            left_closest_point, right_closest_point
    """
    ee_pos = obs["ee_pos"][:]  # (T, 3)
    ee_ori = obs["ee_ori"][:]  # (T, 3)
    gripper_states = obs["gripper_states"][:]  # (T, 2)
    T = ee_pos.shape[0]

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

    # Build id→name and allowed set up front (needed inside the timestep loop).
    id_to_name: dict[int, str] = {int(k): v for k, v in seg_name_map.items()}
    allowed_ids: set[int] | None = set(id_to_name.keys()) if id_to_name else None

    # Per timestep, per object: gripper→object distances.
    # ts_data[t][sid] = {"left_dist", "right_dist", "left_closest", "right_closest"}
    ts_data: list[dict[int, dict]] = []

    # Second-order hysteresis state — maintained inline during the main loop.
    so_active: dict[tuple[int, int], bool] = {}
    so_bufs: dict[tuple[int, int], dict[str, list]] = {}
    second_order_interactions: list[dict] = []

    def _flush_so_pair(
        pair: tuple[int, int], end_t: int, *, enforce_min: bool = True
    ) -> None:
        bufs = so_bufs.get(pair)
        if not bufs or not bufs["timesteps"]:
            return
        if enforce_min and len(bufs["timesteps"]) < MIN_INTERACTION_STEPS:
            return
        primary_sid, secondary_sid = pair
        primary_name = id_to_name.get(primary_sid, str(primary_sid))
        secondary_name = id_to_name.get(secondary_sid, str(secondary_sid))
        second_order_interactions.append(
            {
                "order": 2,
                "object_id": secondary_sid,
                "object_name": secondary_name,
                "via_object_id": primary_sid,
                "via_object_name": primary_name,
                "interaction_chain": f"gripper → {primary_name} → {secondary_name}",
                "start_timestep": bufs["timesteps"][0],
                "end_timestep": end_t,
                "timesteps": np.array(bufs["timesteps"], dtype=np.int32),
                "left_finger_pos": np.array(bufs["left_pos"], dtype=np.float32),
                "right_finger_pos": np.array(bufs["right_pos"], dtype=np.float32),
                "gripper_to_primary_dist": np.array(
                    bufs["gripper_primary"], dtype=np.float32
                ),
                "primary_to_secondary_dist": np.array(
                    bufs["primary_sec"], dtype=np.float32
                ),
                "closest_on_primary": np.array(bufs["closest_prim"], dtype=np.float32),
                "closest_on_secondary": np.array(bufs["closest_sec"], dtype=np.float32),
            }
        )

    print(f"    Processing {T} timesteps…", flush=True)
    for t in range(T):
        if t % 50 == 0:
            print(f"      t={t}/{T}", end="\r", flush=True)

        av_pcd = build_segmented_pointcloud(
            av_rgb[t], av_dep[t], erode_segmentation(av_seg[t]), av_K[t], av_E[t]
        )
        eih_pcd = build_segmented_pointcloud(
            eih_rgb[t], eih_dep[t], erode_segmentation(eih_seg[t]), eih_K[t], eih_E[t]
        )
        pcd = merge_pointclouds(av_pcd, eih_pcd)

        obj_dists = _compute_object_distances(pcd, left_tips[t], right_tips[t])
        ts_data.append(obj_dists)

        # Second-order: for each primary object (gripper < 1 cm), apply hysteresis
        # inline against each neighbouring secondary object.
        active_so_pairs_this_t: set[tuple[int, int]] = set()
        for sid, info in obj_dists.items():
            if allowed_ids is not None and sid not in allowed_ids:
                continue
            gripper_to_primary = min(info["left_dist"], info["right_dist"])
            if gripper_to_primary >= SECOND_ORDER_PRIMARY_M:
                continue
            for secondary_sid, sec_info in _compute_object_to_object_distances(
                pcd, sid, allowed_ids
            ).items():
                pair = (sid, secondary_sid)
                active_so_pairs_this_t.add(pair)
                dist = sec_info["dist"]

                if not so_active.get(pair, False):
                    if dist < start_thresh:
                        so_active[pair] = True
                        so_bufs[pair] = {
                            "timesteps": [],
                            "left_pos": [],
                            "right_pos": [],
                            "gripper_primary": [],
                            "primary_sec": [],
                            "closest_prim": [],
                            "closest_sec": [],
                        }

                if so_active.get(pair, False):
                    b = so_bufs[pair]
                    b["timesteps"].append(t)
                    b["left_pos"].append(left_tips[t].astype(np.float32))
                    b["right_pos"].append(right_tips[t].astype(np.float32))
                    b["gripper_primary"].append(float(gripper_to_primary))
                    b["primary_sec"].append(float(dist))
                    b["closest_prim"].append(sec_info["closest_on_primary"])
                    b["closest_sec"].append(sec_info["closest_on_secondary"])

                    if dist > stop_thresh:
                        _flush_so_pair(pair, t)
                        so_active[pair] = False

        # Any active pair whose primary was not within 1 cm this timestep has
        # effectively infinite secondary distance → stop it.
        for pair, is_active in list(so_active.items()):
            if is_active and pair not in active_so_pairs_this_t:
                _flush_so_pair(pair, t)
                so_active[pair] = False

        if debug:
            _debug_log_timestep(
                t=t,
                av_rgb=av_rgb[t],
                av_seg=av_seg[t],
                eih_rgb=eih_rgb[t],
                eih_seg=eih_seg[t],
                pcd=pcd,
                left_tip=left_tips[t],
                right_tip=right_tips[t],
                obj_dists=obj_dists,
                id_to_name=id_to_name,
                start_thresh=start_thresh,
                stop_thresh=stop_thresh,
            )

    # End of trajectory — flush remaining active second-order interactions.
    for pair, is_active in so_active.items():
        if is_active:
            _flush_so_pair(pair, T, enforce_min=False)

    print(f"      t={T}/{T} done.        ")

    all_obj_ids: set[int] = set()
    for obj_dists in ts_data:
        for sid in obj_dists:
            if allowed_ids is None or sid in allowed_ids:
                all_obj_ids.add(sid)

    # Apply hysteresis per object to detect interaction windows.
    interactions: list[dict] = []

    for sid in sorted(all_obj_ids):
        active = False

        # Buffers for one active interaction.
        buf_timesteps: list[int] = []
        buf_left_pos: list[np.ndarray] = []
        buf_right_pos: list[np.ndarray] = []
        buf_left_dist: list[float] = []
        buf_right_dist: list[float] = []
        buf_min_dist: list[float] = []
        buf_left_closest: list[np.ndarray] = []
        buf_right_closest: list[np.ndarray] = []

        def _flush_interaction(end_t: int, *, enforce_min: bool = True) -> None:
            """Flush the current buffer as a completed interaction."""
            if enforce_min and len(buf_timesteps) < MIN_INTERACTION_STEPS:  # noqa: B023
                return
            obj_name = id_to_name.get(sid, str(sid))  # noqa: B023
            interactions.append(
                {
                    "order": 1,
                    "object_id": sid,  # noqa: B023
                    "object_name": obj_name,
                    "via_object_id": None,
                    "via_object_name": None,
                    "interaction_chain": f"gripper → {obj_name}",
                    "start_timestep": buf_timesteps[0],  # noqa: B023
                    "end_timestep": end_t,
                    "timesteps": np.array(buf_timesteps, dtype=np.int32),  # noqa: B023
                    "left_finger_pos": np.array(buf_left_pos, dtype=np.float32),  # noqa: B023
                    "right_finger_pos": np.array(buf_right_pos, dtype=np.float32),  # noqa: B023
                    "left_finger_dist": np.array(buf_left_dist, dtype=np.float32),  # noqa: B023
                    "right_finger_dist": np.array(buf_right_dist, dtype=np.float32),  # noqa: B023
                    "min_dist": np.array(buf_min_dist, dtype=np.float32),  # noqa: B023
                    "left_closest_point": np.array(buf_left_closest, dtype=np.float32),  # noqa: B023
                    "right_closest_point": np.array(
                        buf_right_closest,  # noqa: B023
                        dtype=np.float32,
                    ),
                }
            )

        for t in range(T):
            obj_info = ts_data[t].get(sid)

            if obj_info is None:
                # Object not visible this timestep — treat as max distance.
                min_d = np.inf
            else:
                min_d = min(obj_info["left_dist"], obj_info["right_dist"])

            if not active:
                if min_d < start_thresh:
                    active = True
                    buf_timesteps = []
                    buf_left_pos = []
                    buf_right_pos = []
                    buf_left_dist = []
                    buf_right_dist = []
                    buf_min_dist = []
                    buf_left_closest = []
                    buf_right_closest = []
            # (no else — fall through so we always append when active)

            if active:
                # Use obj_info if available, else fill with NaN.
                if obj_info is not None:
                    buf_timesteps.append(t)
                    buf_left_pos.append(left_tips[t].astype(np.float32))
                    buf_right_pos.append(right_tips[t].astype(np.float32))
                    buf_left_dist.append(obj_info["left_dist"])
                    buf_right_dist.append(obj_info["right_dist"])
                    buf_min_dist.append(min_d)
                    buf_left_closest.append(obj_info["left_closest"])
                    buf_right_closest.append(obj_info["right_closest"])
                else:
                    buf_timesteps.append(t)
                    nan3 = np.full(3, np.nan, dtype=np.float32)
                    buf_left_pos.append(left_tips[t].astype(np.float32))
                    buf_right_pos.append(right_tips[t].astype(np.float32))
                    buf_left_dist.append(np.nan)
                    buf_right_dist.append(np.nan)
                    buf_min_dist.append(np.nan)
                    buf_left_closest.append(nan3)
                    buf_right_closest.append(nan3)

                # Check stop condition.
                if min_d > stop_thresh:
                    _flush_interaction(t)
                    active = False

        # End of trajectory — flush any open interaction (no minimum length enforced).
        if active:
            _flush_interaction(T, enforce_min=False)

    if debug:
        _debug_log_interactions(interactions + second_order_interactions, T)

    return interactions + second_order_interactions


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------


def iter_hdf5_files(input_path: Path) -> Iterator[Path]:
    """Yield HDF5 files from a path (file or directory)."""
    if input_path.is_file():
        yield input_path
    else:
        for p in sorted(input_path.rglob("*.hdf5")):
            yield p


def process_dataset(
    input_path: Path,
    output_path: Path,
    demo_limit: int | None = None,
    start_thresh: float = INTERACTION_START_M,
    stop_thresh: float = INTERACTION_STOP_M,
    max_order: int = 2,
    *,
    debug: bool = False,
    rerun_connect: str | None = None,
) -> None:
    """Process all HDF5 files in *input_path* and write interactions to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if debug:
        _debug_init_rerun(connect_addr=rerun_connect)
        _debug_send_blueprint()

    interaction_count = 0

    with h5py.File(output_path, "w") as out_f:
        out_grp = out_f.require_group("interactions")

        for hdf5_file in iter_hdf5_files(input_path):
            print(f"\nProcessing: {hdf5_file}")
            with h5py.File(hdf5_file, "r") as f:
                data = f["data"]
                _seg_map_json = data.attrs.get("segmentation_name_mapping", None)
                seg_name_map: dict[str, str] = (
                    json.loads(_seg_map_json) if _seg_map_json else {}
                )

                demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
                if demo_limit is not None:
                    demo_keys = demo_keys[:demo_limit]

                for demo_key in demo_keys:
                    print(f"  [{demo_key}]")
                    demo = data[demo_key]
                    obs = demo["obs"]

                    if debug:
                        _debug_log_static(
                            seg_name_map=seg_name_map,
                            start_thresh=start_thresh,
                            stop_thresh=stop_thresh,
                            demo_key=demo_key,
                            source_file=str(hdf5_file.resolve()),
                        )

                    try:
                        interactions = extract_demo_interactions(
                            obs,
                            seg_name_map,
                            start_thresh=start_thresh,
                            stop_thresh=stop_thresh,
                            debug=debug,
                        )
                    except Exception as exc:
                        print(f"    ERROR: {exc}")
                        continue

                    interactions = [i for i in interactions if i["order"] <= max_order]

                    n1 = sum(1 for i in interactions if i["order"] == 1)
                    n2 = sum(1 for i in interactions if i["order"] == 2)
                    print(
                        f"    Found {len(interactions)} interaction(s) ({n1} first-order, {n2} second-order)."
                    )
                    for interaction in interactions:
                        order_tag = "[1st]" if interaction["order"] == 1 else "[2nd]"
                        t0 = interaction["start_timestep"]
                        t1 = interaction["end_timestep"]
                        print(
                            f"      {order_tag} {interaction['interaction_chain']}  (t={t0}..{t1})"
                        )

                    for interaction in interactions:
                        idx = interaction_count
                        interaction_count += 1

                        grp = out_grp.create_group(str(idx))

                        # Metadata as attributes.
                        grp.attrs["source_file"] = str(hdf5_file.resolve())
                        grp.attrs["demo_key"] = demo_key
                        grp.attrs["interaction_order"] = interaction["order"]
                        grp.attrs["object_id"] = interaction["object_id"]
                        grp.attrs["object_name"] = interaction["object_name"]
                        grp.attrs["interaction_chain"] = interaction[
                            "interaction_chain"
                        ]
                        grp.attrs["start_timestep"] = interaction["start_timestep"]
                        grp.attrs["end_timestep"] = interaction["end_timestep"]
                        if interaction["order"] == 2:
                            grp.attrs["via_object_id"] = interaction["via_object_id"]
                            grp.attrs["via_object_name"] = interaction[
                                "via_object_name"
                            ]

                        # Per-timestep arrays — schema depends on interaction order.
                        grp.create_dataset(
                            "timesteps",
                            data=interaction["timesteps"],
                            compression="gzip",
                        )
                        grp.create_dataset(
                            "left_finger_pos",
                            data=interaction["left_finger_pos"],
                            compression="gzip",
                        )
                        grp.create_dataset(
                            "right_finger_pos",
                            data=interaction["right_finger_pos"],
                            compression="gzip",
                        )
                        if interaction["order"] == 1:
                            grp.create_dataset(
                                "left_finger_dist",
                                data=interaction["left_finger_dist"],
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "right_finger_dist",
                                data=interaction["right_finger_dist"],
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "min_dist",
                                data=interaction["min_dist"],
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "left_closest_point",
                                data=interaction["left_closest_point"],
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "right_closest_point",
                                data=interaction["right_closest_point"],
                                compression="gzip",
                            )
                        else:
                            grp.create_dataset(
                                "gripper_to_primary_dist",
                                data=interaction["gripper_to_primary_dist"],
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "primary_to_secondary_dist",
                                data=interaction["primary_to_secondary_dist"],
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "closest_on_primary",
                                data=interaction["closest_on_primary"],
                                compression="gzip",
                            )
                            grp.create_dataset(
                                "closest_on_secondary",
                                data=interaction["closest_on_secondary"],
                                compression="gzip",
                            )

        out_f.attrs["total_interactions"] = interaction_count
        out_f.attrs["interaction_start_thresh_m"] = start_thresh
        out_f.attrs["interaction_stop_thresh_m"] = stop_thresh
        out_f.attrs["min_interaction_steps"] = MIN_INTERACTION_STEPS
        out_f.attrs["second_order_primary_thresh_m"] = SECOND_ORDER_PRIMARY_M

    print(f"\nDone. Wrote {interaction_count} interactions → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract gripper-object interaction sub-trajectories from LIBERO HDF5 datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to a single .hdf5 file or a directory of .hdf5 files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("datasets/interactions/interactions.hdf5"),
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--demo-limit",
        type=int,
        default=1,
        metavar="N",
        help="Process only the first N demos per file (useful for testing).",
    )
    parser.add_argument(
        "--start-thresh",
        type=float,
        default=INTERACTION_START_M,
        metavar="M",
        help="Distance threshold (m) to start an interaction.",
    )
    parser.add_argument(
        "--stop-thresh",
        type=float,
        default=INTERACTION_STOP_M,
        metavar="M",
        help="Distance threshold (m) to end an interaction (hysteresis).",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=1,
        metavar="N",
        help="Maximum interaction order to include in the output (e.g. 1 = first-order only, 2 = up to second-order).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Stream all intermediate data to a Rerun viewer: RGB images, "
            "segmentation overlays, merged point cloud, finger-tip positions, "
            "per-object distances, threshold reference lines, and detected "
            "interaction windows. Pair with --demo-limit 1 for fast iteration."
        ),
    )
    parser.add_argument(
        "--rerun-connect",
        type=str,
        default=None,
        metavar="ADDR",
        help=(
            "gRPC address of a running Rerun viewer (e.g. 127.0.0.1:9876). "
            "If omitted, a new viewer process is spawned automatically."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for python/numpy/torch/jax RNGs.",
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    process_dataset(
        args.input,
        args.output,
        demo_limit=args.demo_limit,
        start_thresh=args.start_thresh,
        stop_thresh=args.stop_thresh,
        max_order=args.max_order,
        debug=args.debug,
        rerun_connect=args.rerun_connect,
    )


if __name__ == "__main__":
    main()

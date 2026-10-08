"""Break a demo trajectory into segments using post-condition lambdas.

The iterative task-analysis strategy proposes per-segment ``post_condition``
lambdas (signature: ``lambda ee, keypoints, ee_R: <scalar cost>``;
satisfied ⇔ cost ≤ 0). Segments are *chained* — segment ``i`` ends at the
first time ``post_i`` becomes satisfied for at least ``min_segment_length``
consecutive frames; segment ``i+1`` begins at that timestep.

Two helpers live here:

- :func:`load_demo_pose_and_keypoints` reads one HDF5 demo and returns
  ``(ee_pos, ee_R, keypoints)`` — everything needed to evaluate a constraint
  lambda at every timestep without recomputing point clouds elsewhere.
- :func:`segment_demo_by_conditions` runs the chained scan and returns one
  :class:`SegmentBoundary` per proposed segment (or a fallback for unmet
  conditions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import h5py
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation


@dataclass
class SegmentBoundary:
    """One demo segment derived from chained post-condition lambdas."""

    subtraj_idx: int
    t_start: int
    t_end: int  # exclusive
    fallback: bool = False
    reason: str | None = None
    """Diagnostic when ``fallback`` is True (e.g. ``"post never satisfied"``)."""


def load_demo_pose_and_keypoints(
    hdf5_path: str,
    demo_key: str,
    entity_ids: set[int],
    erode_px: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]] | None:
    """Read one demo and return per-timestep EE pose + entity centroids.

    Parameters
    ----------
    hdf5_path:    Demo HDF5 path.
    demo_key:     Demo key inside ``f["data"]``.
    entity_ids:   Segmentation IDs whose centroids we want at every timestep.
    erode_px:     Erosion kernel applied to the seg mask before centroiding.

    Returns
    -------
    ``(ee_pos[T,3], ee_R[T,3,3], keypoints[entity][T,3])`` or ``None`` on
    failure. Frames where an entity has no points contain ``NaN`` centroids.
    """
    from policy_abstraction.envs.libero.config import (
        LIBERO_DEPTH_FAR,
        LIBERO_DEPTH_NEAR,
    )
    from policy_abstraction.utils.depth import mujoco_depth_to_metric
    from policy_abstraction.utils.segmentation import (
        build_segmented_pointcloud,
        erode_segmentation,
    )

    try:
        with h5py.File(hdf5_path, "r") as f:
            obs = f["data"][demo_key]["obs"]  # type: ignore[index]
            av_rgb = obs["agentview_rgb"][:][:, ::-1, ::-1, :]
            av_dep = obs["agentview_depth"][:][:, ::-1, ::-1]
            av_seg = obs["agentview_segmentation"][:][:, ::-1, ::-1]
            av_K = obs["agentview_intrinsic"][:]
            av_E = obs["agentview_extrinsic"][:]
            ee_pos = obs["ee_pos"][:].astype(np.float32)
            ee_ori = obs["ee_ori"][:].astype(np.float64)  # axis-angle
    except Exception as exc:
        logger.warning(
            f"[segmentation_from_conditions] Could not load HDF5 {hdf5_path}/{demo_key}: {exc}"
        )
        return None

    T = av_rgb.shape[0]
    if av_dep.max() <= 1.0:
        av_dep = mujoco_depth_to_metric(
            av_dep, near=LIBERO_DEPTH_NEAR, far=LIBERO_DEPTH_FAR
        )

    ee_R = Rotation.from_rotvec(ee_ori).as_matrix().astype(np.float32)  # [T, 3, 3]

    keypoints: dict[int, np.ndarray] = {
        eid: np.full((T, 3), np.nan, dtype=np.float32) for eid in entity_ids
    }
    if not entity_ids:
        return ee_pos, ee_R, keypoints

    for t in range(T):
        try:
            seg = av_seg[t]
            if erode_px > 0:
                seg = erode_segmentation(seg, erode_px)
            K = av_K[t]
            K3 = K[:3, :3] if K.shape == (4, 4) else K
            pcd = build_segmented_pointcloud(av_rgb[t], av_dep[t], seg, K3, av_E[t])
            if pcd.shape[0] == 0:
                continue
            seg_col = pcd[:, 6].astype(np.int32)
            for eid in entity_ids:
                mask = seg_col == eid
                if mask.any():
                    keypoints[eid][t] = pcd[mask, :3].mean(axis=0)
        except Exception as exc:
            logger.debug(
                f"[segmentation_from_conditions] step {t} centroid failed: {exc}"
            )

    # Carry-forward fill so lambdas see a stable centroid even on dropped frames.
    for eid in entity_ids:
        kp = keypoints[eid]
        for t in range(1, T):
            nan_rows = np.isnan(kp[t])
            if nan_rows.any():
                kp[t, nan_rows] = kp[t - 1, nan_rows]

    return ee_pos, ee_R, keypoints


class _CarryForwardKeypoints(dict):
    """Dict that returns a zero 3-vector for any missing key.

    Constraint lambdas may reference entity IDs that aren't actually present
    in this demo (e.g. after pair re-mapping drops some). Returning zeros keeps
    the lambda from raising; the resulting cost may be wrong but the segmenter
    will surface the failure via ``fallback=True`` if no consistent transition
    is found.
    """

    def __missing__(self, key):  # type: ignore[override]
        return np.zeros(3, dtype=np.float32)


def segment_demo_by_conditions(
    ee_pos: np.ndarray,
    ee_R: np.ndarray,
    keypoints: dict[int, np.ndarray],
    post_conditions: list[Callable[[np.ndarray, dict, np.ndarray], float]],
    min_segment_length: int = 5,
) -> list[SegmentBoundary]:
    """Chained left-to-right segmentation by post-condition lambdas.

    For each ``i`` (in order), find the first ``t > cursor`` such that
    ``post_i(ee, kp, ee_R) <= 0`` for ``min_segment_length`` consecutive
    frames; that ``t`` is the segment end. Segment ``i+1`` starts at the
    same ``t`` (chained). If a segment's post never fires for the required
    streak, the segment is marked fallback and downstream segments are also
    fallback (their boundaries can't be derived once the chain breaks).

    Parameters
    ----------
    ee_pos:              ``[T, 3]`` end-effector world positions.
    ee_R:                ``[T, 3, 3]`` world-from-EE rotation matrices.
    keypoints:           ``{seg_id: [T, 3]}`` per-timestep centroids.
    post_conditions:     List of compiled lambdas, one per segment.
    min_segment_length:  Minimum number of consecutive frames the post must
                         remain satisfied to register a transition.

    Returns
    -------
    List of :class:`SegmentBoundary`, one per element of *post_conditions*.
    """
    T = ee_pos.shape[0]
    K = len(post_conditions)
    if K == 0:
        return []

    # Pre-evaluate each post at every timestep.
    sat = np.zeros((K, T), dtype=bool)
    eval_failures: dict[int, int] = {}
    for i, fn in enumerate(post_conditions):
        for t in range(T):
            kp_t = _CarryForwardKeypoints({eid: kp[t] for eid, kp in keypoints.items()})
            try:
                cost = float(fn(ee_pos[t], kp_t, ee_R[t]))
            except Exception:
                eval_failures[i] = eval_failures.get(i, 0) + 1
                continue
            if not np.isfinite(cost):
                continue
            sat[i, t] = cost <= 0.0
    for i, n in eval_failures.items():
        logger.debug(
            f"[segmentation_from_conditions] post {i} raised in {n}/{T} frames"
        )

    segments: list[SegmentBoundary] = []
    cursor = 0
    chain_broken = False

    for i in range(K):
        if chain_broken:
            segments.append(
                SegmentBoundary(
                    subtraj_idx=i,
                    t_start=cursor,
                    t_end=cursor,
                    fallback=True,
                    reason="upstream segment failed",
                )
            )
            continue

        # Find first window of length min_segment_length where post_i is True.
        t_end = -1
        run = 0
        for t in range(cursor, T):
            if sat[i, t]:
                run += 1
                if run >= min_segment_length:
                    t_end = t + 1  # end is exclusive; use the last frame of run
                    break
            else:
                run = 0

        # Edge case: if min_segment_length=1 above already covers it.
        # If post never reached the required streak but did fire briefly,
        # we still treat this as a failure for downstream chaining safety.
        if t_end < 0:
            # Last segment can extend to T even without an explicit post.
            if i == K - 1:
                segments.append(
                    SegmentBoundary(
                        subtraj_idx=i,
                        t_start=cursor,
                        t_end=T,
                        fallback=True,
                        reason="post never satisfied; extending to T",
                    )
                )
            else:
                segments.append(
                    SegmentBoundary(
                        subtraj_idx=i,
                        t_start=cursor,
                        t_end=cursor,
                        fallback=True,
                        reason="post never satisfied",
                    )
                )
                chain_broken = True
            continue

        if t_end <= cursor:
            # Defensive: shouldn't happen given the loop invariants, but
            # guard against a zero-length segment that would break DTW.
            t_end = min(cursor + 1, T)

        segments.append(
            SegmentBoundary(
                subtraj_idx=i,
                t_start=cursor,
                t_end=t_end,
                fallback=False,
                reason=None,
            )
        )
        cursor = t_end

    return segments

"""Build structured guidance parameter arrays from distance pair data.

This module replaces the old closure-based guidance builder.  Instead of
returning a Python callable (which breaks ``jax.jit``), it returns a plain dict
of numpy arrays that can be passed through ``module_jit`` and reconstructed into
a JAX-differentiable function on the model side.

Guidance parameter schema
-------------------------
For N robot-entity-involving pairs (``ee``, ``gripper_l``, ``gripper_r``),
chunk_size T, and max_surface_pts S:

``__gp_ee_pos_0__``          (3,)      float32 — EE position at current timestep
``__gp_expected_dists__``    (N, T)    float32 — ref distances per pair per chunk step
``__gp_expected_stds__``     (N, T)    float32 — per-pair uncertainty (std from multi-demo alignment)

``__gp_dist_types__``        (N,)      uint8   — 0=centroid/axis/plane, 1=surface, 2=orientation, 3=gripper
``__gp_source_type__``       (N,)      uint8   — 0=EE, 1=finger_l, 2=finger_r, 3=orientation, 4=gripper
``__gp_waypoints__``         (N, 3)    float32 — centroid of other entity per pair (zeros for self-ref pairs)
``__gp_has_axis__``          (N,)      uint8   — 1 if axis applies
``__gp_axes__``              (N, 3)    float32 — signed axis: +axis if ee_is_from, -axis if ee_is_to
``__gp_has_plane__``         (N,)      uint8   — 1 if plane applies
``__gp_plane_normals__``     (N, 3)    float32 — zeros if has_plane=0
``__gp_surface_counts__``    (N,)      int32   — number of surface points (0 for non-surface)
``__gp_surface_pts__``       (N, S, 3) float32 — padded surface point cloud
``__gp_guidance_weights__``  (N,)      float32 — per-pair guidance weight (scalar broadcast if all equal)

The detection sentinel is ``"__gp_expected_dists__"``.  If this key is present
in the obs dict, the server will use the params path instead of the callable path.

Mathematical basis
------------------
The model integrates x_t (SO3 velocity actions) to recover EE positions at each
chunk timestep, then computes distances from those positions to the reference
waypoints.  This yields a per-timestep distance trajectory that is compared
against the reference to compute the guidance potential:

    pos[k]       = ee_pos_0 + cumsum(x_t[:k+1, :3])
    dist[i, k]   = distance_fn_i(pos[k], waypoints[i])   (type-dependent)
    error[i, k]  = dist[i, k] - expected[i, k]
    f(x_t)       = -0.5 * sum_{i,k} guidance_factors[i] * error[i, k]^2

``jax.grad(f)`` is applied via autograd through the full integration chain.

Axis sign convention
--------------------
Axes are stored with sign encoding the measurement direction:

    ee_is_from: axis_stored = +axis   →  dist = dot(waypoint - pos, axis)
    ee_is_to:   axis_stored = -axis   →  dist = dot(waypoint - pos, -axis)
                                                = dot(pos - waypoint, axis)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from policy_abstraction.vlm.types import DistancePair

from policy_abstraction.vlm.types import ROBOT_ENTITIES
from policy_steering.params import (  # noqa: F401
    GP_KEYS,
    GP_SENTINEL_KEY,
)

logger = logging.getLogger(__name__)

# Action dimensions corresponding to EE translation (x, y, z)
_EE_DIMS = slice(0, 3)

# Distance type codes packed into uint8
_DIST_TYPE_CENTROID = np.uint8(0)
_DIST_TYPE_SURFACE = np.uint8(1)
_DIST_TYPE_ORIENTATION = np.uint8(2)
_DIST_TYPE_GRIPPER = np.uint8(3)

# Source entity type codes packed into uint8
_SOURCE_EE = np.uint8(0)
_SOURCE_FINGER_L = np.uint8(1)
_SOURCE_FINGER_R = np.uint8(2)
_SOURCE_ORIENTATION = np.uint8(3)
_SOURCE_GRIPPER = np.uint8(4)

# Max surface points to send per surface pair (to cap payload size)
_DEFAULT_MAX_SURFACE_PTS = 256


def _nonfinite_fields(**fields) -> list[str]:
    """Names of the given arrays/scalars that hold a NaN or inf."""
    return [
        name
        for name, val in fields.items()
        if not np.all(np.isfinite(np.asarray(val, dtype=np.float64)))
    ]


def build_distance_guidance_params(
    pairs: list[DistancePair],
    live_distances: dict[str, float],
    entity_centroids: dict[int | str, np.ndarray],
    object_pointclouds: dict[str, np.ndarray] | None,
    ref_traj: np.ndarray,
    pair_labels: list[str],
    aligned_ts: np.ndarray,
    chunk_size: int,
    int_name_map: dict[int, str] | None = None,
    dt: float = 1.0,
    guidance_factor: float = 1.0,
    guidance_weights: np.ndarray | None = None,
    max_surface_pts: int = _DEFAULT_MAX_SURFACE_PTS,
    ref_std: np.ndarray | None = None,
    ee_ori_0: np.ndarray | None = None,
    gripper_0: np.ndarray | None = None,
    relation_filter: frozenset[str] | None = None,
) -> dict | None:
    """Build JIT-compatible guidance parameter arrays from distance pair data.

    The model integrates x_t (action velocities) to get EE position trajectories,
    computes distances to the reference waypoints, and uses JAX autograd through
    the full chain.  Only the reference geometry (waypoints, axes, plane normals,
    surface point clouds) and expected distance trajectories are sent as params.
    Live distances are NOT included — they are re-computed from x_t on the model side.

    Args:
        pairs: Translated distance pairs (from :attr:`AbstractionResult.pairs`).
            Only pairs that involve a robot entity (``ee``, ``gripper_l``,
            or ``gripper_r``) are included in the output. The JAX side then
            integrates velocity actions from ``ee_pos_0`` (and FK-derives the
            finger positions) to evaluate per-pair distances.
        live_distances: Current measured distance per pair label
            (from :attr:`AbstractionResult.live_distances`).
            Used only to filter out pairs that are not currently observable.
        entity_centroids: ``{entity: [3]}`` centroids including ``"ee"``
            (from :attr:`AbstractionResult.entity_centroids`).
        object_pointclouds: ``{name: [M, 7]}`` point clouds for each visible
            object (from :attr:`AbstractionResult.object_pointclouds`).
            Required for surface pairs; may be ``None`` to disable surface guidance.
        ref_traj: ``(T_ref, n_pairs)`` reference distance trajectory
            (from :attr:`RolloutAbstractor.ref_traj`).
        pair_labels: Column labels for ``ref_traj``
            (from :attr:`RolloutAbstractor.pair_labels`).
        aligned_ts: Per-pair alignment indices ``(n_pairs,)`` int32
            (from :attr:`AbstractionResult.aligned_ts`).  Each pair uses its
            own temporal position to slice the reference trajectory.
        chunk_size: Number of action steps to generate guidance for
            (typically ``n_chunks * action_horizon``).
        int_name_map: ``{seg_id: name}`` for surface point cloud lookups.
        dt: Duration of each action step in seconds (used to scale velocity integration).
        guidance_factor: Per-pair guidance weight scalar.  Applied uniformly to all N robot-entity-involving pairs.
        guidance_weights: Per-pair guidance weights.  May be:

            - A single ``float`` (applied uniformly to all N robot-entity-involving pairs).
            - A ``np.ndarray`` of shape ``(len(pairs),)`` with one weight per
              input pair (same indexing as *pairs*; unresolvable pairs are
              dropped before the array is consumed).

        max_surface_pts: Maximum number of surface points per surface pair.
        ref_std: ``(T_ref, n_pairs)`` per-pair uncertainty (std-dev) from
            multi-demo DTW alignment.  When provided, the output will contain
            ``__gp_expected_stds__`` ``(N, T)`` alongside ``__gp_expected_dists__``.
            ``None`` omits the key (backward-compatible).
        relation_filter: Optional set of relation categories to keep — a subset
            of ``{"positional", "orientation", "gripper"}``. Each pair is
            classified by its ``distance_type`` (ORIENTATION→orientation,
            GRIPPER_APERTURE / finger-pair→gripper, CENTROID/SURFACE→positional)
            and dropped when its category is not in the set. ``None`` (default)
            keeps every category — bit-for-bit identical to the unfiltered path.
            Used by the per-relation guidance ablation.

    Returns:
        Dict of numpy arrays (see module docstring), or ``None`` when no valid
        robot-entity-involving pair can be resolved.
    """
    from policy_abstraction.vlm.types import DistanceType

    # ``ee_pos_0`` seeds the JAX-side velocity integration (finger positions
    # are recovered via FK from this EE pose + gripper state).  It is always
    # needed, even for gripper-only pairs, so bail out if it is missing.
    ee_pos = entity_centroids.get("ee")
    if ee_pos is None:
        logger.warning(
            "[guidance_builder] 'ee' not found in entity_centroids; skipping."
        )
        return None

    int_name_map = int_name_map or {}
    T_ref = ref_traj.shape[0]

    # Build col_index lookup: pair_label → column index in ref_traj
    label_to_col: dict[str, int] = {lbl: i for i, lbl in enumerate(pair_labels)}

    # Collect per-pair data
    dist_types: list[int] = []
    source_types: list[int] = []
    expected_dists: list[np.ndarray] = []  # each (chunk_size,)
    expected_stds: list[
        np.ndarray
    ] = []  # each (chunk_size,) — uncertainty per pair per step
    waypoints: list[np.ndarray] = []
    has_axis: list[int] = []
    axes: list[np.ndarray] = []
    has_plane: list[int] = []
    plane_normals: list[np.ndarray] = []
    surface_counts: list[int] = []
    surface_pts_list: list[np.ndarray] = []  # each (M, 3), variable M
    ee_vectors: list[np.ndarray] = []  # each (3,), body-frame vector for orientation
    pair_weight: list[
        float
    ] = []  # per-pair guidance factor (scalar fallback applied later)

    for pair_idx, pair in enumerate(pairs):
        # Filter to robot-entity-involving pairs only.
        from_is_robot = pair.from_entity in ROBOT_ENTITIES
        to_is_robot = pair.to_entity in ROBOT_ENTITIES
        if not (from_is_robot or to_is_robot):
            continue
        ee_is_from = from_is_robot

        # gripper_l <-> gripper_r is always an aperture pair regardless of the
        # VLM-declared distance_type: the two fingers move symmetrically, so
        # the only gradient that produces the intended closure/opening is on
        # the gripper channel. Treating it as centroid would yield spurious
        # EE-translation and orientation gradients via FK.
        is_finger_pair = (
            from_is_robot
            and to_is_robot
            and pair.from_entity in ("gripper_l", "gripper_r")
            and pair.to_entity in ("gripper_l", "gripper_r")
            and pair.from_entity != pair.to_entity
        )

        # Per-relation ablation: classify the pair into one of the three
        # relation categories and drop it when the caller restricted the set.
        if relation_filter is not None:
            if pair.distance_type == DistanceType.ORIENTATION:
                _category = "orientation"
            elif pair.distance_type == DistanceType.GRIPPER_APERTURE or is_finger_pair:
                _category = "gripper"
            else:
                _category = "positional"
            if _category not in relation_filter:
                continue

        # Determine source entity type for JAX-side guidance
        if is_finger_pair or pair.distance_type == DistanceType.GRIPPER_APERTURE:
            src_type = int(_SOURCE_GRIPPER)
        elif pair.distance_type == DistanceType.ORIENTATION:
            src_type = int(_SOURCE_ORIENTATION)
        elif pair.from_entity == "gripper_l" or (
            not ee_is_from and pair.to_entity == "gripper_l"
        ):
            src_type = int(_SOURCE_FINGER_L)
        elif pair.from_entity == "gripper_r" or (
            not ee_is_from and pair.to_entity == "gripper_r"
        ):
            src_type = int(_SOURCE_FINGER_R)
        else:
            src_type = int(_SOURCE_EE)

        # Skip pairs with zero guidance weight — they are irrelevant for the
        # current sub-trajectory and must not be sent to the policy.
        if guidance_weights is not None and float(guidance_weights[pair_idx]) == 0.0:
            continue

        # Use live_distances only to filter unobservable pairs — not for computation
        live_d = live_distances.get(pair.label)
        if live_d is None or not np.isfinite(live_d):
            continue

        # Reference distances: slice from ref_traj using per-pair aligned_ts
        col_idx = label_to_col.get(pair.label)
        if col_idx is None:
            continue
        pair_t = int(aligned_ts[col_idx])
        pair_t_end = min(pair_t + chunk_size, T_ref)
        ref_slice = ref_traj[pair_t:pair_t_end, col_idx]  # (T_window,)
        if len(ref_slice) < chunk_size:
            pad_val = float(ref_slice[-1]) if len(ref_slice) > 0 else float(live_d)
            ref_slice = np.concatenate(
                [
                    ref_slice,
                    np.full(chunk_size - len(ref_slice), pad_val, dtype=np.float32),
                ]
            )
        ref_exp = ref_slice[:chunk_size].astype(np.float32)  # (chunk_size,)

        # Per-pair uncertainty from multi-demo alignment
        if ref_std is not None:
            std_slice = ref_std[pair_t:pair_t_end, col_idx]
            if len(std_slice) < chunk_size:
                pad_val = float(std_slice[-1]) if len(std_slice) > 0 else 0.0
                std_slice = np.concatenate(
                    [
                        std_slice,
                        np.full(chunk_size - len(std_slice), pad_val, dtype=np.float32),
                    ]
                )
            std_exp = std_slice[:chunk_size].astype(np.float32)
        else:
            std_exp = np.zeros(chunk_size, dtype=np.float32)

        # Waypoint: centroid of the other entity (or zeros for self-referencing pairs)
        if (
            pair.distance_type
            in (
                DistanceType.ORIENTATION,
                DistanceType.GRIPPER_APERTURE,
            )
            or is_finger_pair
        ):
            # Self-referencing pairs have no scene waypoint; use zeros placeholder
            waypoint = np.zeros(3, dtype=np.float32)
        else:
            other_entity = pair.to_entity if ee_is_from else pair.from_entity
            waypoint = entity_centroids.get(other_entity)
            if waypoint is None:
                continue

        # Surface points: for surface-type pairs, sample from point cloud
        is_surface = (
            pair.distance_type == DistanceType.SURFACE
            and pair.axis is None
            and pair.plane is None
        )
        surf_pts = np.zeros((0, 3), dtype=np.float32)
        if is_surface and object_pointclouds is not None:
            other_entity = pair.to_entity if ee_is_from else pair.from_entity
            name = (
                int_name_map.get(int(other_entity), str(other_entity))
                if isinstance(other_entity, int)
                else str(other_entity)
            )
            pcd = object_pointclouds.get(name)
            if pcd is not None and pcd.shape[0] > 0:
                surf_pts = pcd[:, :3].astype(np.float32)  # (M, 3)
                if surf_pts.shape[0] > max_surface_pts:
                    # Deterministic stride-based subsample so JAX never sees
                    # randomly-ordered arrays (which could affect trace caching).
                    idx = np.linspace(
                        0, surf_pts.shape[0] - 1, max_surface_pts, dtype=int
                    )
                    surf_pts = surf_pts[idx]

        # Determine dist_type code
        if pair.distance_type == DistanceType.ORIENTATION:
            dt_code = int(_DIST_TYPE_ORIENTATION)
        elif pair.distance_type == DistanceType.GRIPPER_APERTURE or is_finger_pair:
            dt_code = int(_DIST_TYPE_GRIPPER)
        elif is_surface:
            dt_code = int(_DIST_TYPE_SURFACE)
        else:
            dt_code = int(_DIST_TYPE_CENTROID)

        # Axis: stored with sign encoding direction for positional pairs.
        #   ee_is_from: stored = +axis  →  dist = dot(waypoint - pos, axis)
        #   ee_is_to:   stored = -axis  →  dist = dot(waypoint - pos, -axis) = dot(pos - waypoint, axis)
        #   For orientation pairs: axis is stored unsigned (it is the world-
        #   frame direction the gripper z-axis is projected onto); plane is
        #   stored unsigned in the plane_normals slot and indicates the
        #   in-plane-magnitude variant.  The JAX-side dispatch on has_plane
        #   selects between the two interpretations.
        ax = np.zeros(3, dtype=np.float32)
        pn = np.zeros(3, dtype=np.float32)
        if pair.distance_type == DistanceType.ORIENTATION:
            if pair.axis is None and pair.plane is None:
                logger.debug(
                    f"[guidance_builder] Orientation pair {pair.label!r} has no "
                    "axis or plane direction; skipping."
                )
                continue
            if pair.axis is not None:
                v = np.asarray(pair.axis, dtype=np.float64)
                n = np.linalg.norm(v)
                if n <= 1e-9:
                    logger.debug(
                        f"[guidance_builder] Orientation pair {pair.label!r} has "
                        "zero-norm axis; skipping."
                    )
                    continue
                ax = (v / n).astype(np.float32)
            if pair.plane is not None:
                v = np.asarray(pair.plane, dtype=np.float64)
                n = np.linalg.norm(v)
                if n <= 1e-9:
                    logger.debug(
                        f"[guidance_builder] Orientation pair {pair.label!r} has "
                        "zero-norm plane normal; skipping."
                    )
                    continue
                pn = (v / n).astype(np.float32)
        else:
            if pair.axis is not None:
                v = np.asarray(pair.axis, dtype=np.float64)
                n = np.linalg.norm(v)
                if n > 1e-9:
                    signed = (v / n) if ee_is_from else -(v / n)
                    ax = signed.astype(np.float32)
            if pair.plane is not None:
                v = np.asarray(pair.plane, dtype=np.float64)
                n = np.linalg.norm(v)
                if n > 1e-9:
                    pn = (v / n).astype(np.float32)

        weight = (
            float(guidance_weights[pair_idx]) if guidance_weights is not None else 1.0
        )
        # One NaN or inf in any of these becomes a NaN in every action channel
        # of the policy's chunk, and the env server then rejects the whole
        # step. Drop the pair instead and say which value was bad.
        nonfinite = _nonfinite_fields(
            weight=weight, expected=ref_exp, waypoint=waypoint, surface_pts=surf_pts
        )
        if nonfinite:
            logger.warning(
                f"[guidance_builder] Dropping pair {pair.label!r}: non-finite "
                f"{nonfinite}."
            )
            continue

        dist_types.append(dt_code)
        source_types.append(src_type)
        expected_dists.append(ref_exp)
        expected_stds.append(std_exp)
        waypoints.append(np.asarray(waypoint, dtype=np.float32))
        has_axis.append(1 if pair.axis is not None else 0)
        axes.append(ax)
        has_plane.append(1 if pair.plane is not None else 0)
        plane_normals.append(pn)
        # Body-frame vector for orientation pairs (default: z-axis [0,0,1])
        if (
            pair.distance_type == DistanceType.ORIENTATION
            and pair.ee_vector is not None
        ):
            ev = np.asarray(pair.ee_vector, dtype=np.float64)
            ev_n = np.linalg.norm(ev)
            if ev_n > 1e-9:
                ee_vec = (ev / ev_n).astype(np.float32)
            else:
                ee_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            ee_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        surface_counts.append(int(surf_pts.shape[0]))
        surface_pts_list.append(surf_pts)
        ee_vectors.append(ee_vec)
        pair_weight.append(weight)

    if not expected_dists:
        logger.debug(
            "[guidance_builder] No valid robot-entity-involving pairs; guidance disabled."
        )
        return None

    # The shared scalars reach every pair, so a bad one disables guidance for
    # this chunk rather than one pair.
    nonfinite = _nonfinite_fields(
        ee_pos_0=ee_pos,
        ee_ori_0=ee_ori_0 if ee_ori_0 is not None else 0.0,
        guidance_factor=guidance_factor,
        dt=dt,
    )
    if nonfinite:
        logger.warning(
            f"[guidance_builder] Non-finite {nonfinite}; guidance disabled for "
            "this chunk."
        )
        return None

    N = len(expected_dists)
    logger.debug(
        f"[guidance_builder] N={N} active pairs, S_max={max_surface_pts} (fixed)"
    )
    # Always pad to max_surface_pts so the array shape is constant across calls.
    # Variable S_max causes JAX to retrace the compiled inference function on
    # every policy query (7+ second penalty each time).
    S_max = max_surface_pts

    # Pack surface points into padded (N, S_max, 3) array
    surf_padded = np.zeros((N, S_max, 3), dtype=np.float32)
    for i, sp in enumerate(surface_pts_list):
        if sp.shape[0] > 0:
            surf_padded[i, : sp.shape[0]] = sp

    out = {
        "__gp_guidance_factor__": np.float32(guidance_factor),  # scalar fallback
        "__gp_ee_pos_0__": np.asarray(ee_pos, dtype=np.float32),  # (3,)
        "__gp_dt__": np.float32(dt),  # scalar
        "__gp_guidance_weights__": np.array(pair_weight, dtype=np.float32),  # (N,)
        "__gp_expected_dists__": np.stack(expected_dists, axis=0).astype(
            np.float32
        ),  # (N, T)
        "__gp_expected_stds__": np.stack(expected_stds, axis=0).astype(
            np.float32
        ),  # (N, T)
        "__gp_dist_types__": np.array(dist_types, dtype=np.uint8),  # (N,)
        "__gp_source_type__": np.array(source_types, dtype=np.uint8),  # (N,)
        "__gp_waypoints__": np.stack(waypoints, axis=0).astype(np.float32),  # (N, 3)
        "__gp_has_axis__": np.array(has_axis, dtype=np.uint8),  # (N,)
        "__gp_axes__": np.stack(axes, axis=0).astype(np.float32),  # (N, 3) signed
        "__gp_has_plane__": np.array(has_plane, dtype=np.uint8),  # (N,)
        "__gp_plane_normals__": np.stack(plane_normals, axis=0).astype(
            np.float32
        ),  # (N, 3)
        "__gp_surface_counts__": np.array(surface_counts, dtype=np.int32),  # (N,)
        "__gp_surface_pts__": surf_padded,  # (N, S, 3)
        "__gp_ee_vectors__": np.stack(ee_vectors, axis=0).astype(np.float32),  # (N, 3)
    }

    # EE orientation and gripper state for FK-based finger guidance
    if ee_ori_0 is not None:
        out["__gp_ee_ori_0__"] = np.asarray(ee_ori_0, dtype=np.float32)  # (3,)
    else:
        out["__gp_ee_ori_0__"] = np.zeros(3, dtype=np.float32)
    if gripper_0 is not None:
        out["__gp_gripper_0__"] = np.asarray(gripper_0, dtype=np.float32).ravel()[
            :1
        ]  # (1,)
    else:
        out["__gp_gripper_0__"] = np.zeros(1, dtype=np.float32)

    return out


def merge_guidance_params(a: dict | None, b: dict | None) -> dict | None:
    """Merge two guidance parameter dicts into one by concatenating pair arrays.

    Both dicts must have been produced by :func:`build_distance_guidance_params`.
    Scalars (``__gp_ee_pos_0__``, ``__gp_dt__``) are taken from *a*; pair-level
    arrays (``N`` axis = axis 0) are concatenated.  The surface-point padding
    dimension ``S`` is broadcast to the larger of the two dicts.

    Parameters
    ----------
    a, b:
        Guidance parameter dicts.  Either may be ``None``; if both are ``None``
        the function returns ``None``.

    Returns
    -------
    dict or ``None``
    """
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    # Pair-level array keys that need concatenation along axis 0
    pair_keys_1d = (
        "__gp_guidance_weights__",
        "__gp_dist_types__",
        "__gp_source_type__",
        "__gp_has_axis__",
        "__gp_has_plane__",
        "__gp_surface_counts__",
    )
    pair_keys_2d = (
        "__gp_expected_dists__",
        "__gp_expected_stds__",
        "__gp_waypoints__",
        "__gp_axes__",
        "__gp_plane_normals__",
        "__gp_ee_vectors__",
    )

    merged: dict = {}

    # Scalars: take from a
    merged["__gp_ee_pos_0__"] = a["__gp_ee_pos_0__"]
    merged["__gp_dt__"] = a["__gp_dt__"]

    # 1-D pair arrays
    for key in pair_keys_1d:
        # Backward compat: if source_type is missing, default to 0 (EE)
        a_val = a.get(key)
        b_val = b.get(key)
        if a_val is None:
            n_a = a["__gp_guidance_weights__"].shape[0]
            a_val = np.zeros(n_a, dtype=np.uint8)
        if b_val is None:
            n_b = b["__gp_guidance_weights__"].shape[0]
            b_val = np.zeros(n_b, dtype=np.uint8)
        merged[key] = np.concatenate([a_val, b_val], axis=0)

    # 2-D pair arrays
    for key in pair_keys_2d:
        # Backward compat: __gp_ee_vectors__ may be absent in older params
        n_a = a["__gp_guidance_weights__"].shape[0]
        n_b = b["__gp_guidance_weights__"].shape[0]
        _default_2d = {
            "__gp_ee_vectors__": lambda n: np.tile(
                np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
            ),
        }
        if key in _default_2d:
            arr_a = a.get(key, _default_2d[key](n_a))
            arr_b = b.get(key, _default_2d[key](n_b))
        else:
            arr_a = a[key]
            arr_b = b[key]
        # Pad time dimension if chunk_sizes differ (take max)
        if arr_a.shape[1] != arr_b.shape[1]:
            T = max(arr_a.shape[1], arr_b.shape[1])

            def _pad_T(arr: np.ndarray, T: int) -> np.ndarray:
                if arr.shape[1] < T:
                    pad = np.repeat(arr[:, -1:, ...], T - arr.shape[1], axis=1)
                    arr = np.concatenate([arr, pad], axis=1)
                return arr

            arr_a = _pad_T(arr_a, T)
            arr_b = _pad_T(arr_b, T)
        merged[key] = np.concatenate([arr_a, arr_b], axis=0)

    # Surface points: (N, S, 3) — pad S to max
    surf_a = a["__gp_surface_pts__"]
    surf_b = b["__gp_surface_pts__"]
    S = max(surf_a.shape[1], surf_b.shape[1])
    if surf_a.shape[1] < S:
        pad = np.zeros((surf_a.shape[0], S - surf_a.shape[1], 3), dtype=np.float32)
        surf_a = np.concatenate([surf_a, pad], axis=1)
    if surf_b.shape[1] < S:
        pad = np.zeros((surf_b.shape[0], S - surf_b.shape[1], 3), dtype=np.float32)
        surf_b = np.concatenate([surf_b, pad], axis=1)
    merged["__gp_surface_pts__"] = np.concatenate([surf_a, surf_b], axis=0)

    # Carry through scalar guidance keys if present in either dict
    for key in (
        "__gp_ee_ori_0__",
        "__gp_gripper_0__",
    ):
        if key in a:
            merged[key] = a[key]
        elif key in b:
            merged[key] = b[key]

    return merged

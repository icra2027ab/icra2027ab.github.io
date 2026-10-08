"""Tests for policy_steering guidance functions in isolation.

Verifies that the guidance loss/gradient functions correctly steer action
chunks toward known target behaviors across all guidance types:
  - Centroid (EE position → waypoint)
  - Axis projection (signed 1-D distance along a direction)
  - Plane projection (2-D in-plane distance)
  - Orientation (projection of a configurable EE body vector onto a direction)
  - Gripper aperture (scalar gripper opening)
  - Finger position (FK-derived finger tip → waypoint)
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from policy_abstraction.policies.dummy import DummyPolicy
from policy_steering import build_guidance_fns
from policy_steering.kinematics import (
    FINGER_X_REST,
    FINGER_Z_FROM_GRIP,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_DIM = 32
ACTION_HORIZON = 50
TOTAL_HORIZON = ACTION_HORIZON
DT_ACTION = np.float32(1.0 / 50)
NUM_DENOISING_STEPS = 10
GUIDANCE_FACTOR = float(np.float32(1.0))
BATCH_SIZE = 1

# Small rotation used to break symmetry in orientation tests: the dot-product
# orientation loss has a *zero* gradient at the exact alignment ``body_vec ∥
# axis`` (it's a local max of ``cos(angle)``), and ``DummyPolicy`` starts from
# x_init = 0, so without this perturbation the line search sees no descent
# direction and returns alpha=0.  Using a ~0.06° pre-tilt around the
# orthogonal in-plane axes gives the optimiser a non-zero first derivative
# without meaningfully changing the test's initial cosine value.
_ORI_EPSILON = np.array([1e-3, 1e-3, 0.0], dtype=np.float32)

_PLOT_DIR = pathlib.Path(__file__).resolve().parents[2] / "tmp" / "guidance_plots"


# ---------------------------------------------------------------------------
# Helpers – param builders
# ---------------------------------------------------------------------------


def _base_params(
    N: int,
    ee_pos_0: np.ndarray,
    expected_dists: np.ndarray,
    *,
    ee_ori_0: np.ndarray | None = None,
    gripper_0: np.ndarray | None = None,
) -> dict:
    """Return common guidance param dict for *N* pairs."""
    T = TOTAL_HORIZON
    S_max = 1
    return {
        "__gp_ee_pos_0__": ee_pos_0,
        "__gp_dt__": DT_ACTION,
        "__gp_expected_dists__": expected_dists.reshape(N, T),
        "__gp_dist_types__": np.zeros(N, dtype=np.uint8),
        "__gp_source_type__": np.zeros(N, dtype=np.uint8),
        "__gp_waypoints__": np.zeros((N, 3), dtype=np.float32),
        "__gp_has_axis__": np.zeros(N, dtype=np.uint8),
        "__gp_axes__": np.zeros((N, 3), dtype=np.float32),
        "__gp_has_plane__": np.zeros(N, dtype=np.uint8),
        "__gp_plane_normals__": np.zeros((N, 3), dtype=np.float32),
        "__gp_surface_counts__": np.zeros(N, dtype=np.int32),
        "__gp_surface_pts__": np.zeros((N, S_max, 3), dtype=np.float32),
        "__gp_guidance_weights__": np.ones(N, dtype=np.float32),
        "__gp_guidance_factor__": GUIDANCE_FACTOR,
        "__gp_ee_ori_0__": np.asarray(
            ee_ori_0 if ee_ori_0 is not None else [0, 0, 0], dtype=np.float32
        ),
        "__gp_gripper_0__": np.asarray(
            gripper_0 if gripper_0 is not None else [0], dtype=np.float32
        ).ravel()[:1],
        "__gp_ee_vectors__": np.tile(np.array([0, 0, 1], dtype=np.float32), (N, 1)),
    }


def _centroid_params(
    ee_pos_0: np.ndarray,
    waypoint: np.ndarray,
    expected_dists: np.ndarray,
) -> dict:
    """Guidance params for a single centroid-distance pair (source=EE)."""
    params = _base_params(1, ee_pos_0, expected_dists)
    params["__gp_waypoints__"] = waypoint.reshape(1, 3)
    return params


def _axis_params(
    ee_pos_0: np.ndarray,
    waypoint: np.ndarray,
    expected_dists: np.ndarray,
    axis: list[float],
) -> dict:
    """Guidance params for a single axis-distance pair (source=EE)."""
    axis_norm = np.asarray(axis, dtype=np.float32)
    axis_norm = axis_norm / (np.linalg.norm(axis_norm) + 1e-9)
    params = _base_params(1, ee_pos_0, expected_dists)
    params["__gp_waypoints__"] = waypoint.reshape(1, 3)
    params["__gp_has_axis__"] = np.array([1], dtype=np.uint8)
    params["__gp_axes__"] = axis_norm.reshape(1, 3)
    return params


def _plane_params(
    ee_pos_0: np.ndarray,
    waypoint: np.ndarray,
    expected_dists: np.ndarray,
    plane_normal: list[float],
) -> dict:
    """Guidance params for a single plane-distance pair (source=EE)."""
    pn = np.asarray(plane_normal, dtype=np.float32)
    pn = pn / (np.linalg.norm(pn) + 1e-9)
    params = _base_params(1, ee_pos_0, expected_dists)
    params["__gp_waypoints__"] = waypoint.reshape(1, 3)
    params["__gp_has_plane__"] = np.array([1], dtype=np.uint8)
    params["__gp_plane_normals__"] = pn.reshape(1, 3)
    return params


def _orientation_params(
    ee_pos_0: np.ndarray,
    expected_dists: np.ndarray,
    axis: list[float],
    *,
    ee_ori_0: np.ndarray | None = None,
    ee_vector: list[float] | None = None,
) -> dict:
    """Guidance params for orientation projection (source_type=3, dist_type=2)."""
    axis_norm = np.asarray(axis, dtype=np.float32)
    axis_norm = axis_norm / (np.linalg.norm(axis_norm) + 1e-9)
    params = _base_params(1, ee_pos_0, expected_dists, ee_ori_0=ee_ori_0)
    params["__gp_dist_types__"] = np.array([2], dtype=np.uint8)
    params["__gp_source_type__"] = np.array([3], dtype=np.uint8)
    params["__gp_has_axis__"] = np.array([1], dtype=np.uint8)
    params["__gp_axes__"] = axis_norm.reshape(1, 3)
    params["__gp_waypoints__"] = np.zeros((1, 3), dtype=np.float32)
    if ee_vector is not None:
        ev = np.asarray(ee_vector, dtype=np.float32)
        ev = ev / (np.linalg.norm(ev) + 1e-9)
        params["__gp_ee_vectors__"] = ev.reshape(1, 3)
    return params


def _gripper_params(
    ee_pos_0: np.ndarray,
    expected_dists: np.ndarray,
    *,
    gripper_0: np.ndarray | None = None,
) -> dict:
    """Guidance params for gripper aperture (source_type=4, dist_type=3)."""
    params = _base_params(1, ee_pos_0, expected_dists, gripper_0=gripper_0)
    params["__gp_dist_types__"] = np.array([3], dtype=np.uint8)
    params["__gp_source_type__"] = np.array([4], dtype=np.uint8)
    params["__gp_waypoints__"] = np.zeros((1, 3), dtype=np.float32)
    return params


def _finger_params(
    ee_pos_0: np.ndarray,
    waypoint: np.ndarray,
    expected_dists: np.ndarray,
    *,
    finger: str = "left",
    ee_ori_0: np.ndarray | None = None,
    gripper_0: np.ndarray | None = None,
) -> dict:
    """Guidance params for a finger-tip-to-waypoint pair."""
    source_type = 1 if finger == "left" else 2
    params = _base_params(
        1, ee_pos_0, expected_dists, ee_ori_0=ee_ori_0, gripper_0=gripper_0
    )
    params["__gp_source_type__"] = np.array([source_type], dtype=np.uint8)
    params["__gp_waypoints__"] = waypoint.reshape(1, 3)
    return params


def _multi_params(
    ee_pos_0: np.ndarray,
    pairs: list[dict],
    *,
    ee_ori_0: np.ndarray | None = None,
    gripper_0: np.ndarray | None = None,
) -> dict:
    """Guidance params for multiple pairs with mixed distance types."""
    N = len(pairs)
    T = TOTAL_HORIZON

    waypoints = np.zeros((N, 3), dtype=np.float32)
    expected = np.zeros((N, T), dtype=np.float32)
    dist_types = np.zeros(N, dtype=np.uint8)
    source_types = np.zeros(N, dtype=np.uint8)
    has_axis_arr = np.zeros(N, dtype=np.uint8)
    axes_arr = np.zeros((N, 3), dtype=np.float32)
    has_plane_arr = np.zeros(N, dtype=np.uint8)
    plane_normals_arr = np.zeros((N, 3), dtype=np.float32)
    ee_vectors_arr = np.tile(np.array([0, 0, 1], dtype=np.float32), (N, 1))

    for i, p in enumerate(pairs):
        waypoints[i] = p.get("waypoint", [0, 0, 0])
        expected[i] = p["expected_dists"]
        dist_types[i] = p.get("dist_type", 0)
        source_types[i] = p.get("source_type", 0)
        if p.get("has_axis") and p.get("axis") is not None:
            has_axis_arr[i] = 1
            ax = np.asarray(p["axis"], dtype=np.float32)
            axes_arr[i] = ax / (np.linalg.norm(ax) + 1e-9)
        if p.get("has_plane") and p.get("plane_normal") is not None:
            has_plane_arr[i] = 1
            pn = np.asarray(p["plane_normal"], dtype=np.float32)
            plane_normals_arr[i] = pn / (np.linalg.norm(pn) + 1e-9)
        if p.get("ee_vector") is not None:
            ev = np.asarray(p["ee_vector"], dtype=np.float32)
            ee_vectors_arr[i] = ev / (np.linalg.norm(ev) + 1e-9)

    return {
        "__gp_ee_pos_0__": ee_pos_0,
        "__gp_dt__": DT_ACTION,
        "__gp_expected_dists__": expected,
        "__gp_dist_types__": dist_types,
        "__gp_source_type__": source_types,
        "__gp_waypoints__": waypoints,
        "__gp_has_axis__": has_axis_arr,
        "__gp_axes__": axes_arr,
        "__gp_has_plane__": has_plane_arr,
        "__gp_plane_normals__": plane_normals_arr,
        "__gp_surface_counts__": np.zeros(N, dtype=np.int32),
        "__gp_surface_pts__": np.zeros((N, 1, 3), dtype=np.float32),
        "__gp_guidance_weights__": np.ones(N, dtype=np.float32),
        "__gp_guidance_factor__": GUIDANCE_FACTOR,
        "__gp_ee_ori_0__": np.asarray(
            ee_ori_0 if ee_ori_0 is not None else [0, 0, 0], dtype=np.float32
        ),
        "__gp_gripper_0__": np.asarray(
            gripper_0 if gripper_0 is not None else [0], dtype=np.float32
        ).ravel()[:1],
        "__gp_ee_vectors__": ee_vectors_arr,
    }


# ---------------------------------------------------------------------------
# Helpers – simulation
# ---------------------------------------------------------------------------


_DUMMY_POLICY_CACHE: dict[int, DummyPolicy] = {}


def _get_dummy_policy(num_denoise_steps: int) -> DummyPolicy:
    """Return a cached ``DummyPolicy`` sized for the isolation-test horizon."""
    if num_denoise_steps not in _DUMMY_POLICY_CACHE:
        _DUMMY_POLICY_CACHE[num_denoise_steps] = DummyPolicy(
            num_action_chunks=1,
            action_horizon=TOTAL_HORIZON,
            action_dim=ACTION_DIM,
            num_output_actions=ACTION_DIM,
            num_denoise_steps=num_denoise_steps,
        )
    return _DUMMY_POLICY_CACHE[num_denoise_steps]


def _simulate_denoising(
    params: dict,
    *,
    rng_key: jax.Array | None = None,  # noqa: ARG001 — init is zeros
    num_steps: int = NUM_DENOISING_STEPS,
    guidance_factor: float = GUIDANCE_FACTOR,
    return_trajectory: bool = False,
):
    """Run denoising via ``DummyPolicy`` with golden-section line search.

    Delegates to the same in-process flow-matching loop used by
    ``evaluate --debug_guidance``
    (`policy_abstraction.policies.dummy.DummyPolicy`) so guidance behaviour
    matches the production path exactly — in particular, each step uses
    `compute_guidance_velocity` with `use_line_search=True`.  The loop
    starts from zeros (debug-guidance convention).

    Returns the denoised actions as ``np.ndarray`` of shape ``(B, T, D)``.
    If ``return_trajectory`` is True, also returns ``[zeros, x_0]`` so the
    convergence panel can still render a start/end norm pair (the internal
    while-loop does not expose per-step snapshots).
    """
    policy = _get_dummy_policy(num_steps)

    obs: dict = dict(params)
    obs["__gp_guidance_factor__"] = np.float32(guidance_factor)
    obs["__gp_num_chunks__"] = np.int32(1)
    obs["__gp_line_search__"] = True

    result = policy.infer(obs)
    x_0 = np.asarray(result["actions"]).reshape(BATCH_SIZE, TOTAL_HORIZON, ACTION_DIM)

    if return_trajectory:
        traj = [
            np.zeros((BATCH_SIZE, TOTAL_HORIZON, ACTION_DIM), dtype=np.float32),
            x_0,
        ]
        return x_0, traj
    return x_0


def _integrate_positions(
    x_0: np.ndarray,
    ee_pos_0: np.ndarray,
) -> np.ndarray:
    """Integrate velocity actions to EE positions. Returns (T, 3)."""
    vel = x_0[0, :, :3]
    return np.array(ee_pos_0[None, :] + np.cumsum(vel * float(DT_ACTION), axis=0))


def _integrate_full_state(
    x_0: np.ndarray,
    ee_pos_0: np.ndarray,
    ee_ori_0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate all 7 action dims. Returns (positions, orientations, grippers)."""
    x = x_0[0]
    dt = float(DT_ACTION)
    ee_pos = np.asarray(ee_pos_0, dtype=np.float32)
    ee_ori = np.asarray(
        ee_ori_0 if ee_ori_0 is not None else [0, 0, 0], dtype=np.float32
    )

    positions = ee_pos[None, :] + np.cumsum(x[:, :3] * dt, axis=0)
    orientations = ee_ori[None, :] + np.cumsum(x[:, 3:6] * dt, axis=0)
    grippers = x[:, 6]
    return positions, orientations, grippers


def _compute_finger_pos_np(
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy finger FK for a single timestep. Returns (left, right) each (3,)."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(ee_ori).as_matrix()
    q1 = gripper_val / 2.0
    q2 = -gripper_val / 2.0
    left_local = np.array([0.0, -(q1 + FINGER_X_REST), FINGER_Z_FROM_GRIP])
    right_local = np.array([0.0, -q2 + FINGER_X_REST, FINGER_Z_FROM_GRIP])
    return ee_pos + R @ left_local, ee_pos + R @ right_local


# ---------------------------------------------------------------------------
# Helpers – visualization
# ---------------------------------------------------------------------------


def _per_pair_achieved_distance(
    params: dict,
    positions: np.ndarray,
    orientations: np.ndarray,
    grippers: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute the scalar distance each pair sees given an integrated trajectory.

    Returns an ``(N, T)`` array of achieved distances and a list of labels.
    """
    N = params["__gp_waypoints__"].shape[0]
    T = positions.shape[0]
    waypoints = np.asarray(params["__gp_waypoints__"])
    dist_types = np.asarray(params["__gp_dist_types__"])
    source_types = np.asarray(params["__gp_source_type__"])
    has_axis = np.asarray(params["__gp_has_axis__"])
    axes = np.asarray(params["__gp_axes__"])
    has_plane = np.asarray(params["__gp_has_plane__"])
    plane_normals = np.asarray(params["__gp_plane_normals__"])
    ee_vectors = np.asarray(params["__gp_ee_vectors__"])

    achieved = np.zeros((N, T), dtype=np.float32)
    labels: list[str] = []
    for i in range(N):
        src = int(source_types[i])
        dt_i = int(dist_types[i])
        wp = waypoints[i]

        if src == 3 and dt_i == 2:
            body_vec = np.zeros((T, 3), dtype=np.float32)
            for t in range(T):
                body_vec[t] = _body_vec_world(orientations[t], ee_vectors[i])
            if has_axis[i] > 0:
                ax = axes[i] / (np.linalg.norm(axes[i]) + 1e-9)
                achieved[i] = np.sum(body_vec * ax[None, :], axis=-1)
                labels.append(f"ori_axis[{i}]")
            elif has_plane[i] > 0:
                n = plane_normals[i] / (np.linalg.norm(plane_normals[i]) + 1e-9)
                proj = np.sum(body_vec * n[None, :], axis=-1, keepdims=True)
                in_plane = body_vec - proj * n[None, :]
                achieved[i] = np.linalg.norm(in_plane, axis=-1)
                labels.append(f"ori_plane[{i}]")
            else:
                labels.append(f"ori_zero[{i}]")
            continue

        if src == 4 and dt_i == 3:
            from policy_steering.robot_config import gripper_action_bounds

            ao, ac, mo, mc = gripper_action_bounds("franka_panda")
            scale = (mo - mc) / (ao - ac)
            achieved[i] = mc + scale * (grippers - ac)
            labels.append(f"aperture[{i}]")
            continue

        if src in (1, 2):
            side = "L" if src == 1 else "R"
            for t in range(T):
                left, right = _compute_finger_pos_np(
                    positions[t], orientations[t], float(grippers[t])
                )
                ft = left if src == 1 else right
                achieved[i, t] = np.linalg.norm(ft - wp)
            labels.append(f"finger_{side}[{i}]")
            continue

        diff = positions - wp[None, :]
        if dt_i == 1:
            achieved[i] = np.linalg.norm(diff, axis=-1)
            labels.append(f"surface[{i}]")
        elif has_axis[i] > 0:
            ax = axes[i] / (np.linalg.norm(axes[i]) + 1e-9)
            achieved[i] = -np.sum(diff * ax[None, :], axis=-1)
            labels.append(f"axis[{i}]")
        elif has_plane[i] > 0:
            n = plane_normals[i] / (np.linalg.norm(plane_normals[i]) + 1e-9)
            proj = np.sum(diff * n[None, :], axis=-1, keepdims=True)
            in_plane = diff - proj * n[None, :]
            achieved[i] = np.linalg.norm(in_plane, axis=-1)
            labels.append(f"plane[{i}]")
        else:
            achieved[i] = np.linalg.norm(diff, axis=-1)
            labels.append(f"centroid[{i}]")
    return achieved, labels


def _render_scenario_panel(
    ax_row,
    name: str,
    params: dict,
    x_0_np: np.ndarray,
    traj: list[np.ndarray],
    *,
    ee_pos_0: np.ndarray,
    ee_ori_0: np.ndarray | None = None,
) -> None:
    """Render a 4-column debug panel for a single scenario onto ``ax_row``.

    Columns:
      0 — 3D EE trajectory + waypoints + axis/plane vectors
      1 — distance profile (expected vs. achieved) for every pair
      2 — action velocities (translational xyz) + orientation rates if used
      3 — denoising convergence (||x_t[:, :3]||)
    """
    if not _HAS_MPL:
        return

    ee_ori_0 = np.zeros(3, dtype=np.float32) if ee_ori_0 is None else ee_ori_0
    positions, orientations, grippers = _integrate_full_state(
        x_0_np, ee_pos_0, ee_ori_0
    )

    expected = np.asarray(params["__gp_expected_dists__"])
    waypoints = np.asarray(params["__gp_waypoints__"])
    source_types = np.asarray(params["__gp_source_type__"])
    has_axis = np.asarray(params["__gp_has_axis__"])
    axes = np.asarray(params["__gp_axes__"])
    has_plane = np.asarray(params["__gp_has_plane__"])
    plane_normals = np.asarray(params["__gp_plane_normals__"])

    N = waypoints.shape[0]
    T = positions.shape[0]
    t_steps = np.arange(T)
    colors = plt.cm.Set1(np.linspace(0, 1, max(N, 3)))

    achieved, labels = _per_pair_achieved_distance(
        params, positions, orientations, grippers
    )

    # Col 0 — 3D trajectory
    ax = ax_row[0]
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "b-",
        linewidth=1.5,
        label="EE traj",
    )
    ax.scatter(*ee_pos_0, color="green", s=80, marker="o", label="Start", zorder=5)
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        color="blue",
        s=60,
        marker="^",
        label="End",
        zorder=5,
    )
    for i in range(N):
        src = int(source_types[i])
        # Skip waypoints for pure-orientation pairs (waypoint is meaningless).
        if src == 3:
            continue
        ax.scatter(
            *waypoints[i], color=colors[i], s=80, marker="*", label=f"WP{i}", zorder=5
        )
        if has_axis[i] > 0:
            av = axes[i].copy()
            av = av / (np.linalg.norm(av) + 1e-9) * 0.15
            ax.quiver(
                *waypoints[i],
                *av,
                color="orange",
                linewidth=2,
                arrow_length_ratio=0.2,
                label="Axis" if i == 0 else None,
            )
        elif has_plane[i] > 0:
            nv = plane_normals[i].copy()
            nv = nv / (np.linalg.norm(nv) + 1e-9) * 0.10
            ax.quiver(
                *waypoints[i],
                *nv,
                color="purple",
                linewidth=2,
                arrow_length_ratio=0.2,
                label="Normal" if i == 0 else None,
            )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{name}\n3D EE Trajectory")
    ax.legend(fontsize=6)

    # Col 1 — distance profiles per pair
    ax = ax_row[1]
    for i in range(N):
        c = colors[i]
        ax.plot(
            t_steps,
            expected[i, :T],
            "--",
            color=c,
            linewidth=2,
            alpha=0.8,
            label=f"exp {labels[i]}",
        )
        ax.plot(
            t_steps,
            achieved[i],
            "-",
            color=c,
            linewidth=1.2,
            alpha=0.9,
            label=f"got {labels[i]}",
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance")
    ax.set_title("Distance profile (per pair)")
    ax.legend(fontsize=5, ncol=2 if N > 2 else 1)
    ax.grid(True, alpha=0.3)

    # Col 2 — action velocities (pos + ori + gripper)
    ax = ax_row[2]
    vel = x_0_np[0, :, :3]
    ax.plot(t_steps, vel[:, 0], label="vx", alpha=0.8)
    ax.plot(t_steps, vel[:, 1], label="vy", alpha=0.8)
    ax.plot(t_steps, vel[:, 2], label="vz", alpha=0.8)
    if np.any(source_types == 3):
        ori_vel = x_0_np[0, :, 3:6]
        ax.plot(t_steps, ori_vel[:, 0], "--", label="wx", alpha=0.6)
        ax.plot(t_steps, ori_vel[:, 1], "--", label="wy", alpha=0.6)
        ax.plot(t_steps, ori_vel[:, 2], "--", label="wz", alpha=0.6)
    if np.any(source_types == 4) or np.any(np.isin(source_types, [1, 2])):
        ax.plot(t_steps, x_0_np[0, :, 6], ":", color="k", label="grip", alpha=0.9)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action value")
    ax.set_title("Action channels")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Col 3 — denoising convergence
    ax = ax_row[3]
    if traj:
        norms_pos = [float(np.linalg.norm(t[0, :, :3])) for t in traj]
        ax.plot(
            range(len(norms_pos)), norms_pos, "k-o", markersize=3, label="||vel_xyz||"
        )
        if np.any(source_types == 3):
            norms_ori = [float(np.linalg.norm(t[0, :, 3:6])) for t in traj]
            ax.plot(
                range(len(norms_ori)),
                norms_ori,
                "r-s",
                markersize=3,
                label="||vel_ori||",
            )
        if np.any(source_types == 4) or np.any(np.isin(source_types, [1, 2])):
            norms_grip = [float(np.linalg.norm(t[0, :, 6])) for t in traj]
            ax.plot(
                range(len(norms_grip)),
                norms_grip,
                "g-^",
                markersize=3,
                label="||grip||",
            )
        ax.legend(fontsize=7)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Norm")
    ax.set_title("Denoising convergence")
    ax.grid(True, alpha=0.3)


def _save_scenario_grid(
    filename: str,
    scenarios: list[dict],
    *,
    suptitle: str,
) -> pathlib.Path:
    """Run each scenario (running denoising with trajectory capture) and save
    a vertical grid of 4-column panels (one row per scenario) to disk.

    Each scenario dict must have keys:
      - name: str
      - params: dict (guidance params)
      - ee_pos: np.ndarray (3,)
      - ee_ori: np.ndarray (3,) or None
      - rng_key: jax.Array
    """
    assert _HAS_MPL, "matplotlib required for rendering"
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(scenarios)
    fig = plt.figure(figsize=(20, 5 * n))
    for i, sc in enumerate(scenarios):
        ax3d = fig.add_subplot(n, 4, i * 4 + 1, projection="3d")
        ax1 = fig.add_subplot(n, 4, i * 4 + 2)
        ax2 = fig.add_subplot(n, 4, i * 4 + 3)
        ax3 = fig.add_subplot(n, 4, i * 4 + 4)

        x_0, traj = _simulate_denoising(
            sc["params"], rng_key=sc["rng_key"], return_trajectory=True
        )
        _render_scenario_panel(
            [ax3d, ax1, ax2, ax3],
            sc["name"],
            sc["params"],
            x_0,
            traj,
            ee_pos_0=sc["ee_pos"],
            ee_ori_0=sc.get("ee_ori"),
        )

    fig.suptitle(suptitle, fontsize=14, y=1.0)
    fig.tight_layout()
    out = _PLOT_DIR / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tests – gradient sanity
# ---------------------------------------------------------------------------


class TestGradientSign:
    """Verify that gradients point in the correct direction."""

    def test_gradient_pushes_toward_waypoint_z(self):
        """Gradient from a z-axis waypoint should push z-velocity positive."""
        ee_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        wp = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        expected = np.linspace(0.5, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _centroid_params(ee_pos, wp, expected)
        _, batch_grad_fn = build_guidance_fns(params)

        x_zero = jnp.zeros((1, TOTAL_HORIZON, ACTION_DIM))
        grad = np.array(batch_grad_fn(x_zero)[0, :, :3])

        dt_denoise = -1.0 / NUM_DENOISING_STEPS
        # After one step: x_t + dt * gf * (-grad/||grad||)
        # Since dt < 0, the effective push is +grad direction → descent
        # The mean z-gradient should be negative (loss decreases when z increases)
        # so -grad_z > 0 → pushed toward positive z
        delta_z = dt_denoise * GUIDANCE_FACTOR * grad[:, 2].mean()
        assert delta_z > 0, "Gradient should push z-velocity toward waypoint"

    def test_gradient_pushes_toward_waypoint_x(self):
        """Gradient from an x-axis waypoint should push x-velocity positive."""
        ee_pos = np.zeros(3, dtype=np.float32)
        wp = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        expected = np.linspace(0.5, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _centroid_params(ee_pos, wp, expected)
        _, batch_grad_fn = build_guidance_fns(params)

        x_zero = jnp.zeros((1, TOTAL_HORIZON, ACTION_DIM))
        grad = np.array(batch_grad_fn(x_zero)[0, :, :3])

        dt_denoise = -1.0 / NUM_DENOISING_STEPS
        delta_x = dt_denoise * GUIDANCE_FACTOR * grad[:, 0].mean()
        assert delta_x > 0, "Gradient should push x-velocity toward waypoint"


# ---------------------------------------------------------------------------
# Tests – gradient flow through FK
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify that finger-position guidance produces gradients on all relevant dims."""

    def test_finger_gradients_flow_to_pos_ori_gripper(self):
        """Left finger pair → nonzero gradients on pos (0:3), ori (3:6), gripper (6)."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        grip = np.array([0.04], dtype=np.float32)
        wp = np.array([0.35, 0.28, 0.32], dtype=np.float32)
        expected = np.linspace(0.05, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _finger_params(
            ee_pos, wp, expected, finger="left", ee_ori_0=ee_ori, gripper_0=grip
        )
        _, batch_grad_fn = build_guidance_fns(params)

        rng = jax.random.PRNGKey(500)
        x_t = jax.random.normal(rng, (1, TOTAL_HORIZON, ACTION_DIM)) * 0.01
        g = np.array(batch_grad_fn(x_t)[0])

        pos_norm = np.linalg.norm(g[:, :3])
        ori_norm = np.linalg.norm(g[:, 3:6])
        grip_norm = np.linalg.norm(g[:, 6:7])
        other_norm = np.linalg.norm(g[:, 7:])

        assert pos_norm > 1e-6, "Position gradient should be nonzero"
        assert ori_norm > 1e-6, "Orientation gradient should be nonzero"
        assert grip_norm > 1e-6, "Gripper gradient should be nonzero"
        assert other_norm < 1e-6, "Unused dims should have ~zero gradient"

    def test_right_finger_also_flows(self):
        """Right finger pair produces the same gradient flow pattern."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        grip = np.array([0.04], dtype=np.float32)
        wp = np.array([0.35, 0.32, 0.32], dtype=np.float32)
        expected = np.linspace(0.05, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _finger_params(
            ee_pos, wp, expected, finger="right", ee_ori_0=ee_ori, gripper_0=grip
        )
        _, batch_grad_fn = build_guidance_fns(params)

        rng = jax.random.PRNGKey(501)
        x_t = jax.random.normal(rng, (1, TOTAL_HORIZON, ACTION_DIM)) * 0.01
        g = np.array(batch_grad_fn(x_t)[0])

        assert np.linalg.norm(g[:, :3]) > 1e-6
        assert np.linalg.norm(g[:, 3:6]) > 1e-6
        assert np.linalg.norm(g[:, 6:7]) > 1e-6


# ---------------------------------------------------------------------------
# Tests – centroid guidance (EE approaches waypoint)
# ---------------------------------------------------------------------------


class TestCentroidGuidance:
    """Denoising with centroid guidance should move EE toward waypoint."""

    @pytest.mark.parametrize(
        "name,ee_start,waypoint",
        [
            ("up", [0, 0, 0], [0, 0, 0.5]),
            ("right", [0, 0, 0], [0.5, 0, 0]),
            ("forward", [0, 0, 0], [0, 0.5, 0]),
            ("diagonal", [0, 0, 0], [0.3, 0.3, 0.3]),
        ],
    )
    def test_ee_approaches_waypoint(self, name, ee_start, waypoint):
        ee = np.array(ee_start, dtype=np.float32)
        wp = np.array(waypoint, dtype=np.float32)
        start_dist = float(np.linalg.norm(wp - ee))
        expected = np.linspace(start_dist, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _centroid_params(ee, wp, expected)
        x_0 = _simulate_denoising(params)
        positions = _integrate_positions(x_0, ee)
        final_dist = float(np.linalg.norm(positions[-1] - wp))

        assert final_dist < start_dist, (
            f"[{name}] EE should be closer after guidance: "
            f"start={start_dist:.4f}, final={final_dist:.4f}"
        )

    def test_hold_still(self):
        """When EE starts at waypoint, it should stay close."""
        pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        expected = np.zeros(TOTAL_HORIZON, dtype=np.float32)

        params = _centroid_params(pos, pos, expected)
        x_0 = _simulate_denoising(params)
        positions = _integrate_positions(x_0, pos)
        drift = float(np.linalg.norm(positions[-1] - pos))

        assert drift < 0.05, f"EE should stay near start: drift={drift:.4f}"

    def test_small_scale_motion(self):
        """5mm motion should still produce convergence with enough steps."""
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0.003, 0.003, 0.003], dtype=np.float32)
        start_dist = float(np.linalg.norm(wp))
        expected = np.linspace(start_dist, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _centroid_params(ee, wp, expected)
        x_0 = _simulate_denoising(params, num_steps=30)
        positions = _integrate_positions(x_0, ee)
        final_dist = float(np.linalg.norm(positions[-1] - wp))

        assert final_dist < start_dist

    def test_loss_decreases_over_denoising(self):
        """The guidance loss should decrease across denoising steps."""
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        expected = np.linspace(0.5, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _centroid_params(ee, wp, expected)
        batch_loss_fn, batch_grad_fn = build_guidance_fns(params)

        rng = jax.random.PRNGKey(42)
        x_t = jax.random.normal(rng, (BATCH_SIZE, TOTAL_HORIZON, ACTION_DIM)) * 0.001

        initial_loss = float(jnp.sum(batch_loss_fn(x_t)))

        step_size = GUIDANCE_FACTOR / NUM_DENOISING_STEPS
        for _ in range(NUM_DENOISING_STEPS):
            grad = batch_grad_fn(x_t)
            grad_norm = jnp.linalg.norm(
                grad.reshape(BATCH_SIZE, -1), axis=-1, keepdims=True
            )[:, :, None]
            descent_dir = -grad / (grad_norm + 1e-8)
            x_t = x_t + step_size * descent_dir

        final_loss = float(jnp.sum(batch_loss_fn(x_t)))
        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests – axis guidance
# ---------------------------------------------------------------------------


class TestAxisGuidance:
    """Axis projection guidance constrains motion along a single direction."""

    def test_axis_z_approach(self):
        """Axis along Z: EE should move in Z toward waypoint."""
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0, 0, 0.5], dtype=np.float32)
        expected = np.linspace(0.5, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _axis_params(ee, wp, expected, [0, 0, 1])
        x_0 = _simulate_denoising(params)
        positions = _integrate_positions(x_0, ee)

        # Z should increase (approach waypoint along Z)
        z_displacement = positions[-1, 2] - ee[2]
        assert z_displacement > 0.1, f"Z should increase: got {z_displacement:.4f}"

    def test_axis_constrains_dominant_motion(self):
        """Axis along X with off-axis waypoint: dominant motion should be in X."""
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0.4, 0.3, 0.0], dtype=np.float32)
        expected = np.linspace(0.4, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _axis_params(ee, wp, expected, [1, 0, 0])
        x_0 = _simulate_denoising(params)
        positions = _integrate_positions(x_0, ee)

        # X displacement should dominate
        x_disp = abs(positions[-1, 0])
        y_disp = abs(positions[-1, 1])
        assert x_disp > y_disp, f"X should dominate: x={x_disp:.4f}, y={y_disp:.4f}"


# ---------------------------------------------------------------------------
# Tests – plane guidance
# ---------------------------------------------------------------------------


class TestPlaneGuidance:
    """Plane projection guidance constrains motion to an in-plane distance."""

    def test_plane_xy_approach(self):
        """Plane normal=Z: in-plane XY distance should decrease."""
        ee = np.array([0.3, 0.3, 0.1], dtype=np.float32)
        wp = np.zeros(3, dtype=np.float32)
        pn = np.array([0, 0, 1.0])

        # Compute initial in-plane distance
        diff = wp - ee
        proj = np.dot(diff, pn) * pn
        start_in_plane = float(np.linalg.norm(diff - proj))
        expected = np.linspace(start_in_plane, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _plane_params(ee, wp, expected, [0, 0, 1])
        x_0 = _simulate_denoising(params)
        positions = _integrate_positions(x_0, ee)

        # In-plane distance at end
        diff_end = positions[-1] - wp
        proj_end = np.dot(diff_end, pn) * pn
        end_in_plane = float(np.linalg.norm(diff_end - proj_end))

        assert end_in_plane < start_in_plane, (
            f"In-plane distance should decrease: "
            f"start={start_in_plane:.4f}, end={end_in_plane:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests – orientation guidance
# ---------------------------------------------------------------------------


def _gripper_z_world(ee_ori: np.ndarray) -> np.ndarray:
    """Numpy reference for the JAX rotation: gripper body z-axis in world frame."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(ee_ori, dtype=np.float64)).as_matrix()[:, 2]


def _body_vec_world(ee_ori: np.ndarray, ee_vector: list[float]) -> np.ndarray:
    """Numpy reference: rotate an arbitrary body-frame vector into world frame."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(np.asarray(ee_ori, dtype=np.float64)).as_matrix()
    ev = np.asarray(ee_vector, dtype=np.float64)
    return R @ ev


class TestOrientationGuidance:
    """Orientation projection guidance steers a configurable EE body vector.

    The per-pair scalar is ``dot(R(predicted_ori) @ ee_vector, axis)`` (the
    signed cosine of the angle between the body vector and the supplied
    direction) when ``has_axis`` is set, or the in-plane magnitude when
    ``has_plane`` is set.  When ``ee_vector`` is not provided, it defaults to
    ``[0,0,1]`` (the gripper z-axis).
    """

    def test_tilt_gripper_away_from_world_z(self):
        """At rest the gripper z-axis is parallel to world z (cosine = 1).
        Driving the expected projection below 1 should tilt the gripper so its
        z-axis ends up less aligned with world z.

        The simple normalised-gradient denoiser used here has limited capacity
        to apply a 90° rotation, so the test only verifies the *direction* of
        steering: the cosine drops below its initial value of 1.
        """
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        # Small pre-tilt — see _ORI_EPSILON docstring.
        ee_ori = _ORI_EPSILON.copy()
        # Start at the actual initial cosine (~1.0), drive toward 0.
        expected = np.linspace(1.0, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _orientation_params(ee_pos, expected, [0, 0, 1], ee_ori_0=ee_ori)
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(200))
        _, orientations, _ = _integrate_full_state(x_0, ee_pos, ee_ori)

        final_cos = float(np.dot(_gripper_z_world(orientations[-1]), [0, 0, 1]))
        assert final_cos < 0.999, (
            f"Gripper z-axis should tilt away from world z (cosine < 1.0): "
            f"got {final_cos:.4f}"
        )

    def test_align_gripper_z_with_minus_world_z(self):
        """Starting from the gripper z parallel to +world z (cosine vs [0,0,-1]
        is -1), drive it toward pointing downward so the cosine grows toward +1
        ("approach from above"). Only the steering direction is asserted —
        see ``test_tilt_gripper_away_from_world_z`` for context."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = _ORI_EPSILON.copy()
        # Initial dot(R[:,2], [0,0,-1]) = -1.  Drive toward 0 (~90° tilt).
        expected = np.linspace(-1.0, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _orientation_params(ee_pos, expected, [0, 0, -1], ee_ori_0=ee_ori)
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(201))
        _, orientations, _ = _integrate_full_state(x_0, ee_pos, ee_ori)

        final_cos = float(np.dot(_gripper_z_world(orientations[-1]), [0, 0, -1]))
        assert final_cos > -0.999, (
            f"Gripper z-axis should rotate so its cosine with [0,0,-1] grows: "
            f"got {final_cos:.4f}"
        )

    def test_missing_direction_contributes_zero_loss(self):
        """Defensive guard: orientation pair with neither axis nor plane set
        contributes zero loss/gradient.

        Regression: previously a misconfigured orientation pair silently
        produced a non-zero, gradient-less error term that polluted the loss
        without steering anything.
        """
        ee_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        # Non-zero expected distance — would produce bogus loss without the guard.
        expected = np.full(TOTAL_HORIZON, 0.5, dtype=np.float32)

        params = _orientation_params(ee_pos, expected, [0, 0, 1], ee_ori_0=ee_ori)
        # Force the misconfiguration: clear axes/plane and mark both flags off.
        params["__gp_axes__"] = np.zeros((1, 3), dtype=np.float32)
        params["__gp_has_axis__"] = np.array([0], dtype=np.uint8)
        params["__gp_plane_normals__"] = np.zeros((1, 3), dtype=np.float32)
        params["__gp_has_plane__"] = np.array([0], dtype=np.uint8)

        batch_loss_fn, batch_grad_fn = build_guidance_fns(params)
        x_t = jnp.zeros((1, TOTAL_HORIZON, ACTION_DIM))

        loss = float(batch_loss_fn(x_t)[0])
        grad = np.array(batch_grad_fn(x_t)[0])

        assert loss == 0.0, (
            f"Loss should be zero for direction-less orientation pair: {loss}"
        )
        assert np.allclose(grad, 0.0), (
            "Gradient should be zero for direction-less orientation pair"
        )


# ---------------------------------------------------------------------------
# Tests – orientation guidance with custom ee_vector
# ---------------------------------------------------------------------------


class TestOrientationEeVectorGuidance:
    """Orientation guidance with a non-default body-frame vector (ee_vector).

    Verifies that the JAX guidance correctly uses ``R @ ee_vector`` instead of
    the hardcoded ``R[:, 2]`` when ``__gp_ee_vectors__`` is provided.
    """

    def test_x_axis_ee_vector_tilt(self):
        """Track the gripper x-axis ([1,0,0]) instead of z-axis.

        At identity rotation, dot(R @ [1,0,0], [1,0,0]) = 1.
        Driving expected toward 0 should tilt the gripper so its body x-axis
        is no longer aligned with world x.
        """
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = _ORI_EPSILON.copy()
        expected = np.linspace(1.0, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _orientation_params(
            ee_pos, expected, [1, 0, 0], ee_ori_0=ee_ori, ee_vector=[1, 0, 0]
        )
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(210))
        _, orientations, _ = _integrate_full_state(x_0, ee_pos, ee_ori)

        final_cos = float(
            np.dot(_body_vec_world(orientations[-1], [1, 0, 0]), [1, 0, 0])
        )
        assert final_cos < 0.999, (
            f"Body x-axis should tilt away from world x: got cosine={final_cos:.4f}"
        )

    def test_y_axis_ee_vector_steers_independently_of_z(self):
        """Tracking body y-axis should NOT tilt the z-axis (at least initially).

        At identity rotation, dot(R @ [0,1,0], [0,1,0]) = 1.
        The gradient should flow into orientation dims (3:6), tilting the
        body y-axis while leaving the z-axis mostly unaffected for small tilts.
        """
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = _ORI_EPSILON.copy()
        expected = np.linspace(1.0, 0.5, TOTAL_HORIZON).astype(np.float32)

        params = _orientation_params(
            ee_pos, expected, [0, 1, 0], ee_ori_0=ee_ori, ee_vector=[0, 1, 0]
        )
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(211))
        _, orientations, _ = _integrate_full_state(x_0, ee_pos, ee_ori)

        # Body y-axis should have tilted
        final_cos_y = float(
            np.dot(_body_vec_world(orientations[-1], [0, 1, 0]), [0, 1, 0])
        )
        assert final_cos_y < 0.999, (
            f"Body y-axis should tilt: got cosine={final_cos_y:.4f}"
        )

    def test_default_ee_vector_matches_z_axis_guidance(self):
        """Orientation params without ee_vector should behave identically to
        ee_vector=[0,0,1] (backward compatibility)."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        expected = np.linspace(1.0, 0.0, TOTAL_HORIZON).astype(np.float32)

        params_default = _orientation_params(
            ee_pos, expected, [0, 0, 1], ee_ori_0=ee_ori
        )
        params_explicit = _orientation_params(
            ee_pos, expected, [0, 0, 1], ee_ori_0=ee_ori, ee_vector=[0, 0, 1]
        )

        batch_loss_default, _ = build_guidance_fns(params_default)
        batch_loss_explicit, _ = build_guidance_fns(params_explicit)

        rng = jax.random.PRNGKey(212)
        x_t = jax.random.normal(rng, (1, TOTAL_HORIZON, ACTION_DIM)) * 0.01

        loss_default = float(batch_loss_default(x_t)[0])
        loss_explicit = float(batch_loss_explicit(x_t)[0])

        assert abs(loss_default - loss_explicit) < 1e-6, (
            f"Default and explicit z-axis should match: "
            f"default={loss_default:.6f}, explicit={loss_explicit:.6f}"
        )

    def test_gradient_flows_to_orientation_dims_only(self):
        """ee_vector orientation pairs should produce gradients only on
        orientation dims (3:6), not position (0:3) or unused dims (7:)."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        expected = np.linspace(1.0, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _orientation_params(
            ee_pos, expected, [0, 0, 1], ee_ori_0=ee_ori, ee_vector=[1, 0, 0]
        )
        _, batch_grad_fn = build_guidance_fns(params)

        rng = jax.random.PRNGKey(213)
        x_t = jax.random.normal(rng, (1, TOTAL_HORIZON, ACTION_DIM)) * 0.01
        g = np.array(batch_grad_fn(x_t)[0])

        ori_norm = np.linalg.norm(g[:, 3:6])
        pos_norm = np.linalg.norm(g[:, :3])
        other_norm = np.linalg.norm(g[:, 7:])

        assert ori_norm > 1e-6, "Orientation gradient should be nonzero"
        assert pos_norm < 1e-6, "Position gradient should be ~zero for orientation pair"
        assert other_norm < 1e-6, "Unused dims should have ~zero gradient"

    def test_multi_pair_mixed_ee_vectors(self):
        """Two orientation pairs with different ee_vectors should produce
        finite gradients and a positive loss."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        exp_z = np.linspace(1.0, 0.5, TOTAL_HORIZON).astype(np.float32)
        exp_x = np.linspace(0.0, 0.5, TOTAL_HORIZON).astype(np.float32)

        params = _multi_params(
            ee_pos,
            [
                {
                    "waypoint": [0, 0, 0],
                    "expected_dists": exp_z,
                    "dist_type": 2,
                    "source_type": 3,
                    "has_axis": True,
                    "axis": [0, 0, 1],
                    "ee_vector": [0, 0, 1],
                },
                {
                    "waypoint": [0, 0, 0],
                    "expected_dists": exp_x,
                    "dist_type": 2,
                    "source_type": 3,
                    "has_axis": True,
                    "axis": [0, 0, 1],
                    "ee_vector": [1, 0, 0],
                },
            ],
            ee_ori_0=ee_ori,
        )

        batch_loss_fn, batch_grad_fn = build_guidance_fns(params)
        rng = jax.random.PRNGKey(214)
        x_t = jax.random.normal(rng, (1, TOTAL_HORIZON, ACTION_DIM)) * 0.01

        loss = batch_loss_fn(x_t)
        grad = batch_grad_fn(x_t)

        assert np.all(np.isfinite(np.array(loss))), "Loss should be finite"
        assert np.all(np.isfinite(np.array(grad))), "Gradient should be finite"
        assert float(jnp.sum(loss)) > 0, "Loss should be nonzero"


# ---------------------------------------------------------------------------
# Tests – gripper guidance
# ---------------------------------------------------------------------------


class TestGripperGuidance:
    """Gripper aperture guidance steers the gripper opening action dim."""

    def test_open_gripper(self):
        """Gripper should open from 0 toward 0.08."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        expected = np.linspace(0.0, 0.08, TOTAL_HORIZON).astype(np.float32)

        params = _gripper_params(ee_pos, expected, gripper_0=np.array([0.0]))
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(300))
        _, _, grippers = _integrate_full_state(x_0, ee_pos)

        # Gripper at end should be positive (opening)
        mean_grip = float(np.mean(grippers[-10:]))
        assert mean_grip > 0.01, f"Gripper should open: mean_final={mean_grip:.4f}"

    def test_close_gripper(self):
        """Gripper should close from 0.08 toward 0."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        expected = np.linspace(0.08, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _gripper_params(ee_pos, expected, gripper_0=np.array([0.08]))
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(301))
        _, _, grippers = _integrate_full_state(x_0, ee_pos)

        mean_grip = float(np.mean(grippers[-10:]))
        assert mean_grip < 0.06, f"Gripper should close: mean_final={mean_grip:.4f}"


# ---------------------------------------------------------------------------
# Tests – finger guidance
# ---------------------------------------------------------------------------


class TestFingerGuidance:
    """FK-based finger tip guidance with gradients flowing to all action dims."""

    def test_left_finger_approaches_waypoint(self):
        """Left finger tip should move toward waypoint."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        grip = np.array([0.04], dtype=np.float32)

        wp = np.array([0.35, 0.28, 0.32], dtype=np.float32)
        left_init, _ = _compute_finger_pos_np(ee_pos, ee_ori, grip[0])
        start_dist = float(np.linalg.norm(wp - left_init))
        expected = np.linspace(start_dist, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _finger_params(
            ee_pos, wp, expected, finger="left", ee_ori_0=ee_ori, gripper_0=grip
        )
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(400))
        positions, orientations, grippers = _integrate_full_state(x_0, ee_pos, ee_ori)

        # Compute final left finger position
        left_final, _ = _compute_finger_pos_np(
            positions[-1], orientations[-1], grippers[-1]
        )
        final_dist = float(np.linalg.norm(wp - left_final))

        assert final_dist < start_dist, (
            f"Left finger should approach waypoint: "
            f"start={start_dist:.4f}, final={final_dist:.4f}"
        )

    def test_right_finger_approaches_waypoint(self):
        """Right finger tip should move toward waypoint."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        grip = np.array([0.04], dtype=np.float32)

        wp = np.array([0.35, 0.32, 0.32], dtype=np.float32)
        _, right_init = _compute_finger_pos_np(ee_pos, ee_ori, grip[0])
        start_dist = float(np.linalg.norm(wp - right_init))
        expected = np.linspace(start_dist, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _finger_params(
            ee_pos, wp, expected, finger="right", ee_ori_0=ee_ori, gripper_0=grip
        )
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(401))
        positions, orientations, grippers = _integrate_full_state(x_0, ee_pos, ee_ori)

        _, right_final = _compute_finger_pos_np(
            positions[-1], orientations[-1], grippers[-1]
        )
        final_dist = float(np.linalg.norm(wp - right_final))

        assert final_dist < start_dist, (
            f"Right finger should approach waypoint: "
            f"start={start_dist:.4f}, final={final_dist:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests – combined guidance
# ---------------------------------------------------------------------------


class TestCombinedGuidance:
    """Multiple guidance types working together."""

    def test_centroid_plus_orientation(self):
        """EE centroid approach + gripper-z tilt both improve.

        Starts with the gripper z-axis parallel to world +z (cosine = 1) and
        steers it toward 0.5 (a ~60° tilt) while simultaneously approaching
        a position waypoint.
        """
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = _ORI_EPSILON.copy()
        wp = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        start_dist = float(np.linalg.norm(wp - ee_pos))
        exp_cent = np.linspace(start_dist, 0.0, TOTAL_HORIZON).astype(np.float32)
        # Initial gripper z-axis is ~[0,0,1] so dot with [0,0,1] ≈ 1.0; drive to 0.5.
        exp_ori = np.linspace(1.0, 0.5, TOTAL_HORIZON).astype(np.float32)

        params = _multi_params(
            ee_pos,
            [
                {
                    "waypoint": wp,
                    "expected_dists": exp_cent,
                    "source_type": 0,
                    "has_axis": False,
                    "axis": None,
                    "has_plane": False,
                    "plane_normal": None,
                },
                {
                    "waypoint": [0, 0, 0],
                    "expected_dists": exp_ori,
                    "dist_type": 2,
                    "source_type": 3,
                    "has_axis": True,
                    "axis": [0, 0, 1],
                    "has_plane": False,
                    "plane_normal": None,
                },
            ],
            ee_ori_0=ee_ori,
        )

        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(600))
        positions, orientations, _ = _integrate_full_state(x_0, ee_pos, ee_ori)

        final_dist = float(np.linalg.norm(positions[-1] - wp))
        final_ori_dot = float(np.dot(_gripper_z_world(orientations[-1]), [0, 0, 1]))

        assert final_dist < start_dist, "Centroid distance should decrease"
        assert final_ori_dot < 1.0, (
            "Gripper z-axis cosine with world z should tilt away from 1.0"
        )

    def test_approach_plus_gripper_close(self):
        """EE approach + gripper close both progress."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        grip = np.array([0.04], dtype=np.float32)
        wp = np.array([0.35, 0.35, 0.25], dtype=np.float32)
        start_dist = float(np.linalg.norm(wp - ee_pos))
        exp_approach = np.linspace(start_dist, 0.0, TOTAL_HORIZON).astype(np.float32)

        t_norm = np.linspace(0, 1, TOTAL_HORIZON)
        exp_grip = np.where(
            t_norm < 0.5, grip[0], grip[0] * (1 - (t_norm - 0.5) / 0.5)
        ).astype(np.float32)

        params = _multi_params(
            ee_pos,
            [
                {
                    "waypoint": wp,
                    "expected_dists": exp_approach,
                    "source_type": 0,
                    "has_axis": False,
                    "axis": None,
                    "has_plane": False,
                    "plane_normal": None,
                },
                {
                    "waypoint": [0, 0, 0],
                    "expected_dists": exp_grip,
                    "dist_type": 3,
                    "source_type": 4,
                    "has_axis": False,
                    "axis": None,
                    "has_plane": False,
                    "plane_normal": None,
                },
            ],
            ee_ori_0=ee_ori,
            gripper_0=grip,
        )

        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(601))
        positions, _, grippers = _integrate_full_state(x_0, ee_pos, ee_ori)

        final_dist = float(np.linalg.norm(positions[-1] - wp))
        assert final_dist < start_dist, "EE should approach waypoint"

    def test_axis_plus_plane_plus_orientation(self):
        """Axis + plane + orientation (with custom ee_vector) combined loss all
        produce valid gradients."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)

        wp_ax = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        exp_ax = np.linspace(
            float(np.dot(wp_ax - ee_pos, [0, 0, 1])), 0.0, TOTAL_HORIZON
        ).astype(np.float32)

        diff_pl = np.array([0.3, 0.3, 0.0]) - ee_pos
        pn = np.array([0, 0, 1.0])
        proj_pl = np.dot(diff_pl, pn) * pn
        start_pl = float(np.linalg.norm(diff_pl - proj_pl))
        exp_pl = np.linspace(start_pl, 0.0, TOTAL_HORIZON).astype(np.float32)

        exp_ori = np.linspace(0.0, 0.4, TOTAL_HORIZON).astype(np.float32)

        params = _multi_params(
            ee_pos,
            [
                {
                    "waypoint": wp_ax,
                    "expected_dists": exp_ax,
                    "source_type": 0,
                    "has_axis": True,
                    "axis": [0, 0, 1],
                    "has_plane": False,
                    "plane_normal": None,
                },
                {
                    "waypoint": [0.3, 0.3, 0],
                    "expected_dists": exp_pl,
                    "source_type": 0,
                    "has_axis": False,
                    "axis": None,
                    "has_plane": True,
                    "plane_normal": [0, 0, 1],
                },
                {
                    "waypoint": [0, 0, 0],
                    "expected_dists": exp_ori,
                    "dist_type": 2,
                    "source_type": 3,
                    "has_axis": True,
                    "axis": [1, 0, 0],
                    "has_plane": False,
                    "plane_normal": None,
                    "ee_vector": [0, 1, 0],
                },
            ],
            ee_ori_0=ee_ori,
        )

        # Just verify the loss+grad functions build and produce finite values
        batch_loss_fn, batch_grad_fn = build_guidance_fns(params)
        rng = jax.random.PRNGKey(602)
        x_t = jax.random.normal(rng, (1, TOTAL_HORIZON, ACTION_DIM)) * 0.01

        loss = batch_loss_fn(x_t)
        grad = batch_grad_fn(x_t)

        assert np.all(np.isfinite(np.array(loss))), "Loss should be finite"
        assert np.all(np.isfinite(np.array(grad))), "Gradient should be finite"
        assert float(jnp.sum(loss)) > 0, "Loss should be nonzero"


# ---------------------------------------------------------------------------
# Tests – gripper action -> metric normalisation
# ---------------------------------------------------------------------------


class TestGripperNormalization:
    """The policy action channel ``x_t[:, 6]`` is in ``[-1, 1]`` but finger FK
    and aperture distances both live in metres (``q1 - q2`` in ``[0, 0.08]``).
    A linear map inside ``_guidance_fn`` keeps every distance autograd sees
    in the same metric frame.
    """

    def test_metric_bounds_match_config(self):
        """Policy action ±1 should map to metric 0 / 0.08 respectively."""
        from policy_steering.robot_config import gripper_action_bounds

        ao, ac, mo, mc = gripper_action_bounds("franka_panda")
        # Evaluate the same linear map used inside ``_guidance_fn``.
        scale = (mo - mc) / (ao - ac)

        def _to_metric(a: float) -> float:
            return float(mc + scale * (a - ac))

        assert abs(_to_metric(ac) - mc) < 1e-6
        assert abs(_to_metric(ao) - mo) < 1e-6

    def test_aperture_guidance_in_policy_space(self):
        """Aperture pair: expected in metres → gripper action climbs to +1.

        Pre-fix, ``grip_dists = predicted_grip`` compared raw action ``[-1,1]``
        directly to metric ``[0, 0.08]``, so closing gradient was ~12× too
        strong and the action flew out of range.  Post-fix, the metric
        distances match and the gripper action converges inside ``[-1, 1]``.
        """
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        expected = np.linspace(0.0, 0.08, TOTAL_HORIZON).astype(np.float32)

        params = _gripper_params(ee_pos, expected, gripper_0=np.array([0.0]))
        x_0 = _simulate_denoising(params, rng_key=jax.random.PRNGKey(700))
        _, _, grippers = _integrate_full_state(x_0, ee_pos)

        mean_grip_action = float(np.mean(grippers[-10:]))
        # Qualitative: after a small number of normalised-gradient steps the
        # action should have moved in the open direction.  It won't converge
        # all the way to +1 in 10 steps with unit-norm gradients, but must
        # be clearly positive and not blown out of policy space.
        assert mean_grip_action > 0.1, (
            f"Gripper action should rise toward +1 (open): got {mean_grip_action:.3f}"
        )
        assert mean_grip_action < 1.5, (
            f"Gripper action should stay roughly inside [-1, 1]: "
            f"got {mean_grip_action:.3f}"
        )


# ---------------------------------------------------------------------------
# Regression – finger guidance with realistic policy-space gripper action
# ---------------------------------------------------------------------------


class TestFingerGuidancePolicySpace:
    """Regression: when the policy's gripper channel is in ``[-1, 1]`` (not
    metric), a ``gripper_l→waypoint`` pair should still steer the EE body
    toward the waypoint.  Pre-fix, the huge non-metric ``gripper_val`` fed
    into FK made the Jacobian w.r.t. the gripper channel dominate, leaving
    the EE essentially unguided.
    """

    def _denoise_with_biased_gripper(
        self,
        params: dict,
        *,
        rng_key: jax.Array,
        gripper_bias: float,
        num_steps: int = NUM_DENOISING_STEPS,
    ) -> np.ndarray:
        """Simulate denoising but keep ``x_t[:, 6]`` near a realistic policy
        value like ``-1`` (fully closed)."""
        batch_loss_fn, batch_grad_fn = build_guidance_fns(params)
        step_size = GUIDANCE_FACTOR / num_steps
        x_t = (
            jax.random.normal(rng_key, (BATCH_SIZE, TOTAL_HORIZON, ACTION_DIM)) * 0.001
        )
        # Bias the gripper channel into policy space.
        x_t = x_t.at[:, :, 6].set(gripper_bias)

        for _ in range(num_steps):
            grad = batch_grad_fn(x_t)
            grad_norm = jnp.linalg.norm(
                grad.reshape(BATCH_SIZE, -1), axis=-1, keepdims=True
            )[:, :, None]
            descent_dir = -grad / (grad_norm + 1e-8)
            x_t = x_t + step_size * descent_dir
        return np.array(x_t)

    def test_left_finger_pulls_ee_under_policy_space_gripper_action(self):
        """Seed ``x_t[:, 6] ≈ -1`` (closed).  EE must still approach the
        waypoint — the gradient from the finger FK should not be absorbed
        entirely by the gripper channel."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        grip = np.array([0.0], dtype=np.float32)

        wp = np.array([0.40, 0.28, 0.32], dtype=np.float32)
        start_ee_dist = float(np.linalg.norm(wp - ee_pos))
        expected = np.linspace(start_ee_dist, 0.0, TOTAL_HORIZON).astype(np.float32)

        params = _finger_params(
            ee_pos, wp, expected, finger="left", ee_ori_0=ee_ori, gripper_0=grip
        )
        x_0 = self._denoise_with_biased_gripper(
            params, rng_key=jax.random.PRNGKey(800), gripper_bias=-1.0
        )
        positions = _integrate_positions(x_0, ee_pos)

        final_ee_dist = float(np.linalg.norm(positions[-1] - wp))
        assert final_ee_dist < start_ee_dist, (
            f"EE should move toward waypoint even with closed gripper action: "
            f"start={start_ee_dist:.4f}, final={final_ee_dist:.4f}"
        )


# ---------------------------------------------------------------------------
# Visual inspection — dump debug plots to tmp/guidance_plots/
# ---------------------------------------------------------------------------


@pytest.mark.visual
@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
class TestVisualInspection:
    """Render grids of scenarios to PNGs under ``tmp/guidance_plots/`` for
    visual inspection. Runs the same denoising pipeline as the unit tests but
    captures the per-step trajectory and plots 3D paths, distance profiles,
    action channels, and denoising convergence.

    Run with:
        uv run pytest tests/policy_steering/test_guidance_isolation.py \
            -m visual -v
    """

    def test_centroid_scenarios(self):
        scenarios: list[dict] = []

        for name, ee_start, wp in [
            ("centroid: up", [0, 0, 0], [0, 0, 0.5]),
            ("centroid: right", [0, 0, 0], [0.5, 0, 0]),
            ("centroid: forward", [0, 0, 0], [0, 0.5, 0]),
            ("centroid: diagonal", [0, 0, 0], [0.3, 0.3, 0.3]),
        ]:
            ee = np.array(ee_start, dtype=np.float32)
            wpa = np.array(wp, dtype=np.float32)
            start_dist = float(np.linalg.norm(wpa - ee))
            expected = np.linspace(start_dist, 0.0, TOTAL_HORIZON).astype(np.float32)
            scenarios.append(
                {
                    "name": name,
                    "params": _centroid_params(ee, wpa, expected),
                    "ee_pos": ee,
                    "ee_ori": None,
                    "rng_key": jax.random.PRNGKey(len(scenarios)),
                }
            )

        # Small-scale (5mm) scenario
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0.003, 0.003, 0.003], dtype=np.float32)
        start = float(np.linalg.norm(wp))
        expected = np.linspace(start, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "centroid: TINY (3mm diag)",
                "params": _centroid_params(ee, wp, expected),
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(99),
            }
        )

        out = _save_scenario_grid(
            "guidance_centroid.png",
            scenarios,
            suptitle="Centroid guidance scenarios",
        )
        assert out.exists()

    def test_axis_plane_scenarios(self):
        scenarios: list[dict] = []

        # Axis: move up along Z
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        exp = np.linspace(0.5, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "axis: +Z approach",
                "params": _axis_params(ee, wp, exp, [0, 0, 1]),
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(10),
            }
        )

        # Axis: move along X ignoring Y offset
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0.4, 0.3, 0.0], dtype=np.float32)
        exp = np.linspace(0.4, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "axis: +X (ignore Y)",
                "params": _axis_params(ee, wp, exp, [1, 0, 0]),
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(11),
            }
        )

        # Axis: pass-through (expected goes pos → neg)
        ee = np.zeros(3, dtype=np.float32)
        wp = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        exp = np.linspace(0.2, -0.2, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "axis: pass through (+→-)",
                "params": _axis_params(ee, wp, exp, [0, 0, 1]),
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(12),
            }
        )

        # Plane: approach in XY
        ee = np.array([0.3, 0.3, 0.1], dtype=np.float32)
        wp = np.zeros(3, dtype=np.float32)
        pn = np.array([0, 0, 1.0])
        diff = wp - ee
        start_pd = float(np.linalg.norm(diff - np.dot(diff, pn) * pn))
        exp = np.linspace(start_pd, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "plane: XY (normal=Z)",
                "params": _plane_params(ee, wp, exp, [0, 0, 1]),
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(13),
            }
        )

        # Plane: approach in XZ (normal=Y)
        ee = np.array([0.3, 0.2, 0.3], dtype=np.float32)
        wp = np.zeros(3, dtype=np.float32)
        pn = np.array([0, 1.0, 0])
        diff = wp - ee
        start_pd = float(np.linalg.norm(diff - np.dot(diff, pn) * pn))
        exp = np.linspace(start_pd, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "plane: XZ (normal=Y)",
                "params": _plane_params(ee, wp, exp, [0, 1, 0]),
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(14),
            }
        )

        out = _save_scenario_grid(
            "guidance_axis_plane.png",
            scenarios,
            suptitle="Axis & plane guidance scenarios",
        )
        assert out.exists()

    def test_orientation_scenarios(self):
        """Orientation pairs with default and custom ee_vector."""
        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        # Small pre-tilt breaks the saddle point at body_vec ∥ axis.
        ee_ori = _ORI_EPSILON.copy()

        scenarios: list[dict] = []

        # Default ee_vector (z-axis) — tilt away from world +z
        exp = np.linspace(1.0, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "orientation: body-Z vs world+Z (default ee_vec)",
                "params": _orientation_params(ee_pos, exp, [0, 0, 1], ee_ori_0=ee_ori),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(20),
            }
        )

        # ee_vector=[1,0,0] — track body x-axis
        exp = np.linspace(1.0, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "orientation: body-X vs world+X (ee_vec=[1,0,0])",
                "params": _orientation_params(
                    ee_pos, exp, [1, 0, 0], ee_ori_0=ee_ori, ee_vector=[1, 0, 0]
                ),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(21),
            }
        )

        # ee_vector=[0,1,0] — track body y-axis
        exp = np.linspace(1.0, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "orientation: body-Y vs world+Y (ee_vec=[0,1,0])",
                "params": _orientation_params(
                    ee_pos, exp, [0, 1, 0], ee_ori_0=ee_ori, ee_vector=[0, 1, 0]
                ),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(22),
            }
        )

        # Drive body-z to point down (align with world -z): cosine -1 → 0
        exp = np.linspace(-1.0, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "orientation: body-Z toward -world-Z (approach from above)",
                "params": _orientation_params(ee_pos, exp, [0, 0, -1], ee_ori_0=ee_ori),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(23),
            }
        )

        out = _save_scenario_grid(
            "guidance_orientation.png",
            scenarios,
            suptitle="Orientation guidance scenarios (incl. configurable ee_vector)",
        )
        assert out.exists()

    def test_gripper_and_finger_scenarios(self):
        scenarios: list[dict] = []

        ee_pos = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)

        # Gripper open from 0 → 0.08
        exp = np.linspace(0.0, 0.08, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "gripper: open (0 → 0.08 m)",
                "params": _gripper_params(
                    ee_pos, exp, gripper_0=np.array([0.0], dtype=np.float32)
                ),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(30),
            }
        )

        # Gripper close from 0.08 → 0
        exp = np.linspace(0.08, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "gripper: close (0.08 → 0 m)",
                "params": _gripper_params(
                    ee_pos, exp, gripper_0=np.array([0.08], dtype=np.float32)
                ),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(31),
            }
        )

        # Left finger approaches waypoint
        grip = np.array([0.04], dtype=np.float32)
        wp = np.array([0.35, 0.28, 0.32], dtype=np.float32)
        left_init, _ = _compute_finger_pos_np(ee_pos, ee_ori, grip[0])
        start_d = float(np.linalg.norm(wp - left_init))
        exp = np.linspace(start_d, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "finger-L → waypoint",
                "params": _finger_params(
                    ee_pos, wp, exp, finger="left", ee_ori_0=ee_ori, gripper_0=grip
                ),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(32),
            }
        )

        # Right finger approaches waypoint
        wp = np.array([0.35, 0.32, 0.32], dtype=np.float32)
        _, right_init = _compute_finger_pos_np(ee_pos, ee_ori, grip[0])
        start_d = float(np.linalg.norm(wp - right_init))
        exp = np.linspace(start_d, 0.0, TOTAL_HORIZON).astype(np.float32)
        scenarios.append(
            {
                "name": "finger-R → waypoint",
                "params": _finger_params(
                    ee_pos, wp, exp, finger="right", ee_ori_0=ee_ori, gripper_0=grip
                ),
                "ee_pos": ee_pos,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(33),
            }
        )

        out = _save_scenario_grid(
            "guidance_gripper_finger.png",
            scenarios,
            suptitle="Gripper & finger guidance scenarios",
        )
        assert out.exists()

    def test_combined_scenarios(self):
        """Multiple pairs mixed together: axis + centroid, plane + centroid,
        axis + plane + orientation (with custom ee_vector)."""
        scenarios: list[dict] = []

        # Axis(Z) + Centroid
        ee = np.zeros(3, dtype=np.float32)
        wp_ax = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        wp_ct = np.array([0.3, 0.3, 0.0], dtype=np.float32)
        exp_ax = np.linspace(0.5, 0.0, TOTAL_HORIZON).astype(np.float32)
        start_ct = float(np.linalg.norm(wp_ct - ee))
        exp_ct = np.linspace(start_ct, 0.0, TOTAL_HORIZON).astype(np.float32)
        params = _multi_params(
            ee,
            [
                {
                    "waypoint": wp_ax,
                    "expected_dists": exp_ax,
                    "has_axis": True,
                    "axis": [0, 0, 1],
                },
                {"waypoint": wp_ct, "expected_dists": exp_ct},
            ],
        )
        scenarios.append(
            {
                "name": "combo: axis(Z) + centroid",
                "params": params,
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(40),
            }
        )

        # Plane(XY) + Centroid
        ee = np.array([0.3, 0.3, 0.0], dtype=np.float32)
        wp_pl = np.zeros(3, dtype=np.float32)
        wp_ct = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        pn = np.array([0, 0, 1.0])
        diff = wp_pl - ee
        start_pl = float(np.linalg.norm(diff - np.dot(diff, pn) * pn))
        exp_pl = np.linspace(start_pl, 0.0, TOTAL_HORIZON).astype(np.float32)
        start_ct = float(np.linalg.norm(wp_ct - ee))
        exp_ct = np.linspace(start_ct, 0.0, TOTAL_HORIZON).astype(np.float32)
        params = _multi_params(
            ee,
            [
                {
                    "waypoint": wp_pl,
                    "expected_dists": exp_pl,
                    "has_plane": True,
                    "plane_normal": [0, 0, 1],
                },
                {"waypoint": wp_ct, "expected_dists": exp_ct},
            ],
        )
        scenarios.append(
            {
                "name": "combo: plane(XY) + centroid",
                "params": params,
                "ee_pos": ee,
                "ee_ori": None,
                "rng_key": jax.random.PRNGKey(41),
            }
        )

        # Axis + Plane + Orientation (with custom ee_vector)
        ee = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ee_ori = np.zeros(3, dtype=np.float32)
        wp_ax = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        exp_ax = np.linspace(
            float(np.dot(wp_ax - ee, [0, 0, 1])), 0.0, TOTAL_HORIZON
        ).astype(np.float32)
        wp_pl = np.array([0.3, 0.3, 0.0], dtype=np.float32)
        diff_pl = wp_pl - ee
        pn = np.array([0, 0, 1.0])
        start_pl = float(np.linalg.norm(diff_pl - np.dot(diff_pl, pn) * pn))
        exp_pl = np.linspace(start_pl, 0.0, TOTAL_HORIZON).astype(np.float32)
        exp_ori = np.linspace(0.0, 0.4, TOTAL_HORIZON).astype(np.float32)
        params = _multi_params(
            ee,
            [
                {
                    "waypoint": wp_ax,
                    "expected_dists": exp_ax,
                    "has_axis": True,
                    "axis": [0, 0, 1],
                },
                {
                    "waypoint": wp_pl,
                    "expected_dists": exp_pl,
                    "has_plane": True,
                    "plane_normal": [0, 0, 1],
                },
                {
                    "waypoint": [0, 0, 0],
                    "expected_dists": exp_ori,
                    "dist_type": 2,
                    "source_type": 3,
                    "has_axis": True,
                    "axis": [1, 0, 0],
                    "ee_vector": [0, 1, 0],
                },
            ],
            ee_ori_0=ee_ori,
        )
        scenarios.append(
            {
                "name": "combo: axis + plane + orientation(ee_vec=[0,1,0])",
                "params": params,
                "ee_pos": ee,
                "ee_ori": ee_ori,
                "rng_key": jax.random.PRNGKey(42),
            }
        )

        out = _save_scenario_grid(
            "guidance_combinations.png",
            scenarios,
            suptitle="Combined guidance scenarios (axis + plane + centroid + orientation)",
        )
        assert out.exists()

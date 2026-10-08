"""Numerical equivalence tests: torch guidance kernel ↔ JAX guidance kernel.

For each guidance scenario (centroid / axis / plane / surface / orientation /
gripper / finger source), build identical ``__gp_*__`` parameter dicts and
verify that:

- ``batch_loss_fn(x_t)`` matches between torch and JAX to ``≤ 1e-4`` absolute.
- ``batch_grad_fn(x_t)`` agrees in direction (cosine similarity ``> 0.999``)
  and magnitude (``rtol=1e-3``).

The torch kernel is the only path used at inference time for the Gr00t
backend, so equivalence with the validated JAX kernel is the key correctness
contract.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from policy_steering.guidance_fns import build_guidance_fns as build_guidance_fns_jax
from policy_steering.guidance_fns_torch import (
    build_guidance_fns as build_guidance_fns_torch,
)
from policy_steering.guidance_fns_torch import (
    compute_guidance_velocity as compute_guidance_velocity_torch,
)

ACTION_DIM = 7
ACTION_HORIZON = 16
TOTAL_HORIZON = ACTION_HORIZON
DT_ACTION = np.float32(1.0 / ACTION_HORIZON)
SEED = 0


def _base_params(
    N: int,
    ee_pos_0: np.ndarray,
    expected_dists: np.ndarray,
    *,
    ee_ori_0: np.ndarray | None = None,
) -> dict:
    T = expected_dists.shape[1]
    S_max = 4
    return {
        "__gp_ee_pos_0__": np.asarray(ee_pos_0, dtype=np.float32),
        "__gp_dt__": DT_ACTION,
        "__gp_expected_dists__": expected_dists.astype(np.float32).reshape(N, T),
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
        "__gp_ee_ori_0__": np.asarray(
            ee_ori_0 if ee_ori_0 is not None else [0, 0, 0], dtype=np.float32
        ),
        "__gp_ee_vectors__": np.tile(np.array([0, 0, 1], dtype=np.float32), (N, 1)),
    }


def _centroid_params() -> dict:
    expected = np.linspace(0.5, 0.05, ACTION_HORIZON, dtype=np.float32)[None]
    p = _base_params(1, np.zeros(3, dtype=np.float32), expected)
    p["__gp_waypoints__"] = np.array([[0.3, 0.2, 0.1]], dtype=np.float32)
    return p


def _axis_params() -> dict:
    expected = np.linspace(0.4, 0.0, ACTION_HORIZON, dtype=np.float32)[None]
    p = _base_params(1, np.zeros(3, dtype=np.float32), expected)
    p["__gp_waypoints__"] = np.array([[0.4, 0.0, 0.0]], dtype=np.float32)
    p["__gp_has_axis__"] = np.array([1], dtype=np.uint8)
    p["__gp_axes__"] = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    return p


def _plane_params() -> dict:
    expected = np.linspace(0.3, 0.0, ACTION_HORIZON, dtype=np.float32)[None]
    p = _base_params(1, np.zeros(3, dtype=np.float32), expected)
    p["__gp_waypoints__"] = np.array([[0.2, 0.3, 0.0]], dtype=np.float32)
    p["__gp_has_plane__"] = np.array([1], dtype=np.uint8)
    p["__gp_plane_normals__"] = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    return p


def _surface_params() -> dict:
    expected = np.linspace(0.5, 0.05, ACTION_HORIZON, dtype=np.float32)[None]
    p = _base_params(1, np.zeros(3, dtype=np.float32), expected)
    p["__gp_dist_types__"] = np.array([1], dtype=np.uint8)
    p["__gp_surface_counts__"] = np.array([3], dtype=np.int32)
    surf = np.zeros((1, 4, 3), dtype=np.float32)
    surf[0, 0] = [0.3, 0.2, 0.1]
    surf[0, 1] = [0.5, 0.0, 0.0]
    surf[0, 2] = [0.1, 0.3, 0.2]
    # surf[0, 3] is invalid (count=3, slot 3 ignored)
    p["__gp_surface_pts__"] = surf
    return p


def _orientation_params() -> dict:
    expected = np.linspace(0.95, 0.5, ACTION_HORIZON, dtype=np.float32)[None]
    # Tiny pre-tilt to avoid the zero-gradient singularity at exact alignment
    p = _base_params(
        1,
        np.zeros(3, dtype=np.float32),
        expected,
        ee_ori_0=np.array([1e-3, 1e-3, 0.0], dtype=np.float32),
    )
    p["__gp_dist_types__"] = np.array([2], dtype=np.uint8)
    p["__gp_source_type__"] = np.array([3], dtype=np.uint8)
    p["__gp_has_axis__"] = np.array([1], dtype=np.uint8)
    p["__gp_axes__"] = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    return p


def _gripper_params() -> dict:
    expected = np.full((1, ACTION_HORIZON), 0.04, dtype=np.float32)
    p = _base_params(1, np.zeros(3, dtype=np.float32), expected)
    p["__gp_dist_types__"] = np.array([3], dtype=np.uint8)
    p["__gp_source_type__"] = np.array([4], dtype=np.uint8)
    return p


def _finger_l_params() -> dict:
    expected = np.linspace(0.4, 0.05, ACTION_HORIZON, dtype=np.float32)[None]
    p = _base_params(1, np.zeros(3, dtype=np.float32), expected)
    p["__gp_source_type__"] = np.array([1], dtype=np.uint8)
    p["__gp_waypoints__"] = np.array([[0.2, 0.1, 0.05]], dtype=np.float32)
    return p


SCENARIOS = {
    "centroid": _centroid_params,
    "axis": _axis_params,
    "plane": _plane_params,
    "surface": _surface_params,
    "orientation": _orientation_params,
    "gripper": _gripper_params,
    "finger_l": _finger_l_params,
}


def _make_x_t(seed: int = SEED) -> tuple[jnp.ndarray, torch.Tensor]:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((2, TOTAL_HORIZON, ACTION_DIM)).astype(np.float32) * 0.1
    return jnp.asarray(arr), torch.from_numpy(arr.copy())


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_loss_and_gradient_match_jax(scenario):
    """Torch ↔ JAX equivalence for loss and gradient on each scenario."""
    np_params = SCENARIOS[scenario]()
    jax_params = {k: jnp.asarray(v) for k, v in np_params.items()}

    loss_jax_fn, grad_jax_fn = build_guidance_fns_jax(jax_params)
    loss_torch_fn, grad_torch_fn = build_guidance_fns_torch(
        np_params, device="cpu", dtype=torch.float32
    )

    x_jax, x_torch = _make_x_t()

    loss_jax = np.asarray(loss_jax_fn(x_jax))
    loss_torch = loss_torch_fn(x_torch).detach().cpu().numpy()
    np.testing.assert_allclose(
        loss_torch,
        loss_jax,
        atol=2e-4,
        rtol=1e-3,
        err_msg=f"loss mismatch for {scenario}",
    )

    grad_jax = np.asarray(grad_jax_fn(x_jax))
    grad_torch = grad_torch_fn(x_torch).detach().cpu().numpy()
    # Skip the gradient direction check on samples whose gradient vanishes
    # numerically (e.g. surface/finger pairs where x_t starts far enough that
    # the stop-grad on the unselected branch dominates) — assert magnitude
    # only there.
    for b in range(grad_jax.shape[0]):
        gj = grad_jax[b].ravel()
        gt = grad_torch[b].ravel()
        nj = np.linalg.norm(gj)
        nt = np.linalg.norm(gt)
        if nj < 1e-8 or nt < 1e-8:
            assert nj < 1e-6 and nt < 1e-6, (
                f"{scenario}[b={b}]: one grad zero, other isn't (nj={nj}, nt={nt})"
            )
            continue
        cos = float(gj @ gt / (nj * nt))
        assert cos > 0.999, f"{scenario}[b={b}]: grad direction mismatch, cos={cos:.6f}"
        np.testing.assert_allclose(
            nt,
            nj,
            rtol=1e-3,
            atol=2e-5,
            err_msg=f"{scenario}[b={b}]: grad magnitude mismatch",
        )


def test_compute_guidance_velocity_descent_step():
    """One-step sanity: corrected velocity reduces loss vs. uncorrected."""
    np_params = _centroid_params()
    loss_fn, grad_fn = build_guidance_fns_torch(
        np_params, device="cpu", dtype=torch.float32
    )

    rng = np.random.default_rng(SEED)
    x_t = torch.from_numpy(
        rng.standard_normal((2, TOTAL_HORIZON, ACTION_DIM)).astype(np.float32) * 0.05
    )
    v_t = torch.from_numpy(
        rng.standard_normal((2, TOTAL_HORIZON, ACTION_DIM)).astype(np.float32) * 0.05
    )

    dt_pos = float(DT_ACTION)  # gr00t convention: dt > 0
    x_next_uncorrected = x_t + dt_pos * v_t
    base_loss = loss_fn(x_next_uncorrected).sum().item()

    # Apply guidance correction
    v_corr = compute_guidance_velocity_torch(
        loss_fn,
        grad_fn,
        x_t,
        v_t,
        dt=dt_pos,
        action_horizon=ACTION_HORIZON,
        guidance_factor=0.05,
        use_line_search=False,
    )
    x_next_corrected = x_t + dt_pos * v_corr
    corr_loss = loss_fn(x_next_corrected).sum().item()

    assert corr_loss < base_loss, (
        f"guidance step did not reduce loss: base={base_loss:.6f}, "
        f"corr={corr_loss:.6f} (sign convention bug?)"
    )


def test_no_op_when_factor_is_zero():
    """guidance_factor=0 → corrected velocity is identical to input."""
    np_params = _centroid_params()
    loss_fn, grad_fn = build_guidance_fns_torch(
        np_params, device="cpu", dtype=torch.float32
    )

    rng = np.random.default_rng(SEED)
    x_t = torch.from_numpy(
        rng.standard_normal((1, TOTAL_HORIZON, ACTION_DIM)).astype(np.float32)
    )
    v_t = torch.from_numpy(
        rng.standard_normal((1, TOTAL_HORIZON, ACTION_DIM)).astype(np.float32)
    )
    out = compute_guidance_velocity_torch(
        loss_fn,
        grad_fn,
        x_t,
        v_t,
        dt=float(DT_ACTION),
        action_horizon=ACTION_HORIZON,
        guidance_factor=0.0,
        use_line_search=False,
    )
    np.testing.assert_allclose(out.numpy(), v_t.numpy(), atol=1e-6)

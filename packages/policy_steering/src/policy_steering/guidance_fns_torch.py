"""PyTorch port of ``guidance_fns.py``.

This module mirrors :mod:`policy_steering.guidance_fns` semantically so that
the torch-based Gr00t flow-matching action head can consume the same
``__gp_*__`` wire-format guidance parameters that pi0grounded uses on the
JAX side. Distance computations (centroid / axis / plane / surface /
orientation / gripper aperture) and source-type selection (EE / left finger /
right finger / orientation / gripper) follow the JAX kernel exactly.

Sign convention difference vs the JAX kernel:

The JAX denoising loop in pi0 uses ``dt < 0`` (denoising convention from
``time = 1`` toward ``time = 0``), so adding ``+grad_hat`` to the velocity
yields a downhill position-space step. Gr00t's flow-matching loop uses
``dt > 0`` (``dt = 1 / num_inference_timesteps``), so the corresponding
descent step requires *subtracting* ``grad_hat`` from the velocity. This
module enforces descent regardless of the sign of ``dt`` (see
:func:`compute_guidance_velocity`).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from policy_steering.kinematics_torch import (
    torch_axis_angle_to_matrix_batch,
    torch_finger_positions_vmap,
)
from policy_steering.line_search_torch import golden_section_line_search
from policy_steering.robot_config import (
    DEFAULT_GRIPPER,
    GripperGeometry,
    load_gripper_geometry,
)

_DEFAULT_GEOMETRY = load_gripper_geometry(DEFAULT_GRIPPER)


# ---------------------------------------------------------------------------
# Opt-in guidance line-search landscape capture (diagnostic only).
# Enabled by setting PA_GUIDANCE_CAPTURE=<dir>. When unset, all of the code
# below is bypassed by a single cached-None check — zero overhead. See
# scripts/debug/analyze_guidance_landscape.py for the offline plotter.
# ---------------------------------------------------------------------------
_CAPTURE_DIR = os.environ.get("PA_GUIDANCE_CAPTURE")
_CAPTURE_INFER_EVERY = int(os.environ.get("PA_GUIDANCE_CAPTURE_INFER_EVERY", "1"))
_CAPTURE_DENOISE_EVERY = int(os.environ.get("PA_GUIDANCE_CAPTURE_DENOISE_EVERY", "1"))
_CAPTURE_ALPHA_MIN = float(os.environ.get("PA_GUIDANCE_CAPTURE_ALPHA_MIN", "-2.0"))
_CAPTURE_ALPHA_MULT = float(os.environ.get("PA_GUIDANCE_CAPTURE_ALPHA_MULT", "64.0"))
_CAPTURE_ALPHA_N = int(os.environ.get("PA_GUIDANCE_CAPTURE_ALPHA_N", "257"))
_capture_inference_idx = -1


def guidance_capture_enabled() -> bool:
    """True when PA_GUIDANCE_CAPTURE is set (diagnostic landscape dump)."""
    return _CAPTURE_DIR is not None


def guidance_capture_next_inference() -> int:
    """Increment + return the per-inference (trajectory-moment) counter.

    Called once per denoising loop entry in the GR00T grounded head so each
    captured npz can be grouped by trajectory moment. The counter lives in the
    server process (one process per eval run).
    """
    global _capture_inference_idx
    _capture_inference_idx += 1
    return _capture_inference_idx


def _capture_landscape(
    inference_idx: int,
    denoise_step: int,
    *,
    per_pair_loss_fn: Callable,
    per_pair_grad_fn: Callable,
    x_first: torch.Tensor,
    direction_shared: torch.Tensor,
    raw_grad: torch.Tensor,
    probe_alphas: np.ndarray,
    probe_losses: np.ndarray,
    chosen_alpha: float,
    do_no_harm_fired: bool,
    max_alpha: float,
    use_normalization: bool,
    guidance_factor: float,
    dt: float,
    num_inference_timesteps: int,
    guidance_params: dict,
) -> None:
    """Dense-probe the guidance loss along the search line and dump one npz.

    Captures (in float32, upcast from any bf16 live path): the shared-direction
    total + per-pair loss sweeps, each pair's own-gradient-direction sweep, the
    line-search probes, and labeling metadata. Side-effect only.
    """
    f32 = torch.float32
    xf = x_first.detach().to(f32)  # (B, H, D)
    dsh = direction_shared.detach().to(f32)
    alphas = torch.linspace(
        _CAPTURE_ALPHA_MIN,
        _CAPTURE_ALPHA_MULT * float(max_alpha),
        _CAPTURE_ALPHA_N,
        device=xf.device,
        dtype=f32,
    )

    per_pair_sweep = []
    with torch.no_grad():
        for a in alphas:
            per_pair_sweep.append(
                per_pair_loss_fn(xf + a * dsh).sum(dim=0).to(f32).cpu().numpy()  # (N,)
            )
    per_pair_sweep = np.stack(per_pair_sweep, axis=0)  # (A, N)
    total_sweep = per_pair_sweep.sum(axis=1).astype(np.float32)  # (A,) == Σ pairs
    loss_at_zero = float(per_pair_loss_fn(xf).sum().item())

    # Per-pair OWN gradient directions and their per-pair sweeps.
    g_pp = per_pair_grad_fn(xf)  # (B, N, H, D)
    B, N = g_pp.shape[0], g_pp.shape[1]
    norms = torch.linalg.norm(g_pp.reshape(B, N, -1), dim=-1).clamp_min(1e-8)
    if use_normalization:
        d_pp = -g_pp / norms[:, :, None, None]  # unit per-pair descent dirs
    else:
        d_pp = -g_pp
    per_pair_own = np.empty((N, _CAPTURE_ALPHA_N), np.float32)
    with torch.no_grad():
        for n in range(N):
            dn = d_pp[:, n]  # (B, H, D)
            for ai, a in enumerate(alphas):
                ln = per_pair_loss_fn(xf + a * dn)[:, n].sum()  # pair-n loss
                per_pair_own[n, ai] = float(ln.item())

    def _np(key, default=None):
        v = guidance_params.get(key, default)
        return None if v is None else np.asarray(getattr(v, "cpu", lambda: v)())

    os.makedirs(_CAPTURE_DIR, exist_ok=True)
    out = os.path.join(
        _CAPTURE_DIR, f"infer{inference_idx:04d}_step{denoise_step:02d}.npz"
    )
    np.savez_compressed(
        out,
        inference_idx=inference_idx,
        denoise_step=denoise_step,
        num_inference_timesteps=num_inference_timesteps,
        x_t_first=xf.cpu().numpy(),
        direction_shared=dsh.cpu().numpy(),
        raw_grad=raw_grad.detach().to(f32).cpu().numpy(),
        per_pair_dirs=d_pp.detach().to(f32).cpu().numpy(),
        alphas=alphas.cpu().numpy(),
        total_loss_sweep=total_sweep,
        per_pair_loss_sweep=per_pair_sweep,
        per_pair_own_sweep=per_pair_own,
        probe_alphas=probe_alphas,
        probe_losses=probe_losses,
        chosen_alpha=np.float32(chosen_alpha),
        do_no_harm_fired=bool(do_no_harm_fired),
        loss_at_zero=np.float32(loss_at_zero),
        max_alpha=np.float32(max_alpha),
        bracket_max=np.float32(32.0 * max_alpha),
        use_normalization=bool(use_normalization),
        guidance_factor=np.float32(guidance_factor),
        dt=np.float32(dt),
        dist_types=_np("__gp_dist_types__"),
        source_type=_np("__gp_source_type__"),
        has_axis=_np("__gp_has_axis__"),
        has_plane=_np("__gp_has_plane__"),
        guidance_weights=_np("__gp_guidance_weights__"),
    )


def _to_tensor(
    x: Any, *, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Coerce a numpy array / torch tensor / Python scalar to a tensor."""
    if isinstance(x, torch.Tensor):
        out = x.to(device=device)
        return out.to(dtype=dtype) if dtype is not None else out
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def build_guidance_fns(
    guidance_params: dict[str, Any],
    dynamics_fn: Callable | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    with_per_pair: bool = False,
    gripper_geometry: GripperGeometry | None = None,
) -> tuple[Callable, ...]:
    """Build batched loss and gradient functions from guidance parameter arrays.

    Args:
        guidance_params: Dict with ``__gp_*__`` keys (numpy or torch arrays).
        dynamics_fn: Optional differentiable dynamics function with signature
            ``dynamics_fn(x_t, ee_pos_0, ee_ori_0, dt) -> (positions,
            orientations, gripper)``. ``None`` uses Euler integration (cumsum).
        device: Torch device to place params on.
        dtype: Floating point dtype for params.
        gripper_geometry: Geometry of the gripper *being guided*. Defaults to
            the Panda; pass the live robot's geometry for cross-embodiment
            rollouts, where the demo and the live arm differ.

    Returns:
        ``(batch_loss_fn, batch_grad_fn)`` where:

        - ``batch_loss_fn(x_t: (B, total_horizon, D)) -> (B,)``
        - ``batch_grad_fn(x_t: (B, total_horizon, D)) -> (B, total_horizon, D)``
    """
    device = torch.device(device)
    geom = _DEFAULT_GEOMETRY if gripper_geometry is None else gripper_geometry

    def _t(
        key: str, *, default: Any = None, target_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        val = guidance_params.get(key, default)
        if val is None:
            raise KeyError(f"build_guidance_fns: missing required key {key!r}")
        td = target_dtype if target_dtype is not None else dtype
        return _to_tensor(val, device=device, dtype=td)

    ee_pos_0 = _t("__gp_ee_pos_0__")  # (3,)
    dt_param = float(np.asarray(guidance_params["__gp_dt__"]).item())  # python scalar
    expected = _t("__gp_expected_dists__")  # (N, T)
    dist_types = _to_tensor(
        guidance_params["__gp_dist_types__"], device=device, dtype=torch.long
    )  # (N,)
    N = expected.shape[0]
    source_types = _to_tensor(
        guidance_params.get("__gp_source_type__", np.zeros(N, dtype=np.uint8)),
        device=device,
        dtype=torch.long,
    )  # (N,)
    waypoints = _t("__gp_waypoints__")  # (N, 3)
    has_axis = _to_tensor(
        guidance_params["__gp_has_axis__"], device=device, dtype=torch.long
    )  # (N,)
    axes = _t("__gp_axes__")  # (N, 3) signed
    has_plane = _to_tensor(
        guidance_params["__gp_has_plane__"], device=device, dtype=torch.long
    )  # (N,)
    plane_norms = _t("__gp_plane_normals__")  # (N, 3)
    surf_pts = _t("__gp_surface_pts__")  # (N, S, 3)
    surf_counts = _to_tensor(
        guidance_params["__gp_surface_counts__"], device=device, dtype=torch.long
    )  # (N,)

    # Per-pair weights shape gradient direction only — the gradient is
    # normalized before being applied, and `guidance_factor` enters separately
    # in `compute_guidance_velocity` as the step-size / intensity multiplier.
    gf_weights = _t("__gp_guidance_weights__")  # (N,)

    ee_ori_0 = _t("__gp_ee_ori_0__", default=np.zeros(3, dtype=np.float32))  # (3,)
    ee_vectors = _t(
        "__gp_ee_vectors__",
        default=np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (N, 1)),
    )  # (N, 3)

    # Pre-compute boolean masks
    is_finger_l = source_types == 1  # (N,)
    is_finger_r = source_types == 2  # (N,)
    has_axis_mask = has_axis > 0  # (N,)
    has_plane_mask = has_plane > 0  # (N,)

    # Surface-validity mask: (N, S) — True for valid surface points
    S = surf_pts.shape[1]
    surf_idx = torch.arange(S, device=device)
    valid_surf_mask = surf_idx[None, :] < surf_counts[:, None]  # (N, S)

    is_surf = (dist_types == 1).unsqueeze(0)  # (1, N)
    is_ori = (dist_types == 2).unsqueeze(0)  # (1, N)
    is_grip = (dist_types == 3).unsqueeze(0)  # (1, N)
    ha = has_axis_mask.unsqueeze(0)  # (1, N)
    hp = has_plane_mask.unsqueeze(0)  # (1, N)
    ori_invalid = (is_ori & ~ha & ~hp).squeeze(0)  # (N,)

    T_expected = expected.shape[1]
    _SURF_SENTINEL = float(1e10)

    def _batched_guidance_fn(
        x_t: torch.Tensor, per_pair: bool = False, per_timestep: bool = False
    ) -> torch.Tensor:
        """``x_t``: ``(B, total_horizon, action_dim)`` → losses ``(B,)``.

        When ``per_pair`` is True, returns the un-summed per-distance-pair
        losses ``(B, N)`` instead; ``loss.sum(dim=1)`` recovers the ``(B,)``
        total exactly (the loss is additively decomposable over pairs).
        """
        B, total_horizon, _ = x_t.shape

        # --- Integrate to recover EE state trajectories (T = T_expected) -------
        if dynamics_fn is not None:
            positions, predicted_ori, predicted_grip = dynamics_fn(
                x_t[:, :T_expected], ee_pos_0, ee_ori_0, dt_param
            )
        else:
            vel = x_t[..., :3]  # (B, H, 3)
            positions_full = ee_pos_0[None, None, :] + torch.cumsum(
                vel * dt_param, dim=1
            )  # (B, H, 3)
            positions = positions_full[:, :T_expected]  # (B, T, 3)
            ori_vel = x_t[:, :T_expected, 3:6]
            predicted_ori = ee_ori_0[None, None, :] + torch.cumsum(
                ori_vel * dt_param, dim=1
            )  # (B, T, 3)
            predicted_grip = x_t[:, :T_expected, 6:7]  # (B, T, 1) — absolute

        # --- Gripper action → metric metres ---------------------------------
        grip_metric = geom.action_to_metric(predicted_grip[..., 0])  # (B, T)

        # --- Finger FK (batched over (B, T)) ---------------------------------
        T = positions.shape[1]
        flat_pos = positions.reshape(B * T, 3)
        flat_ori = predicted_ori.reshape(B * T, 3)
        flat_grip = grip_metric.reshape(B * T)
        left_flat, right_flat = torch_finger_positions_vmap(
            flat_pos, flat_ori, flat_grip, geometry=geom
        )
        left_fingers = left_flat.reshape(B, T, 3)
        right_fingers = right_flat.reshape(B, T, 3)

        # --- Per-pair source positions ---------------------------------------
        # source_pos: (B, T, N, 3)
        is_fl = is_finger_l[None, None, :, None]  # (1, 1, N, 1)
        is_fr = is_finger_r[None, None, :, None]
        source_pos = torch.where(
            is_fl,
            left_fingers[:, :, None, :].expand(B, T, N, 3),
            torch.where(
                is_fr,
                right_fingers[:, :, None, :].expand(B, T, N, 3),
                positions[:, :, None, :].expand(B, T, N, 3),
            ),
        )  # (B, T, N, 3)

        diffs = source_pos - waypoints[None, None, :, :]  # (B, T, N, 3)

        # Centroid distances
        centroid_dists = torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-8)  # (B, T, N)

        # Axis distances: dot(waypoint - source, signed_axis)
        axis_dists = -torch.sum(diffs * axes[None, None, :, :], dim=-1)  # (B, T, N)

        # Plane distances: ||in-plane component||
        proj = torch.sum(
            diffs * plane_norms[None, None, :, :], dim=-1, keepdim=True
        )  # (B, T, N, 1)
        in_plane = diffs - proj * plane_norms[None, None, :, :]
        plane_dists = torch.sqrt(torch.sum(in_plane**2, dim=-1) + 1e-8)  # (B, T, N)

        # Surface distances: min over valid surface points (sentinel-masked)
        src_exp = source_pos.unsqueeze(-2)  # (B, T, N, 1, 3)
        surf_exp = surf_pts[None, None, :, :, :]  # (1, 1, N, S, 3)
        surf_d2 = torch.sum((src_exp - surf_exp) ** 2, dim=-1)  # (B, T, N, S)
        # Broadcast valid_surf_mask: (N, S) → (1, 1, N, S)
        surf_d2 = torch.where(
            valid_surf_mask[None, None, :, :],
            surf_d2,
            torch.full_like(surf_d2, _SURF_SENTINEL),
        )
        surface_dists = torch.sqrt(torch.amin(surf_d2, dim=-1) + 1e-8)  # (B, T, N)

        # --- Orientation projection -----------------------------------------
        flat_ori_b = predicted_ori.reshape(B * T, 3)
        R_flat = torch_axis_angle_to_matrix_batch(flat_ori_b)  # (B*T, 3, 3)
        R_t = R_flat.reshape(B, T, 3, 3)
        # body_vecs_world[b, t, n, i] = sum_j R_t[b,t,i,j] * ee_vectors[n,j]
        body_vecs_world = torch.einsum("btij,nj->btni", R_t, ee_vectors)  # (B, T, N, 3)
        ori_axis_dists = torch.sum(
            body_vecs_world * axes[None, None, :, :], dim=-1
        )  # (B, T, N)
        ori_proj = torch.sum(
            body_vecs_world * plane_norms[None, None, :, :], dim=-1, keepdim=True
        )
        ori_in_plane = body_vecs_world - ori_proj * plane_norms[None, None, :, :]
        ori_plane_dists = torch.sqrt(
            torch.sum(ori_in_plane**2, dim=-1) + 1e-8
        )  # (B, T, N)
        ori_dists = torch.where(
            has_plane_mask[None, None, :], ori_plane_dists, ori_axis_dists
        )  # (B, T, N)

        # --- Gripper distances (broadcast metric over T) ---------------------
        grip_dists = grip_metric.unsqueeze(-1).expand(B, T, N)  # (B, T, N)

        # --- Select per-pair distance based on dist_type ---------------------
        is_surf_b = is_surf.unsqueeze(0)  # (1, 1, N)
        is_ori_b = is_ori.unsqueeze(0)
        is_grip_b = is_grip.unsqueeze(0)
        ha_b = ha.unsqueeze(0)
        hp_b = hp.unsqueeze(0)

        pos_dists = torch.where(
            is_surf_b,
            surface_dists,
            torch.where(
                ha_b,
                axis_dists,
                torch.where(hp_b, plane_dists, centroid_dists),
            ),
        )
        dists = torch.where(
            is_ori_b, ori_dists, torch.where(is_grip_b, grip_dists, pos_dists)
        )  # (B, T, N)

        # errors[b, n, t] = dists[b, t, n] - expected[n, t]
        errors = dists.transpose(1, 2) - expected[None, :, :T]  # (B, N, T)

        error_mask = torch.where(
            ori_invalid,
            torch.zeros((), dtype=dtype, device=device),
            torch.ones((), dtype=dtype, device=device),
        )  # (N,)
        errors = errors * error_mask[None, :, None]
        weighted = errors * gf_weights[None, :, None]  # (B, N, T)
        if per_pair:
            # Per-pair loss: 0.5 * sum over T only → (B, N)
            return 0.5 * torch.sum(weighted**2, dim=-1)
        if per_timestep:
            # Per-timestep loss: 0.5 * sum over N only → (B, T)
            return 0.5 * torch.sum(weighted**2, dim=1)
        # Per-sample loss: 0.5 * sum over (N, T)
        return 0.5 * torch.sum(weighted**2, dim=(-2, -1))  # (B,)

    def batch_loss_fn(x_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _batched_guidance_fn(x_t.to(dtype=dtype, device=device))

    def batch_grad_fn(x_t: torch.Tensor) -> torch.Tensor:
        # Per-sample gradients: ``losses[b]`` depends only on ``x_t[b]``, so
        # ``grad(sum(losses), x_t)`` equals stacked per-sample gradients.
        x_in = x_t.detach().to(dtype=dtype, device=device).requires_grad_(True)
        with torch.enable_grad():
            losses = _batched_guidance_fn(x_in)
            grads = torch.autograd.grad(
                losses.sum(), x_in, create_graph=False, retain_graph=False
            )[0]
        return grads.detach()

    if not with_per_pair:
        return batch_loss_fn, batch_grad_fn

    def per_pair_loss_fn(x_t: torch.Tensor) -> torch.Tensor:
        """``(B, total_horizon, D)`` → per-pair losses ``(B, N)``."""
        with torch.no_grad():
            return _batched_guidance_fn(
                x_t.to(dtype=dtype, device=device), per_pair=True
            )

    def per_pair_grad_fn(x_t: torch.Tensor) -> torch.Tensor:
        """Per-pair gradients ``(B, N, total_horizon, D)`` (N small; loop)."""
        x_in = x_t.detach().to(dtype=dtype, device=device).requires_grad_(True)
        with torch.enable_grad():
            per_pair = _batched_guidance_fn(x_in, per_pair=True)  # (B, N)
            n_pairs = per_pair.shape[1]
            grads = []
            for n in range(n_pairs):
                gn = torch.autograd.grad(
                    per_pair[:, n].sum(), x_in, retain_graph=(n < n_pairs - 1)
                )[0]
                grads.append(gn.detach())
        return torch.stack(grads, dim=1)  # (B, N, total_horizon, D)

    def per_timestep_loss_fn(x_t: torch.Tensor) -> torch.Tensor:
        """``(B, total_horizon, D)`` → per-timestep losses ``(B, T)``."""
        with torch.no_grad():
            return _batched_guidance_fn(
                x_t.to(dtype=dtype, device=device), per_timestep=True
            )

    def per_timestep_grad_fn(x_t: torch.Tensor) -> torch.Tensor:
        """Per-timestep gradients ``(B, T, total_horizon, D)`` (T = horizon; loop)."""
        x_in = x_t.detach().to(dtype=dtype, device=device).requires_grad_(True)
        with torch.enable_grad():
            per_ts = _batched_guidance_fn(x_in, per_timestep=True)  # (B, T)
            n_ts = per_ts.shape[1]
            grads = []
            for t in range(n_ts):
                gt = torch.autograd.grad(
                    per_ts[:, t].sum(), x_in, retain_graph=(t < n_ts - 1)
                )[0]
                grads.append(gt.detach())
        return torch.stack(grads, dim=1)  # (B, T, total_horizon, D)

    return (
        batch_loss_fn,
        batch_grad_fn,
        per_pair_loss_fn,
        per_pair_grad_fn,
        per_timestep_loss_fn,
        per_timestep_grad_fn,
    )


def compute_guidance_velocity(
    batch_loss_fn: Callable | None,
    batch_grad_fn: Callable,
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    dt: float,
    action_horizon: int,
    guidance_factor: float,
    use_line_search: bool,
    use_normalization: bool = True,
    max_alpha: float = 1.0,
    *,
    per_pair_loss_fn: Callable | None = None,
    per_pair_grad_fn: Callable | None = None,
    capture_inference_idx: int | None = None,
    capture_denoise_step: int | None = None,
    capture_num_steps: int | None = None,
    capture_guidance_params: dict | None = None,
) -> torch.Tensor:
    """Apply guidance gradient correction to the predicted velocity field.

    Mirror of :func:`policy_steering.guidance_fns.compute_guidance_velocity`,
    but adapted to Gr00t's positive ``dt`` denoising convention. Operates on
    the first ``action_horizon`` timesteps; remaining chunk slots receive
    zero correction.

    Args:
        batch_loss_fn: ``(B, T, D) -> (B,)`` per-sample loss (or ``None``).
        batch_grad_fn: ``(B, T, D) -> (B, T, D)`` per-sample gradients.
        x_t: Current denoising state ``(B, total_horizon, D)``.
        v_t: Predicted velocity ``(B, total_horizon, D)``.
        dt: Denoising step size. Pi0 uses ``dt < 0``; Gr00t uses ``dt > 0``.
        action_horizon: Number of steps in the first chunk.
        guidance_factor: Scalar guidance strength.
        use_line_search: Whether to run golden-section line search.
        use_normalization: When True (default), the gradient is L2-normalized
            to a unit direction before being applied. When False, the raw
            gradient is used.
        max_alpha: Initial upper bound for the golden-section line-search
            bracket (the search may expand it ~32x). Mirrors the JAX path so
            the GR00T head can tune step size; raise above 1.0 when
            ``use_normalization=False`` to reach a meaningful raw-gradient step.

    Returns:
        Corrected ``v_t`` with the same shape and dtype as the input.
    """
    B = x_t.shape[0]

    x_t_first = x_t[:, :action_horizon, :]
    v_t_first = v_t[:, :action_horizon, :]
    g = batch_grad_fn(x_t_first + v_t_first * dt)  # (B, H, D)

    if use_normalization:
        grad_norm = torch.linalg.norm(g.reshape(B, -1), dim=-1, keepdim=True).unsqueeze(
            -1
        )  # (B, 1, 1)
        grad_hat_first = g / (grad_norm + 1e-8)
    else:
        grad_hat_first = g  # raw gradient

    if use_line_search and batch_loss_fn is not None:
        direction_first = -grad_hat_first  # unit descent direction
        do_capture = (
            _CAPTURE_DIR is not None
            and per_pair_loss_fn is not None
            and per_pair_grad_fn is not None
            and capture_inference_idx is not None
            and capture_inference_idx % _CAPTURE_INFER_EVERY == 0
            and (capture_denoise_step or 0) % _CAPTURE_DENOISE_EVERY == 0
        )
        probe_log: list | None = [] if do_capture else None
        alpha = golden_section_line_search(
            batch_loss_fn,
            x_t_first,
            direction_first,
            max_alpha=max_alpha,
            probe_log=probe_log,
        )
        if do_capture:
            _capture_landscape(
                int(capture_inference_idx),
                int(capture_denoise_step or 0),
                per_pair_loss_fn=per_pair_loss_fn,
                per_pair_grad_fn=per_pair_grad_fn,
                x_first=x_t_first,
                direction_shared=direction_first,
                raw_grad=g,
                probe_alphas=np.asarray([p[0] for p in probe_log], np.float32),
                probe_losses=np.asarray([p[1] for p in probe_log], np.float32),
                chosen_alpha=float(alpha),
                do_no_harm_fired=bool(float(alpha) == 0.0 and len(probe_log) > 0),
                max_alpha=float(max_alpha),
                use_normalization=use_normalization,
                guidance_factor=guidance_factor,
                dt=dt,
                num_inference_timesteps=int(capture_num_steps or 0),
                guidance_params=capture_guidance_params or {},
            )
        # Padded direction over the full horizon
        if x_t.shape[1] > action_horizon:
            zero_tail = torch.zeros_like(x_t[:, action_horizon:, :])
            direction = torch.cat([direction_first, zero_tail], dim=1)
        else:
            direction = direction_first
        # Position-space step: dt * delta_v = alpha * direction → descent step
        # of magnitude ``alpha`` along ``direction``. Sign-correct for both
        # backends because alpha/dt has the right sign in either case.
        return v_t + (alpha / dt) * direction * guidance_factor

    # Non-line-search path: descent in position space requires subtracting
    # ``grad_hat`` for positive ``dt`` (Gr00t) and adding for negative ``dt``
    # (pi0). Use ``copysign`` so the same code is correct for either sign.
    sign = math.copysign(1.0, dt)
    # When dt > 0: subtract grad_hat; when dt < 0: add grad_hat.
    coeff = -sign * guidance_factor
    if x_t.shape[1] > action_horizon:
        zero_tail = torch.zeros_like(x_t[:, action_horizon:, :])
        grad_hat = torch.cat([grad_hat_first, zero_tail], dim=1)
    else:
        grad_hat = grad_hat_first
    return v_t + coeff * grad_hat


def compute_guidance_velocity_multi_alpha(
    batch_loss_fn: Callable,
    per_pair_loss_fn: Callable,
    per_pair_grad_fn: Callable,
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    dt: float,
    action_horizon: int,
    guidance_factor: float,
    use_normalization: bool = True,
    max_alpha: float = 1.0,
    combine: str = "sequential",
    guidance_weights: torch.Tensor | None = None,
    per_timestep_loss_fn: Callable | None = None,
    per_timestep_grad_fn: Callable | None = None,
) -> torch.Tensor:
    """Per-distance-pair multi-alpha guidance correction.

    Each pair ``n`` gets its own descent direction ``d_n = -grad(L_n)`` and its
    own line-searched step ``alpha_n``; the per-pair displacements are merged
    into one position-space step (then converted to a velocity correction with
    the same convention as :func:`compute_guidance_velocity`):

      - ``"sequential"`` (default): greedy coordinate descent — each pair line-
        searches along its own direction to reduce the **total** loss from the
        running point, so the total loss is non-increasing (do-no-harm safe).
        Pairs are visited in descending current-loss order.
      - ``"sum"``: ``alpha_n`` minimizes pair ``n``'s own loss; steps summed.
      - ``"weighted"``: like ``sum`` but scaled by per-pair guidance weights.
      - ``"per_timestep"``: NOT per-pair — the timestep-analog of
        ``"sequential"``. Each chunk timestep ``t`` gets its OWN descent
        direction ``-grad(loss_t)`` (``loss_t`` = the per-timestep potential,
        distance pairs combined by their weights) and its own line-searched step
        ``alpha_t``, searched against the TOTAL loss from the running point
        (coordinate descent — do-no-harm, total loss non-increasing). Timesteps
        visited in descending per-timestep-loss order. Requires
        ``per_timestep_loss_fn`` / ``per_timestep_grad_fn``.

    Motivated by the landscape analysis: the per-pair loss optima along the
    shared direction disagree widely, so a single shared alpha is a poor
    compromise.
    """
    x_first = x_t[:, :action_horizon, :]
    v_first = v_t[:, :action_horizon, :]
    base = x_first + v_first * dt
    g_pp = per_pair_grad_fn(base)  # (B, N, H, D)
    B, N = g_pp.shape[0], g_pp.shape[1]
    if use_normalization:
        norms = torch.linalg.norm(g_pp.reshape(B, N, -1), dim=-1).clamp_min(1e-8)
        d_pp = -g_pp / norms[:, :, None, None]
    else:
        d_pp = -g_pp

    disp = torch.zeros_like(x_first)
    if combine == "sequential":
        x_cur = base
        with torch.no_grad():
            cur_losses = per_pair_loss_fn(base).sum(dim=0)  # (N,)
        order = torch.argsort(cur_losses, descending=True).tolist()
        for n in order:
            dn = d_pp[:, n]
            an = golden_section_line_search(
                batch_loss_fn, x_cur, dn, max_alpha=max_alpha
            )
            step = an * dn
            x_cur = x_cur + step
            disp = disp + step
    elif combine == "per_timestep":
        # Coordinate descent over timesteps (timestep-analog of "sequential"):
        # each timestep's own-loss direction -grad(loss_t) is line-searched
        # against the TOTAL loss from the RUNNING point x_cur (which already
        # includes the earlier timesteps' steps), so the total loss is
        # non-increasing (do-no-harm; cannot diverge — a purely independent
        # search would overshoot under the velocity→position cumsum coupling).
        # Timesteps visited in descending per-timestep-loss order.
        g_pt = per_timestep_grad_fn(base)  # (B, T, H, D)
        if use_normalization:
            B_, T_ = g_pt.shape[0], g_pt.shape[1]
            pt_norms = torch.linalg.norm(g_pt.reshape(B_, T_, -1), dim=-1).clamp_min(
                1e-8
            )
            d_pt = -g_pt / pt_norms[:, :, None, None]
        else:
            d_pt = -g_pt
        x_cur = base
        with torch.no_grad():
            cur_losses = per_timestep_loss_fn(base).sum(dim=0)  # (T,)
        order = torch.argsort(cur_losses, descending=True).tolist()
        for t in order:
            d_t = d_pt[:, t]  # (B, H, D)
            at = golden_section_line_search(
                batch_loss_fn, x_cur, d_t, max_alpha=max_alpha
            )
            step = at * d_t
            x_cur = x_cur + step
            disp = disp + step
    else:
        for n in range(N):
            dn = d_pp[:, n]

            def _loss_n(y, _n=n):
                return per_pair_loss_fn(y)[:, _n]

            an = golden_section_line_search(_loss_n, base, dn, max_alpha=max_alpha)
            w = 1.0
            if combine == "weighted" and guidance_weights is not None:
                w = float(guidance_weights[n])
            disp = disp + w * an * dn

    if x_t.shape[1] > action_horizon:
        zero_tail = torch.zeros_like(x_t[:, action_horizon:, :])
        disp = torch.cat([disp, zero_tail], dim=1)
    # ``disp`` already includes the per-pair alphas (position-space step).
    return v_t + (disp / dt) * guidance_factor

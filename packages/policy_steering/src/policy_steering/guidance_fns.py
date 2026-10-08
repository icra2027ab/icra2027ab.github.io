"""JAX-differentiable guidance loss, gradient, and velocity correction.

This module provides the core guidance functions used by the denoising loop
in ``Pi0Grounded.sample_actions``.  All functions are JAX-traceable and
compatible with ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from policy_steering.kinematics import (
    jax_axis_angle_to_matrix,
    make_jax_finger_positions_vmap,
)
from policy_steering.line_search import golden_section_line_search
from policy_steering.robot_config import (
    DEFAULT_GRIPPER,
    GripperGeometry,
    load_gripper_geometry,
)

_DEFAULT_GEOMETRY = load_gripper_geometry(DEFAULT_GRIPPER)


def build_guidance_fns(
    guidance_params: dict,
    dynamics_fn=None,
    with_per_pair: bool = False,
    gripper_geometry: GripperGeometry | None = None,
):
    """Build batched loss AND gradient functions from structured guidance parameter arrays.

    Integrates x_t (SO3 velocity actions) to recover EE state trajectories,
    computes distances from those states (or derived quantities like finger
    positions) to the reference waypoints, then returns both a loss and a
    gradient of a squared-error potential via JAX autograd.

    Source entity types (per pair, selected by ``__gp_source_type__``):
        - 0 (EE):        distance from EE position
        - 1 (finger_l):  distance from left finger tip (via FK)
        - 2 (finger_r):  distance from right finger tip (via FK)
        - 3 (orientation): projection of a configurable body-frame vector
          ``g = R(predicted_ori) @ ee_vector`` (default ``[0,0,1]``) onto
          ``axis`` (signed cosine, in ``[-1, 1]``) or in-plane magnitude when
          ``plane`` is set.
        - 4 (gripper):    gripper aperture scalar

    For source types 0-2, the distance computation is selected by
    ``dist_type`` (centroid/axis/plane/surface).  For types 3-4, dedicated
    distance semantics apply.

    Args:
        guidance_params: Dict with JAX arrays (keys match ``GP_KEYS``).
        dynamics_fn: Optional JAX-differentiable dynamics function with signature
            ``integrate_actions(x_t, ee_pos_0, ee_ori_0, dt) -> (positions, orientations, gripper)``.
            When ``None``, the default Euler integration (cumsum) is used.

    Returns:
        When ``with_per_pair`` is False: ``(batch_loss_fn, batch_grad_fn)``:
        - batch_loss_fn(x_t) -> (B,) per-sample losses
        - batch_grad_fn(x_t) -> (B, T, D) per-sample gradients

        When ``with_per_pair`` is True, additionally the per-pair and
        per-timestep decompositions (all summing to the scalar potential):
        - per_pair_loss_fn(x_t) -> (B, N), per_pair_grad_fn(x_t) -> (B, N, T, D)
        - per_timestep_loss_fn(x_t) -> (B, T), per_timestep_grad_fn(x_t) -> (B, T, T, D)
    """
    # Gripper geometry of the robot *being guided* — the live embodiment, which
    # is not necessarily the one the demo was recorded on. Bound into the
    # closure so it stays static under jit.
    geom = _DEFAULT_GEOMETRY if gripper_geometry is None else gripper_geometry
    finger_positions_vmap = make_jax_finger_positions_vmap(geom)

    ee_pos_0 = jnp.array(guidance_params["__gp_ee_pos_0__"])  # (3,)
    dt = jnp.array(guidance_params["__gp_dt__"])  # scalar
    expected = jnp.array(guidance_params["__gp_expected_dists__"])  # (N, T)
    dist_types = jnp.array(guidance_params["__gp_dist_types__"])  # (N,) uint8
    source_types = jnp.array(
        guidance_params.get(
            "__gp_source_type__",
            jnp.zeros(expected.shape[0], dtype=jnp.uint8),
        )
    )  # (N,) uint8
    waypoints = jnp.array(guidance_params["__gp_waypoints__"])  # (N, 3)
    has_axis = jnp.array(guidance_params["__gp_has_axis__"])  # (N,) uint8
    axes = jnp.array(guidance_params["__gp_axes__"])  # (N, 3) signed
    has_plane = jnp.array(guidance_params["__gp_has_plane__"])  # (N,) uint8
    plane_norms = jnp.array(guidance_params["__gp_plane_normals__"])  # (N, 3)
    surf_pts = jnp.array(guidance_params["__gp_surface_pts__"])  # (N, S, 3)
    surf_counts = jnp.array(guidance_params["__gp_surface_counts__"])  # (N,) int32

    # Per-pair weights shape gradient direction only — the gradient is
    # normalized before being applied, and `guidance_factor` enters separately
    # in `compute_guidance_velocity` as the step-size / intensity multiplier.
    gf_weights = jnp.array(guidance_params["__gp_guidance_weights__"])  # (N,)

    # EE orientation (used to FK-rotate finger offsets into the world frame).
    # Initial gripper state is not needed in the default path because the
    # gripper channel is absolute (not integrated) — we map the raw action to
    # metric metres via ``gripper_action_to_metric`` inside ``_guidance_fn``.
    ee_ori_0 = jnp.array(guidance_params.get("__gp_ee_ori_0__", jnp.zeros(3)))

    # Per-pair body-frame vector for orientation guidance (default: z-axis)
    ee_vectors = jnp.array(
        guidance_params.get(
            "__gp_ee_vectors__",
            jnp.tile(jnp.array([0.0, 0.0, 1.0]), (expected.shape[0], 1)),
        )
    )  # (N, 3)
    # Per-pair body-frame offset of an EEF-attached object (source_type 5):
    # source_pos = ee_pos + R_eef @ offset. Zeros for all other pairs.
    ee_offsets = jnp.array(
        guidance_params.get("__gp_ee_offsets__", jnp.zeros((expected.shape[0], 3)))
    )  # (N, 3)

    # Pre-compute boolean masks for source type selection (avoid repeated comparisons)
    is_finger_l = source_types == 1  # (N,)
    is_finger_r = source_types == 2  # (N,)
    is_attached = source_types == 5  # (N,) — object rigidly fixed to the EEF
    has_axis_mask = has_axis > 0  # (N,)
    has_plane_mask = has_plane > 0  # (N,)

    def _weighted_errors(x_t):
        # x_t: (total_horizon, action_dim) — unbatched. Returns the weighted
        # per-pair, per-timestep errors ``(N, T)`` (pre-square-sum), shared by
        # the scalar total potential and the per-pair (multi-alpha) potential.
        T = expected.shape[1]

        # --- Integrate actions to get EE state trajectories ------------------
        if dynamics_fn is not None:
            positions, predicted_ori, predicted_grip = dynamics_fn(
                x_t[:T], ee_pos_0, ee_ori_0, dt
            )
        else:
            # Default: Euler integration (cumsum)
            vel = x_t[:, :3]  # (H, 3)
            positions = ee_pos_0[None, :] + jnp.cumsum(vel * dt, axis=0)  # (H, 3)
            positions = positions[:T]  # (T, 3)
            ori_vel = x_t[:T, 3:6]  # (T, 3)
            predicted_ori = ee_ori_0[None, :] + jnp.cumsum(
                ori_vel * dt, axis=0
            )  # (T, 3)
            predicted_grip = x_t[:T, 6:7]  # (T, 1) — absolute, not integrated

        # --- Normalise gripper action to metric aperture (metres) -----------
        # The policy's gripper channel is in normalised action space (whose
        # bounds differ per robot: [-1, 1] for the Panda, [0, 1] for the
        # Robotiq) but finger FK and aperture expected distances both live in
        # metres.  Applying this linear map keeps every distance the autograd
        # sees in the same metric frame, so gradient from finger-position pairs
        # flows proportionally to EE pos/ori rather than being absorbed by the
        # gripper channel.
        grip_metric = geom.action_to_metric(predicted_grip[:, 0])  # (T,)

        # --- Compute finger positions via FK (only when needed) --------------
        # Always compute to keep shapes static for JIT; the cost is minimal
        # when no finger pairs exist (results are unused).
        left_fingers, right_fingers = finger_positions_vmap(
            positions, predicted_ori, grip_metric
        )  # each (T, 3)

        # --- EEF orientation matrices (hoisted) ------------------------------
        # Needed by both the attached-object source (below) and the orientation
        # distance block (later). Depends only on predicted_ori.
        R_t = jax.vmap(jax_axis_angle_to_matrix)(predicted_ori)  # (T, 3, 3)
        # Attached-object world positions: obj rigidly fixed to the EEF at
        # body-frame offset ``ee_offsets[n]`` → ee_pos + R_eef @ offset.
        attached_pos = positions[:, None, :] + jnp.einsum(
            "tij,nj->tni", R_t, ee_offsets
        )  # (T, N, 3)

        # --- Build per-pair source positions based on source_type ------------
        # source_types: 0=EE, 1=finger_l, 2=finger_r, 3=ori, 4=grip, 5=attached
        # For types 3,4 the position is unused but must have valid shape.
        source_pos = jnp.where(
            is_attached[None, :, None],
            attached_pos,
            jnp.where(
                is_finger_l[None, :, None],
                left_fingers[:, None, :],
                jnp.where(
                    is_finger_r[None, :, None],
                    right_fingers[:, None, :],
                    positions[:, None, :],  # default: EE position
                ),
            ),
        )  # (T, N, 3)

        diffs = source_pos - waypoints[None, :, :]  # (T, N, 3)

        # Centroid distances: ||source - waypoint||
        centroid_dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)  # (T, N)

        # Axis distances: dot(waypoint - source, signed_axis)
        axis_dists = -jnp.sum(diffs * axes[None, :, :], axis=-1)  # (T, N)

        # Plane distances: ||in_plane(waypoint - source)||
        proj = jnp.sum(
            diffs * plane_norms[None, :, :], axis=-1, keepdims=True
        )  # (T, N, 1)
        in_plane = diffs - proj * plane_norms[None, :, :]  # (T, N, 3)
        plane_dists = jnp.sqrt(jnp.sum(in_plane**2, axis=-1) + 1e-8)  # (T, N)

        # Surface distances: min ||source - surf_pt|| over valid points.
        # We mask invalid slots with a large *finite* sentinel rather than inf,
        # because jnp.where with inf produces NaN gradients via the unselected
        # branch even though the forward value is fine.
        src_exp = source_pos[:, :, None, :]  # (T, N, 1, 3)
        surf_exp = surf_pts[None, :, :, :]  # (1, N, S, 3)
        surf_d2 = jnp.sum((src_exp - surf_exp) ** 2, axis=-1)  # (T, N, S)
        S = surf_pts.shape[1]
        valid_mask = jnp.arange(S)[None, :] < surf_counts[:, None]  # (N, S)
        _SURF_SENTINEL = jnp.float32(1e10)
        surf_d2 = jnp.where(valid_mask[None, :, :], surf_d2, _SURF_SENTINEL)
        surface_dists = jnp.sqrt(
            jnp.min(surf_d2, axis=-1) + 1e-8
        )  # (T, N), always finite

        # --- Orientation "distance" = projection of body-frame vector --------
        # Build per-timestep world-from-EE rotation matrices and rotate each
        # pair's body-frame ee_vector into world coordinates.  Then project
        # onto the supplied per-pair direction (axis = signed cosine; plane =
        # in-plane magnitude perpendicular to the normal).
        # Per-pair rotated body vector: R_t[t] @ ee_vectors[n] → (T, N, 3)
        # (R_t hoisted above for the attached-object source.)
        body_vecs_world = jnp.einsum("tij,nj->tni", R_t, ee_vectors)  # (T, N, 3)
        ori_axis_dists = jnp.sum(body_vecs_world * axes[None, :, :], axis=-1)  # (T, N)
        ori_proj = jnp.sum(
            body_vecs_world * plane_norms[None, :, :],
            axis=-1,
            keepdims=True,
        )  # (T, N, 1)
        ori_in_plane = body_vecs_world - ori_proj * plane_norms[None, :, :]  # (T, N, 3)
        ori_plane_dists = jnp.sqrt(jnp.sum(ori_in_plane**2, axis=-1) + 1e-8)  # (T, N)
        ori_dists = jnp.where(
            has_plane_mask[None, :], ori_plane_dists, ori_axis_dists
        )  # (T, N)

        # --- Gripper distances -----------------------------------------------
        # Aperture pairs compare the metric gripper opening (metres) to the
        # expected reference trajectory (also metres), so use ``grip_metric``.
        grip_dists = jnp.broadcast_to(
            grip_metric[:, None], (T, expected.shape[0])
        )  # (T, N)

        # --- Select distance per pair based on dist_type ---------------------
        # dist_types: 0=centroid, 1=surface, 2=orientation, 3=gripper
        is_surf = (dist_types == 1)[None, :]  # (1, N)
        is_ori = (dist_types == 2)[None, :]  # (1, N)
        is_grip = (dist_types == 3)[None, :]  # (1, N)
        ha = has_axis_mask[None, :]  # (1, N)
        hp = has_plane_mask[None, :]  # (1, N)

        # Position-type selection (centroid/axis/plane/surface)
        pos_dists = jnp.where(
            is_surf,
            surface_dists,
            jnp.where(ha, axis_dists, jnp.where(hp, plane_dists, centroid_dists)),
        )  # (T, N)

        # Final selection across all types
        dists = jnp.where(
            is_ori, ori_dists, jnp.where(is_grip, grip_dists, pos_dists)
        )  # (T, N)

        errors = dists.T - expected[:, :T]  # (N, T)

        # Defensive mask: orientation pairs without any direction (no axis
        # AND no plane) would otherwise contribute a non-zero, gradient-less
        # error.  Zero them out so a misconfigured pair fails silent-zero
        # rather than silent-bogus.
        ori_invalid = (is_ori & ~ha & ~hp)[0]  # (N,)
        error_mask = jnp.where(ori_invalid, 0.0, 1.0)  # (N,)
        errors = errors * error_mask[:, None]

        weighted = errors * gf_weights[:, None]  # (N, T)
        return weighted

    def _guidance_fn(x_t):
        # Scalar total potential: 0.5 * Σ_{N,T} weighted².
        w = _weighted_errors(x_t)
        return 0.5 * jnp.sum(w**2)

    def _guidance_fn_per_pair(x_t):
        # Per-pair potential ``(N,)``: 0.5 * Σ_T weighted² with NO sum over
        # pairs, so each pair's loss/gradient can be line-searched on its own
        # (multi-alpha guidance). Sums to ``_guidance_fn``.
        w = _weighted_errors(x_t)
        return 0.5 * jnp.sum(w**2, axis=1)

    def _guidance_fn_per_timestep(x_t):
        # Per-timestep potential ``(T,)``: 0.5 * Σ_N weighted² with NO sum over
        # timesteps (pairs are combined normally with their weights), so each
        # chunk timestep can be line-searched against its OWN loss
        # (``per_timestep`` multi-alpha mode). Sums to ``_guidance_fn``.
        w = _weighted_errors(x_t)
        return 0.5 * jnp.sum(w**2, axis=0)

    batch_loss_fn = jax.vmap(_guidance_fn)
    batch_grad_fn = jax.vmap(jax.grad(_guidance_fn))
    if not with_per_pair:
        return batch_loss_fn, batch_grad_fn

    # Per-pair variants for multi-alpha guidance (mirrors the torch head's
    # ``build_guidance_fns(..., with_per_pair=True)``).
    per_pair_loss_fn = jax.vmap(_guidance_fn_per_pair)  # (B,T,D) -> (B,N)
    per_pair_grad_fn = jax.vmap(
        jax.jacrev(_guidance_fn_per_pair)
    )  # (B,T,D) -> (B,N,T,D)
    # Per-timestep variants for the ``per_timestep`` mode (each chunk timestep
    # gets its own descent direction + line-searched step against its own loss).
    per_timestep_loss_fn = jax.vmap(_guidance_fn_per_timestep)  # (B,T,D) -> (B,T)
    per_timestep_grad_fn = jax.vmap(
        jax.jacrev(_guidance_fn_per_timestep)
    )  # (B,T,D)->(B,T,T,D)
    return (
        batch_loss_fn,
        batch_grad_fn,
        per_pair_loss_fn,
        per_pair_grad_fn,
        per_timestep_loss_fn,
        per_timestep_grad_fn,
    )


def compute_guidance_velocity(
    batch_loss_fn,
    batch_grad_fn,
    x_t,
    v_t,
    dt: float,
    action_horizon: int,
    guidance_factor: float,
    use_line_search: bool,
    use_normalization: bool = True,
    max_alpha: float = 1.0,
):
    """Apply guidance gradient correction to the predicted velocity field.

    Computes guidance gradients on the first action chunk only, optionally
    normalizes them to a unit direction, optionally runs a line search for
    the optimal step size, and blends the correction into ``v_t``.  Remaining
    chunks receive zero guidance (padded with zeros).

    Args:
        batch_loss_fn: ``(B, T, D) -> (B,)`` per-sample loss, or ``None``.
        batch_grad_fn: ``(B, T, D) -> (B, T, D)`` per-sample gradients.
        x_t: Current noisy actions ``(B, total_horizon, D)``.
        v_t: Predicted velocity ``(B, total_horizon, D)``.
        dt: Denoising step size (negative scalar).
        action_horizon: Number of steps in the first chunk.
        guidance_factor: Scalar multiplier for guidance strength.
        use_line_search: Whether to run golden-section line search.
        use_normalization: When True (default), the gradient is L2-normalized
            to a unit ascent direction before being applied. When False, the
            raw gradient is used as the descent direction — magnitude scales
            with the loss surface, so ``guidance_factor`` then behaves as a
            classic gradient-descent step size.

    Returns:
        Corrected ``v_t`` with the same shape as the input.
    """
    batch_size = x_t.shape[0]

    # Guidance on first chunk only
    x_t_first = x_t[:, :action_horizon, :]
    guidance_grad_first = batch_grad_fn(x_t_first + v_t[:, :action_horizon, :] * dt)

    # Normalize-or-not is selected with jnp.where so the predicate may be a
    # JAX tracer (it arrives via guidance_params, which is part of the JIT
    # pytree). Both expressions are cheap to compute, so unconditionally
    # building them is fine.
    grad_norm = jnp.linalg.norm(
        guidance_grad_first.reshape(batch_size, -1), axis=-1, keepdims=True
    )[:, :, None]  # (B, 1, 1)
    normalize_scale = jnp.where(
        use_normalization,
        1.0 / (grad_norm + 1e-8),
        jnp.ones_like(grad_norm),
    )
    grad_hat_first = guidance_grad_first * normalize_scale

    tail_zeros = jnp.zeros_like(x_t[:, action_horizon:, :])

    def _v_with_line_search(_):
        direction_first = -grad_hat_first  # descent direction (unit if normalized)
        alpha = golden_section_line_search(
            batch_loss_fn,
            x_t_first,
            direction_first,
            max_alpha=max_alpha,
        )
        direction = jnp.concatenate([direction_first, tail_zeros], axis=1)
        # Convert position-space step to velocity: dt * v_guidance = alpha * direction
        return v_t + (alpha / dt) * direction * guidance_factor

    def _v_no_line_search(_):
        grad_hat = jnp.concatenate([grad_hat_first, tail_zeros], axis=1)
        # Blend (normalized or raw) gradient into velocity field. Since dt < 0,
        # adding +grad_hat to v_t -> dt * grad_hat moves in -grad (descent).
        return v_t + guidance_factor * grad_hat

    if batch_loss_fn is None:
        # No loss function available — line search is impossible regardless of
        # the toggle. Fall through to the fixed-step branch.
        return _v_no_line_search(None)

    # use_line_search may be a JAX tracer (from guidance_params) — lax.cond
    # accepts tracer predicates and compiles both branches.
    return jax.lax.cond(
        use_line_search, _v_with_line_search, _v_no_line_search, operand=None
    )


def compute_guidance_velocity_multi_alpha(
    batch_loss_fn,
    per_pair_loss_fn,
    per_pair_grad_fn,
    x_t,
    v_t,
    dt: float,
    action_horizon: int,
    guidance_factor,
    use_normalization: bool = True,
    max_alpha: float = 1.0,
    combine=0,
    guidance_weights=None,
    per_timestep_loss_fn=None,
    per_timestep_grad_fn=None,
):
    """Per-distance-pair multi-alpha guidance correction (JAX).

    JAX mirror of :func:`guidance_fns_torch.compute_guidance_velocity_multi_alpha`.
    Each pair ``n`` gets its own descent direction ``d_n = -grad(L_n)`` and its
    own line-searched step ``alpha_n``; the per-pair displacements merge into one
    position-space step, then convert to a velocity correction with the same
    convention as :func:`compute_guidance_velocity`.

    ``combine`` (traced int from ``__gp_multi_alpha_combine__``):
      - 0 ``sequential``: greedy coordinate descent — each pair line-searches
        along its own direction to reduce the TOTAL loss from the running point
        (do-no-harm safe). Pairs visited in descending current-loss order.
      - 1 ``sum``: ``alpha_n`` minimizes pair ``n``'s own loss; steps summed.
      - 2 ``weighted``: like ``sum`` but scaled by per-pair guidance weights.
      - 3 ``per_timestep``: NOT per-pair — the timestep-analog of ``sequential``.
        Each chunk timestep ``t`` gets its OWN descent direction ``-grad(loss_t)``
        (``loss_t`` = the per-timestep potential, distance pairs combined by their
        weights) and its own line-searched step ``alpha_t``, searched against the
        TOTAL loss from the running point (coordinate descent — do-no-harm, so
        the total loss is non-increasing). Timesteps visited in descending
        per-timestep-loss order. Requires ``per_timestep_loss_fn`` /
        ``per_timestep_grad_fn``.

    Runs inside the jitted denoising scan, so all control flow is pure JAX
    (``lax.scan`` over pairs/timesteps, ``lax.switch`` over combine modes).
    """
    x_first = x_t[:, :action_horizon, :]
    v_first = v_t[:, :action_horizon, :]
    base = x_first + v_first * dt
    g_pp = per_pair_grad_fn(base)  # (B, N, H, D)
    batch_size, n_pairs = g_pp.shape[0], g_pp.shape[1]

    norms = jnp.maximum(
        jnp.linalg.norm(g_pp.reshape(batch_size, n_pairs, -1), axis=-1), 1e-8
    )
    d_pp = jnp.where(
        use_normalization, -g_pp / norms[:, :, None, None], -g_pp
    )  # (B, N, H, D)

    zeros_disp = jnp.zeros_like(base)

    def _sequential():
        # Greedy coordinate descent in descending current-loss order. Each pair
        # line-searches along its own direction against the TOTAL loss, so the
        # running point's total loss is non-increasing (do-no-harm safe).
        cur_losses = jnp.sum(per_pair_loss_fn(base), axis=0)  # (N,)
        order = jnp.argsort(-cur_losses)  # (N,) descending

        def _body(carry, n_idx):
            x_cur, disp = carry
            dn = jnp.take(d_pp, n_idx, axis=1)  # (B, H, D)
            an = golden_section_line_search(
                batch_loss_fn, x_cur, dn, max_alpha=max_alpha
            )
            step = an * dn
            return (x_cur + step, disp + step), None

        (_, disp), _ = jax.lax.scan(_body, (base, zeros_disp), order)
        return disp

    def _summed(weighted_mode: bool):
        wvec = (
            guidance_weights
            if guidance_weights is not None
            else jnp.ones((n_pairs,), dtype=base.dtype)
        )

        def _body(disp, n):
            dn = jnp.take(d_pp, n, axis=1)  # (B, H, D)

            def _loss_n(y):
                return jnp.take(per_pair_loss_fn(y), n, axis=1)  # (B,)

            an = golden_section_line_search(_loss_n, base, dn, max_alpha=max_alpha)
            coeff = an * (jnp.take(wvec, n) if weighted_mode else 1.0)
            return disp + coeff * dn, None

        disp, _ = jax.lax.scan(_body, zeros_disp, jnp.arange(n_pairs))
        return disp

    def _per_timestep():
        # Coordinate descent over timesteps (timestep-analog of ``_sequential``):
        # each timestep ``t`` gets its own descent direction ``-grad(loss_t)``
        # (loss_t = the per-timestep potential, distance pairs combined with
        # their weights) and its own line-searched step, but searched against the
        # TOTAL loss from the RUNNING point ``x_cur`` (which already includes the
        # earlier timesteps' steps). The total loss is therefore non-increasing
        # (do-no-harm — it cannot diverge); a purely independent-from-``base``
        # search would overshoot under the velocity→position cumsum coupling.
        # Timesteps are visited in descending per-timestep-loss order.
        # ``n_ts`` (guidance horizon = number of expected-distance timesteps) may
        # be SHORTER than ``action_horizon``; key off the grad shape.
        g_pt = per_timestep_grad_fn(base)  # (B, T, H, D)
        n_ts = g_pt.shape[1]
        pt_norms = jnp.maximum(
            jnp.linalg.norm(g_pt.reshape(batch_size, n_ts, -1), axis=-1),
            1e-8,
        )  # (B, T)
        d_pt = jnp.where(
            use_normalization, -g_pt / pt_norms[:, :, None, None], -g_pt
        )  # (B, T, H, D)
        cur_losses = jnp.sum(per_timestep_loss_fn(base), axis=0)  # (T,)
        order = jnp.argsort(-cur_losses)  # descending per-timestep loss

        def _body(carry, t):
            x_cur, disp = carry
            d_t = jnp.take(d_pt, t, axis=1)  # (B, H, D)
            at = golden_section_line_search(
                batch_loss_fn, x_cur, d_t, max_alpha=max_alpha
            )
            step = at * d_t
            return (x_cur + step, disp + step), None

        (_, disp), _ = jax.lax.scan(_body, (base, zeros_disp), order)
        return disp

    disp = jax.lax.switch(
        combine,
        [_sequential, lambda: _summed(False), lambda: _summed(True), _per_timestep],
    )

    if x_t.shape[1] > action_horizon:
        zero_tail = jnp.zeros_like(x_t[:, action_horizon:, :])
        disp = jnp.concatenate([disp, zero_tail], axis=1)
    # ``disp`` already includes the per-pair alphas (position-space step).
    return v_t + (disp / dt) * guidance_factor

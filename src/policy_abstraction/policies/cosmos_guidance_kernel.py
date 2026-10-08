"""Backend-independent guidance kernel for Cosmos.

Holds the two helpers that don't depend on any cosmos-policy import:

- :func:`pop_guidance_from_options` — splits ``__gp_*__`` keys out of the
  per-call ``options`` dict.
- :func:`apply_guidance_to_x0` — applies the action-space feature loss as
  a gradient-descent step on Cosmos's x0 latent prediction.

Keeping these here lets the project root venv import the kernel for unit
tests (the cosmos venv is needed only for the model subclass that uses
it). The model subclass in :mod:`cosmos_grounded_model` imports both and
re-exports the names so callers can keep their existing imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from policy_steering.params import _BOOL_KEYS, GP_KEYS, GP_SENTINEL_KEY


def pop_guidance_from_options(
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split ``__gp_*__`` keys out of ``options`` if present.

    Returns ``(guidance_params, cleaned_options)``. If no sentinel key is
    present, returns ``(None, options)`` unchanged. Duplicates the helper
    in :mod:`policy_abstraction.policies.groot_grounded_head` so this
    module stays self-contained and can be imported without gr00t.
    """
    if options is None or GP_SENTINEL_KEY not in options:
        return None, options
    guidance_params: dict[str, Any] = {}
    cleaned: dict[str, Any] = {}
    for k, v in options.items():
        if k in GP_KEYS:
            guidance_params[k] = bool(v) if k in _BOOL_KEYS else v
        else:
            cleaned[k] = v
    return guidance_params, (cleaned if cleaned else None)


def apply_guidance_to_x0(
    noise_pred: torch.Tensor,
    *,
    action_latent_idx: int,
    action_horizon: int,
    action_dim: int,
    unnorm_scale: torch.Tensor,
    unnorm_bias: torch.Tensor,
    batch_loss_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    batch_grad_fn: Callable[[torch.Tensor], torch.Tensor],
    guidance_factor: float,
    use_line_search: bool = False,
    debug_guidance: bool = False,
    step_idx: int = 0,
) -> torch.Tensor:
    """Apply action-space guidance to a Cosmos x0 prediction.

    Args:
        noise_pred: x0 estimate from the cosmos denoiser. Shape
            ``(B, C, T', H', W')``, dtype is whatever the network used
            (typically bfloat16).
        action_latent_idx: Index of the action frame in ``T'``.
        action_horizon: Action chunk length (``cfg.chunk_size``; 16 for LIBERO).
        action_dim: Per-step action dimensionality (7 for LIBERO).
        unnorm_scale, unnorm_bias: ``(action_dim,)`` tensors implementing
            ``real = norm * scale + bias``. Differentiable.
        batch_loss_fn, batch_grad_fn: Callables returned by
            :func:`policy_steering.guidance_fns_torch.build_guidance_fns`.
            Operate on action tensors shaped ``(B, action_horizon, action_dim)``.
        guidance_factor: Scalar guidance strength. Multiplies the unit
            descent direction; the latent step is independent of the
            absolute gradient norm.
        use_line_search: Currently a no-op (line search on the latent
            space is a follow-up; the action-space line search in
            ``compute_guidance_velocity`` does not directly apply because
            we descend in latent space, not action space).
        debug_guidance: If True, log per-step loss + grad-norm.
        step_idx: Denoising step index, only used for the debug log.

    Returns:
        A new ``noise_pred`` tensor (same shape/dtype as the input) with
        the gradient-descent correction applied.

    The chain-rule path is::

        leaf (B,C,T',H,W) f32  --extract+unnormalize-->  actions_real (B,Ah,D)
                                                                       |
                                  batch_grad_fn (policy_steering)      v
                                                              g_actions (B,Ah,D)
                                                                       |
                                              torch.autograd.grad      v
                                                              grad_latent (B,C,T',H,W)
    """
    del use_line_search  # placeholder for future line-search support
    # Cast to f32 + detach to break out of any inference_mode-flagged
    # ancestor. The detach() return is a regular tensor on which we can
    # set requires_grad=True regardless of the parent context.
    leaf = noise_pred.detach().to(torch.float32).requires_grad_(True)
    B = leaf.shape[0]

    with torch.enable_grad():
        action_indices = torch.full(
            (B,), action_latent_idx, dtype=torch.long, device=leaf.device
        )
        batch_indices = torch.arange(B, device=leaf.device)
        action_frame = leaf[batch_indices, :, action_indices, :, :]  # (B, C, H, W)
        flat = action_frame.reshape(B, -1)
        num_action_elements = action_horizon * action_dim
        if flat.shape[1] < num_action_elements:
            raise ValueError(
                f"Cosmos action latent frame has {flat.shape[1]} elements but "
                f"action_horizon*action_dim={num_action_elements}. Check "
                f"action_latent_idx={action_latent_idx} and the model config."
            )
        num_action_chunks = flat.shape[1] // num_action_elements
        chunks = flat[:, : num_action_chunks * num_action_elements].reshape(
            B, num_action_chunks, action_horizon, action_dim
        )
        actions_norm = chunks.mean(dim=1)  # (B, action_horizon, action_dim)
        actions_real = actions_norm * unnorm_scale + unnorm_bias  # (B, T, D)

    # Action-space gradient. ``batch_grad_fn`` internally detaches/copies
    # actions_real, so the returned tensor is not on the leaf's graph. We
    # use it as the co-tangent for a VJP that yields the latent gradient.
    g_actions = batch_grad_fn(actions_real)  # (B, T, D), detached f32

    grad_latent = torch.autograd.grad(
        outputs=actions_real,
        inputs=leaf,
        grad_outputs=g_actions,
        create_graph=False,
        retain_graph=False,
    )[0]  # (B, C, T', H, W) f32

    # Unit-normalize per sample. The action loss is roughly invariant to
    # the latent's absolute scale, so a unit-direction step makes the
    # ``guidance_factor`` interpretable across denoising steps.
    grad_flat = grad_latent.reshape(B, -1)
    grad_norm = torch.linalg.norm(grad_flat, dim=-1, keepdim=True)  # (B, 1)
    # Broadcast back to the 5D latent shape.
    shape = (B,) + (1,) * (grad_latent.ndim - 1)
    grad_hat = grad_latent / (grad_norm.reshape(shape) + 1e-8)

    if debug_guidance and batch_loss_fn is not None:
        with torch.no_grad():
            loss_val = batch_loss_fn(actions_real).mean().item()
        # Importing loguru lazily keeps the smoke test free of loguru.
        from loguru import logger

        logger.debug(
            f"[cosmos guidance] step={step_idx} loss={loss_val:.6f} "
            f"|grad_latent|={grad_norm.mean().item():.4e}"
        )

    correction = (guidance_factor * grad_hat).to(noise_pred.dtype)
    return noise_pred - correction

"""Guided variant of Gr00t N1.7's flow-matching action head.

Mirror of openpi's ``Pi0Grounded``: receives ``__gp_*__`` guidance arrays
through the per-call ``options`` dict and steers the Euler integration of
the flow-matching diffusion process toward demo-derived distance
constraints. When ``options`` does not contain guidance, the override
delegates to the upstream parent so behaviour is byte-identical to a
vanilla Gr00t server.

The override is a near-copy of
``gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7ActionHead.get_action_with_features``
with three surgical changes:

1. The whole denoising loop runs without the ``@torch.no_grad()`` decorator
   so we can compute autograd gradients of the guidance loss w.r.t. the
   actions tensor (the network forward passes are still wrapped in
   ``torch.no_grad()`` for efficiency).
2. After the velocity is predicted at each denoising step, we apply
   ``compute_guidance_velocity`` from :mod:`policy_steering.guidance_fns_torch`
   to inject a guidance gradient correction.
3. ``build_guidance_fns`` is hoisted outside the for-loop so we don't
   re-bind tensors per step.

Important: the upstream policy wrapper (``Gr00tPolicy._get_action``) calls
the model under ``torch.inference_mode()``. Because inference-mode tensors
cannot participate in autograd, the corresponding policy-level subclass
(see ``groot_grounded_policy.py``) replaces that with ``torch.no_grad()``.
"""

from __future__ import annotations

from typing import Any

import torch
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from transformers.feature_extraction_utils import BatchFeature

from policy_steering.guidance_fns_torch import (
    build_guidance_fns,
    compute_guidance_velocity,
    compute_guidance_velocity_multi_alpha,
    guidance_capture_enabled,
    guidance_capture_next_inference,
)
from policy_steering.params import _BOOL_KEYS, GP_KEYS, GP_SENTINEL_KEY


def _pop_guidance_from_options(
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split ``__gp_*__`` keys out of ``options`` if present.

    Returns ``(guidance_params, cleaned_options)``. If no sentinel key is
    present, returns ``(None, options)`` unchanged.
    """
    if options is None or GP_SENTINEL_KEY not in options:
        return None, options
    guidance_params: dict[str, Any] = {}
    cleaned: dict[str, Any] = {}
    for k, v in options.items():
        if k in GP_KEYS:
            if k in _BOOL_KEYS:
                guidance_params[k] = bool(v)
            else:
                guidance_params[k] = v
        else:
            cleaned[k] = v
    return guidance_params, (cleaned if cleaned else None)


class Gr00tN1d7GroundedActionHead(Gr00tN1d7ActionHead):
    """Gr00t N1.7 action head that consumes ``__gp_*__`` guidance options."""

    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        guidance_params, options = _pop_guidance_from_options(options)
        if guidance_params is None:
            return super().get_action_with_features(
                backbone_features=backbone_features,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
                action_input=action_input,
                options=options,
            )
        return self._guided_denoise(
            backbone_features=backbone_features,
            state_features=state_features,
            embodiment_id=embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
            guidance_params=guidance_params,
        )

    def _guided_denoise(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None,
        guidance_params: dict[str, Any],
    ) -> BatchFeature:
        vl_embeds = backbone_features

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            intermediate_steps = (
                options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            )
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)
            ramp = ramp[1:-1]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Hoist guidance kernel construction outside the loop.
        guidance_factor = float(guidance_params.get("__gp_guidance_factor__", 1.0))
        use_line_search = bool(guidance_params.get("__gp_line_search__", True))
        use_normalization = bool(guidance_params.get("__gp_normalize__", True))
        max_alpha = float(guidance_params.get("__gp_max_alpha__", 1.0))
        use_multi_alpha = bool(guidance_params.get("__gp_multi_alpha__", False))
        combine_name = {
            0: "sequential",
            1: "sum",
            2: "weighted",
            3: "per_timestep",
        }.get(int(guidance_params.get("__gp_multi_alpha_combine__", 0)), "sequential")
        # Truncate expected distances to the first chunk (matches pi0grounded
        # behaviour: guidance is computed only on the first action_horizon).
        guidance_params = dict(guidance_params)
        if "__gp_expected_dists__" in guidance_params:
            guidance_params["__gp_expected_dists__"] = guidance_params[
                "__gp_expected_dists__"
            ][:, : self.action_horizon]
        # Reuse the actions tensor's dtype/device to avoid bf16/f32 mismatches.
        batch_loss_fn, batch_grad_fn = build_guidance_fns(
            guidance_params, device=device, dtype=actions.dtype
        )
        per_pair_loss_fn = per_pair_grad_fn = None
        per_timestep_loss_fn = per_timestep_grad_fn = None
        capture_inference_idx = None
        # Diagnostic capture only applies to the single-alpha line-search path.
        _capture = guidance_capture_enabled() and not use_multi_alpha
        if use_multi_alpha:
            # Live per-pair + per-timestep fns at the action dtype for the
            # multi-alpha kernel.
            (
                _,
                _,
                per_pair_loss_fn,
                per_pair_grad_fn,
                per_timestep_loss_fn,
                per_timestep_grad_fn,
            ) = build_guidance_fns(
                guidance_params, device=device, dtype=actions.dtype, with_per_pair=True
            )
        elif _capture:
            # Landscape sweep is computed in float32 (separate from the live
            # guidance fns) so true optima aren't quantized away.
            (
                _,
                _,
                per_pair_loss_fn,
                per_pair_grad_fn,
                per_timestep_loss_fn,
                per_timestep_grad_fn,
            ) = build_guidance_fns(
                guidance_params,
                device=device,
                dtype=torch.float32,
                with_per_pair=True,
            )
            capture_inference_idx = guidance_capture_next_inference()

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Network forward: keep grad disabled for efficiency, since we
            # only need autograd through our own loss kernel (acts on
            # ``pred_velocity`` after the network call).
            with torch.no_grad():
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(
                    actions, timesteps_tensor, embodiment_id
                )
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                sa_embs = torch.cat((state_features, action_features), dim=1)

                if self.config.use_alternate_vl_dit:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    )
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                    )
                pred = self.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -self.action_horizon :]

            # Apply guidance correction *before* the vel_strength multiply so
            # RTC's frozen-step mask still zeros it out where appropriate.
            if use_multi_alpha:
                pred_velocity = compute_guidance_velocity_multi_alpha(
                    batch_loss_fn,
                    per_pair_loss_fn,
                    per_pair_grad_fn,
                    x_t=actions,
                    v_t=pred_velocity,
                    dt=dt,
                    action_horizon=self.action_horizon,
                    guidance_factor=guidance_factor,
                    use_normalization=use_normalization,
                    max_alpha=max_alpha,
                    combine=combine_name,
                    guidance_weights=guidance_params.get("__gp_guidance_weights__"),
                    per_timestep_loss_fn=per_timestep_loss_fn,
                    per_timestep_grad_fn=per_timestep_grad_fn,
                )
            else:
                pred_velocity = compute_guidance_velocity(
                    batch_loss_fn,
                    batch_grad_fn,
                    x_t=actions,
                    v_t=pred_velocity,
                    dt=dt,
                    action_horizon=self.action_horizon,
                    guidance_factor=guidance_factor,
                    use_line_search=use_line_search,
                    use_normalization=use_normalization,
                    max_alpha=max_alpha,
                    per_pair_loss_fn=per_pair_loss_fn,
                    per_pair_grad_fn=per_pair_grad_fn,
                    capture_inference_idx=capture_inference_idx,
                    capture_denoise_step=t,
                    capture_num_steps=self.num_inference_timesteps,
                    capture_guidance_params=guidance_params if _capture else None,
                )

            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

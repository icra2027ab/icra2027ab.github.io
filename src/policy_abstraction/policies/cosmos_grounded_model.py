"""Guided variant of Cosmos's ``CosmosPolicyVideo2WorldModel``.

Mirror of :mod:`policy_abstraction.policies.groot_grounded_head`: receives
``__gp_*__`` guidance arrays through ``self._pa_guidance_params`` (stashed
on the model instance by ``cosmos_grounded_runtime.run_guided_get_action``)
and steers the FlowUniPC denoising loop toward demo-derived distance
constraints. When the stash is absent, the override falls through to the
upstream parent so behaviour is byte-identical to a vanilla Cosmos run.

Three architectural quirks of Cosmos drive this file:

1. Cosmos denoises VAE latents, not actions. The action chunk lives at
   ``latent[:, :, action_latent_idx, :, :]`` and is recovered by a
   flatten/reshape/mean ([cosmos_utils.extract_action_chunk_from_latent_sequence]).
   Those operations are linear, so a feature-space loss on the recovered
   chunk has a well-defined VJP onto the latent.

2. The upstream FlowUniPC loop sits inside ``with torch.no_grad():`` *and*
   the surrounding ``get_action`` body sits inside
   ``torch.inference_mode()``. To enable autograd inside our hook we
   reimplement the FlowUniPC body without the outer no_grad wrapper and
   rely on the runtime helper to neutralise inference_mode at the
   ``get_action`` boundary.

3. The action chunk emerges in *normalized* training space; the
   policy_steering feature loss is in real action space. The runtime
   helper stashes ``dataset_stats`` so we can compose a differentiable
   affine unnormalize inside the hook.

Sign convention: FlowUniPC's ``sample_scheduler.step(noise_pred, ...)``
consumes ``noise_pred`` as an *x0 prediction* (clean-latent estimate),
not a velocity. The descent step on ``noise_pred`` is therefore
``noise_pred - guidance_factor * grad_hat`` — the cosmos analog of GR00T's
``v_t - guidance_factor * grad_hat`` (positive ``dt`` convention).
"""

from __future__ import annotations

from typing import Any

import torch
import tqdm
from cosmos_policy._src.imaginaire.flags import INTERNAL
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy._src.imaginaire.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
    find_split,
)
from cosmos_policy._src.predict2.models.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from einops import rearrange

from policy_abstraction.policies.cosmos_guidance_kernel import (
    apply_guidance_to_x0,
    pop_guidance_from_options,
)
from policy_steering.guidance_fns_torch import build_guidance_fns

# Re-export under the leading-underscore name kept by serve_cosmos.py.
_pop_guidance_from_options = pop_guidance_from_options


class CosmosPolicyVideo2WorldModelGrounded(CosmosPolicyVideo2WorldModel):
    """``CosmosPolicyVideo2WorldModel`` with in-loop action-space guidance.

    The runtime helper sets the following attributes on the instance
    before calling ``generate_samples_from_batch``:

    - ``_pa_guidance_params``: dict of ``__gp_*__`` arrays (or ``None``).
    - ``_pa_dataset_stats``: cosmos dataset_stats dict (for unnormalize).
    - ``_pa_action_horizon``: int, number of action steps to recover
      (``cfg.chunk_size``; 16 for LIBERO).
    - ``_pa_action_dim``: int, action dimensionality (7 for LIBERO).

    ``action_latent_idx`` is read directly from ``data_batch`` (cosmos's
    own ``get_action`` puts it there at line 1082 of ``cosmos_utils``),
    so the runtime helper does not need to stash it.

    When ``_pa_guidance_params`` is ``None`` (or missing), the override
    delegates to the parent and behaviour is byte-identical to a vanilla
    Cosmos forward pass.
    """

    _pa_guidance_params: dict[str, Any] | None = None
    _pa_dataset_stats: dict[str, Any] | None = None
    _pa_action_horizon: int = 16
    _pa_action_dim: int = 7

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: str = "2ab",
        x_sigma_max: torch.Tensor | None = None,
        sigma_max: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        guidance_params = self._pa_guidance_params
        if guidance_params is None:
            return super().generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                is_negative_prompt=is_negative_prompt,
                num_steps=num_steps,
                solver_option=solver_option,
                x_sigma_max=x_sigma_max,
                sigma_max=sigma_max,
                **kwargs,
            )
        if not self.config.use_flowunipc_scheduler:
            raise NotImplementedError(
                "Guided cosmos sampling currently only supports the FlowUniPC "
                "scheduler (use_flowunipc_scheduler=True). The Cosmos-Policy "
                "LIBERO checkpoint enables it by default; if you hit this, "
                "either fall back to unguided inference or extend "
                "_guided_generate_flowunipc to cover the EDM sampler path."
            )
        samples, orig_clean_latent_frames = self._guided_generate_flowunipc(
            data_batch,
            guidance=guidance,
            seed=seed,
            state_shape=state_shape,
            n_sample=n_sample,
            is_negative_prompt=is_negative_prompt,
            num_steps=num_steps,
            guidance_params=guidance_params,
        )
        # get_action unpacks (samples, orig_clean_latent_frames) when it passes
        # return_orig_clean_latent_frames=True; mirror the parent's contract.
        if kwargs.get("return_orig_clean_latent_frames", False):
            return samples, orig_clean_latent_frames
        return samples

    # ------------------------------------------------------------------
    # FlowUniPC denoising loop with action-space guidance
    # ------------------------------------------------------------------

    def _guided_generate_flowunipc(
        self,
        data_batch: dict,
        guidance: float,
        seed: int,
        state_shape: tuple | None,
        n_sample: int | None,
        is_negative_prompt: bool,
        num_steps: int,
        guidance_params: dict[str, Any],
    ) -> torch.Tensor:
        """Near-copy of the FlowUniPC branch in
        :meth:`CosmosPolicyVideo2WorldModel.generate_samples_from_batch`,
        with three surgical changes:

        1. The outer ``with torch.no_grad():`` wrapper is dropped so we can
           run autograd on the action-space loss; only the heavy network
           forward (``x0_fn``) stays inside ``no_grad`` for efficiency.
        2. After each ``x0_fn`` call we route ``noise_pred`` through
           :func:`_apply_guidance_to_x0` which adds a descent step in
           latent space corresponding to the action-space feature loss.
        3. The guidance kernel is constructed once outside the loop.

        The caller (``run_guided_get_action``) is responsible for ensuring
        ``torch.inference_mode`` is off; otherwise the autograd-enabled
        sections inside this method will not be able to allocate
        gradient-tracked tensors.
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        sample_scheduler.set_timesteps(
            num_steps, device=self.tensor_kwargs["device"], shift=5
        )
        timesteps = sample_scheduler.timesteps

        # Network forward stays no_grad for efficiency; only our guidance
        # hook needs autograd, and it owns its own enable_grad block.
        with torch.no_grad():
            _x0_fn_upstream = self.get_x0_fn_from_batch(
                data_batch, guidance, is_negative_prompt=is_negative_prompt
            )
        # Upstream's FlowUniPC ``x0_fn`` unconditionally evaluates
        # ``denoise_with_velocity(..., uncondition)``, but this checkpoint does
        # always-conditional generation (``uncondition is None``), so that path
        # raises ``'NoneType' object has no attribute 'is_video'``. Rebuild the
        # velocity ``x0_fn`` from the upstream closure — reusing all of its
        # (intricate) condition construction — and guard the CFG combine on
        # ``uncondition``, mirroring upstream's own EDM branch.
        _free = dict(
            zip(
                _x0_fn_upstream.__code__.co_freevars,
                (c.cell_contents for c in (_x0_fn_upstream.__closure__ or ())),
            )
        )
        _condition = _free["condition"]
        _uncondition = _free.get("uncondition")
        _cfg_scale = _free.get("guidance", guidance)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_velocity = self.denoise_with_velocity(noise_x, sigma, _condition)
            if _uncondition is None:
                return cond_velocity
            uncond_velocity = self.denoise_with_velocity(noise_x, sigma, _uncondition)
            return uncond_velocity + _cfg_scale * (cond_velocity - uncond_velocity)

        latents = noise

        use_spatial_split = False
        after_split_shape = None
        if self.net.is_context_parallel_enabled:
            cp_size = len(
                torch.distributed.get_process_group_ranks(
                    self.get_context_parallel_group()
                )
            )
            use_spatial_split = (
                cp_size > latents.shape[2] or latents.shape[2] % cp_size != 0
            )
            after_split_shape = (
                find_split(latents.shape, cp_size) if use_spatial_split else None
            )
            if use_spatial_split:
                latents = rearrange(latents, "b c t h w -> b c (t h w)")
                latents = broadcast_split_tensor(
                    latents, seq_dim=2, process_group=self.get_context_parallel_group()
                )
                latents = rearrange(
                    latents,
                    "b c (t h w) -> b c t h w",
                    t=after_split_shape[0],
                    h=after_split_shape[1],
                )

        # Hoist guidance kernel out of the denoising loop. Truncate the
        # reference trajectory to ``action_horizon`` to mirror pi0/groot:
        # guidance is only meaningful inside the first action chunk.
        gp = dict(guidance_params)
        if "__gp_expected_dists__" in gp:
            gp["__gp_expected_dists__"] = gp["__gp_expected_dists__"][
                :, : self._pa_action_horizon
            ]
        device = self.tensor_kwargs["device"]
        batch_loss_fn, batch_grad_fn = build_guidance_fns(
            gp, device=device, dtype=torch.float32
        )
        guidance_factor = float(gp.get("__gp_guidance_factor__", 1.0))
        use_line_search = bool(gp.get("__gp_line_search__", False))
        debug_guidance = bool(gp.get("__gp_debug_guidance__", False))

        # Pre-compute unnormalize constants on-device as tensors.
        ds = self._pa_dataset_stats
        if ds is None:
            raise RuntimeError(
                "Guided cosmos sampling requires self._pa_dataset_stats to be "
                "set before generate_samples_from_batch is invoked. The runtime "
                "helper run_guided_get_action does this automatically."
            )
        actions_min = torch.as_tensor(
            ds["actions_min"], device=device, dtype=torch.float32
        )
        actions_max = torch.as_tensor(
            ds["actions_max"], device=device, dtype=torch.float32
        )
        unnorm_scale = 0.5 * (actions_max - actions_min)
        unnorm_bias = 0.5 * (actions_max + actions_min)

        # Cosmos's get_action stashes the action-frame index in data_batch.
        # All batch entries share the same value in our single-suite setup.
        if "action_latent_idx" not in data_batch:
            raise KeyError(
                "data_batch is missing 'action_latent_idx'; cosmos's get_action "
                "always sets this. Did you call generate_samples_from_batch "
                "outside the standard inference pipeline?"
            )
        action_latent_idx = int(data_batch["action_latent_idx"][0].item())

        if INTERNAL:
            timesteps_iter = timesteps
        else:
            timesteps_iter = tqdm.tqdm(
                timesteps, desc="Generating samples (guided)", total=len(timesteps)
            )

        for step_idx, t in enumerate(timesteps_iter):
            latent_model_input = latents
            timestep = torch.stack([t]) / 1000

            # Network forward: stay grad-free; we only differentiate the
            # local extraction inside apply_guidance_to_x0.
            with torch.no_grad():
                noise_pred = x0_fn(latent_model_input, timestep.unsqueeze(0))

            noise_pred = apply_guidance_to_x0(
                noise_pred,
                action_latent_idx=action_latent_idx,
                action_horizon=self._pa_action_horizon,
                action_dim=self._pa_action_dim,
                unnorm_scale=unnorm_scale,
                unnorm_bias=unnorm_bias,
                batch_loss_fn=batch_loss_fn,
                batch_grad_fn=batch_grad_fn,
                guidance_factor=guidance_factor,
                use_line_search=use_line_search,
                debug_guidance=debug_guidance,
                step_idx=step_idx,
            )

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            if use_spatial_split:
                latents = rearrange(latents, "b c t h w -> b c (t h w)")
            latents = cat_outputs_cp(
                latents, seq_dim=2, cp_group=self.get_context_parallel_group()
            )
            if use_spatial_split:
                latents = rearrange(
                    latents,
                    "b c (t h w) -> b c t h w",
                    t=state_shape[1],
                    h=state_shape[2],
                )
        # Mirror the parent FlowUniPC path's return contract: the original
        # (pre-injection) clean latent frames accompany the generated latents.
        return latents, _condition.orig_gt_frames

"""Runtime entry point for guided Cosmos action inference.

The vanilla Cosmos inference path runs the whole pipeline inside
``torch.inference_mode()`` (see
``cosmos_policy.experiments.robot.cosmos_utils.get_action`` line 887).
Inference-mode tagging is permanent on tensors and prevents autograd from
operating on them, so the gradient-based guidance hook in
:mod:`cosmos_grounded_model` would never see a usable graph.

To enable guidance without duplicating the ~250-line preprocessing body of
``get_action``, this module surgically swaps ``torch.inference_mode`` for
``torch.no_grad`` for the duration of one call. ``no_grad`` disables
gradient *tracking* on outputs but still allows autograd to be re-enabled
locally (``with torch.enable_grad():``), which is exactly what
``_apply_guidance_to_x0`` needs.

The model class is expected to be a
:class:`CosmosPolicyVideo2WorldModelGrounded` instance (or any subclass
that reads ``self._pa_guidance_params`` and friends); ``serve_cosmos.py``
performs the ``model.__class__ = ...`` rebind right after model load.

This is the only file in the integration that has to touch a torch global.
We keep it contained to a single function so the surface is auditable.
"""

from __future__ import annotations

from typing import Any

import torch

# Import cosmos_utils for its preprocessing + sampling pipeline. The
# import is local to this module so policy_abstraction code that lives
# *outside* the cosmos venv (e.g. evaluation/, episode_runner) is not
# affected at import time.
from cosmos_policy.experiments.robot import cosmos_utils as _cu


def run_guided_get_action(
    cfg,
    model: torch.nn.Module,
    dataset_stats: dict,
    obs: dict[str, Any],
    task_label_or_embedding: Any,
    *,
    guidance_params: dict[str, Any],
    seed: int = 1,
    randomize_seed: bool = False,
    num_denoising_steps_action: int = 5,
    generate_future_state_and_value_in_parallel: bool = False,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run cosmos ``get_action`` with action-space guidance enabled.

    Args:
        cfg: Same ``SimpleNamespace`` ``get_action`` expects.
        model: A ``CosmosPolicyVideo2WorldModelGrounded`` instance (the
            class rebind happens in ``serve_cosmos.py``).
        dataset_stats: Action/proprio min/max for un/normalize.
        obs: Per-step observation dict (primary_image, wrist_image,
            proprio, task_description).
        task_label_or_embedding: Either the task-description string or a
            cached T5 embedding ndarray.
        guidance_params: ``__gp_*__`` dict already split out from the
            wire-level ``options`` by the server. Must contain
            ``__gp_expected_dists__`` (the sentinel).
        seed, randomize_seed, num_denoising_steps_action: Same as the
            upstream ``get_action`` kwargs.
        generate_future_state_and_value_in_parallel, batch_size: Pinned to
            their ``serve_cosmos.py`` defaults; we don't expose
            future-state extraction in v1 because the server doesn't use
            it.

    Returns:
        The same dict shape ``get_action`` returns (``actions`` plus
        bookkeeping keys), so the caller can treat it interchangeably
        with the unguided path.
    """
    # Stash guidance state on the model. The overridden
    # ``generate_samples_from_batch`` reads these to decide whether to
    # take the guided branch and how to perform the latent → action
    # unnormalize.
    model._pa_guidance_params = guidance_params
    model._pa_dataset_stats = dataset_stats
    model._pa_action_horizon = int(cfg.chunk_size)
    # Cosmos uses a hardcoded 7-DoF action layout for LIBERO; reuse the
    # module-level constant rather than redefining it.
    model._pa_action_dim = int(_cu.ACTION_DIM)

    # The grounded guided denoising loop is only implemented for the
    # FlowUniPC scheduler (see
    # CosmosPolicyVideo2WorldModelGrounded.generate_samples_from_batch).
    # The LIBERO inference config ships with it OFF, so enable it for the
    # duration of this guided call and restore it afterward; the unguided
    # path keeps the config's original setting.
    _orig_flowunipc = model.config.use_flowunipc_scheduler
    model.config.use_flowunipc_scheduler = True

    # Swap inference_mode → no_grad for the duration of this call. This
    # lets autograd be locally re-enabled inside ``_apply_guidance_to_x0``
    # without lifting any tensor out of an inference-mode flag.
    _orig_inference_mode = torch.inference_mode
    torch.inference_mode = torch.no_grad  # type: ignore[assignment]
    try:
        result = _cu.get_action(
            cfg,
            model,
            dataset_stats,
            obs,
            task_label_or_embedding=task_label_or_embedding,
            seed=seed,
            randomize_seed=randomize_seed,
            num_denoising_steps_action=num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=generate_future_state_and_value_in_parallel,
            batch_size=batch_size,
        )
    finally:
        torch.inference_mode = _orig_inference_mode  # type: ignore[assignment]
        model.config.use_flowunipc_scheduler = _orig_flowunipc
        # Clear the stash so a subsequent unguided call goes through the
        # parent class unchanged. ``_pa_dataset_stats`` and the action
        # shape attrs can stay; they're inert when _pa_guidance_params is
        # None.
        model._pa_guidance_params = None

    return result

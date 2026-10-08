"""Guided variant of ``Gr00tPolicy``.

Mirrors upstream ``Gr00tPolicy._get_action`` byte-for-byte except for two
surgical changes required to plumb guidance into the flow-matching head:

1. ``torch.inference_mode()`` → ``torch.no_grad()`` so autograd can track
   the guidance loss through the actions tensor inside the head's denoising
   loop. Inference-mode tensors are flagged in a way that prevents them
   from participating in autograd at all, even after exiting the context.
2. ``self.model.get_action(**collated_inputs)`` →
   ``self.model.get_action(**collated_inputs, options=options)`` so the
   ``__gp_*__`` keys in ``options`` reach the action head.

When ``options`` does not contain guidance keys, the grounded action head
delegates to the upstream parent and behaviour is byte-identical to the
unguarded path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


class Gr00tGroundedPolicy(Gr00tPolicy):
    """``Gr00tPolicy`` with options forwarded and ``no_grad`` instead of ``inference_mode``."""

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [
                {"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}
            ]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: forward options into the model so the grounded action head
        # can consume ``__gp_*__`` keys. Use ``no_grad`` (not ``inference_mode``)
        # so the head can locally re-enable grad for its guidance kernel.
        with torch.no_grad():
            model_pred = self.model.get_action(**collated_inputs, options=options)
        normalized_action = model_pred["action_pred"].float()

        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

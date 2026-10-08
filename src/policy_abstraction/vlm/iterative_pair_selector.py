"""VLM client for the iterative pre/post-condition strategy.

Mirrors :class:`policy_abstraction.vlm.task_pair_selector.TaskPairSelector` but
with a different prompt and an explicit ``refine`` method that appends
follow-up turns to an existing conversation.

The legacy :class:`TaskPairSelector` is deliberately untouched so the
``single_turn`` strategy stays bit-for-bit identical to the pre-strategy
pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from policy_abstraction.vlm._client import make_vlm_client
from policy_abstraction.vlm._retry import chat_completion_with_retry
from policy_abstraction.vlm.task_pair_selector import (
    ConstraintValidationError,
    TaskPairSelector,
)
from policy_abstraction.vlm.types import (
    ROBOT_ENTITIES,
    DistancePair,
    DistanceType,
    SubtrajConstraint,
    VLMConversation,
    VLMMessage,
)


class IterativePairSelector:
    """Stateless wrapper that runs one VLM proposal turn at a time."""

    _SYSTEM_PROMPT = (
        Path(__file__).parent / "prompts" / "iterative_pair_selector_system.txt"
    ).read_text()

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        max_validation_retries: int = 2,
        base_url: str | None = None,
    ) -> None:
        self._client = make_vlm_client(api_key, base_url)
        self._model = model
        self._max_validation_retries = max_validation_retries

    # ---- Initial proposal ---------------------------------------------------

    def propose_initial(
        self,
        task: str,
        objects: dict[int, str],
        scene_image: np.ndarray | None = None,
        seg_index_image: np.ndarray | None = None,
    ) -> tuple[
        list[DistancePair], list[SubtrajConstraint], VLMConversation, list[dict], str
    ]:
        """Run the first VLM turn and return parsed results.

        Returns
        -------
        ``(pairs, constraints, conversation, openai_messages, assistant_text)``
        where ``openai_messages`` is the OpenAI chat-format list (so the
        caller can keep extending it across refinement turns) and
        ``assistant_text`` is the raw VLM response.
        """
        objects_str = "\n".join(
            f"  {id_}: {name}" for id_, name in sorted(objects.items())
        )
        text_content = f"Task: {task}\n\nAvailable objects (id: name):\n{objects_str}"

        user_content: list[dict] = []
        for img in (scene_image, seg_index_image):
            if img is not None:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{TaskPairSelector._encode_image(img)}",
                            "detail": "low",
                        },
                    }
                )
        user_content.append({"type": "text", "text": text_content})

        messages: list[dict] = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        conv_messages: list[VLMMessage] = [
            VLMMessage(role="system", text=self._SYSTEM_PROMPT),
            VLMMessage(
                role="user",
                text=text_content,
                images=[
                    img for img in (scene_image, seg_index_image) if img is not None
                ],
            ),
        ]

        pairs, constraints, assistant_text = self._call_with_validation(
            messages, conv_messages
        )
        conversation = VLMConversation(messages=conv_messages, model=self._model)
        return pairs, constraints, conversation, messages, assistant_text

    # ---- Refinement turn ----------------------------------------------------

    def refine(
        self,
        prev_messages: list[dict],
        conversation: VLMConversation,
        feedback_text: str,
    ) -> tuple[
        list[DistancePair], list[SubtrajConstraint], VLMConversation, list[dict], str
    ]:
        """Send a follow-up user message and parse the new proposal."""
        messages = list(prev_messages)
        messages.append({"role": "user", "content": feedback_text})
        new_conv_messages = list(conversation.messages)
        new_conv_messages.append(VLMMessage(role="user", text=feedback_text))

        pairs, constraints, assistant_text = self._call_with_validation(
            messages, new_conv_messages
        )
        new_conversation = VLMConversation(
            messages=new_conv_messages, model=self._model
        )
        return pairs, constraints, new_conversation, messages, assistant_text

    # ---- Internals ----------------------------------------------------------

    def _call_with_validation(
        self,
        openai_messages: list[dict],
        conv_messages: list[VLMMessage],
    ) -> tuple[list[DistancePair], list[SubtrajConstraint], str]:
        """Call the VLM, parse, validate, and retry on validation failures.

        Mutates ``openai_messages`` and ``conv_messages`` in place to reflect
        the (possibly multi-turn) regeneration trace.
        """
        pairs: list[DistancePair] = []
        constraints: list[SubtrajConstraint] = []
        errors: list[tuple[SubtrajConstraint, str]] = []
        assistant_text = ""

        for attempt in range(self._max_validation_retries + 1):
            response = chat_completion_with_retry(
                self._client,
                log_prefix="IterativePairSelector",
                model=self._model,
                messages=openai_messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                temperature=0,
            )
            assistant_text = response.choices[0].message.content or ""
            conv_messages.append(VLMMessage(role="assistant", text=assistant_text))

            data = json.loads(assistant_text)
            raw_subtrajs = data.get(
                "subtrajectories", list(data.values())[0] if data else []
            )
            pairs, constraints = self._parse_iterative(raw_subtrajs)
            errors = TaskPairSelector._validate_constraints(constraints)
            if not errors:
                break

            feedback = TaskPairSelector._format_validation_feedback(errors)
            logger.warning(
                "IterativePairSelector: constraint validation failed on attempt "
                f"{attempt + 1}/{self._max_validation_retries + 1} "
                f"({len(errors)} error(s)); requesting regeneration."
            )
            openai_messages.append({"role": "assistant", "content": assistant_text})
            openai_messages.append({"role": "user", "content": feedback})
            conv_messages.append(VLMMessage(role="user", text=feedback))

        if errors:
            details = "\n".join(f"  - [{c.kind}] {c.label}: {msg}" for c, msg in errors)
            raise ConstraintValidationError(
                "VLM produced invalid constraints after "
                f"{self._max_validation_retries + 1} attempts:\n{details}"
            )
        return pairs, constraints, assistant_text

    @staticmethod
    def _parse_iterative(
        raw_subtrajs: list[dict],
    ) -> tuple[list[DistancePair], list[SubtrajConstraint]]:
        """Parse the iterative JSON schema: pre/post + pairs + optional path."""
        pairs: list[DistancePair] = []
        constraints: list[SubtrajConstraint] = []

        for subtraj_block in raw_subtrajs:
            raw_idx = subtraj_block.get("subtraj_idx")
            subtraj_idx: int | None = int(raw_idx) if raw_idx is not None else None
            reasoning: str | None = subtraj_block.get("reasoning")
            label_prefix = subtraj_block.get("label", f"st{subtraj_idx}")

            for item in subtraj_block.get("pairs", []):
                pairs.append(_parse_pair(item, subtraj_idx, reasoning))

            if subtraj_idx is not None:
                pre = subtraj_block.get("pre_condition")
                post = subtraj_block.get("post_condition")
                if pre is not None:
                    constraints.append(
                        SubtrajConstraint(
                            kind="sub_goal",
                            subtraj_idx=subtraj_idx,
                            label=pre.get("label", f"{label_prefix}_pre"),
                            description=pre.get("description", ""),
                            expression=pre["expression"],
                            pairs_affected=[],
                            reasoning=pre.get("reasoning"),
                        )
                    )
                    # Tag with _pre suffix so consistency.py can pick it out.
                    if not constraints[-1].label.endswith("_pre"):
                        constraints[-1].label = f"{constraints[-1].label}_pre"
                if post is not None:
                    constraints.append(
                        SubtrajConstraint(
                            kind="sub_goal",
                            subtraj_idx=subtraj_idx,
                            label=post.get("label", f"{label_prefix}_post"),
                            description=post.get("description", ""),
                            expression=post["expression"],
                            pairs_affected=[],
                            reasoning=post.get("reasoning"),
                        )
                    )
                    if not constraints[-1].label.endswith("_post"):
                        constraints[-1].label = f"{constraints[-1].label}_post"

                for raw_c in subtraj_block.get("path_constraints", []) or []:
                    constraints.append(
                        SubtrajConstraint(
                            kind="path",
                            subtraj_idx=subtraj_idx,
                            label=raw_c["label"],
                            description=raw_c.get("description", ""),
                            expression=raw_c["expression"],
                            pairs_affected=list(raw_c.get("pairs_affected", [])),
                            reasoning=raw_c.get("reasoning"),
                        )
                    )

        return pairs, constraints


def _parse_pair(
    item: dict, subtraj_idx: int | None, reasoning: str | None
) -> DistancePair:
    from_e = item["from_entity"]
    to_e = item["to_entity"]
    if isinstance(from_e, str) and from_e not in ROBOT_ENTITIES:
        from_e = int(from_e)
    if isinstance(to_e, str) and to_e not in ROBOT_ENTITIES:
        to_e = int(to_e)

    distance_type = DistanceType(item.get("distance_type", DistanceType.CENTROID))
    raw_axis = item.get("axis")
    axis = [float(v) for v in raw_axis] if raw_axis is not None else None
    raw_plane = item.get("plane")
    plane = [float(v) for v in raw_plane] if raw_plane is not None else None
    raw_ee = item.get("ee_vector")
    ee_vector = [float(v) for v in raw_ee] if raw_ee is not None else None

    return DistancePair(
        from_entity=from_e,
        to_entity=to_e,
        label=item["label"],
        distance_type=distance_type,
        axis=axis,
        plane=plane,
        ee_vector=ee_vector,
        subtraj_idx=subtraj_idx,
        reasoning=reasoning,
    )

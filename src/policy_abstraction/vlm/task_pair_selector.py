"""VLM-based selection of informative distance pairs for manipulation tasks."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from policy_abstraction.vlm._client import make_vlm_client
from policy_abstraction.vlm._orientation import to_vlm_orientation
from policy_abstraction.vlm._retry import chat_completion_with_retry
from policy_abstraction.vlm.types import (
    ROBOT_ENTITIES,
    DistancePair,
    DistanceType,
    SubtrajConstraint,
    VLMConversation,
    VLMMessage,
)


class _AnyKeypoints(dict):
    """Dict that returns a zero 3-vector for any missing key (used for validation)."""

    def __missing__(self, key):  # type: ignore[override]
        return np.zeros(3)


class ConstraintValidationError(RuntimeError):
    """Raised when the VLM cannot produce syntactically valid constraints."""


class TaskPairSelector:
    """Queries an LLM to choose the most informative distance pairs for a task."""

    _SYSTEM_PROMPT = (
        Path(__file__).parent / "prompts" / "task_pair_selector_system.txt"
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

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        """Encode a uint8 numpy image (H, W, 3) to a base64 PNG string.

        The PA reference is 180°-rotated relative to scene-frame; we apply
        :func:`to_vlm_orientation` here so every VLM call sees a
        scene-consistent view (image-left == scene-left).
        """
        pil_img = Image.fromarray(to_vlm_orientation(img).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def select_pairs(
        self,
        task: str,
        objects: dict[int, str],
        scene_image: np.ndarray | None = None,
        seg_index_image: np.ndarray | None = None,
        subtrajectories: list[tuple[int, int]] | None = None,
        subtraj_image: np.ndarray | None = None,
    ) -> tuple[list[DistancePair], list[SubtrajConstraint], VLMConversation]:
        """Return relevant distance pairs and constraints for *task*.

        Args:
            task:             Natural-language task description.
            objects:          Mapping from segmentation ID to object name.
            scene_image:      Optional ``[H, W, 3]`` uint8 RGB image of the scene.
            seg_index_image:  Optional ``[H, W, 3]`` uint8 RGB segmentation overlay
                              with instance IDs labelled per region.
            subtrajectories:  Optional list of ``(start, end)`` index pairs defining
                              sub-trajectory segments (end is exclusive).
            subtraj_image:    Optional ``[H, W, 3]`` uint8 RGB image with EE trajectory
                              paths overlaid, each sub-trajectory in a distinct bright
                              colour and labelled ST0, ST1, …

        Returns:
            A 3-tuple ``(pairs, constraints, conversation)`` where *pairs* is a
            list of :class:`DistancePair`, *constraints* is a list of
            :class:`SubtrajConstraint` (sub-goal and path), and *conversation*
            is the full :class:`VLMConversation` for inspection.
        """
        objects_str = "\n".join(
            f"  {id_}: {name}" for id_, name in sorted(objects.items())
        )

        n_subtrajs = len(subtrajectories) if subtrajectories else 0
        subtraj_info = ""
        if subtrajectories:
            lines = [f"\n\nNumber of sub-trajectories: {n_subtrajs}"]
            for i, (s, e) in enumerate(subtrajectories):
                lines.append(f"  ST{i}: timesteps {s}–{e - 1}")
            subtraj_info = "\n".join(lines)

        text_content = f"Task: {task}\n\nAvailable objects (id: name):\n{objects_str}{subtraj_info}"

        user_content: list[dict] = []
        if scene_image is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(scene_image)}",
                        "detail": "low",
                    },
                }
            )
        if seg_index_image is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(seg_index_image)}",
                        "detail": "low",
                    },
                }
            )
        if subtraj_image is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(subtraj_image)}",
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
                    img
                    for img in [scene_image, seg_index_image, subtraj_image]
                    if img is not None
                ],
            ),
        ]

        pairs: list[DistancePair] = []
        constraints: list[SubtrajConstraint] = []
        errors: list[tuple[SubtrajConstraint, str]] = []

        for attempt in range(self._max_validation_retries + 1):
            response = chat_completion_with_retry(
                self._client,
                log_prefix="TaskPairSelector",
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                temperature=0,
            )
            assistant_text = response.choices[0].message.content or ""
            conv_messages.append(VLMMessage(role="assistant", text=assistant_text))

            data = json.loads(assistant_text)
            raw_subtrajs = data.get(
                "subtrajectories", list(data.values())[0] if data else []
            )
            pairs, constraints = self._parse_subtrajectories(raw_subtrajs)
            errors = self._validate_constraints(constraints)
            if not errors:
                break

            feedback = self._format_validation_feedback(errors)
            logger.warning(
                "TaskPairSelector: constraint validation failed on attempt "
                f"{attempt + 1}/{self._max_validation_retries + 1} "
                f"({len(errors)} error(s)); requesting regeneration."
            )
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": feedback})
            conv_messages.append(VLMMessage(role="user", text=feedback))

        if errors:
            details = "\n".join(f"  - [{c.kind}] {c.label}: {msg}" for c, msg in errors)
            raise ConstraintValidationError(
                "VLM produced invalid constraints after "
                f"{self._max_validation_retries + 1} attempts:\n{details}"
            )

        conversation = VLMConversation(messages=conv_messages, model=self._model)
        return pairs, constraints, conversation

    @staticmethod
    def _parse_subtrajectories(
        raw_subtrajs: list[dict],
    ) -> tuple[list[DistancePair], list[SubtrajConstraint]]:
        """Parse the VLM JSON response into typed pairs and constraints."""
        pairs: list[DistancePair] = []
        constraints: list[SubtrajConstraint] = []
        for subtraj_block in raw_subtrajs:
            raw_subtraj = subtraj_block.get("subtraj_idx")
            subtraj_idx: int | None = (
                int(raw_subtraj) if raw_subtraj is not None else None
            )
            reasoning: str | None = subtraj_block.get("reasoning")

            for item in subtraj_block.get("pairs", []):
                from_e = item["from_entity"]
                to_e = item["to_entity"]
                if isinstance(from_e, str) and from_e not in ROBOT_ENTITIES:
                    from_e = int(from_e)
                if isinstance(to_e, str) and to_e not in ROBOT_ENTITIES:
                    to_e = int(to_e)

                distance_type = DistanceType(
                    item.get("distance_type", DistanceType.CENTROID)
                )
                raw_axis = item.get("axis")
                axis: list[float] | None = (
                    [float(v) for v in raw_axis] if raw_axis is not None else None
                )
                raw_plane = item.get("plane")
                plane: list[float] | None = (
                    [float(v) for v in raw_plane] if raw_plane is not None else None
                )
                raw_ee_vec = item.get("ee_vector")
                ee_vector: list[float] | None = (
                    [float(v) for v in raw_ee_vec] if raw_ee_vec is not None else None
                )

                pairs.append(
                    DistancePair(
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
                )

            if subtraj_idx is not None:
                for raw_c in subtraj_block.get("sub_goal_constraints", []):
                    constraints.append(
                        SubtrajConstraint(
                            kind="sub_goal",
                            subtraj_idx=subtraj_idx,
                            label=raw_c["label"],
                            description=raw_c.get("description", ""),
                            expression=raw_c["expression"],
                            pairs_affected=[],
                            reasoning=raw_c.get("reasoning"),
                        )
                    )
                for raw_c in subtraj_block.get("path_constraints", []):
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

    @staticmethod
    def _validate_constraints(
        constraints: list[SubtrajConstraint],
    ) -> list[tuple[SubtrajConstraint, str]]:
        """Check that each constraint compiles and runs on dummy inputs.

        Returns a list of ``(constraint, error_message)`` for failing entries;
        empty if all constraints are valid. Validation mirrors the runtime
        sandbox used by :meth:`SubtrajConstraint.compile`, so any failure here
        would also fail at evaluation time.
        """
        dummy_ee = np.zeros(3)
        dummy_kp = _AnyKeypoints()
        dummy_ee_R = np.eye(3)
        errors: list[tuple[SubtrajConstraint, str]] = []
        for c in constraints:
            try:
                fn = c.compile()
            except Exception as exc:
                errors.append((c, f"failed to compile: {exc}"))
                continue
            try:
                result = fn(dummy_ee, dummy_kp, dummy_ee_R)
            except Exception as exc:
                errors.append((c, f"raised {type(exc).__name__} when called: {exc}"))
                continue
            try:
                float(result)
            except (TypeError, ValueError) as exc:
                errors.append(
                    (c, f"did not return a scalar (got {type(result).__name__}): {exc}")
                )
        return errors

    @staticmethod
    def _format_validation_feedback(
        errors: list[tuple[SubtrajConstraint, str]],
    ) -> str:
        """Build a user message describing which constraints failed and why."""
        lines = [
            "Some of the constraints you produced are invalid and must be fixed. "
            "The lambda is evaluated in a sandbox where only the `np` module is "
            "available (no Python built-ins like max/min/abs/sum — use np.maximum, "
            "np.minimum, np.abs, np.sum). Each lambda must have signature "
            "`lambda ee, keypoints, ee_R: <scalar>` (where `ee_R` is the 3x3 "
            "world-from-EE rotation matrix; `ee_R[:, 2]` is the gripper "
            "approach axis in world frame) and return a scalar.",
            "",
            "Errors:",
        ]
        for c, msg in errors:
            lines.append(f"  - [{c.kind}] subtraj {c.subtraj_idx}, label={c.label!r}")
            lines.append(f"      expression: {c.expression}")
            lines.append(f"      problem: {msg}")
        lines.append("")
        lines.append(
            "Regenerate the FULL analysis JSON (same schema as before) with all "
            "expressions fixed. Keep valid constraints as-is; only change the "
            "broken ones. Return ONLY the JSON object."
        )
        return "\n".join(lines)

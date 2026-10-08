"""Shared base for VLM box/point detectors.

A detection backend reads an image (optionally with a task instruction) and
proposes objects as 2D boxes or center points in normalized ``0-1000`` coords
plus a label. Two backends share this logic and differ only in how the
image+prompt request is sent (:meth:`VlmBoxDetectorBase._generate`):

- :class:`open_perception.models.gemini.GeminiBoxDetector` — Google's native
  ``generateContent`` API (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``).
- :class:`open_perception.models.openrouter.OpenRouterBoxDetector` — any
  OpenAI-wire model, routed through OpenRouter (``OPENROUTER_API_KEY``).

Both implement the ``BoxDetectClient`` protocol in ``stages/detect_gemini.py``.
The prompts, JSON parsing, and 0-1000 → pixel conversion are identical across
backends, so only the request transport is backend-specific.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from ..stages.detect_gemini import BoxDetectResult

_OPEN_VOCAB_PROMPT = """\
You are an object detector for a robot manipulation scene.

Detect every distinct, physically manipulable object relevant to the scene{task}.
Do NOT return the robot, the robot gripper/hand, the table surface, the floor,
the background, or out-of-reach/irrelevant items.

Rules:
- Give each object a unique, lowercase, snake_case label that distinguishes it
  from the others using salient attributes (color, size, relative position),
  e.g. "red_mug", "small_blue_block", "soda_can_left".
- Return at most {max_objects} objects.
- For each object provide a 2D bounding box "box_2d" as
  [ymin, xmin, ymax, xmax] with each value normalized to the range 0-1000
  relative to the image (y is the vertical axis, x is the horizontal axis).

Respond with ONLY a JSON object of the form:
{{"bboxes": [{{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}}]}}
"""

_GROUNDED_PROMPT = """\
You are an object detector for a robot manipulation scene.

Locate the following objects in the image{task}:
{query_list}

Rules:
- Return a bounding box only for objects that are actually visible. Omit any
  requested object you cannot find.
- If several instances of a requested object are visible, return one bounding
  box for EACH instance.
- Use exactly the requested name as the "label" for each detected object.
- For each object provide a 2D bounding box "box_2d" as
  [ymin, xmin, ymax, xmax] with each value normalized to the range 0-1000
  relative to the image (y is the vertical axis, x is the horizontal axis).

Respond with ONLY a JSON object of the form:
{{"bboxes": [{{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}}]}}
"""

_OPEN_VOCAB_POINT_PROMPT = """\
You are an object detector for a robot manipulation scene.

Point to every distinct, physically manipulable object relevant to the scene{task}.
Do NOT return the robot, the robot gripper/hand, the table surface, the floor,
the background, or out-of-reach/irrelevant items.

Rules:
- Give each object a unique, lowercase, snake_case label that distinguishes it
  from the others using salient attributes (color, size, relative position),
  e.g. "red_mug", "small_blue_block", "soda_can_left".
- Return at most {max_objects} objects.
- Place each point at the center of the visible part of the object.
- Provide each point as "point": [y, x] with each value normalized to the
  range 0-1000 relative to the image (y is the vertical axis, x is the
  horizontal axis).

Respond with ONLY a JSON object of the form:
{{"points": [{{"point": [y, x], "label": "object_name"}}]}}
"""

_GROUNDED_POINT_PROMPT = """\
You are an object detector for a robot manipulation scene.

Point to the following objects in the image{task}:
{query_list}

Rules:
- Return a point only for objects that are actually visible. Omit any
  requested object you cannot find.
- If several instances of a requested object are visible, return one point
  for EACH instance.
- Use exactly the requested name as the "label" for each point.
- Place each point at the center of the visible part of the object.
- Provide each point as "point": [y, x] with each value normalized to the
  range 0-1000 relative to the image (y is the vertical axis, x is the
  horizontal axis).

Respond with ONLY a JSON object of the form:
{{"points": [{{"point": [y, x], "label": "object_name"}}]}}
"""


@dataclass
class VlmBoxDetectorBase:
    """Backend-agnostic box/point proposer.

    Subclasses implement :meth:`_generate` (send image+prompt, return raw text).
    """

    model_id: str = ""
    temperature: float | None = None
    max_objects: int = 25
    _client: Any = field(default=None, init=False, repr=False)

    # -- BoxDetectClient protocol --------------------------------------

    def detect(
        self,
        rgb: np.ndarray,
        queries: list[str] | None,
        task: str | None,
        proposal: str = "box",
    ) -> BoxDetectResult:
        H, W = rgb.shape[:2]
        prompt = self._build_prompt(queries, task, proposal)
        text = self._generate(rgb, prompt)
        if proposal == "point":
            return BoxDetectResult(
                points=self._to_pixel_xy(self._parse_json(text, key="points"), H, W)
            )
        return BoxDetectResult(
            boxes=self._to_pixel_xyxy(self._parse_json(text, key="bboxes"), H, W)
        )

    def _generate(self, rgb: np.ndarray, prompt: str) -> str | None:
        """Send ``rgb`` + ``prompt`` to the backend; return the raw text reply."""
        raise NotImplementedError

    # -- shared prompt / parse / convert -------------------------------

    def _build_prompt(
        self, queries: list[str] | None, task: str | None, proposal: str = "box"
    ) -> str:
        task_clause = f' for the task: "{task}"' if task else ""
        if proposal == "point":
            if queries:
                query_list = "\n".join(f"- {q}" for q in queries)
                return _GROUNDED_POINT_PROMPT.format(
                    task=task_clause, query_list=query_list
                )
            return _OPEN_VOCAB_POINT_PROMPT.format(
                task=task_clause, max_objects=self.max_objects
            )
        if queries:
            query_list = "\n".join(f"- {q}" for q in queries)
            return _GROUNDED_PROMPT.format(task=task_clause, query_list=query_list)
        return _OPEN_VOCAB_PROMPT.format(task=task_clause, max_objects=self.max_objects)

    @staticmethod
    def _parse_json(text: str | None, key: str = "bboxes") -> list[dict[str, Any]]:
        if not text:
            return []
        cleaned = text.strip()
        # Strip ```json ... ``` / ``` ... ``` fences if present.
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        parsed = _loads_or_extract(cleaned)
        if parsed is None:
            logger.warning(f"[vlm-detect] could not parse JSON response: {text!r}")
            return []
        if isinstance(parsed, dict):
            items = parsed.get(key, [])
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = []
        return [d for d in items if isinstance(d, dict)]

    @staticmethod
    def _to_pixel_xyxy(
        detections: list[dict[str, Any]], H: int, W: int
    ) -> dict[str, np.ndarray]:
        """Convert 0-1000 ``[ymin,xmin,ymax,xmax]`` boxes to xyxy pixel arrays.

        Boxes sharing a label are stacked into one ``(N, 4)`` array rather
        than overwriting each other.
        """
        grouped: dict[str, list[list[float]]] = {}
        for det in detections:
            box = det.get("box_2d")
            label = det.get("label")
            if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = (float(v) for v in box)
            xyxy = [
                (xmin / 1000.0) * W,
                (ymin / 1000.0) * H,
                (xmax / 1000.0) * W,
                (ymax / 1000.0) * H,
            ]
            key = str(label).replace(" ", "_")
            grouped.setdefault(key, []).append(xyxy)
        return {
            label: np.asarray(boxes, dtype=np.float32)
            for label, boxes in grouped.items()
        }

    @staticmethod
    def _to_pixel_xy(
        detections: list[dict[str, Any]], H: int, W: int
    ) -> dict[str, np.ndarray]:
        """Convert 0-1000 ``[y, x]`` points to (N, 2) xy pixel arrays.

        Points sharing a label are stacked into one ``(N, 2)`` array — one
        row per proposed instance — mirroring ``_to_pixel_xyxy``.
        """
        grouped: dict[str, list[list[float]]] = {}
        for det in detections:
            pt = det.get("point")
            label = det.get("label")
            if not label or not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            y, x = (float(v) for v in pt)
            xy = [(x / 1000.0) * W, (y / 1000.0) * H]
            key = str(label).replace(" ", "_")
            grouped.setdefault(key, []).append(xy)
        return {
            label: np.asarray(pts, dtype=np.float32) for label, pts in grouped.items()
        }


def _loads_or_extract(cleaned: str) -> Any | None:
    """``json.loads``; on failure, extract the first ``{...}`` block and retry.

    Reasoning-style models (e.g. gemini-3) sometimes prepend prose before the
    JSON object despite the "respond with ONLY" instruction.
    """
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

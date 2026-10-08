"""Stub model classes shared by tests.

Kept in a normal module (not conftest.py) so other test files can import
them directly with `from _stubs import StubSam3`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from open_perception.config import PerceptionConfig
from open_perception.encoders.reference import ReferenceBank


class StubSam3:
    model_id = "stub/sam3"
    device = "cpu"

    def __init__(self, masks: np.ndarray, scores: np.ndarray) -> None:
        self._masks = masks
        self._scores = scores

    def resolve_prompts(self, templates: list[str], queries: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for tmpl in templates:
            if "<object_names>" in tmpl:
                for q in queries:
                    p = tmpl.replace("<object_names>", q)
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
            elif tmpl not in seen:
                seen.add(tmpl)
                out.append(tmpl)
        return out

    def segment(self, rgb, text_prompts, score_threshold, mask_threshold):
        return self._masks.copy(), self._scores.copy()

    def segment_per_prompt(self, rgb, text_prompts, score_threshold, mask_threshold):
        # Attribute the full mask pool to the first prompt and empty arrays
        # to the rest — the downstream assign-pipeline tests only care about
        # ctx.instances getting all candidate masks, not which prompt they
        # came from.
        if not text_prompts:
            return []
        empty_masks = np.zeros((0, *self._masks.shape[1:]), dtype=bool)
        empty_scores = np.zeros((0,), dtype=np.float32)
        out = [(self._masks.copy(), self._scores.copy())]
        out.extend((empty_masks.copy(), empty_scores.copy()) for _ in text_prompts[1:])
        return out


class StubSam2:
    """Geometric-prompt SAM2 stub.

    Boxes -> one filled-rectangle mask per box; points -> one small filled
    square (9x9) centred on each prompt group's first point.
    """

    model_id = "stub/sam2"
    device = "cpu"

    def segment_with_boxes(self, rgb, boxes, *, multimask_output: bool = False):
        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        n = boxes_arr.shape[0]
        H, W = rgb.shape[:2]
        if n == 0:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
        masks = np.zeros((n, H, W), dtype=bool)
        for i, (x1, y1, x2, y2) in enumerate(boxes_arr):
            xa, xb = sorted((int(round(x1)), int(round(x2))))
            ya, yb = sorted((int(round(y1)), int(round(y2))))
            xa, ya = max(xa, 0), max(ya, 0)
            xb, yb = min(xb, W), min(yb, H)
            masks[i, ya:yb, xa:xb] = True
        scores = np.full((n,), 0.9, dtype=np.float32)
        return masks, scores

    def segment_with_points(
        self, rgb, points, labels=None, *, multimask_output: bool = False
    ):
        H, W = rgb.shape[:2]
        n = len(points)
        if n == 0:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
        masks = np.zeros((n, H, W), dtype=bool)
        for i, obj_points in enumerate(points):
            x, y = obj_points[0]
            xa, xb = max(int(round(x)) - 4, 0), min(int(round(x)) + 5, W)
            ya, yb = max(int(round(y)) - 4, 0), min(int(round(y)) + 5, H)
            masks[i, ya:yb, xa:xb] = True
        scores = np.full((n,), 0.9, dtype=np.float32)
        return masks, scores


class StubSam3Geometric:
    """SAM3 wrapper stub exposing only the ``tracker`` geometric head."""

    model_id = "stub/sam3"
    device = "cpu"

    def __init__(self) -> None:
        self.tracker = StubSam2()


class StubBoxDetector:
    """Box/point-proposal stub implementing the ``BoxDetectClient`` protocol."""

    def __init__(
        self,
        boxes: dict[str, np.ndarray] | None = None,
        points: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.boxes = boxes or {}
        self.points = points or {}
        self.calls: list[tuple] = []

    def detect(self, rgb, queries, task, proposal: str = "box"):
        from open_perception.stages.detect_gemini import BoxDetectResult

        self.calls.append((queries, task))
        if proposal == "point":
            return BoxDetectResult(
                points={k: np.asarray(v) for k, v in self.points.items()}
            )
        return BoxDetectResult(boxes={k: np.asarray(v) for k, v in self.boxes.items()})


class StubClip:
    def __init__(
        self, crop_embeddings: np.ndarray, name_to_text: dict[str, np.ndarray]
    ) -> None:
        self._crop = crop_embeddings
        self._text = name_to_text
        self._dim = crop_embeddings.shape[1]

    def encode_images(self, crops: list[np.ndarray]) -> np.ndarray:
        n = len(crops)
        if n == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        feats = self._crop[:n].astype(np.float32)
        if feats.shape[0] < n:
            pad = np.zeros((n - feats.shape[0], self._dim), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        return feats

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t in self._text:
                out[i] = self._text[t]
        return out


@dataclass
class StubServices:
    config: PerceptionConfig
    sam3: StubSam3
    clip: StubClip | None = None
    reference_banks: dict[str, ReferenceBank] = field(default_factory=dict)

    def get_sam3(self):
        return self.sam3

    def get_clip(self):
        return self.clip

    def get_dino(self):
        return None

    def get_vlm_client(self):
        return None

    def get_box_detector(self):
        return None

    def get_reference_bank(self, space: str):
        return self.reference_banks.get(space)

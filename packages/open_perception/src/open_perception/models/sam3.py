"""Thin SAM3 image-model wrapper.

Lifted from `Sam3ClipRerankPredictor._run_sam3` in
`scripts/ablations/perception/sam3_clip_rerank_eval.py`: vision features are
computed once per image and reused across multiple text prompts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

from .sam2 import Sam2Wrapper


class Sam3Wrapper:
    """Wraps `transformers.Sam3Model` + `Sam3Processor`.

    Kept small — it exposes a single `segment(rgb, text_prompts, ...)` method
    that returns stacked (N, H, W) boolean masks and per-mask scores. Callers
    handle templating (e.g. "<object_names>" expansion) and score-based sorting.
    """

    OBJECT_NAMES_TOKEN = "<object_names>"

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self.device = device
        self.model_id = model_id
        logger.info(f"[sam3] loading {model_id} on {device}")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        # SAM3's tracker head (SAM2-style geometric prompting) — built lazily
        # on first use so text-only pipelines never load it.
        self._tracker: Sam3TrackerWrapper | None = None

    @property
    def tracker(self) -> "Sam3TrackerWrapper":
        """Geometric-prompt (bbox/point) head, loaded on first access."""
        if self._tracker is None:
            self._tracker = Sam3TrackerWrapper(self.model_id, self.device)
        return self._tracker

    def resolve_prompts(self, templates: list[str], queries: list[str]) -> list[str]:
        """Expand `<object_names>` tokens against the target queries.

        Each template entry containing the token is expanded against every
        query, producing one prompt per (template, query) pair. Entries
        without the token pass through verbatim — callers can mix literal
        context prompts (e.g. ``"object"``) with the expansion to give SAM3
        extra negatives/positives. Duplicates are dropped while preserving
        the first occurrence.
        """
        resolved: list[str] = []
        seen: set[str] = set()
        for tmpl in templates:
            if self.OBJECT_NAMES_TOKEN in tmpl:
                for q in queries:
                    p = tmpl.replace(self.OBJECT_NAMES_TOKEN, q)
                    if p not in seen:
                        seen.add(p)
                        resolved.append(p)
            elif tmpl not in seen:
                seen.add(tmpl)
                resolved.append(tmpl)
        return resolved

    def segment_per_prompt(
        self,
        rgb: np.ndarray,
        text_prompts: list[str],
        score_threshold: float,
        mask_threshold: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (masks, scores) per prompt, aligned 1-1 with ``text_prompts``.

        All prompts run in a single forward pass — the vision encoder is
        evaluated once, its output expanded across the text batch dimension.
        Prompts that produce no masks yield empty ``(0, H, W)`` / ``(0,)``
        arrays, so callers can rely on positional alignment.
        """
        torch = self._torch
        H, W = rgb.shape[:2]
        if not text_prompts:
            return []

        N = len(text_prompts)
        image = Image.fromarray(rgb)
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(
                pixel_values=img_inputs.pixel_values
            )
        vision_embeds_N = _expand_batch(vision_embeds, N)

        text_inputs = self.processor(text=text_prompts, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(vision_embeds=vision_embeds_N, **text_inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=[[H, W]] * N,
        )

        empty_masks = np.zeros((0, H, W), dtype=bool)
        empty_scores = np.zeros((0,), dtype=np.float32)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for r in results:
            masks = r.get("masks")
            scores = r.get("scores")
            if masks is None or len(masks) == 0:
                out.append((empty_masks, empty_scores))
                continue
            inst = _as_bool_array(masks)
            out.append((inst, _as_float_array(scores, inst.shape[0])))
        return out

    def segment(
        self,
        rgb: np.ndarray,
        text_prompts: list[str],
        score_threshold: float,
        mask_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (masks[N, H, W] bool, scores[N] float32) sorted by descending score."""
        H, W = rgb.shape[:2]
        per_prompt = self.segment_per_prompt(
            rgb, text_prompts, score_threshold, mask_threshold
        )
        non_empty = [(m, s) for m, s in per_prompt if m.shape[0] > 0]
        if not non_empty:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
        inst = np.concatenate([m for m, _ in non_empty], axis=0)
        scores = np.concatenate([s for _, s in non_empty], axis=0)
        order = np.argsort(-scores)
        return inst[order], scores[order]


class Sam3TrackerWrapper(Sam2Wrapper):
    """SAM3's tracker head behind the ``Sam2Wrapper`` geometric-prompt API.

    ``Sam3TrackerModel`` / ``Sam3TrackerProcessor`` mirror the SAM2 classes
    (SAM3's tracker *is* a SAM2-style promptable segmenter), so
    ``segment_with_boxes`` / ``segment_with_points`` are inherited verbatim.
    ``segment_auto`` is not supported for this head.
    """

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self._torch = torch
        self.device = device
        self.model_id = model_id
        logger.info(f"[sam3.tracker] loading {model_id} on {device}")
        self.processor = Sam3TrackerProcessor.from_pretrained(model_id)
        self.model = Sam3TrackerModel.from_pretrained(model_id).to(device).eval()
        self._auto_pipeline = None

    def segment_auto(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError(
            "sam3 tracker head only supports bbox/point prompts; "
            "use the sam2 stage for automatic mask generation."
        )


def _as_bool_array(masks: Any) -> np.ndarray:
    if hasattr(masks, "cpu"):
        return masks.cpu().numpy().astype(bool)
    return np.stack([np.asarray(m).astype(bool) for m in masks], axis=0)


def _as_float_array(scores: Any, n: int) -> np.ndarray:
    if scores is None:
        return np.zeros((n,), dtype=np.float32)
    if hasattr(scores, "cpu"):
        return scores.cpu().numpy().astype(np.float32)
    return np.asarray(scores, dtype=np.float32)


def _expand_batch(vision_out: Any, n: int) -> Any:
    """Broadcast batch=1 vision-encoder outputs to batch=N without recomputing.

    Cheaper than running the vision encoder N times on a replicated image —
    all tensor fields share storage via ``expand``.
    """
    if n == 1:
        return vision_out

    def _expand_tensor(t: Any) -> Any:
        if t is None or not hasattr(t, "shape"):
            return t
        if t.shape[0] == n:
            return t
        if t.shape[0] != 1:
            return t
        return t.expand(n, *t.shape[1:])

    new_fields: dict[str, Any] = {}
    for k, v in vision_out.items():
        if isinstance(v, (tuple, list)) and v and hasattr(v[0], "shape"):
            new_fields[k] = type(v)(_expand_tensor(x) for x in v)
        else:
            new_fields[k] = _expand_tensor(v)
    return type(vision_out)(**new_fields)

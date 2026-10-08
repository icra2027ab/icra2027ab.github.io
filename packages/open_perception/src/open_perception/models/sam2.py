"""Thin SAM2 image-model wrapper.

SAM2 cannot segment from text — it accepts dense (point grid) or sparse
(point / bbox) prompts. This wrapper exposes three modes:

- ``segment_auto``  : grid-of-points "propose all masks" via the
  ``mask-generation`` pipeline (matches SAM2's automatic mask generator).
- ``segment_with_boxes``: one mask per supplied bounding box.
- ``segment_with_points``: one mask per supplied (points, labels) prompt.

Used by ``stages/sam2.py`` as a SAM3-shaped segmentation stage so the same
filter/disambiguate/assign pipeline can run with SAM2 as the proposer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from PIL import Image


class Sam2Wrapper:
    """Wraps `transformers.Sam2Model` + `Sam2Processor` for image segmentation.

    Auto-mode reuses the HF ``mask-generation`` pipeline (which already
    handles the points-grid → NMS → filter dance). Point/bbox modes go
    straight through ``Sam2Model`` for a single forward pass per image.
    """

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._torch = torch
        self.device = device
        self.model_id = model_id
        logger.info(f"[sam2] loading {model_id} on {device}")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()
        # Lazily built when ``segment_auto`` is first called — avoids
        # constructing the pipeline machinery if only point/bbox modes are used.
        self._auto_pipeline: Any | None = None

    # ------------------------------------------------------------------

    def segment_auto(
        self,
        rgb: np.ndarray,
        *,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.92,
        crops_nms_thresh: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propose all masks via SAM2's automatic mask generator.

        Returns ``(masks[N, H, W] bool, scores[N] float32)``. ``scores``
        are SAM2 IoU predictions for each kept mask.
        """
        if self._auto_pipeline is None:
            from transformers import pipeline

            # Reuse our pre-loaded model + processor so we don't re-download
            # / re-instantiate the weights for the pipeline.
            self._auto_pipeline = pipeline(
                "mask-generation",
                model=self.model,
                image_processor=self.processor.image_processor,
                device=self.device,
            )
        H, W = rgb.shape[:2]
        out = self._auto_pipeline(
            Image.fromarray(rgb),
            points_per_batch=points_per_batch,
            points_per_crop=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crops_nms_thresh=crops_nms_thresh,
        )
        masks_raw = out.get("masks") or []
        scores_raw = out.get("scores")
        if not masks_raw:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
        masks = np.stack([_to_numpy(m).astype(bool) for m in masks_raw], axis=0)
        scores = _as_float_array(scores_raw, masks.shape[0])
        return masks, scores

    def segment_with_boxes(
        self,
        rgb: np.ndarray,
        boxes: np.ndarray | list[tuple[float, float, float, float]],
        *,
        multimask_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM2 once with one bbox prompt per object.

        ``boxes`` is ``(N, 4)`` in ``(x1, y1, x2, y2)`` pixel coords.
        Returns ``(masks[N, H, W] bool, scores[N] float32)``. With
        ``multimask_output=True`` SAM2 emits 3 candidates per box and we
        keep the highest-scoring one.
        """
        torch = self._torch
        H, W = rgb.shape[:2]
        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if boxes_arr.shape[0] == 0:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)

        inputs = self.processor(
            images=Image.fromarray(rgb),
            input_boxes=[boxes_arr.tolist()],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)
        return self._postprocess(outputs, inputs["original_sizes"], H, W)

    def segment_with_points(
        self,
        rgb: np.ndarray,
        points: list[list[tuple[float, float]]],
        labels: list[list[int]] | None = None,
        *,
        multimask_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM2 once with one (points, labels) prompt per object.

        ``points`` is a list of length ``N`` (one entry per object); each
        entry is a list of ``(x, y)`` pixel coords. ``labels`` is the
        matching list of foreground/background flags (``1`` / ``0``); it
        defaults to all-foreground when omitted.
        """
        torch = self._torch
        H, W = rgb.shape[:2]
        if not points:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)

        if labels is None:
            labels = [[1] * len(p) for p in points]
        if len(labels) != len(points):
            raise ValueError(
                f"len(labels)={len(labels)} must match len(points)={len(points)}"
            )

        # Sam2Processor expects [image, object, point, coord] for points and
        # [image, object, point] for labels.
        input_points = [[[list(pt) for pt in obj] for obj in points]]
        input_labels = [[list(lbl) for lbl in labels]]
        inputs = self.processor(
            images=Image.fromarray(rgb),
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)
        return self._postprocess(outputs, inputs["original_sizes"], H, W)

    # ------------------------------------------------------------------

    def _postprocess(
        self,
        outputs: Any,
        original_sizes: Any,
        H: int,
        W: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # ``pred_masks`` is (B, N_obj, N_masks, h, w); ``iou_scores`` is
        # (B, N_obj, N_masks). Post-process resizes/binarizes back to (H, W).
        masks_post = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes.cpu(), binarize=True
        )[0]
        if hasattr(masks_post, "cpu"):
            masks_post = masks_post.cpu().numpy()
        masks_post = np.asarray(masks_post).astype(bool)
        # Shape: (N_obj, N_masks, H, W) — pick the best candidate per object.
        if masks_post.ndim != 4:
            raise RuntimeError(
                f"unexpected SAM2 mask shape after post_process_masks: {masks_post.shape}"
            )
        iou = outputs.iou_scores.detach().cpu().numpy()[0]  # (N_obj, N_masks)
        best_idx = np.argmax(iou, axis=1)
        n_obj = masks_post.shape[0]
        out_masks = np.stack([masks_post[i, best_idx[i]] for i in range(n_obj)], axis=0)
        out_scores = np.array(
            [float(iou[i, best_idx[i]]) for i in range(n_obj)], dtype=np.float32
        )
        # Sanity: make sure we resolved to the source image's (H, W).
        if out_masks.shape[-2:] != (H, W):
            raise RuntimeError(
                f"SAM2 mask shape {out_masks.shape} doesn't match source "
                f"image ({H}, {W})"
            )
        return out_masks, out_scores


def _to_numpy(mask: Any) -> np.ndarray:
    if hasattr(mask, "cpu"):
        return mask.cpu().numpy()
    return np.asarray(mask)


def _as_float_array(scores: Any, n: int) -> np.ndarray:
    if scores is None:
        return np.zeros((n,), dtype=np.float32)
    if hasattr(scores, "cpu"):
        return scores.cpu().numpy().astype(np.float32)
    return np.asarray(scores, dtype=np.float32)

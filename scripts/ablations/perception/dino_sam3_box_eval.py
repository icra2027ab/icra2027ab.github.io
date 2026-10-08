"""MM Grounding DINO -> bboxes -> SAM3 box prompt, vs GT masks.

Pipeline:
  1. MM Grounding DINO runs once per frame with a dot-joined text prompt
     ("red cube. blue mug. plate.") and returns ``{scores, boxes, text_labels}``.
  2. Each ``text_labels[i]`` is matched to a requested object name; the
     highest-scoring box per name wins.
  3. Each matched box is fed into SAM3 as an ``input_boxes`` prompt. The
     returned mask is the predicted segmentation for that name. Unmatched names
     get an empty mask.

Output schema matches ``sam3_segmentation_eval.py`` so results compare
side-by-side.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import tyro
from _common import BaseArgs, normalize_name, run_pipeline
from loguru import logger
from PIL import Image


@dataclasses.dataclass
class Args(BaseArgs):
    out_dir: str = "tmp/dino_sam3_box_eval"
    dino_model: str = "openmmlab-community/mm_grounding_dino_base_all"
    sam3_model: str = "facebook/sam3"
    box_threshold: float = 0.15
    """Detection confidence threshold for Grounding DINO."""
    text_threshold: float = 0.15
    """Text-matching threshold for Grounding DINO."""
    sam3_mask_threshold: float = 0.5
    """Binarisation threshold for SAM3's semantic_seg output."""


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class DinoSam3BoxPredictor:
    name = "dino_sam3_box"

    def __init__(
        self,
        dino_model_id: str,
        sam3_model_id: str,
        device: str,
        box_threshold: float,
        text_threshold: float,
        sam3_mask_threshold: float,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam3Model,
            Sam3Processor,
        )

        self._torch = torch
        self.device = (
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if self.device != device:
            logger.warning(
                f"requested device={device} unavailable, falling back to {self.device}"
            )
        logger.info(f"loading {dino_model_id} on {self.device}")
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino = (
            AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id)
            .to(self.device)
            .eval()
        )
        logger.info(f"loading {sam3_model_id} on {self.device}")
        self.sam3_processor = Sam3Processor.from_pretrained(sam3_model_id)
        self.sam3 = Sam3Model.from_pretrained(sam3_model_id).to(self.device).eval()

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.sam3_mask_threshold = sam3_mask_threshold

    # ------------------------------------------------------------------

    def _run_dino(
        self, rgb: np.ndarray, prompts: list[str]
    ) -> dict[str, tuple[tuple[float, float, float, float], float]]:
        """Return ``{canonical_name: (box_xyxy, score)}`` for the best detection per name."""
        torch = self._torch
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)
        # Grounding DINO expects dot-terminated, lowercase labels.
        text = ". ".join(p.lower() for p in prompts) + "."
        inputs = self.dino_processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.dino(**inputs)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(H, W)],
        )[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("text_labels") or results.get("labels")
        if boxes is None or len(boxes) == 0 or labels is None:
            return {}
        boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
        scores_np = (
            scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
        )
        labels_list = [str(lbl) for lbl in labels]

        best: dict[str, tuple[tuple[float, float, float, float], float]] = {}
        for i, raw_label in enumerate(labels_list):
            canon = normalize_name(raw_label)
            # DINO may emit sub-phrases; try exact match against each requested prompt.
            match = None
            for p in prompts:
                if p == canon or p in canon or canon in p:
                    match = p
                    break
            if match is None:
                continue
            s = float(scores_np[i])
            if match not in best or s > best[match][1]:
                x0, y0, x1, y1 = (float(v) for v in boxes_np[i].tolist())
                best[match] = ((x0, y0, x1, y1), s)
        return best

    def _run_sam3_box(
        self, rgb: np.ndarray, box: tuple[float, float, float, float]
    ) -> np.ndarray:
        """Return a binary mask[H,W] from SAM3 prompted with a single box.

        Uses ``post_process_semantic_segmentation`` on ``outputs.semantic_seg``
        which is the box-conditioned mask; ``post_process_instance_segmentation``
        returns 200 generic DETR queries and doesn't correctly isolate the
        box-prompted object.
        """
        torch = self._torch
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)
        x0, y0, x1, y1 = box
        inputs = self.sam3_processor(
            images=image,
            input_boxes=[[[x0, y0, x1, y1]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.sam3(**inputs)
        sem = self.sam3_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(H, W)], threshold=self.sam3_mask_threshold
        )[0]
        return sem.cpu().numpy().astype(bool)

    # ------------------------------------------------------------------

    def predict(self, rgb: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        H, W = rgb.shape[:2]
        empty = np.zeros((H, W), dtype=bool)
        out = {p: empty.copy() for p in prompts}

        box_per_name = self._run_dino(rgb, prompts)
        for name, (box, _score) in box_per_name.items():
            out[name] = self._run_sam3_box(rgb, box)
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    predictor = DinoSam3BoxPredictor(
        dino_model_id=args.dino_model,
        sam3_model_id=args.sam3_model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        sam3_mask_threshold=args.sam3_mask_threshold,
    )
    tag = f"b{args.box_threshold:.2f}_t{args.text_threshold:.2f}"
    extra_meta: dict[str, Any] = {
        "dino_model": args.dino_model,
        "sam3_model": args.sam3_model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "sam3_mask_threshold": args.sam3_mask_threshold,
    }
    run_pipeline(args, predictor, tag=tag, extra_meta=extra_meta)


if __name__ == "__main__":
    main(tyro.cli(Args))

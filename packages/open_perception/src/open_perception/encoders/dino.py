"""DINOv2 / DINOv3 image encoder — mean-pooled, L2-normalized features.

Matches `scripts/extract_objects.py::ClipDinoEncoder.encode_dino_image`, so
reference-bank embeddings and runtime crop embeddings live in the same space.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from PIL import Image


class DinoEncoder:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, Dinov2Model

        self._torch = torch
        self.device = device
        self.model_id = model_id
        logger.info(f"[dino] loading {model_id} on {device}")
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = Dinov2Model.from_pretrained(model_id).to(device).eval()

    def encode_images(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        if not crops:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        pil = [Image.fromarray(c) for c in crops]
        with torch.no_grad():
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            outputs = self.model(pixel_values=inputs["pixel_values"])
            feats = outputs.last_hidden_state.mean(dim=1)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

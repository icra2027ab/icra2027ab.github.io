"""CLIP image and text encoder — L2-normalized features.

Port of `_encode_crops` / text-encoding paths in `Sam3ClipRerankPredictor`.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from PIL import Image


class ClipEncoder:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device
        self.model_id = model_id
        logger.info(f"[clip] loading {model_id} on {device}")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def encode_images(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        if not crops:
            return np.zeros((0, self.model.config.projection_dim), dtype=np.float32)
        pil = [Image.fromarray(c) for c in crops]
        with torch.no_grad():
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs).pooler_output
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        if not texts:
            return np.zeros((0, self.model.config.projection_dim), dtype=np.float32)
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            feats = self.model.get_text_features(**inputs).pooler_output
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

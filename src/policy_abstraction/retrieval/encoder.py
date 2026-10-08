"""DINOv2 visual + CLIP text encoders for demo retrieval."""

from __future__ import annotations

import numpy as np
import torch


class DemoEncoder:
    """Encodes demo frames (DINOv2) and task instructions (CLIP text).

    Both outputs are L2-normalised so cosine similarity reduces to a dot product.
    Callers are responsible for any image flipping before passing frames in.

    Parameters
    ----------
    dino_model:
        HuggingFace model ID for DINOv2. Default: ``"facebook/dinov2-base"`` (768-dim).
    clip_model:
        HuggingFace model ID for CLIP. Default: ``"openai/clip-vit-base-patch16"`` (512-dim).
    device:
        Torch device string. If ``None``, auto-detects CUDA.
    """

    DINO_DIM = 768
    CLIP_DIM = 512

    def __init__(
        self,
        dino_model: str = "facebook/dinov2-base",
        clip_model: str = "openai/clip-vit-base-patch16",
        device: str | None = None,
    ) -> None:
        from transformers import (
            AutoImageProcessor,
            CLIPModel,
            CLIPTokenizer,
            Dinov2Model,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # DINOv2 visual encoder
        self._dino: Dinov2Model = Dinov2Model.from_pretrained(dino_model)  # type: ignore[assignment]
        self._dino.eval()
        self._dino.to(device)  # type: ignore[arg-type]
        self._dino_processor = AutoImageProcessor.from_pretrained(
            dino_model, use_fast=True
        )
        self.dino_model_name = dino_model

        # CLIP text encoder
        self._clip: CLIPModel = CLIPModel.from_pretrained(clip_model)  # type: ignore[assignment]
        self._clip.eval()
        self._clip.to(device)  # type: ignore[arg-type]
        self._clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.clip_model_name = clip_model

    # ------------------------------------------------------------------
    # Visual encoding
    # ------------------------------------------------------------------

    def encode_frames(
        self,
        frames: list[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode a sequence of RGB frames into a single (768,) embedding.

        Parameters
        ----------
        frames:
            List of ``(H, W, 3)`` uint8 RGB arrays. Must already be flipped
            (LIBERO images need ``[::-1, ::-1, :]`` applied by the caller).
        batch_size:
            Number of frames per DINOv2 forward pass.

        Returns
        -------
        np.ndarray
            Shape ``(768,)`` float32, L2-normalised.
        """
        all_feats: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                inputs = self._dino_processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                outputs = self._dino(pixel_values=pixel_values)
                # avg-pool over patch tokens: (B, seq_len, 768) → (B, 768)
                feats = outputs.last_hidden_state.mean(dim=1)
                all_feats.append(feats.cpu().numpy())

        # average over temporal dimension
        frame_matrix = np.concatenate(all_feats, axis=0)  # (T, 768)
        visual_emb = frame_matrix.mean(axis=0)  # (768,)
        return _l2_normalize(visual_emb)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a task instruction string into a (512,) embedding.

        Parameters
        ----------
        text:
            Human-readable task description, e.g. ``"pick up the cup"``.

        Returns
        -------
        np.ndarray
            Shape ``(512,)`` float32, L2-normalised.
        """
        inputs = self._clip_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # transformers >=5 returns BaseModelOutputWithPooling here, with
            # the pooled (1, 512) text projection on .pooler_output; 4.x
            # returns that projection as a bare tensor. Accept both so the
            # cache schema does not depend on the installed version.
            out = self._clip.get_text_features(**inputs)
            text_features = getattr(out, "pooler_output", out)
        text_emb = text_features[0].cpu().numpy()
        return _l2_normalize(text_emb)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def encode_demo(
        self,
        frames: list[np.ndarray],
        task_text: str,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode both visual frames and task text.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(visual_emb, text_emb)`` with shapes ``(768,)`` and ``(512,)``.
        """
        return self.encode_frames(frames, batch_size=batch_size), self.encode_text(
            task_text
        )


def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalise a 1-D vector with epsilon to avoid division by zero."""
    norm = float(np.linalg.norm(v))
    return (v / max(norm, eps)).astype(np.float32)

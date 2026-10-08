"""Dynamic point-cloud reconstruction from image sequences using Pi3."""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Ensure Pi3 is importable regardless of whether it is installed as a package.
_PI3_ROOT = Path(__file__).parents[4] / "third_party" / "Pi3"
if str(_PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(_PI3_ROOT))


@dataclass
class Pi3Result:
    """Outputs of a Pi3 reconstruction pass.

    All tensors have a leading frame dimension ``[N, ...]`` (no batch dim).

    Attributes:
        points:        ``[N, H, W, 3]`` float32 — world-frame 3-D point map.
        local_points:  ``[N, H, W, 3]`` float32 — camera-frame 3-D point map.
        camera_poses:  ``[N, 4, 4]``   float32 — camera-to-world SE(3) poses.
        conf:          ``[N, H, W]``   float32 — sigmoid confidence in [0, 1].
        mask:          ``[N, H, W]``   bool    — valid-pixel mask after
                       confidence threshold and depth-edge filtering.
    """

    points: torch.Tensor
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    conf: torch.Tensor
    mask: torch.Tensor


class Pi3Reconstructor:
    """Reconstruct a dynamic point cloud from a sequence of RGB images.

    Wraps ``Pi3`` (or ``Pi3X`` for the extended model) and handles:
    - model loading from HuggingFace or a local checkpoint,
    - image pre-processing (resize to Pi3-compatible resolution, normalise),
    - chunked inference for long sequences via ``Pi3XVO``,
    - confidence + depth-edge masking.

    Args:
        model_name:    ``"pi3"`` (default) or ``"pi3x"``.
        checkpoint:    Path to a local ``.safetensors`` or ``.pt`` checkpoint.
                       When ``None`` the model is downloaded from HuggingFace.
        device:        Torch device string.  Auto-selects CUDA if available.
        conf_threshold: Minimum confidence to keep a point (default ``0.1``).
        depth_edge_rtol: Relative tolerance for depth-discontinuity masking
                          (default ``0.03`` — same as the Pi3 examples).
        pixel_limit:   Maximum pixels per frame before downscaling (must be a
                       multiple of ``14 * 14``).  Default matches Pi3's default.
        use_vo:        When ``True`` and ``model_name="pi3x"``, use the
                       ``Pi3XVO`` sliding-window pipeline for long sequences.
        chunk_size:    Frames per chunk for VO pipeline (default ``16``).
        overlap:       Overlap between consecutive chunks (default ``6``).
    """

    _HF_IDS = {
        "pi3": "yyfz233/Pi3",
        "pi3x": "yyfz233/Pi3X",
    }

    def __init__(
        self,
        model_name: str = "pi3",
        checkpoint: str | None = None,
        device: str | None = None,
        conf_threshold: float = 0.1,
        depth_edge_rtol: float = 0.03,
        pixel_limit: int = 255000,
        use_vo: bool = False,
        chunk_size: int = 16,
        overlap: int = 6,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.depth_edge_rtol = depth_edge_rtol
        self.pixel_limit = pixel_limit
        self.use_vo = use_vo
        self.chunk_size = chunk_size
        self.overlap = overlap

        self._dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        logger.info("[Pi3Reconstructor] Loading %s on %s …", model_name, device)
        self._model = self._load_model(model_name, checkpoint)
        self._vo_pipe = None

        if use_vo and model_name == "pi3x":
            from pi3.pipe.pi3x_vo import Pi3XVO

            self._vo_pipe = Pi3XVO(self._model)

        self._to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        images: Sequence[np.ndarray] | torch.Tensor,
    ) -> Pi3Result:
        """Reconstruct a dynamic point cloud from a sequence of RGB images.

        Args:
            images: Either a list / array of ``[H, W, 3]`` uint8 numpy arrays,
                    or a pre-stacked ``[N, 3, H, W]`` float tensor in [0, 1].

        Returns:
            :class:`Pi3Result` with per-frame tensors (no batch dimension).
        """
        imgs_tensor = self._prepare_images(images)  # [N, 3, H, W] float [0,1]
        imgs_batched = imgs_tensor.unsqueeze(0)  # [1, N, 3, H, W]

        if self._vo_pipe is not None:
            raw = self._run_vo(imgs_batched)
        else:
            raw = self._run_standard(imgs_batched)

        return self._postprocess(raw, imgs_tensor)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str, checkpoint: str | None):
        key = model_name.lower()
        if key == "pi3":
            from pi3.models.pi3 import Pi3 as ModelClass
        elif key == "pi3x":
            from pi3.models.pi3x import Pi3X as ModelClass
        else:
            raise ValueError(f"Unknown model_name '{model_name}'. Use 'pi3' or 'pi3x'.")

        if checkpoint is not None:
            model = ModelClass().to(self.device).eval()
            ckpt_path = Path(checkpoint)
            if ckpt_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(ckpt_path))
            else:
                state = torch.load(
                    str(ckpt_path), map_location=self.device, weights_only=False
                )
            model.load_state_dict(state, strict=False)
        else:
            hf_id = self._HF_IDS.get(key)
            if hf_id is None:
                raise ValueError(f"No HuggingFace ID registered for '{model_name}'.")
            model = ModelClass.from_pretrained(hf_id).to(self.device).eval()

        return model

    def _prepare_images(
        self,
        images: Sequence[np.ndarray] | torch.Tensor,
    ) -> torch.Tensor:
        """Convert input to a [N, 3, H, W] float tensor on the target device."""
        if isinstance(images, torch.Tensor):
            # Already a tensor — just move to device.
            imgs = images.to(self.device).float()
            if imgs.ndim == 3:  # [3, H, W] → [1, 3, H, W]
                imgs = imgs.unsqueeze(0)
            return imgs

        # --- numpy / PIL list ---
        pil_list = [
            Image.fromarray(np.asarray(img, dtype=np.uint8))
            if not isinstance(img, Image.Image)
            else img
            for img in images
        ]

        # Compute a Pi3-compatible target resolution (multiples of 14).
        W0, H0 = pil_list[0].size  # type: ignore[misc]
        scale = math.sqrt(self.pixel_limit / max(W0 * H0, 1))
        k = round(W0 * scale / 14)
        m = round(H0 * scale / 14)
        while (k * 14) * (m * 14) > self.pixel_limit:
            if k / max(m, 1) > W0 / max(H0, 1):
                k -= 1
            else:
                m -= 1
        TW, TH = max(1, k) * 14, max(1, m) * 14

        tensors = []
        for img in pil_list:
            resized = img.resize((TW, TH), Image.Resampling.LANCZOS)  # type: ignore[arg-type]
            tensors.append(self._to_tensor(resized))  # [3, H, W] in [0, 1]

        return torch.stack(tensors).to(self.device)  # [N, 3, H, W]

    @torch.no_grad()
    def _run_standard(self, imgs: torch.Tensor) -> dict:
        """Run a single-pass Pi3 forward on imgs [1, N, 3, H, W]."""
        with torch.amp.autocast(self.device.type, dtype=self._dtype):  # type: ignore[attr-defined]
            return self._model(imgs)

    @torch.no_grad()
    def _run_vo(self, imgs: torch.Tensor) -> dict:
        """Run chunked Pi3XVO inference on imgs [1, N, 3, H, W]."""
        assert self._vo_pipe is not None
        return self._vo_pipe(  # type: ignore[misc]
            imgs=imgs,
            dtype=self._dtype,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            conf_thre=self.conf_threshold,
        )

    def _postprocess(self, raw: dict, imgs_tensor: torch.Tensor) -> Pi3Result:
        """Build a :class:`Pi3Result` from the raw model output dict."""
        from pi3.utils.geometry import depth_edge

        # Squeeze out the batch dimension (all shapes become [N, ...]).
        points = raw["points"][0].float()  # [N, H, W, 3]
        local_points = raw["local_points"][0].float()  # [N, H, W, 3]
        camera_poses = raw["camera_poses"][0].float()  # [N, 4, 4]
        conf_raw = raw["conf"][0].float()  # [N, H, W, 1] or [N, H, W]

        if conf_raw.ndim == 4:
            conf_raw = conf_raw[..., 0]
        conf = torch.sigmoid(conf_raw)  # [N, H, W]

        local_depth = local_points[..., 2]  # [N, H, W]
        non_edge = ~depth_edge(local_depth, rtol=self.depth_edge_rtol)
        mask = (conf > self.conf_threshold) & non_edge  # [N, H, W]

        return Pi3Result(
            points=points,
            local_points=local_points,
            camera_poses=camera_poses,
            conf=conf,
            mask=mask,
        )

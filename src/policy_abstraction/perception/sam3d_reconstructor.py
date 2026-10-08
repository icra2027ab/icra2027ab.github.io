from __future__ import annotations

import logging
from typing import Any

import numpy as np

from policy_abstraction.evaluation.obs_encoder import BaseObsEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAM3D: segment objects by keypoint prompt → 3-D Gaussian splat
# ---------------------------------------------------------------------------


class SAM3DObsEncoder(BaseObsEncoder):
    """Segments objects via SAM point prompts and reconstructs them as 3-D
    Gaussian splats using SAM 3D Objects (facebook/sam-3d-objects).

    No camera parameters are required — SAM 3D Objects reconstructs geometry
    purely from the RGB image and a segmentation mask.

    Pipeline (per object):
        1. Flip raw OpenGL image to display convention.
        2. Feed RGB + keypoint prompts into SAM → binary mask [H, W].
        3. Merge mask into the image alpha channel and feed into the SAM 3D
           Objects ``Inference`` pipeline → Gaussian splat.

    Output keys added to the returned obs dict::

        "sam3d_masks"     — list[np.ndarray [H, W] bool], one per object
        "sam3d_gaussians" — list[dict | None], one per object; each dict is the
                            raw output from the SAM 3D Objects pipeline
                            (contains ``"gs"`` for the Gaussian splat, plus
                            ``"rotation"``, ``"translation"``, ``"scale"``, …)

    Args:
        keypoints:          Per-object point prompts, each a list of (row, col)
                            pixel coordinates in **display convention** (row 0 =
                            top of image, as shown to a human viewer).
        camera_name:        Which camera to use (reads ``{camera_name}_image``
                            from obs).
        sam_model_name:     HuggingFace model ID for SAM segmentation.
        sam3d_config_path:  Path to ``pipeline.yaml`` from the SAM 3D Objects
                            repo (``checkpoints/hf/pipeline.yaml``).  When
                            *None* the model is downloaded automatically via
                            ``huggingface_hub`` from ``facebook/sam-3d-objects``.
        device:             Torch device.  Auto-selects CUDA when available.
    """

    def __init__(
        self,
        keypoints: list[list[tuple[int, int]]],
        camera_name: str = "agentview",
        sam_model_name: str = "facebook/sam-vit-base",
        sam3d_config_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._keypoints = keypoints  # list[list[tuple[int,int]]] (row, col) display
        self._camera_name = camera_name

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        logger.info("[SAM3DObsEncoder] Loading SAM: %s on %s", sam_model_name, device)
        from transformers import SamModel, SamProcessor

        self._sam_processor: SamProcessor = SamProcessor.from_pretrained(sam_model_name)  # type: ignore[assignment]
        self._sam_model: SamModel = SamModel.from_pretrained(sam_model_name)  # type: ignore[assignment]
        self._sam_model = self._sam_model.to(self._device).eval()  # type: ignore[arg-type]

        self._sam3d = self._load_sam3d(sam3d_config_path)

    # ------------------------------------------------------------------

    def _load_sam3d(self, config_path: str | None):
        """Download (if needed) and instantiate the SAM 3D Objects pipeline.

        The ``Inference`` class lives in ``notebook/inference.py`` inside the
        SAM 3D Objects repository cloned at ``third_party/sam-3d-objects/``.
        Weights are downloaded via ``huggingface_hub`` when *config_path* is
        ``None``.
        """
        import os
        import sys
        from pathlib import Path

        # notebook/inference.py lives in the cloned GitHub repo.
        repo_src = Path(__file__).parents[3] / "third_party" / "sam-3d-objects"
        notebook_dir = str(repo_src / "notebook")
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)

        # notebook/inference.py sets CUDA_HOME from CONDA_PREFIX at import time,
        # which fails outside a conda environment.  Ensure both vars are set.
        if "CONDA_PREFIX" not in os.environ:
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            os.environ.setdefault("CUDA_HOME", cuda_home)
            os.environ["CONDA_PREFIX"] = cuda_home

        if config_path is None:
            from huggingface_hub import snapshot_download

            hf_dir = snapshot_download("facebook/sam-3d-objects")
            config_path = os.path.join(hf_dir, "checkpoints", "pipeline.yaml")

        from inference import Inference  # notebook/inference.py from the repo

        logger.info("[SAM3DObsEncoder] Loading SAM 3D Objects: %s", config_path)
        return Inference(config_path, compile=False)

    # ------------------------------------------------------------------

    def encode(
        self,
        obs: dict[str, Any],
        robot_state: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = dict(obs)
        result["sam3d_masks"] = []
        result["sam3d_gaussians"] = []

        rgb_key = f"{self._camera_name}_image"

        if rgb_key not in obs:
            logger.warning("[SAM3DObsEncoder] '%s' not in obs, skipping.", rgb_key)
            return result

        # LIBERO observations use OpenGL convention (row 0 = bottom).
        # Flip rows to display convention (row 0 = top) for SAM.
        rgb_gl: np.ndarray = np.asarray(obs[rgb_key])  # [H, W, 3] uint8
        rgb_disp: np.ndarray = rgb_gl[::-1].copy()  # display convention

        for obj_idx, obj_kps in enumerate(self._keypoints):
            try:
                mask = self._segment(rgb_disp, obj_kps)  # [H, W] bool, display
            except Exception as exc:
                logger.warning(
                    "[SAM3DObsEncoder] SAM failed for object %d: %s", obj_idx, exc
                )
                result["sam3d_masks"].append(None)
                result["sam3d_gaussians"].append(None)
                continue

            result["sam3d_masks"].append(mask)

            try:
                # SAM 3D Objects: reconstruct from RGB + mask, no camera params needed.
                sam3d_output = self._sam3d(rgb_disp, mask)
            except Exception as exc:
                logger.warning(
                    "[SAM3DObsEncoder] SAM 3D Objects failed for object %d: %s",
                    obj_idx,
                    exc,
                )
                result["sam3d_gaussians"].append(None)
                continue

            result["sam3d_gaussians"].append(sam3d_output)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _segment(self, rgb: np.ndarray, keypoints: list[tuple[int, int]]) -> np.ndarray:
        """Run SAM with point prompts and return the best mask.

        Args:
            rgb:       ``[H, W, 3]`` uint8 in display convention.
            keypoints: List of ``(row, col)`` prompts in display convention.

        Returns:
            ``[H, W]`` bool mask.
        """
        import torch
        from PIL import Image

        pil_img = Image.fromarray(rgb)

        # SAM processor expects input_points as [x, y] = [col, row].
        input_points = [[[float(c), float(r)] for r, c in keypoints]]
        input_labels = [[1] * len(keypoints)]

        inputs = self._sam_processor(
            images=pil_img,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        # post_process_masks returns List[Tensor[num_pred, num_masks, H, W]]
        # iou_scores shape: [batch, num_pred, num_masks]
        masks = self._sam_processor.image_processor.post_process_masks(  # type: ignore[attr-defined]
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # Select the candidate mask with the highest IoU score.
        # masks[0]: [num_pred, num_masks, H, W]  scores[0]: [num_pred, num_masks]
        mask_tensor = masks[0]  # [num_pred, num_masks, H, W]
        score_tensor = outputs.iou_scores[0]  # [num_pred, num_masks]
        best = score_tensor.flatten().argmax()
        flat_masks = mask_tensor.reshape(-1, *mask_tensor.shape[-2:])  # [N, H, W]
        mask_np = flat_masks[best].numpy().astype(bool)
        return mask_np

"""ReKep keypoint-extraction observation encoder.

Reproduces the keypoint proposal pipeline from:

    Huang, W. et al. "ReKep: Spatio-Temporal Reasoning of Relational Keypoint
    Constraints for Robotic Manipulation." NeurIPS 2024.

Pipeline (§ keypoint proposals in the paper)
--------------------------------------------
1. Run **DINOv2-with-registers (ViT-S/14)** to obtain dense patch features
   ``F_patch ∈ R^{h′×w′×d}``.
2. **Bilinear-interpolate** the patch grid to the original image resolution,
   yielding ``F_interp ∈ R^{h×w×d}``.
3. Run **SAM** in automatic mode to extract all scene masks
   ``M = {m_1, …, m_n}``.
4. For each mask *m_i*:

   a. Gather the features at the masked pixels.
   b. Project to 3 dimensions via **PCA** (suppresses texture artefacts).
   c. Cluster with **k-means (k = 5, Euclidean distance)**.
   d. Use the **median pixel coordinate** of each cluster as a keypoint
      candidate.

5. **Unproject** each 2-D candidate to a world-frame 3-D point using the
   registered depth image and calibrated camera intrinsics / extrinsics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as SklearnPCA

from policy_abstraction.evaluation.obs_encoder import BaseObsEncoder

logger = logging.getLogger(__name__)


class RekepObsEncoder(BaseObsEncoder):
    """Extracts relational keypoints from RGB(-D) observations using the
    ReKep keypoint proposal pipeline.

    Output keys added to the observation dict
    -----------------------------------------
    * ``{output_key}``          — ``[K, 3]`` float32 world-frame keypoints
      (NaN where depth is invalid).  Only written when a depth image *and*
      matching camera parameters are available.
    * ``{output_key}_2d``       — ``[K, 2]`` int32 pixel ``(row, col)``
      coordinates in **display convention** (row 0 = top of image).
    * ``{output_key}_mask_ids`` — ``[K]`` int32 array mapping each keypoint
      to its source SAM mask index.

    Image convention
    ----------------
    Raw LIBERO observations use **OpenGL convention** (row 0 = bottom of the
    scene).  The encoder flips images to top-left-origin before running
    DINOv2 / SAM, and converts 2-D pixel coordinates back to OpenGL
    convention before the depth look-up.

    Camera parameters
    -----------------
    Dicts follow the same schema as ``env.get_camera_params()``:

    * ``"intrinsic"`` — ``[3, 3]`` pinhole matrix ``K`` (pixel units,
      top-left origin, OpenCV convention).
    * ``"extrinsic"`` — ``[4, 4]`` **camera-to-world** pose ``E``
      (robosuite convention with MuJoCo axis correction applied).
    """

    # ViT-S/14 patch size in pixels; also readable from model.config.patch_size.
    _PATCH_SIZE: int = 14

    def __init__(
        self,
        camera_params: dict[str, dict[str, np.ndarray]],
        rgb_camera: str = "agentview",
        depth_camera: str = "agentview",
        dino_model_name: str = "facebook/dinov2-small",
        sam_model_name: str = "facebook/sam-vit-base",
        n_clusters: int = 5,
        pca_components: int = 3,
        depth_scale: float = 1.0,
        max_depth: float = 3.0,
        device: str | None = None,
        output_key: str = "rekep_keypoints",
    ) -> None:
        """
        Args:
            camera_params: Mapping ``{camera_name: {"intrinsic": K, "extrinsic": E}}``.
                Update at run-time with :meth:`set_camera_params`.
            rgb_camera: Prefix for the colour-image observation key
                (reads ``{rgb_camera}_image``).
            depth_camera: Prefix for the depth observation key
                (reads ``{depth_camera}_depth``).  May differ from
                ``rgb_camera`` when different cameras supply colour / depth.
            dino_model_name: HuggingFace model ID for DINOv2 (ViT-S/14).
                Default: ``"facebook/dinov2-small"`` (compatible with
                transformers ≥ 4.x).  Pass
                ``"facebook/dinov2-with-registers-small"`` when using
                transformers ≥ 4.46 for the exact paper variant.
            sam_model_name: HuggingFace model ID for SAM.
                Default: ``"facebook/sam-vit-base"``.
            n_clusters: Number of k-means clusters per mask (*k* in the paper,
                default 5).
            pca_components: PCA output dimensionality applied before clustering
                (3 in the paper).
            depth_scale: Scalar multiplier converting raw depth values to
                metres.  Use ``1.0`` (default) when depth is already metric.
            max_depth: Points with depth (metres) beyond this value are set to
                NaN.
            device: Torch device string.  Auto-selects ``"cuda"`` when
                available, otherwise ``"cpu"``.
            output_key: Base key written into the returned obs dict.
        """
        self._camera_params = camera_params
        self._rgb_camera = rgb_camera
        self._depth_camera = depth_camera
        self._n_clusters = n_clusters
        self._pca_components = pca_components
        self._depth_scale = float(depth_scale)
        self._max_depth = float(max_depth)
        self._output_key = output_key

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # ── DINOv2 (with-registers variant requires transformers ≥ 4.46) ─────
        logger.info("[RekepObsEncoder] Loading DINOv2: %s", dino_model_name)
        from transformers import AutoImageProcessor

        self._dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)

        # Try the generic AutoModel first; fall back to the concrete class for
        # older transformers versions that don't register every DINOv2 variant.
        try:
            from transformers import AutoModel

            self._dino_model = AutoModel.from_pretrained(dino_model_name)
        except (ValueError, KeyError):
            from transformers import Dinov2Model

            self._dino_model = Dinov2Model.from_pretrained(dino_model_name)

        self._dino_model = self._dino_model.to(self._device).eval()  # type: ignore[arg-type]

        # ── SAM automatic mask generator ───────────────────────────────────
        logger.info("[RekepObsEncoder] Loading SAM: %s", sam_model_name)
        from transformers import pipeline as hf_pipeline

        # transformers pipeline expects an integer device index for GPU or -1 for CPU.
        sam_device = 0 if self._device.startswith("cuda") else -1
        self._sam_pipeline = hf_pipeline(
            "mask-generation",
            model=sam_model_name,
            device=sam_device,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_camera_params(
        self, camera_params: dict[str, dict[str, np.ndarray]]
    ) -> None:
        """Replace the stored camera intrinsics / extrinsics.

        Call this at every replan step when the wrist-camera extrinsic has
        changed.

        Args:
            camera_params: Same format as the constructor argument.
        """
        self._camera_params = camera_params

    # ------------------------------------------------------------------
    # BaseObsEncoder interface
    # ------------------------------------------------------------------

    def encode(
        self,
        obs: dict[str, Any],
        robot_state: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the full ReKep keypoint-extraction pipeline on *obs*.

        Args:
            obs:         Raw observation dict.  Must contain at least
                         ``{rgb_camera}_image`` (uint8 ``[H, W, 3]``).
                         When ``{depth_camera}_depth`` is also present and
                         camera parameters are configured, keypoints are
                         additionally unprojected to world-frame 3-D.
            robot_state: Not used; accepted for interface compatibility.

        Returns:
            Shallow copy of *obs* augmented with ReKep keypoint entries
            (see class docstring for key names).
        """
        result = dict(obs)

        rgb_key = f"{self._rgb_camera}_image"
        if rgb_key not in obs:
            logger.warning(
                "[RekepObsEncoder] Key '%s' not found in obs; skipping.", rgb_key
            )
            return result

        # Raw LIBERO images use OpenGL convention (row 0 = bottom of scene).
        # Flip to display/standard convention (row 0 = top) for DINOv2 & SAM.
        rgb_opengl: np.ndarray = np.asarray(obs[rgb_key])  # [H, W, 3] uint8
        rgb_display: np.ndarray = rgb_opengl[::-1].copy()  # row-flip

        # ── Steps 1 & 2: DINOv2 → dense feature map ────────────────────────
        F_interp = self._extract_dino_features(rgb_display)  # [H, W, d]

        # ── Step 3: SAM automatic segmentation ─────────────────────────────
        masks = self._get_sam_masks(rgb_display)  # list[bool [H, W]]

        # ── Steps 4: PCA + k-means → 2-D keypoint candidates ───────────────
        keypoints_2d_display, mask_ids = self._extract_keypoints_2d(F_interp, masks)
        # keypoints_2d_display: [K, 2] (row, col) in display convention

        result[f"{self._output_key}_2d"] = keypoints_2d_display
        result[f"{self._output_key}_mask_ids"] = mask_ids

        # ── Step 5: Unproject to world-frame 3-D ────────────────────────────
        depth_key = f"{self._depth_camera}_depth"
        cam_params = self._camera_params.get(self._depth_camera)
        if depth_key in obs and cam_params is not None:
            depth = np.asarray(obs[depth_key], dtype=np.float64)
            if depth.ndim == 3:
                depth = depth[..., 0]  # [H, W, 1] → [H, W]
            depth = depth * self._depth_scale

            K = np.asarray(cam_params["intrinsic"], dtype=np.float64)
            E = np.asarray(cam_params["extrinsic"], dtype=np.float64)

            # Convert display (top-left origin) rows back to OpenGL (bottom-
            # left origin) for the depth look-up.
            H = rgb_opengl.shape[0]
            keypoints_2d_opengl = keypoints_2d_display.copy()
            if len(keypoints_2d_opengl) > 0:
                keypoints_2d_opengl[:, 0] = (H - 1) - keypoints_2d_display[:, 0]

            keypoints_3d = self._project_to_3d(keypoints_2d_opengl, depth, K, E)
            result[self._output_key] = keypoints_3d

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_dino_features(self, image: np.ndarray) -> np.ndarray:
        """Run DINOv2 and bilinear-interpolate patch tokens to full resolution.

        Args:
            image: ``[H, W, 3]`` uint8 RGB in **display convention**
                   (row 0 = top of image).

        Returns:
            ``F_interp``: ``[H, W, d]`` float32 dense feature map.
        """
        H, W = image.shape[:2]
        pil_img = Image.fromarray(image)

        inputs = self._dino_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._dino_model(**inputs)  # type: ignore[operator]

        # Determine processed spatial dimensions from the preprocessed tensor.
        # pixel_values: [1, 3, h_proc, w_proc]
        h_proc, w_proc = inputs["pixel_values"].shape[2:]
        patch_size = getattr(self._dino_model.config, "patch_size", self._PATCH_SIZE)  # type: ignore[attr-defined]
        h_patches = h_proc // patch_size
        w_patches = w_proc // patch_size

        # last_hidden_state: [1, T, d]
        # Token layout for DINOv2-with-registers:
        #   [CLS, reg_0, …, reg_{r-1}, patch_0, …, patch_{h*w - 1}]
        # We strip all prefix tokens (CLS + registers) to isolate patch tokens.
        hidden = outputs.last_hidden_state  # [1, T, d]
        num_prefix = hidden.shape[1] - h_patches * w_patches
        patch_tokens = hidden[:, num_prefix:, :]  # [1, h_patches*w_patches, d]

        d = patch_tokens.shape[-1]
        # Reshape to a 2-D spatial grid: [1, d, h_patches, w_patches]
        F_patch = patch_tokens.reshape(1, h_patches, w_patches, d).permute(0, 3, 1, 2)

        # Bilinear interpolation to the original image resolution.
        F_interp = F.interpolate(
            F_patch.float(), size=(H, W), mode="bilinear", align_corners=False
        )  # [1, d, H, W]

        return F_interp.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, d]

    def _get_sam_masks(self, image: np.ndarray) -> list[np.ndarray]:
        """Return all binary scene masks via SAM automatic segmentation.

        Args:
            image: ``[H, W, 3]`` uint8 RGB in **display convention**.

        Returns:
            List of ``[H, W]`` bool arrays.  Returns an empty list when SAM
            produces no segments or raises an exception.
        """
        try:
            pil_img = Image.fromarray(image)
            outputs = self._sam_pipeline(pil_img, points_per_batch=64)

            # The transformers mask-generation pipeline output format varies
            # slightly across versions; handle all common shapes defensively.
            if isinstance(outputs, dict):
                raw_masks = outputs.get("masks", [])
            else:
                raw_masks = outputs  # some versions return a bare list

            masks: list[np.ndarray] = []
            for m in raw_masks or []:
                if isinstance(m, dict):
                    # Transformers ≥ 4.29 uses "mask"; older code uses "segmentation".
                    seg = m.get("mask", m.get("segmentation"))
                    if seg is not None:
                        masks.append(np.asarray(seg, dtype=bool))
                elif isinstance(m, (np.ndarray, bool)):
                    masks.append(np.asarray(m, dtype=bool))
                elif hasattr(m, "__array__"):
                    masks.append(np.asarray(m, dtype=bool))
            return masks
        except Exception as exc:
            logger.warning("[RekepObsEncoder] SAM mask generation failed: %s", exc)
            return []

    def _extract_keypoints_2d(
        self,
        F_interp: np.ndarray,
        masks: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply PCA + k-means within each mask to produce 2-D keypoints.

        For each mask the procedure (matching the paper) is:

        1. Gather the feature vectors of all masked pixels.
        2. Apply PCA to project to ``pca_components`` dimensions.
        3. Cluster the PCA-projected features with k-means (k = ``n_clusters``,
           Euclidean distance).
        4. Use the **median pixel coordinate** of each cluster as a keypoint
           candidate (the paper calls these "median centroids").

        Args:
            F_interp: ``[H, W, d]`` dense feature map.
            masks:    List of ``[H, W]`` bool arrays from SAM.

        Returns:
            keypoints_2d: ``[K, 2]`` int32 ``(row, col)`` in **display
                          convention** (row 0 = top).
            mask_ids:     ``[K]`` int32 — SAM mask index for each keypoint.
        """
        all_keypoints: list[np.ndarray] = []
        all_mask_ids: list[int] = []

        d = F_interp.shape[2]

        for mask_idx, mask in enumerate(masks):
            pixel_coords = np.argwhere(mask)  # [n, 2]: (row, col)
            n_pixels = len(pixel_coords)

            # Need enough pixels for both PCA and k-means initialisation.
            min_pixels = max(self._n_clusters, self._pca_components + 1)
            if n_pixels < min_pixels:
                continue

            features = F_interp[mask].astype(np.float32)  # [n, d]

            # ── PCA ─────────────────────────────────────────────────────────
            n_components = min(self._pca_components, n_pixels - 1, d)
            if n_components < 1:
                continue
            try:
                pca = SklearnPCA(n_components=n_components)
                features_pca = pca.fit_transform(features)  # [n, n_components]
            except Exception as exc:
                logger.debug(
                    "[RekepObsEncoder] PCA failed for mask %d: %s", mask_idx, exc
                )
                continue

            # ── k-means ──────────────────────────────────────────────────────
            k = min(self._n_clusters, n_pixels)
            try:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # type: ignore[arg-type]
                labels = kmeans.fit_predict(features_pca)  # [n]
            except Exception as exc:
                logger.debug(
                    "[RekepObsEncoder] k-means failed for mask %d: %s", mask_idx, exc
                )
                continue

            # ── Median centroid per cluster ───────────────────────────────────
            for cluster_id in range(k):
                cluster_pixels = pixel_coords[labels == cluster_id]  # [m, 2]
                if len(cluster_pixels) == 0:
                    continue
                # Median over pixel coordinates → robust to outliers.
                median_rc = np.median(cluster_pixels, axis=0).astype(np.int32)
                all_keypoints.append(median_rc)
                all_mask_ids.append(mask_idx)

        if not all_keypoints:
            return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.int32)

        return (
            np.stack(all_keypoints, axis=0),  # [K, 2]
            np.array(all_mask_ids, dtype=np.int32),  # [K]
        )

    def _project_to_3d(
        self,
        keypoints_2d: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
    ) -> np.ndarray:
        """Unproject 2-D pixel coordinates to world-frame 3-D points.

        Uses the standard pinhole model and robosuite / LIBERO camera
        conventions (same as :class:`PointCloudObsEncoder`).

        Args:
            keypoints_2d: ``[N, 2]`` int array of ``(row, col)`` in
                          **OpenGL convention** (row 0 = bottom of scene).
            depth:        ``[H, W]`` float64 metric depth in OpenGL convention.
            K:            ``[3, 3]`` intrinsic matrix (top-left origin, OpenCV).
            E:            ``[4, 4]`` camera-to-world extrinsic matrix.

        Returns:
            ``[N, 3]`` float32 world-frame points.  Points with zero,
            non-finite, or out-of-range depths are set to ``NaN``.
        """
        if len(keypoints_2d) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        H = depth.shape[0]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        rows = np.clip(keypoints_2d[:, 0].astype(np.int64), 0, depth.shape[0] - 1)
        cols = np.clip(keypoints_2d[:, 1].astype(np.int64), 0, depth.shape[1] - 1)

        d = depth[rows, cols]  # metric depth at each keypoint, [N]

        # OpenGL row → K-space vertical coordinate:
        #   v_K = H - 1 - row_opengl
        u_k = cols.astype(np.float64)
        v_k = (H - 1) - rows.astype(np.float64)

        x_c = (u_k - cx) * d / fx
        y_c = (v_k - cy) * d / fy
        z_c = d

        pts_cam_h = np.stack([x_c, y_c, z_c, np.ones_like(z_c)], axis=-1)  # [N, 4]
        pts_world = (E @ pts_cam_h.T).T[:, :3]  # [N, 3]

        # Invalidate points with bad depth readings.
        invalid = (d <= 0.0) | ~np.isfinite(d) | (d > self._max_depth)
        pts_world[invalid] = np.nan

        return pts_world.astype(np.float32)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        encoded_obs: dict[str, Any],
        radius: int = 6,
        alpha: float = 0.45,
        show_mask_id: bool = True,
        font_scale: float = 0.35,
    ) -> np.ndarray:
        """Render the extracted keypoints overlaid on the RGB image.

        Each keypoint is drawn as a filled circle whose colour encodes the SAM
        mask it came from (one hue per unique mask id, cycling through a fixed
        palette).  Optionally, the numeric mask-id label is printed next to
        every keypoint.  When 3-D positions are available, the world-frame
        ``(x, y, z)`` coordinates are shown in the tooltip text.

        Args:
            encoded_obs: The dict returned by :meth:`encode`.  Must contain
                         ``{output_key}_2d``, ``{output_key}_mask_ids``, and
                         (optionally) ``{output_key}`` (world-frame XYZ) as
                         well as the raw ``{rgb_camera}_image``.
            radius:      Radius in pixels of each keypoint circle (default 6).
            alpha:       Opacity of the filled circle overlay blended on top of
                         the image.  ``1.0`` = fully opaque (default 0.45).
            show_mask_id: When ``True``, draw the integer mask-id next to each
                          keypoint circle.
            font_scale:  OpenCV font scale for the mask-id label (default 0.35).

        Returns:
            ``[H, W, 3]`` uint8 BGR image with keypoints annotated, ready for
            ``cv2.imwrite`` / ``cv2.imshow``.  Returns the original image
            (flipped to display convention) unchanged when no keypoints were
            found.
        """
        import cv2

        rgb_key = f"{self._rgb_camera}_image"
        kp_2d_key = f"{self._output_key}_2d"
        kp_id_key = f"{self._output_key}_mask_ids"
        kp_3d_key = self._output_key

        # Obtain base image in display convention (row 0 = top of scene).
        rgb_raw = np.asarray(
            encoded_obs.get(rgb_key, np.zeros((128, 128, 3), np.uint8))
        )
        # Raw LIBERO obs is stored OpenGL-convention; flip to display.
        img_display = rgb_raw[::-1].copy()
        # Convert RGB → BGR for OpenCV.
        canvas = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

        keypoints_2d = encoded_obs.get(kp_2d_key)
        mask_ids = encoded_obs.get(kp_id_key)
        keypoints_3d = encoded_obs.get(kp_3d_key)  # may be None

        if keypoints_2d is None or len(keypoints_2d) == 0:
            return canvas
        if mask_ids is None:
            return canvas

        # Fixed BGR colour palette (one colour per unique mask id, cycling).
        # Chosen to be perceptually distinct on typical scene backgrounds.
        _PALETTE_BGR = [
            (0, 200, 0),  # green
            (0, 0, 220),  # red
            (220, 100, 0),  # blue
            (0, 180, 220),  # yellow
            (180, 0, 180),  # magenta
            (0, 180, 100),  # teal-ish
            (255, 120, 0),  # azure
            (100, 255, 100),  # light green
            (100, 100, 255),  # salmon
            (180, 255, 0),  # lime
        ]

        unique_ids = sorted(set(mask_ids.tolist()))
        id_to_colour = {
            mid: _PALETTE_BGR[i % len(_PALETTE_BGR)] for i, mid in enumerate(unique_ids)
        }

        overlay = canvas.copy()

        for rc, mid in zip(keypoints_2d, mask_ids, strict=False):
            row, col = int(rc[0]), int(rc[1])
            colour = id_to_colour[int(mid)]

            # Filled circle on the overlay layer.
            cv2.circle(
                overlay, (col, row), radius, colour, thickness=-1, lineType=cv2.LINE_AA
            )
            # Thin white border for legibility.
            cv2.circle(
                overlay,
                (col, row),
                radius,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # Blend the overlay with the original canvas.
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

        # Draw mask-id labels (and optional 3-D coordinates) after blending.
        for k_idx, (rc, mid) in enumerate(zip(keypoints_2d, mask_ids, strict=False)):
            row, col = int(rc[0]), int(rc[1])
            colour = id_to_colour[int(mid)]

            if show_mask_id:
                label = f"m{int(mid)}"
                if keypoints_3d is not None and k_idx < len(keypoints_3d):
                    xyz = keypoints_3d[k_idx]
                    if not np.any(np.isnan(xyz)):
                        label += f" ({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})"

                text_x = col + radius + 2
                text_y = row + 4

                # Shadow for contrast.
                cv2.putText(
                    canvas,
                    label,
                    (text_x + 1, text_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

        return canvas


# ---------------------------------------------------------------------------
# Self-contained smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run the full ReKep pipeline on the first frame of an episode file.

    Usage (from repo root, with the project venv active):

        python -m policy_abstraction.perception.rekep_encoder \\
            data/libero_spatial/episodes/<episode>.vla

        # or supply a plain RGB image (PNG / JPG) instead of a .vla file:
        python -m policy_abstraction.perception.rekep_encoder \\
            path/to/image.png

    The annotated image is saved to tmp/rekep_test_output.png.
    """
    import argparse
    import importlib.util
    import sys
    from pathlib import Path

    import cv2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="ReKep encoder smoke-test: annotate keypoints on a frame."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "Path to a .vla episode file OR a plain image file (PNG/JPG). "
            "When omitted, a synthetic random RGB image is used."
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Timestep index to use from a .vla episode (default: 0).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera name prefix for the RGB key (default: agentview).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tmp/rekep_test_output.png",
        help="Output path for the annotated image (default: tmp/rekep_test_output.png).",
    )
    args = parser.parse_args()

    # ── Build obs dict ──────────────────────────────────────────────────────
    obs: dict = {}
    source = Path(args.source) if args.source else None

    if source is None:
        # Synthetic 256×256 random RGB image.
        logger.info("No source provided – using a synthetic 256×256 random image.")
        rng = np.random.default_rng(42)
        fake_img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        obs[f"{args.camera}_image"] = fake_img

    elif source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        # Plain image file.
        bgr = cv2.imread(str(source))
        if bgr is None:
            sys.exit(f"ERROR: Could not read image: {source}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        obs[f"{args.camera}_image"] = rgb
        logger.info("Loaded plain image %s  shape=%s", source.name, rgb.shape)

    elif source.suffix.lower() == ".vla":
        # robodm VLA episode – dynamically load VLALoader to avoid hard
        # dependency on the full package __init__.
        _repo_root = Path(__file__).resolve().parent.parent.parent
        _loader_path = Path(__file__).resolve().parent / "loader.py"
        spec = importlib.util.spec_from_file_location("loader", _loader_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec from {_loader_path}")
        _loader_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_loader_mod)  # type: ignore[attr-defined]
        VLALoader = _loader_mod.VLALoader

        with VLALoader(str(source)) as loader:
            frame = loader.get_frame(args.frame)
        obs = frame["obs"]
        logger.info(
            "Loaded .vla episode %s  frame=%d  obs keys=%s",
            source.name,
            args.frame,
            list(obs.keys()),
        )
        # VLA observations use agentview_image directly (already mapped by
        # VLALoader); override the camera arg to match.
        if "agentview_image" in obs and args.camera == "agentview":
            args.camera = "agentview"
        elif obs and args.camera + "_image" not in obs:
            # Fall back to the first available image key.
            img_keys = [k for k in obs if k.endswith("_image")]
            if img_keys:
                args.camera = img_keys[0].replace("_image", "")
                logger.info("Auto-selected camera prefix: %s", args.camera)
    else:
        sys.exit(f"ERROR: Unrecognised source type: {source.suffix}")

    # ── Instantiate encoder (no camera params → 2-D only) ──────────────────
    logger.info("Instantiating RekepObsEncoder …")
    encoder = RekepObsEncoder(
        camera_params={},  # no depth/extrinsics available in this test
        rgb_camera=args.camera,
        depth_camera=args.camera,
    )

    # ── Encode ─────────────────────────────────────────────────────────────
    logger.info("Running encode() …")
    encoded = encoder.encode(obs)

    kp_2d = encoded.get(f"{encoder._output_key}_2d", np.zeros((0, 2), dtype=np.int32))
    mask_ids = encoded.get(
        f"{encoder._output_key}_mask_ids", np.zeros(0, dtype=np.int32)
    )
    n_masks = int(mask_ids.max()) + 1 if len(mask_ids) else 0

    logger.info("Keypoints found: %d  (from %d SAM masks)", len(kp_2d), n_masks)
    if len(kp_2d):
        logger.info("Sample keypoints (row, col):\n%s", kp_2d[:10])

    # ── Visualise ──────────────────────────────────────────────────────────
    logger.info("Running visualize() …")
    annotated_bgr = encoder.visualize(encoded)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated_bgr)
    logger.info("Saved annotated image → %s", out_path.resolve())

    # ── Assertions (basic sanity checks) ───────────────────────────────────
    assert annotated_bgr.ndim == 3 and annotated_bgr.shape[2] == 3, (
        "visualize() must return a [H, W, 3] image."
    )
    assert annotated_bgr.dtype == np.uint8, "visualize() must return uint8."
    assert len(kp_2d) == len(mask_ids), (
        "keypoints_2d and mask_ids must have the same length."
    )
    if len(kp_2d):
        H, W = annotated_bgr.shape[:2]
        assert np.all(kp_2d[:, 0] < H) and np.all(kp_2d[:, 1] < W), (
            "All keypoint pixel coordinates must be within image bounds."
        )

    logger.info("All assertions passed. Test complete.")

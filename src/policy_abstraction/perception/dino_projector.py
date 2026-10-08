"""DINOv2 feature extraction with PCA projection for object interactions.

This module provides tools to extract DINOv2 features from segmented object regions,
compute mean-centered features, and apply PCA for dimensionality reduction. The PCA
is computed on the first interaction frame and then applied consistently across
subsequent frames.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import pipeline


class DinoPCAProjector:
    """Extracts DINOv2 features from segmented objects and applies PCA projection.

    This class:
    1. Loads a DINOv2 model via Transformers pipeline for feature extraction
    2. Extracts features from pixels in a segmented region
    3. Computes mean-subtracted features for the segmented object
    4. Performs PCA on the first interaction frame (stores the PCA model)
    5. Applies the same PCA to subsequent frames
    6. Visualizes projected features overlaid on the image

    Usage
    -----
    ::
        projector = DinoPCAProjector(
            model_name="facebook/dinov2",
            n_pca_components=3
        )

        # First frame with object interaction
        vis_img = projector.process_frame(
            rgb_image,           # (H, W, 3) uint8 RGB
            seg_mask,            # (H, W) int32 segmentation IDs
            interacting_ids      # set of IDs to process
        )

        # Subsequent frames - uses same PCA
        vis_img = projector.process_frame(rgb_image, seg_mask, interacting_ids)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2",
        n_pca_components: int = 3,
        device: Optional[int] = None,
    ):
        """Initialize the DINOv2 PCA projector using Transformers pipeline.

        Args:
            model_name: HuggingFace model name. ref: https://github.com/facebookresearch/dinov3
            n_pca_components: Number of PCA components (default 3 for RGB visualization)
            device: Device ID for CUDA (0, 1, etc.) or -1 for CPU, None for auto-detect
        """
        self.n_pca_components = n_pca_components

        # Auto-detect device if not specified
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        # Load DINOv2 model using transformers pipeline
        print(f"Loading DINOv2 model via Transformers: {model_name}")
        print(f"Device: {'CUDA' if device >= 0 else 'CPU'}")

        self.feature_extractor = pipeline(
            model=model_name,
            task="image-feature-extraction",
            device=device,
        )

        # Store model info
        self.model_name = model_name

        # PCA state (computed on first interaction frame)
        self.pca_model: Optional[PCA] = None
        self.feature_dim: Optional[int] = None

        # Cache for last processed features (for debugging)
        self._last_features_2d = None
        self._last_pca_features = None

        print("DINOv2 model loaded successfully")

    def _preprocess_image(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image.

        Args:
            image: (H, W, 3) uint8 RGB image

        Returns:
            PIL Image
        """
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    def _extract_features(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Extract DINOv2 features from image using Transformers pipeline.

        Args:
            image: (H, W, 3) uint8 RGB image

        Returns:
            features: (feat_h, feat_w, feat_dim) feature map
            original_shape: (H, W) original image shape
        """
        H_orig, W_orig = image.shape[:2]

        # Convert to PIL Image
        pil_image = self._preprocess_image(image)

        # Extract features using pipeline
        # The pipeline returns a list with shape info
        features_output = self.feature_extractor(pil_image)

        # Extract the feature tensor
        # Output format: [{'features': tensor of shape [1, num_patches, feat_dim]}]
        if isinstance(features_output, list) and len(features_output) > 0:
            features_dict = features_output[0]
            if isinstance(features_dict, dict) and "features" in features_dict:
                features_tensor = features_dict["features"]
            else:
                # Fallback: assume the output is the features directly
                features_tensor = features_output[0]
        else:
            features_tensor = features_output

        # Convert to numpy
        if isinstance(features_tensor, torch.Tensor):
            features_np = features_tensor.cpu().numpy()
        else:
            features_np = np.array(features_tensor)

        # Reshape from (1, num_patches, feat_dim) or (num_patches, feat_dim) to (feat_h, feat_w, feat_dim)
        # Note: DINOv2 includes a CLS token at position 0, so we need to skip it
        if features_np.ndim == 3:
            # Shape: (batch_size, num_tokens, feat_dim)
            batch_size, num_tokens, feat_dim = features_np.shape
            # Skip CLS token (first token)
            patch_tokens = features_np[0, 1:, :]  # (num_patches, feat_dim)
        elif features_np.ndim == 2:
            # Shape: (num_tokens, feat_dim)
            num_tokens, feat_dim = features_np.shape
            # Skip CLS token (first token)
            patch_tokens = features_np[1:, :]  # (num_patches, feat_dim)
        else:
            # Unexpected shape - use as is
            patch_tokens = features_np
            feat_dim = features_np.shape[-1]

        # Reshape to spatial grid
        num_patches = patch_tokens.shape[0]
        feat_h = feat_w = int(np.sqrt(num_patches))
        features = patch_tokens.reshape(feat_h, feat_w, feat_dim)

        self.feature_dim = feat_dim
        return features, (H_orig, W_orig)

    def _upsample_features(
        self, features: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:
        """Upsample feature map to match original image size.

        Args:
            features: (feat_h, feat_w, feat_dim) feature map
            target_shape: (H, W) target spatial dimensions

        Returns:
            (H, W, feat_dim) upsampled features
        """
        feat_h, feat_w, feat_dim = features.shape
        H, W = target_shape

        # Reshape to (feat_dim, feat_h, feat_w) for cv2.resize
        features_t = np.transpose(features, (2, 0, 1))

        # Upsample each feature channel
        upsampled = np.zeros((feat_dim, H, W), dtype=np.float32)
        for i in range(feat_dim):
            upsampled[i] = cv2.resize(
                features_t[i], (W, H), interpolation=cv2.INTER_LINEAR
            )

        # Reshape back to (H, W, feat_dim)
        return np.transpose(upsampled, (1, 2, 0))

    def _extract_segmented_features(
        self, features: np.ndarray, seg_mask: np.ndarray, interacting_ids: set[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features for pixels in interacting segments.

        Args:
            features: (H, W, feat_dim) upsampled feature map
            seg_mask: (H, W) or (H, W, 1) segmentation mask
            interacting_ids: Set of segment IDs to extract

        Returns:
            seg_features: (N, feat_dim) features for N segmented pixels
            seg_mask_binary: (H, W) binary mask of extracted pixels
        """
        # Handle (H, W, 1) format
        if seg_mask.ndim == 3:
            seg_mask = seg_mask[..., 0]

        # Create binary mask for all interacting IDs
        binary_mask = np.zeros_like(seg_mask, dtype=bool)
        for seg_id in interacting_ids:
            binary_mask |= seg_mask == seg_id

        # Extract features
        seg_features = features[binary_mask]  # (N, feat_dim)

        return seg_features, binary_mask.astype(np.uint8)

    def _mean_center_features(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subtract mean from features.

        Args:
            features: (N, feat_dim) features

        Returns:
            centered_features: (N, feat_dim) mean-centered features
            mean: (feat_dim,) computed mean
        """
        mean = features.mean(axis=0, keepdims=True)  # (1, feat_dim)
        centered = features - mean
        return centered, mean[0]

    def _fit_pca(self, centered_features: np.ndarray) -> PCA:
        """Fit PCA on mean-centered features (first interaction frame).

        Args:
            centered_features: (N, feat_dim) mean-centered features

        Returns:
            Fitted PCA model
        """
        pca = PCA(n_components=self.n_pca_components)
        pca.fit(centered_features)

        explained_var = pca.explained_variance_ratio_
        print(
            f"PCA fitted: {self.n_pca_components} components, "
            f"explained variance: {explained_var.sum():.2%} "
            f"({', '.join(f'{v:.1%}' for v in explained_var)})"
        )

        return pca

    def _project_pca(
        self, centered_features: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Project mean-centered features using PCA.

        Args:
            centered_features: (N, feat_dim) mean-centered features
            mask: (H, W) binary mask indicating which pixels have features

        Returns:
            (H, W, n_pca_components) PCA-projected feature map
        """
        if self.pca_model is None:
            raise RuntimeError("PCA model not fitted. Process first frame first.")

        # Project
        projected = self.pca_model.transform(centered_features)  # (N, n_pca_components)

        # Reconstruct spatial map
        H, W = mask.shape
        pca_map = np.zeros((H, W, self.n_pca_components), dtype=np.float32)
        pca_map[mask > 0] = projected

        return pca_map

    def _visualize_pca_map(
        self,
        pca_map: np.ndarray,
        mask: np.ndarray,
        background_image: np.ndarray,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """Visualize PCA-projected features as RGB overlay.

        Args:
            pca_map: (H, W, n_pca_components) PCA features
            mask: (H, W) binary mask of valid pixels
            background_image: (H, W, 3) RGB image to overlay on
            alpha: Blending weight for overlay (0=background, 1=overlay)

        Returns:
            (H, W, 3) uint8 RGB visualization
        """
        # Normalize each PCA component to [0, 1] across valid pixels
        pca_rgb = np.zeros_like(background_image, dtype=np.float32)

        for i in range(min(3, self.n_pca_components)):
            component = pca_map[:, :, i]
            valid_vals = component[mask > 0]

            if len(valid_vals) > 0:
                # Normalize to [0, 1] using percentile clipping for robustness
                vmin, vmax = np.percentile(valid_vals, [2, 98])
                if vmax > vmin:
                    normalized = np.clip((component - vmin) / (vmax - vmin), 0, 1)
                else:
                    normalized = np.zeros_like(component)

                pca_rgb[:, :, i] = normalized

        # Convert to uint8
        pca_rgb_u8 = (pca_rgb * 255).astype(np.uint8)

        # Blend with background only on masked region
        result = background_image.copy()
        mask_3d = np.stack([mask] * 3, axis=-1).astype(bool)
        result[mask_3d] = (
            alpha * pca_rgb_u8[mask_3d] + (1 - alpha) * background_image[mask_3d]
        ).astype(np.uint8)

        return result

    def _extract_features_batch(
        self,
        images: list[np.ndarray],
    ) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Extract DINOv2 features for a batch of images in one pipeline call.

        Mirrors :meth:`_extract_features` but passes all images at once so the
        model processes them in parallel on the GPU/CPU.

        Args:
            images: List of (H, W, 3) uint8 RGB images.

        Returns:
            List of ``(features, original_shape)`` tuples (same length as *images*).
            Each ``features`` array has shape ``(feat_h, feat_w, feat_dim)``.
        """
        orig_shapes = [img.shape[:2] for img in images]
        pil_images = [self._preprocess_image(img) for img in images]

        # Single batched forward pass through the model.
        batch_output = self.feature_extractor(pil_images)

        results: list[tuple[np.ndarray, tuple[int, int]]] = []
        for output_item, orig_shape in zip(
            batch_output or [], orig_shapes, strict=False
        ):
            # Unwrap: pipeline returns a list per image when called with a list.
            if isinstance(output_item, list) and len(output_item) > 0:
                output_item = output_item[0]

            if isinstance(output_item, dict) and "features" in output_item:
                features_tensor = output_item["features"]  # type: ignore[literal-required]
            else:
                features_tensor = output_item

            if isinstance(features_tensor, torch.Tensor):
                features_np = features_tensor.cpu().numpy()
            else:
                features_np = np.array(features_tensor)

            # Discard CLS token and reshape to spatial grid (mirrors _extract_features).
            if features_np.ndim == 3:
                patch_tokens = features_np[0, 1:, :]
            elif features_np.ndim == 2:
                patch_tokens = features_np[1:, :]
            else:
                patch_tokens = features_np

            feat_dim = patch_tokens.shape[-1]
            if self.feature_dim is None:
                self.feature_dim = feat_dim

            num_patches = patch_tokens.shape[0]
            feat_h = feat_w = int(np.sqrt(num_patches))
            features = patch_tokens.reshape(feat_h, feat_w, feat_dim)
            results.append((features, orig_shape))

        return results

    def process_frames_batch(
        self,
        rgb_images: list[np.ndarray],
        seg_masks: list[np.ndarray],
        interacting_ids: set[int],
        overlay_alpha: float = 0.7,
    ) -> list[np.ndarray]:
        """Process a batch of frames with a single DINOv2 forward pass.

        All images are run through the model in parallel, then PCA projection
        and visualization are applied per-frame.  The PCA model is fitted on
        the first frame that contains at least one pixel from *interacting_ids*
        (or uses the already-fitted model if :meth:`process_frame` was called
        previously).

        Args:
            rgb_images:      List of (H, W, 3) uint8 RGB frames.
            seg_masks:       Per-frame segmentation masks, (H, W) or (H, W, 1).
            interacting_ids: Segment IDs to include in PCA.
            overlay_alpha:   Blending weight for the PCA visualization.

        Returns:
            List of (H, W, 3) uint8 visualizations (same length as *rgb_images*).
            Frames with no pixels in *interacting_ids* are returned unchanged.
        """
        if not rgb_images:
            return []

        # Single batched DINOv2 forward pass.
        batch_lowres = self._extract_features_batch(rgb_images)

        results: list[np.ndarray] = []
        for rgb, seg, (features_2d_lowres, orig_shape) in zip(
            rgb_images, seg_masks, batch_lowres, strict=False
        ):
            features_2d = self._upsample_features(features_2d_lowres, orig_shape)

            seg_features, binary_mask = self._extract_segmented_features(
                features_2d, seg, interacting_ids
            )

            if seg_features.shape[0] == 0:
                results.append(rgb)
                continue

            centered_features, _ = self._mean_center_features(seg_features)

            # Fit PCA on the first valid frame in the batch.
            if self.pca_model is None:
                print(f"Fitting PCA on batch frame with {seg_features.shape[0]} pixels")
                self.pca_model = self._fit_pca(centered_features)

            pca_map = self._project_pca(centered_features, binary_mask)
            vis = self._visualize_pca_map(
                pca_map, binary_mask, rgb, alpha=overlay_alpha
            )
            results.append(vis)

        return results

    def process_frame(
        self,
        rgb_image: np.ndarray,
        seg_mask: np.ndarray,
        interacting_ids: set[int],
        overlay_alpha: float = 0.7,
    ) -> np.ndarray:
        """Process a single frame: extract features, apply PCA, and visualize.

        On the first call, this fits a new PCA model on the mean-centered features.
        On subsequent calls, it applies the stored PCA model.

        Args:
            rgb_image: (H, W, 3) uint8 RGB image
            seg_mask: (H, W) or (H, W, 1) segmentation mask
            interacting_ids: Set of segment IDs to process
            overlay_alpha: Blending weight for visualization overlay

        Returns:
            (H, W, 3) uint8 RGB visualization with PCA features overlaid
        """
        if len(interacting_ids) == 0:
            return rgb_image  # Nothing to process

        # 1. Extract DINOv2 features
        features_2d_lowres, orig_shape = self._extract_features(rgb_image)

        # 2. Upsample features to match image resolution
        features_2d = self._upsample_features(features_2d_lowres, orig_shape)
        self._last_features_2d = features_2d

        # 3. Extract features for segmented regions
        seg_features, binary_mask = self._extract_segmented_features(
            features_2d, seg_mask, interacting_ids
        )

        if seg_features.shape[0] == 0:
            return rgb_image  # No pixels in interacting segments

        # 4. Mean-center features
        centered_features, mean_vec = self._mean_center_features(seg_features)

        # 5. Fit or apply PCA
        if self.pca_model is None:
            # First frame: fit PCA
            print(
                f"First interaction frame: fitting PCA on {seg_features.shape[0]} pixels"
            )
            self.pca_model = self._fit_pca(centered_features)

        # 6. Project features
        pca_map = self._project_pca(centered_features, binary_mask)
        self._last_pca_features = pca_map

        # 7. Visualize
        vis_image = self._visualize_pca_map(
            pca_map, binary_mask, rgb_image, alpha=overlay_alpha
        )

        return vis_image

    def reset_pca(self):
        """Reset the PCA model (e.g., for a new demo/object)."""
        self.pca_model = None
        print("PCA model reset")

"""Pixelwise SD-DINO fused feature extractor and semantic correspondence finder.

Implements the SD-DINO pipeline (Zhang et al., 2024):
  - DINOv3 patch features via torch.hub local load (third_party/dinov3)
  - Stable Diffusion UNet multi-scale features (s5/s4/s3) via diffusers forward hooks
  - Joint co-PCA across image pairs per SD scale for consistent feature spaces
  - Feature fusion: sd_weight * SD_copca concatenated with dino_weight * DINO
  - Masked mutual nearest-neighbour correspondence matching (chunked L2 distance)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DINOV3_REPO_DIR = _REPO_ROOT / "third_party" / "dinov3"

# ImageNet normalisation used by DINOv3
_DINOV3_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv3 patch feature extractor
# ─────────────────────────────────────────────────────────────────────────────


class DINOv3PatchExtractor:
    """Extracts patch-level DINOv3 features via torch.hub (local repo).

    Uses the last transformer block's patch tokens (get_intermediate_layers),
    equivalent to the 'token' facet in the original sd-dino DINOv2 extractor.
    patch_size=16, embed_dim=384 for dinov3_vits16.
    """

    PATCH_SIZE = 16

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        weights_path = str(_DINOV3_REPO_DIR / "checkpoints" / f"{model_name}.pth")
        print(f"Loading DINOv3 from local hub: {model_name}")
        self.model = torch.hub.load(
            str(_DINOV3_REPO_DIR),
            model_name,
            source="local",
            weights=weights_path,
        )
        self.model.eval().to(device)
        print("DINOv3 loaded.")

    @torch.no_grad()
    def extract(self, image_pil: Image.Image, img_size: int = 448) -> torch.Tensor:
        """Extract DINOv3 patch features from a single image.

        Args:
            image_pil: RGB PIL image.
            img_size: Side length to resize to (snapped to nearest patch multiple).

        Returns:
            Tensor of shape (h_patches, w_patches, feat_dim) on CPU.
        """
        ps = self.PATCH_SIZE
        img_size = (img_size // ps) * ps
        image_resized = image_pil.resize((img_size, img_size), Image.LANCZOS)

        batch = _DINOV3_TRANSFORM(image_resized).unsqueeze(0).to(self.device)
        # get_intermediate_layers returns tuple of (B, N_patches, D) tensors
        (patch_tokens,) = self.model.get_intermediate_layers(batch, n=1)
        patch_tokens = patch_tokens.squeeze(0)  # (N_patches, D)

        h_p = w_p = img_size // ps
        return patch_tokens.reshape(h_p, w_p, -1).cpu()  # (H_p, W_p, D)


# ─────────────────────────────────────────────────────────────────────────────
# Stable Diffusion multi-scale UNet feature extractor
# ─────────────────────────────────────────────────────────────────────────────


class SDUNetExtractor:
    """Extracts multi-scale UNet features from Stable Diffusion.

    Hooks into the last ResNet of each of the first 3 up_blocks (decoder),
    corresponding to ODISE block_indices (2, 5, 8).
    Returns s5 (deepest/smallest), s4, s3 (shallowest/largest spatial resolution).
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
        timestep: int = 100,
    ):
        from diffusers import DDIMScheduler, StableDiffusionPipeline  # noqa: PLC0415

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.timestep = timestep

        print(f"Loading Stable Diffusion: {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(1000)

        # Pre-compute null text embedding (empty string, guidance-free extraction)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        with torch.no_grad():
            tokens = tokenizer(
                "", padding="max_length", max_length=77, return_tensors="pt"
            )
            self._null_emb = text_encoder(tokens.input_ids.to(device))[0].half()

        del pipe.tokenizer, pipe.text_encoder
        print("Stable Diffusion loaded.")

    @torch.no_grad()
    def extract(
        self, image_pil: Image.Image, img_size: int = 512
    ) -> dict[str, torch.Tensor]:
        """Extract multi-scale SD UNet features from a single image.

        Hooks into up_blocks[i].resnets[-1] for i=0,1,2:
          - s5: up_blocks[0] → smallest spatial resolution (deepest semantics)
          - s4: up_blocks[1]
          - s3: up_blocks[2] → largest of the three

        Args:
            image_pil: RGB PIL image.
            img_size: Processing resolution (snapped to nearest multiple of 8).

        Returns:
            Dict with keys 's5', 's4', 's3', each (1, C, H, W) float32 on CPU.
        """
        img_size = (img_size // 8) * 8

        image = image_pil.resize((img_size, img_size), Image.LANCZOS)
        img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        img_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .half()
            .to(self.device)
        )

        latents = self.vae.encode(img_tensor).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        t = torch.tensor([self.timestep], device=self.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        collected: dict[str, torch.Tensor] = {}
        hooks = []

        def _make_hook(name: str):
            def _fn(module, inp, out):
                feat = out[0] if isinstance(out, tuple) else out
                collected[name] = feat.detach().float().cpu()

            return _fn

        for i, name in enumerate(["s5", "s4", "s3"]):
            h = (
                self.unet.up_blocks[i]
                .resnets[-1]
                .register_forward_hook(_make_hook(name))
            )
            hooks.append(h)

        self.unet(noisy_latents, t, encoder_hidden_states=self._null_emb)

        for h in hooks:
            h.remove()

        # Ensure all outputs are 4-D (batch, C, H, W)
        for k in collected:
            if collected[k].dim() == 3:
                collected[k] = collected[k].unsqueeze(0)

        return collected


# ─────────────────────────────────────────────────────────────────────────────
# Co-PCA (joint PCA across image pairs per SD scale)
# ─────────────────────────────────────────────────────────────────────────────


def co_pca(
    features1: dict[str, torch.Tensor],
    features2: dict[str, torch.Tensor],
    dim: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint PCA on SD multi-scale features from both images.

    For each scale (s5, s4, s3), concatenates both images' features along the
    spatial dimension, applies truncated SVD, then splits them back.
    Finally upsamples and cascades s5→s4→s3 into a single descriptor at s3
    spatial resolution.

    Args:
        features1, features2: dicts with 's5', 's4', 's3' each (1, C, H, W).
        dim: Target PCA dimensions for [s5, s4, s3]. Defaults to [128, 128, 128].

    Returns:
        (feat1, feat2): each (1, D_total, H_s3, W_s3) where
        D_total = dim[0] + dim[1] + dim[2].
    """
    if dim is None:
        dim = [128, 128, 128]

    processed1: dict[str, torch.Tensor] = {}
    processed2: dict[str, torch.Tensor] = {}
    sizes = {name: features1[name].shape[-1] for name in ["s5", "s4", "s3"]}

    for name, target_dim in zip(["s5", "s4", "s3"], dim, strict=False):
        f1 = features1[name].reshape(
            features1[name].shape[0], features1[name].shape[1], -1
        )
        f2 = features2[name].reshape(
            features2[name].shape[0], features2[name].shape[1], -1
        )

        # Concat spatially, permute → (1, t1+t2, C)
        features = torch.cat([f1, f2], dim=-1).permute(0, 2, 1)

        # Joint truncated SVD on mean-centered tokens
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered = features[0] - mean  # (t1+t2, C)
        _, _, V = torch.pca_lowrank(centered, q=target_dim)
        reduced = torch.matmul(centered, V[:, :target_dim])  # (t1+t2, d)
        features = reduced.unsqueeze(0).permute(0, 2, 1)  # (1, d, t1+t2)

        t1 = f1.shape[-1]
        s = sizes[name]
        processed1[name] = features[:, :, :t1].reshape(1, target_dim, s, s)
        processed2[name] = features[:, :, t1:].reshape(1, target_dim, s, s)

    # Cascade: upsample s5 → s4 spatial, concat [s4 | s5_up]
    for proc in [processed1, processed2]:
        s5_up = F.interpolate(
            proc["s5"], size=proc["s4"].shape[-2:], mode="bilinear", align_corners=False
        )
        proc["s5"] = torch.cat([proc["s4"], s5_up], dim=1)
        proc["s4"] = proc["s3"]
        del proc["s3"]

    # Final: concat old s3 (now s4) with cascaded [s4+s5] upsampled to s3 resolution
    feat1 = torch.cat(
        [
            processed1["s4"],
            F.interpolate(
                processed1["s5"], size=processed1["s4"].shape[-2:], mode="bilinear"
            ),
        ],
        dim=1,
    )
    feat2 = torch.cat(
        [
            processed2["s4"],
            F.interpolate(
                processed2["s5"], size=processed2["s4"].shape[-2:], mode="bilinear"
            ),
        ],
        dim=1,
    )

    return feat1, feat2  # each (1, D_total, H_s3, W_s3)


# ─────────────────────────────────────────────────────────────────────────────
# Pixelwise SD-DINO feature extractor
# ─────────────────────────────────────────────────────────────────────────────


class SDDinoFeatureExtractor:
    """Pixelwise feature extractor combining Stable Diffusion and DINOv3.

    Extracts fused features for an image pair with joint co-PCA on the SD
    multi-scale features, then bilinear-resizes DINOv3 patch features to match
    SD spatial resolution before concatenation.

    Args:
        sd_weight: Scale factor applied to the SD feature block. Default 1.0.
        dino_weight: Scale factor applied to the DINO feature block. Default 1.0.
        pca_dims: PCA target dims for [s5, s4, s3]. Default [128, 128, 128].
        img_size_dino: Input resolution for DINOv3 extraction. Default 448.
        img_size_sd: Input resolution for SD extraction. Default 512.
        dino_model: DINOv3 model name (must match a hubconf entry). Default "dinov3_vits16".
        sd_model_id: HuggingFace SD model ID. Default "runwayml/stable-diffusion-v1-5".
        device: "cuda" or "cpu". Auto-detected if None.
    """

    def __init__(
        self,
        sd_weight: float = 1.0,
        dino_weight: float = 1.0,
        pca_dims: list[int] | None = None,
        img_size_dino: int = 448,
        img_size_sd: int = 512,
        dino_model: str = "dinov3_vits16",
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sd_weight = sd_weight
        self.dino_weight = dino_weight
        self.pca_dims = pca_dims if pca_dims is not None else [128, 128, 128]
        self.img_size_dino = img_size_dino
        self.img_size_sd = img_size_sd

        self._dino = DINOv3PatchExtractor(model_name=dino_model, device=device)
        self._sd = SDUNetExtractor(model_id=sd_model_id, device=device)

    def extract_pair(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract fused SD-DINO features for an image pair.

        Applies joint co-PCA across both images on the SD features so that the
        feature spaces are consistent for cross-image matching.

        Args:
            img1, img2: RGB PIL images.

        Returns:
            (fused1, fused2): each (H_sd, W_sd, D_total) on CPU, where
            D_total = sum(pca_dims) + DINO_feat_dim.
        """
        dino1 = self._dino.extract(img1, img_size=self.img_size_dino)
        sd1 = self._sd.extract(img1, img_size=self.img_size_sd)

        dino2 = self._dino.extract(img2, img_size=self.img_size_dino)
        sd2 = self._sd.extract(img2, img_size=self.img_size_sd)

        # Joint co-PCA on SD features across the pair
        sd_pca1, sd_pca2 = co_pca(sd1, sd2, dim=self.pca_dims)

        fused1 = self._fuse(sd_pca1, dino1)
        fused2 = self._fuse(sd_pca2, dino2)
        return fused1, fused2

    def _fuse(
        self,
        sd_feat: torch.Tensor,  # (1, D_sd, H, W)
        dino_feat: torch.Tensor,  # (h_p, w_p, D_dino)
    ) -> torch.Tensor:
        """Bilinear-resize DINO to SD spatial resolution and concatenate.

        Returns:
            (H, W, D_sd + D_dino) fused feature map on CPU.
        """
        h_sd, w_sd = sd_feat.shape[-2:]

        dino_4d = dino_feat.permute(2, 0, 1).unsqueeze(0).float()  # (1, D, h_p, w_p)
        dino_resized = F.interpolate(
            dino_4d, size=(h_sd, w_sd), mode="bilinear", align_corners=False
        )

        fused = torch.cat(
            [self.sd_weight * sd_feat, self.dino_weight * dino_resized], dim=1
        )  # (1, D_total, H, W)
        return fused.squeeze(0).permute(1, 2, 0)  # (H, W, D_total)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic correspondence finder
# ─────────────────────────────────────────────────────────────────────────────


class SemanticCorrespondenceFinder:
    """Finds pixel-level semantic correspondences between two images.

    Uses fused SD-DINO features with mutual nearest-neighbour matching via
    chunked L2 distance (never materialises the full N×N matrix).
    Matching can optionally be restricted to masked regions in each image,
    with correspondences reported back in original image pixel coordinates.

    Args:
        extractor: SDDinoFeatureExtractor instance.
        corr_size: Resolution to upsample features to before matching.
                   Higher values give finer keypoint localisation. Default 256.
        chunk_size: Number of query tokens processed per L2 batch. Default 512.
    """

    def __init__(
        self,
        extractor: SDDinoFeatureExtractor,
        corr_size: int = 256,
        chunk_size: int = 512,
    ):
        self.extractor = extractor
        self.corr_size = corr_size
        self.chunk_size = chunk_size

    def find_correspondences(
        self,
        img1: Image.Image,
        img2: Image.Image,
        mask1: np.ndarray | None = None,
        mask2: np.ndarray | None = None,
        n_top: int = 15,
        mutual: bool = True,
    ) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
        """Find top-n semantic correspondences between two images.

        Args:
            img1, img2: RGB PIL images.
            mask1: Optional (H1, W1) bool array. Only masked pixels of img1 are
                   used as query points.
            mask2: Optional (H2, W2) bool array. Only masked pixels of img2 are
                   used as key points.
            n_top: Number of correspondences to return.
            mutual: If True, only keep mutually nearest neighbours.

        Returns:
            List of ((r1, c1), (r2, c2), score) sorted descending by score.
            Coordinates are in original image pixel space.
        """
        fused1, fused2 = self.extractor.extract_pair(img1, img2)
        return self.match_from_features(
            fused1, fused2, img1, img2, mask1, mask2, n_top, mutual
        )

    def match_from_features(
        self,
        fused1: torch.Tensor,
        fused2: torch.Tensor,
        img1: Image.Image,
        img2: Image.Image,
        mask1: np.ndarray | None = None,
        mask2: np.ndarray | None = None,
        n_top: int = 15,
        mutual: bool = True,
    ) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
        """Find top-n correspondences from pre-computed fused feature maps.

        Useful when the caller already holds the feature tensors (e.g. for
        simultaneous feature visualisation) and wants to avoid re-extraction.

        Args:
            fused1, fused2: (H, W, D) fused feature tensors from
                            SDDinoFeatureExtractor.extract_pair().
            img1, img2: Original RGB PIL images (used only for coordinate scaling).
            mask1: Optional (H1, W1) bool array for img1.
            mask2: Optional (H2, W2) bool array for img2.
            n_top: Number of correspondences to return.
            mutual: If True, only keep mutually nearest neighbours.

        Returns:
            List of ((r1, c1), (r2, c2), score) sorted descending by score.
            Coordinates are in original image pixel space.

        Raises:
            ValueError: If either valid mask region is empty after downsampling.
        """
        cs = self.corr_size

        # Upsample both feature maps to corr_size × corr_size → (cs*cs, D)
        def _to_flat(feat: torch.Tensor) -> torch.Tensor:
            x = feat.permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(cs, cs), mode="bilinear", align_corners=False)
            return x.squeeze(0).permute(1, 2, 0).reshape(-1, feat.shape[-1])

        f1_full = _to_flat(fused1)  # (cs*cs, D)
        f2_full = _to_flat(fused2)

        # Build valid index sets from masks (downsampled to corr_size)
        valid1 = self._mask_to_valid_indices(mask1, cs)
        valid2 = self._mask_to_valid_indices(mask2, cs)

        f1 = f1_full[valid1] if valid1 is not None else f1_full
        f2 = f2_full[valid2] if valid2 is not None else f2_full
        N1, N2 = f1.shape[0], f2.shape[0]

        if N1 == 0:
            raise ValueError("mask1 has no valid pixels at corr_size resolution.")
        if N2 == 0:
            raise ValueError("mask2 has no valid pixels at corr_size resolution.")

        # Forward pass: nearest neighbour in f2 for each f1 token (chunked)
        nn12 = torch.zeros(N1, dtype=torch.long)
        scores12 = torch.full((N1,), float("-inf"))

        for start in range(0, N1, self.chunk_size):
            end = min(start + self.chunk_size, N1)
            dists = torch.cdist(f1[start:end], f2)  # (chunk, N2)
            best_j = dists.argmin(dim=1)
            best_d = dists[torch.arange(end - start), best_j]
            nn12[start:end] = best_j
            scores12[start:end] = -best_d  # higher = more similar

        if not mutual:
            top_local = scores12.topk(min(n_top, N1)).indices.tolist()
            return [
                self._make_match(
                    i, int(nn12[i]), valid1, valid2, scores12, cs, img1, img2
                )
                for i in top_local
            ]

        # Backward pass: nearest neighbour in f1 for each f2 token
        nn21 = torch.zeros(N2, dtype=torch.long)
        for start in range(0, N2, self.chunk_size):
            end = min(start + self.chunk_size, N2)
            dists = torch.cdist(f2[start:end], f1)
            nn21[start:end] = dists.argmin(dim=1)

        # Mutual filter: keep i only if nn21[nn12[i]] == i
        mutual_mask = nn21[nn12] == torch.arange(N1)
        scores_masked = torch.where(mutual_mask, scores12, torch.tensor(float("-inf")))

        top_local = scores_masked.topk(min(n_top, N1)).indices.tolist()
        matches = [
            self._make_match(i, int(nn12[i]), valid1, valid2, scores12, cs, img1, img2)
            for i in top_local
            if scores_masked[i] > float("-inf")
        ]
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mask_to_valid_indices(mask: np.ndarray | None, cs: int) -> torch.Tensor | None:
        """Downsample a boolean mask to corr_size and return valid flat indices."""
        if mask is None:
            return None
        # Bilinear downsample via PIL, threshold at 0.5
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (cs, cs), Image.BILINEAR
        )
        mask_cs = np.array(mask_pil) > 127
        indices = np.where(mask_cs.flatten())[0]
        return torch.from_numpy(indices).long()

    @staticmethod
    def _make_match(
        local_i: int,
        local_j: int,
        valid1: torch.Tensor | None,
        valid2: torch.Tensor | None,
        scores: torch.Tensor,
        cs: int,
        img1: Image.Image,
        img2: Image.Image,
    ) -> tuple[tuple[int, int], tuple[int, int], float]:
        """Map local subset indices back to original image pixel coordinates."""
        idx1 = int(valid1[local_i]) if valid1 is not None else local_i
        idx2 = int(valid2[local_j]) if valid2 is not None else local_j

        r1_cs, c1_cs = divmod(idx1, cs)
        r2_cs, c2_cs = divmod(idx2, cs)

        # Scale from corr_size grid to image pixel centres
        r1 = int(r1_cs * img1.height / cs + img1.height / cs / 2)
        c1 = int(c1_cs * img1.width / cs + img1.width / cs / 2)
        r2 = int(r2_cs * img2.height / cs + img2.height / cs / 2)
        c2 = int(c2_cs * img2.width / cs + img2.width / cs / 2)

        return (r1, c1), (r2, c2), float(scores[local_i])

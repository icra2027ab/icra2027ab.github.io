"""SAM3 "segment everything" + CLIP/DINOv2 re-rank vs GT masks.

Pipeline:
  1. SAM3 with a generic prompt (default ``"object"``) enumerates instance masks.
  2. Each instance is cropped from the RGB (optionally background-masked).
  3. A visual encoder (CLIP or DINOv2) encodes crops.
  4. Query vectors come from one of three modes. When the reference bank stores
     multiple images per object, per-reference cosine similarities are
     aggregated with ``agg_mode`` (``mean`` or ``max``) — not equivalent to
     comparing against a single averaged embedding.
     - ``image`` (default): pre-computed per-object CLIP image embeddings
       loaded from a single reference file (``datasets/all_objects.hdf5`` by
       default).
     - ``text``: CLIP text embedding of the object name.
     - ``dino``: pre-computed per-object DINOv2 image embeddings from the same
       reference file. Crops are encoded with DINOv2 instead of CLIP.
     For ``image``/``dino`` modes, prompts missing from the bank fall back to
     CLIP text↔image similarity when ``fallback_to_text=True`` (the default);
     otherwise they get a zero query vector.
  5. Hungarian matching on the similarity matrix assigns at most one crop per
     requested name; unassigned names get an empty mask.

The reference bank is NOT built at run time from the evaluation HDF5s — we
load it as-is from the canonical reference file to avoid cross-file leakage
between eval frames and their references.

Output schema matches ``sam3_segmentation_eval.py`` so results can be compared
side-by-side.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import numpy as np
import tyro
from _common import BaseArgs, _overlay_masks, normalize_name, run_pipeline
from loguru import logger
from PIL import Image
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from _common import FailureSample


@dataclasses.dataclass
class Args(BaseArgs):
    out_dir: str = "tmp/sam3_clip_rerank_eval"
    sam3_model: str = "facebook/sam3"
    clip_model: str = "openai/clip-vit-base-patch16"
    dino_model: str = "facebook/dinov2-base"
    """HuggingFace DINOv2 model ID (only used when ``rerank_mode == 'dino'``).
    Must match the encoder used to compute ``dino_embedding`` in the reference
    HDF5 — otherwise crop features and bank features live in different spaces."""
    sam3_prompt: list[str] = dataclasses.field(
        default_factory=lambda: ["object", "<object_name>"]
    )
    """Text prompts used to query SAM3. Each resolved prompt runs SAM3 once and
    the returned instances are pooled before CLIP/DINO re-ranking. The literal
    token ``<object_name>`` expands into one SAM3 query per target object name
    so SAM3 also searches the scene for the specific entity by name."""
    score_threshold: float = 0.15
    mask_threshold: float = 0.4
    min_mask_area: int = 10
    """Drop SAM3 instances with fewer than this many pixels (noise)."""
    max_instances: int = 40
    """Keep at most this many top-scoring SAM3 instances per frame."""
    bbox_pad: int = 4
    """Padding added around each crop's bbox (pixels)."""
    clip_mask_bg: bool = True
    """If True, zero out background pixels in each crop before encoding."""
    rerank_mode: Literal["image", "text", "dino"] = "image"
    """``image`` (default): crops are CLIP-encoded and scored against every
    CLIP image embedding in the reference bank. ``text``: CLIP text embedding
    of the object name. ``dino``: crops are encoded with DINOv2 and scored
    against every DINOv2 image embedding in the bank. Per-object similarities
    across references are aggregated with ``agg_mode``."""
    reference_hdf5: str = "datasets/all_objects.hdf5"
    """Reference file with pre-computed per-object image embeddings. Each
    top-level group is one sanitized object name (e.g. ``"porcelain mug"``)
    containing an ``images/<idx>/`` subgroup per reference crop with
    ``clip_embedding`` / ``dino_embedding`` datasets."""
    fallback_to_text: bool = True
    """When a prompt isn't in the reference bank, score it with CLIP text↔image
    similarity instead of returning a zero vector. In ``dino`` mode this forces
    CLIP to be loaded alongside DINOv2 so crops can be cross-encoded — missing
    prompts are scored in CLIP space while found prompts stay in DINO space
    (the two scales differ, so Hungarian can prefer one over the other)."""
    agg_mode: Literal["mean", "max"] = "max"
    """How to aggregate per-reference similarities into one score per prompt.
    ``mean`` averages across the K references for that object; ``max`` takes
    the best-matching reference."""


# ---------------------------------------------------------------------------
# Reference bank — per-object image embeddings loaded from the canonical
# reference file, used as query vectors in ``image`` / ``dino`` modes.
# ---------------------------------------------------------------------------


def _load_reference_bank(
    hdf5_path: str, embedding_key: str = "clip_embedding"
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    """Load ``(embeddings, images)`` from the canonical reference HDF5.

    Each returned dict is keyed by canonical object name. ``embeddings[name]``
    is an ``(K, D)`` array holding all ``K`` L2-normalised reference embeddings
    that map to this canonical name (groups like ``moka_pot_1`` / ``moka_pot_2``
    collapse to ``moka pot``, so ``K`` is typically 1–4). ``images[name]`` is
    the list of per-group RGB reference crops (``grp["image"]``) that
    contributed, aligned with the same ordering.

    The reference HDF5 uses a nested layout produced by
    ``scripts/ablations/extraction/extract_objects.py``: top-level groups are already sanitized names
    (e.g. ``"porcelain mug"``) and each has an ``images/<idx>/`` subgroup per
    reference crop, with the embedding datasets living inside those subgroups.
    """
    import h5py

    embs_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
    imgs_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            grp = f[key]
            images_grp = grp.get("images")
            if images_grp is None:
                continue
            canon = normalize_name(key)
            for img_key in images_grp.keys():
                sub = images_grp[img_key]
                if embedding_key not in sub:
                    continue
                emb = np.asarray(sub[embedding_key][...], dtype=np.float32)
                norm = float(np.linalg.norm(emb))
                if norm < 1e-8:
                    continue
                embs_by_name[canon].append(emb / norm)
                if "image" in sub:
                    imgs_by_name[canon].append(np.asarray(sub["image"][...]))

    embeddings: dict[str, np.ndarray] = {}
    for name, embs in sorted(embs_by_name.items()):
        embeddings[name] = np.stack(embs, axis=0).astype(np.float32)  # (K, D)

    # Only keep image lists for canonicals that made it into the embedding set.
    images: dict[str, list[np.ndarray]] = {
        name: imgs_by_name[name] for name in embeddings if name in imgs_by_name
    }

    total_refs = sum(int(v.shape[0]) for v in embeddings.values())
    logger.info(
        f"reference bank: loaded {len(embeddings)} objects "
        f"({total_refs} reference embeddings total) from {hdf5_path} "
        f"({embedding_key})"
    )
    return embeddings, images


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Sam3ClipRerankPredictor:
    def __init__(
        self,
        sam3_model_id: str,
        clip_model_id: str,
        dino_model_id: str,
        device: str,
        sam3_prompt: list[str],
        score_threshold: float,
        mask_threshold: float,
        min_mask_area: int,
        max_instances: int,
        bbox_pad: int,
        clip_mask_bg: bool,
        rerank_mode: str = "image",
        reference_embeddings: dict[str, np.ndarray] | None = None,
        fallback_to_text: bool = True,
        agg_mode: str = "mean",
    ) -> None:
        import torch
        from transformers import (
            AutoImageProcessor,
            CLIPModel,
            CLIPProcessor,
            Dinov2Model,
            Sam3Model,
            Sam3Processor,
        )

        if agg_mode not in ("mean", "max"):
            raise ValueError(f"agg_mode must be 'mean' or 'max', got {agg_mode!r}")

        self._torch = torch
        self.device = (
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if self.device != device:
            logger.warning(
                f"requested device={device} unavailable, falling back to {self.device}"
            )
        logger.info(f"loading {sam3_model_id} on {self.device}")
        self.sam3_processor = Sam3Processor.from_pretrained(sam3_model_id)
        self.sam3 = Sam3Model.from_pretrained(sam3_model_id).to(self.device).eval()

        self.rerank_mode = rerank_mode
        self.name = "sam3_dino_rerank" if rerank_mode == "dino" else "sam3_clip_rerank"

        # DINO is needed only in dino mode. CLIP is needed for image/text modes,
        # and also for dino + fallback_to_text (missing prompts cross-encode in
        # CLIP space).
        need_dino = rerank_mode == "dino"
        need_clip = rerank_mode in ("image", "text") or (
            rerank_mode == "dino" and fallback_to_text
        )

        if need_dino:
            logger.info(f"loading {dino_model_id} on {self.device}")
            self.dino_processor = AutoImageProcessor.from_pretrained(
                dino_model_id, use_fast=True
            )
            self.dino = (
                Dinov2Model.from_pretrained(dino_model_id).to(self.device).eval()
            )
        else:
            self.dino_processor = None
            self.dino = None

        if need_clip:
            logger.info(f"loading {clip_model_id} on {self.device}")
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
            self.clip = CLIPModel.from_pretrained(clip_model_id).to(self.device).eval()
        else:
            self.clip_processor = None
            self.clip = None

        self.sam3_prompt = sam3_prompt
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.min_mask_area = min_mask_area
        self.max_instances = max_instances
        self.bbox_pad = bbox_pad
        self.clip_mask_bg = clip_mask_bg
        self.reference_embeddings = reference_embeddings or {}
        self.fallback_to_text = fallback_to_text
        self.agg_mode = agg_mode
        # Populated from the reference HDF5 in ``main()``. Used only by the
        # failure figure — not the re-ranking pipeline.
        self.reference_images: dict[str, list[np.ndarray]] = {}
        # Cache of per-prompt (refs_tensor, space) so we encode each prompt
        # at most once per predictor lifetime. ``space`` is "clip" or "dino".
        self._prompt_feat_cache: dict[str, tuple[Any, str]] = {}
        # Prompts seen so far that aren't in the reference bank. Published so
        # ``run_pipeline`` can flag them in the confusion matrix. Also used to
        # suppress duplicate "missing" warnings.
        self.missing_prompts: set[str] = set()

    # ------------------------------------------------------------------

    _OBJECT_NAME_TOKEN = "<object_name>"

    def _resolve_sam3_prompts(self, target_prompts: list[str]) -> list[str]:
        """Expand ``sam3_prompt`` templates against the target object names.

        Any template containing ``<object_name>`` is duplicated — once per
        target prompt with the token substituted. Order is preserved and
        duplicates are dropped so SAM3 is never re-queried with the same text.
        """
        resolved: list[str] = []
        seen: set[str] = set()
        for tmpl in self.sam3_prompt:
            if self._OBJECT_NAME_TOKEN in tmpl:
                for tp in target_prompts:
                    p = tmpl.replace(self._OBJECT_NAME_TOKEN, tp)
                    if p not in seen:
                        seen.add(p)
                        resolved.append(p)
            elif tmpl not in seen:
                seen.add(tmpl)
                resolved.append(tmpl)
        return resolved

    def _run_sam3(
        self, rgb: np.ndarray, target_prompts: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(instances[N,H,W] bool, scores[N])`` sorted by descending score.

        Runs SAM3 once per resolved prompt (see ``_resolve_sam3_prompts``) and
        pools the instance masks. Vision features are computed once and reused
        across text queries.
        """
        torch = self._torch
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)
        img_inputs = self.sam3_processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            vision_embeds = self.sam3.get_vision_features(
                pixel_values=img_inputs.pixel_values
            )
        target_sizes = img_inputs.get("original_sizes")
        target_sizes = target_sizes.tolist() if target_sizes is not None else [[H, W]]

        resolved_prompts = self._resolve_sam3_prompts(target_prompts)
        if not resolved_prompts:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)

        all_masks: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for prompt in resolved_prompts:
            text_inputs = self.sam3_processor(text=prompt, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.sam3(vision_embeds=vision_embeds, **text_inputs)
            results = self.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )[0]
            masks = results.get("masks")
            scores = results.get("scores")
            if masks is None or len(masks) == 0:
                continue
            if hasattr(masks, "cpu"):
                inst = masks.cpu().numpy().astype(bool)
            else:
                inst = np.stack([np.asarray(m).astype(bool) for m in masks], axis=0)
            if scores is not None and hasattr(scores, "cpu"):
                score_arr = scores.cpu().numpy().astype(np.float32)
            elif scores is not None:
                score_arr = np.asarray(scores, dtype=np.float32)
            else:
                score_arr = np.zeros((inst.shape[0],), dtype=np.float32)
            all_masks.append(inst)
            all_scores.append(score_arr)

        if not all_masks:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)

        inst = np.concatenate(all_masks, axis=0)
        score_arr = np.concatenate(all_scores, axis=0)
        order = np.argsort(-score_arr)
        return inst[order], score_arr[order]

    def _filter_and_crop(
        self, rgb: np.ndarray, instances: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Drop too-small instances, cap to ``max_instances``, return crops.

        Returns ``(instances_kept[M,H,W], crops[M] of variable HxWx3 uint8)``.
        """
        H, W = rgb.shape[:2]
        keep: list[int] = []
        crops: list[np.ndarray] = []
        for i in range(instances.shape[0]):
            m = instances[i]
            area = int(m.sum())
            if area < self.min_mask_area:
                continue
            ys, xs = np.where(m)
            y0 = max(int(ys.min()) - self.bbox_pad, 0)
            y1 = min(int(ys.max()) + self.bbox_pad + 1, H)
            x0 = max(int(xs.min()) - self.bbox_pad, 0)
            x1 = min(int(xs.max()) + self.bbox_pad + 1, W)
            crop = rgb[y0:y1, x0:x1].copy()
            if self.clip_mask_bg:
                local_mask = m[y0:y1, x0:x1]
                crop[~local_mask] = 0
            keep.append(i)
            crops.append(crop)
            if len(keep) >= self.max_instances:
                break
        if not keep:
            return np.zeros((0, H, W), dtype=bool), []
        return instances[keep], crops

    def _prompt_features(
        self, prompts: list[str]
    ) -> tuple[list[Any], list[str], list[str]]:
        """Return ``(per_prompt_refs, spaces, sources)``.

        - ``per_prompt_refs[i]`` is an ``(K_i, D)`` L2-normalised tensor of
          reference vectors for prompt ``i`` in the space indicated by
          ``spaces[i]`` (``"clip"`` or ``"dino"``).
        - ``sources[i]`` is a debug tag: ``"image"`` (CLIP bank), ``"dino"``
          (DINO bank), ``"text"`` (CLIP text fallback), or ``"dino_missing"``
          (dino mode, not in bank, fallback disabled → zero vector).
        """
        torch = self._torch

        per_prompt: list[Any] = [None] * len(prompts)
        spaces: list[str] = [""] * len(prompts)
        sources: list[str] = [""] * len(prompts)

        # Indices of prompts that need a CLIP text encoding (collected for a
        # single batched call at the end).
        text_indices: list[int] = []
        text_prompts: list[str] = []

        for i, p in enumerate(prompts):
            if p in self._prompt_feat_cache:
                t, space = self._prompt_feat_cache[p]
                per_prompt[i] = t
                spaces[i] = space
                if self.rerank_mode == "dino":
                    sources[i] = "dino" if space == "dino" else "text"
                elif self.rerank_mode == "image":
                    sources[i] = (
                        "image"
                        if space == "clip" and p in self.reference_embeddings
                        else "text"
                    )
                else:  # text
                    sources[i] = "text"
                continue

            in_bank = p in self.reference_embeddings
            if self.rerank_mode == "image" and in_bank:
                t = torch.from_numpy(self.reference_embeddings[p]).to(self.device)
                per_prompt[i] = t
                spaces[i] = "clip"
                sources[i] = "image"
                self._prompt_feat_cache[p] = (t, "clip")
            elif self.rerank_mode == "dino" and in_bank:
                t = torch.from_numpy(self.reference_embeddings[p]).to(self.device)
                per_prompt[i] = t
                spaces[i] = "dino"
                sources[i] = "dino"
                self._prompt_feat_cache[p] = (t, "dino")
            elif self.rerank_mode == "dino" and not self.fallback_to_text:
                if p not in self.missing_prompts:
                    logger.warning(
                        f"rerank_mode=dino: prompt '{p}' not in reference bank — using zero vector"
                    )
                self.missing_prompts.add(p)
                dim = (
                    next(iter(self.reference_embeddings.values())).shape[-1]
                    if self.reference_embeddings
                    else 768
                )
                t = torch.zeros(1, dim, device=self.device)
                per_prompt[i] = t
                spaces[i] = "dino"
                sources[i] = "dino_missing"
                self._prompt_feat_cache[p] = (t, "dino")
            else:
                # text mode, or image/dino mode with fallback_to_text=True and
                # prompt not in the bank. Defer to the batched CLIP text pass.
                if self.rerank_mode in ("image", "dino"):
                    if p not in self.missing_prompts:
                        logger.info(
                            f"rerank_mode={self.rerank_mode}: prompt '{p}' not in reference bank — falling back to CLIP text similarity"
                        )
                    self.missing_prompts.add(p)
                spaces[i] = "clip"
                sources[i] = "text"
                text_indices.append(i)
                text_prompts.append(p)

        if text_prompts:
            with torch.no_grad():
                txt_inputs = self.clip_processor(
                    text=text_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(self.device)
                txt_feats = self.clip.get_text_features(**txt_inputs).pooler_output
            txt_feats = torch.nn.functional.normalize(txt_feats, p=2, dim=-1)
            for j, idx in enumerate(text_indices):
                t = txt_feats[j : j + 1]  # keep (1, D) shape
                per_prompt[idx] = t
                self._prompt_feat_cache[prompts[idx]] = (t, "clip")

        return per_prompt, spaces, sources

    def _encode_crops(self, crops: list[np.ndarray], space: str) -> Any:
        """Return L2-normalised image features ``[n_crops, D]`` in ``space``.

        ``space`` is ``"clip"`` or ``"dino"``. DINOv2 features are mean-pooled
        over ``last_hidden_state`` to match the reference bank (see
        ``scripts/ablations/extraction/extract_objects.py``).
        """
        torch = self._torch
        pil_crops = [Image.fromarray(c) for c in crops]
        if space == "dino":
            with torch.no_grad():
                inputs = self.dino_processor(images=pil_crops, return_tensors="pt").to(
                    self.device
                )
                outputs = self.dino(pixel_values=inputs["pixel_values"])
                feats = outputs.last_hidden_state.mean(dim=1)
            return torch.nn.functional.normalize(feats, p=2, dim=-1)
        with torch.no_grad():
            inputs = self.clip_processor(images=pil_crops, return_tensors="pt").to(
                self.device
            )
            feats = self.clip.get_image_features(**inputs).pooler_output
        return torch.nn.functional.normalize(feats, p=2, dim=-1)

    def _aggregate(self, sim_per_ref: Any) -> Any:
        """Collapse ``(M, K)`` per-reference similarities to ``(M,)`` per prompt."""
        if self.agg_mode == "max":
            return sim_per_ref.max(dim=1).values
        return sim_per_ref.mean(dim=1)

    def _clip_similarity(
        self, crops: list[np.ndarray], prompts: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Return ``(sim[n_crops, n_prompts], prompt_sources[n_prompts])``.

        Prompts can live in different encoder spaces (e.g. dino mode with a
        CLIP text fallback for missing prompts) — similarities are computed in
        each prompt's own space and stitched back into a single matrix. CLIP
        and DINO cosine scales differ, so mixing them in one Hungarian can bias
        toward one space.
        """
        torch = self._torch
        per_prompt, spaces, sources = self._prompt_features(prompts)

        M = len(crops)
        P = len(prompts)
        sim = torch.zeros(M, P, device=self.device, dtype=torch.float32)
        if M == 0:
            return sim.cpu().numpy().astype(np.float32), sources

        needed_spaces = set(spaces)
        crop_feats: dict[str, Any] = {
            s: self._encode_crops(crops, s) for s in needed_spaces
        }

        for i, (refs, space) in enumerate(zip(per_prompt, spaces, strict=True)):
            sim_per_ref = crop_feats[space] @ refs.T  # (M, K_i)
            sim[:, i] = self._aggregate(sim_per_ref)

        return sim.cpu().numpy().astype(np.float32), sources

    # ------------------------------------------------------------------

    def predict(self, rgb: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        H, W = rgb.shape[:2]
        empty_mask = np.zeros((H, W), dtype=bool)
        out = {p: empty_mask.copy() for p in prompts}
        # Per-frame debug state for failure-sample rendering. Reset every call.
        self.last_debug: dict[str, Any] = {
            "rgb": rgb,
            "instances": np.zeros((0, H, W), dtype=bool),
            "crops": [],
            "sim": np.zeros((0, len(prompts)), dtype=np.float32),
            "prompts": list(prompts),
            "prompt_sources": ["text"] * len(prompts),
            "assignment": {},  # prompt -> instance_idx
        }

        instances, _scores = self._run_sam3(rgb, prompts)
        if instances.shape[0] == 0:
            return out

        kept, crops = self._filter_and_crop(rgb, instances)
        if not crops:
            return out

        sim, sources = self._clip_similarity(crops, prompts)  # [M, P]
        self.last_debug["instances"] = kept
        self.last_debug["crops"] = crops
        self.last_debug["sim"] = sim
        self.last_debug["prompt_sources"] = sources

        # Hungarian: maximise similarity → minimise negative. Supports M != P.
        row_idx, col_idx = linear_sum_assignment(-sim)
        for r, c in zip(row_idx, col_idx, strict=True):
            out[prompts[c]] = kept[r]
            self.last_debug["assignment"][prompts[c]] = int(r)
        return out


# ---------------------------------------------------------------------------
# Failure visualisation: show SAM3 candidates + CLIP scores + chosen + GT.
# ---------------------------------------------------------------------------


def _failure_extra(predictor: Sam3ClipRerankPredictor, prompt: str) -> dict[str, Any]:
    """Snapshot the predictor's last-frame state for this prompt.

    Must copy arrays — the predictor mutates ``last_debug`` on the next frame.
    """
    d = predictor.last_debug
    prompts = d["prompts"]
    if prompt not in prompts:
        return {}
    col = prompts.index(prompt)
    sim = d["sim"]
    scores_for_prompt = (
        sim[:, col].copy() if sim.size else np.zeros((0,), dtype=np.float32)
    )
    sources = d.get("prompt_sources", ["text"] * len(prompts))
    ref_imgs = predictor.reference_images.get(prompt, [])
    return {
        "instances": d["instances"].copy(),
        "crops": [c.copy() for c in d["crops"]],
        "scores_for_prompt": scores_for_prompt,
        "chosen_idx": d["assignment"].get(prompt),
        "query_source": sources[col],
        "rerank_mode": predictor.rerank_mode,
        # Reference images averaged into the query vector (empty for text
        # mode or when the prompt isn't in the bank).
        "reference_images": [img.copy() for img in ref_imgs],
    }


def _crop_with_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    pad: int = 4,
    color: tuple[int, int, int] = (230, 60, 60),
    alpha: float = 0.45,
) -> np.ndarray:
    """RGB crop around ``mask``'s bbox with the mask tinted in ``color``."""
    H, W = rgb.shape[:2]
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, H)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)
    crop = rgb[y0:y1, x0:x1].astype(np.float32).copy()
    m = mask[y0:y1, x0:x1]
    tint = np.array(color, dtype=np.float32)
    crop[m] = (1.0 - alpha) * crop[m] + alpha * tint
    return np.clip(crop, 0, 255).astype(np.uint8)


def _row_label(ax: Any, text: str) -> None:
    """Render ``text`` rotated 90° inside an axes, which should have no image."""
    ax.text(
        0.5,
        0.5,
        text,
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.axis("off")


def _save_clip_failure_figure(out_path: "Path", s: "FailureSample") -> None:
    """3-row figure for a low-IoU prompt.

    Row 1: pred overlay vs GT overlay on the full RGB frame.
    Row 2: raw SAM3 instance segmentations for the top candidates (mask
           tinted on the RGB crop) with the CLIP/DINO similarity to the
           query. The Hungarian-chosen candidate is outlined.
    Row 3: reference(s) that produced the query vector — the averaged
           dataset crops for ``image``/``dino`` modes, or the text prompt
           rendered as a label for ``text`` mode.
    Rows 2 and 3 have a rotated label axis on the left.
    """
    instances = s.extra.get("instances", np.zeros((0, *s.rgb.shape[:2]), dtype=bool))
    scores_for_prompt = s.extra.get(
        "scores_for_prompt", np.zeros((0,), dtype=np.float32)
    )
    chosen_idx = s.extra.get("chosen_idx")
    reference_images: list[np.ndarray] = s.extra.get("reference_images", [])
    rerank_mode = s.extra.get("rerank_mode", "text")
    query_source = s.extra.get("query_source", "text")

    # Rank candidates by similarity; keep at most 8. Always include the chosen
    # one so the user can see why it was picked even if it ranked low.
    n_cands = int(instances.shape[0])
    if n_cands > 0:
        order = np.argsort(-scores_for_prompt)
        top_k = min(8, n_cands)
        top = order[:top_k].tolist()
        if chosen_idx is not None and chosen_idx not in top:
            top[-1] = chosen_idx
    else:
        top = []

    n_cols = max(len(top), 2)
    # +1 for the leftmost label column. Width ratios give the label column a
    # narrow strip; height ratios keep row 1 (full frames) tallest.
    fig = plt.figure(figsize=(2 + 1.6 * n_cols, 11))
    gs = fig.add_gridspec(
        3,
        n_cols + 1,
        width_ratios=[0.15] + [1.0] * n_cols,
        height_ratios=[3, 2, 2],
    )

    # -- Row 1: pred + GT overlays (span the full width incl. label column). -
    half = (n_cols + 1) // 2
    ax_pred = fig.add_subplot(gs[0, :half])
    pred_overlay = _overlay_masks(
        s.rgb, s.pred_mask[None, ...], colors=np.array([[230, 60, 60]])
    )
    ax_pred.imshow(pred_overlay)
    ax_pred.set_title(
        f"pred '{s.prompt}'  IoU={s.iou:.3f}  area={int(s.pred_mask.sum())}px",
        fontsize=9,
    )
    ax_pred.axis("off")

    ax_gt = fig.add_subplot(gs[0, half:])
    gt_overlay = _overlay_masks(
        s.rgb, s.gt_mask[None, ...], colors=np.array([[60, 200, 90]])
    )
    ax_gt.imshow(gt_overlay)
    ax_gt.set_title(f"GT '{s.prompt}'  area={int(s.gt_mask.sum())}px", fontsize=9)
    ax_gt.axis("off")

    # -- Row 2: raw SAM3 segmentations with similarity scores. ---------------
    _row_label(fig.add_subplot(gs[1, 0]), "SAM3 segmentations")
    for i, idx in enumerate(top):
        ax = fig.add_subplot(gs[1, i + 1])
        viz = _crop_with_mask_overlay(s.rgb, instances[idx])
        ax.imshow(viz)
        ax.set_xticks([])
        ax.set_yticks([])
        sim_val = float(scores_for_prompt[idx]) if idx < len(scores_for_prompt) else 0.0
        tag = f"sim={sim_val:.3f}"
        if idx == chosen_idx:
            tag += "  [chosen]"
            for spine in ax.spines.values():
                spine.set_edgecolor("#ff3366")
                spine.set_linewidth(2.5)
        ax.set_xlabel(tag, fontsize=8)

    # -- Row 3: reference images (or text). ----------------------------------
    _row_label(fig.add_subplot(gs[2, 0]), f"Reference ({query_source})")
    if query_source in ("image", "dino") and reference_images:
        # One reference per column; if more references than columns, truncate;
        # if fewer, leave the rest blank.
        for i, ref in enumerate(reference_images[:n_cols]):
            ax = fig.add_subplot(gs[2, i + 1])
            ax.imshow(ref)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"ref {i + 1}", fontsize=8)
    elif query_source == "text" or rerank_mode == "text":
        # Text fallback (or pure text mode): render the prompt as a big label.
        ax = fig.add_subplot(gs[2, 1:])
        ax.text(
            0.5,
            0.5,
            f'"{s.prompt}"',
            ha="center",
            va="center",
            fontsize=14,
            fontstyle="italic",
            transform=ax.transAxes,
        )
        ax.axis("off")
    else:
        # e.g. ``dino_missing`` — no references, no text fallback.
        ax = fig.add_subplot(gs[2, 1:])
        ax.text(
            0.5,
            0.5,
            f"no reference for '{s.prompt}' (source={query_source})",
            ha="center",
            va="center",
            fontsize=11,
            color="#aa0000",
            transform=ax.transAxes,
        )
        ax.axis("off")

    fig.suptitle(
        f"{s.suite} :: {s.demo}  (mode={rerank_mode}, query={query_source})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    predictor = Sam3ClipRerankPredictor(
        sam3_model_id=args.sam3_model,
        clip_model_id=args.clip_model,
        dino_model_id=args.dino_model,
        device=args.device,
        sam3_prompt=args.sam3_prompt,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_mask_area=args.min_mask_area,
        max_instances=args.max_instances,
        bbox_pad=args.bbox_pad,
        clip_mask_bg=args.clip_mask_bg,
        rerank_mode=args.rerank_mode,
        fallback_to_text=args.fallback_to_text,
        agg_mode=args.agg_mode,
    )

    # Load the reference-image bank once for bank-based re-ranking. Embeddings
    # are pre-computed and stored in the canonical reference HDF5 — we never
    # pull references from the evaluation files. ``image`` → CLIP embeddings;
    # ``dino`` → DINOv2 embeddings; ``text`` needs no bank. Reference images
    # are loaded alongside so failure figures can show them.
    if args.rerank_mode == "image":
        embs, imgs = _load_reference_bank(
            args.reference_hdf5, embedding_key="clip_embedding"
        )
        predictor.reference_embeddings = embs
        predictor.reference_images = imgs
    elif args.rerank_mode == "dino":
        embs, imgs = _load_reference_bank(
            args.reference_hdf5, embedding_key="dino_embedding"
        )
        predictor.reference_embeddings = embs
        predictor.reference_images = imgs

    tag = f"{args.rerank_mode}_s{args.score_threshold:.2f}_m{args.mask_threshold:.2f}"
    extra_meta: dict[str, Any] = {
        "sam3_model": args.sam3_model,
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "sam3_prompt": args.sam3_prompt,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "min_mask_area": args.min_mask_area,
        "max_instances": args.max_instances,
        "bbox_pad": args.bbox_pad,
        "clip_mask_bg": args.clip_mask_bg,
        "rerank_mode": args.rerank_mode,
        "fallback_to_text": args.fallback_to_text,
        "agg_mode": args.agg_mode,
        "reference_hdf5": args.reference_hdf5,
        "n_reference_objects": len(predictor.reference_embeddings),
    }
    run_pipeline(
        args,
        predictor,
        tag=tag,
        extra_meta=extra_meta,
        failure_figure_fn=_save_clip_failure_figure,
        failure_extra_fn=_failure_extra,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

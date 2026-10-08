"""Extract a multi-view object reference dataset from LIBERO-PRO interaction files.

For each interaction in ``datasets/interactions/*.hdf5`` that belongs to the
first demo of its source HDF5 file, this script opens the corresponding demo
file, extracts first / middle / last-frame crops from every camera whose
segmentation shows the object, then computes CLIP (image), DINOv2 (image), and
CLIP (text) embeddings. Object names are sanitised (underscores replaced by
spaces, digits stripped) so that variants like ``moka_pot_1`` and ``moka_pot_2``
collapse onto ``moka pot``. Each sanitised object keeps up to ``--max-images``
image crops; further interactions for a full object are skipped.

Output: ``datasets/all_objects.hdf5`` with layout

    /<sanitised_name>/
        clip_text_embedding   (512,) float32  — L2-normalised CLIP text features
                                                of the sanitised name
        images/
            <idx>/
                image          (h, w, 3) uint8  — masked crop (bg zeroed)
                mask           (H, W)    bool   — binary mask in the source frame
                bbox           (4,)      int32  — [y0, x0, y1, x1]
                clip_embedding (512,)    float32
                dino_embedding (768,)    float32
                attrs:
                    source_dataset
                    source_hdf5
                    source_demo
                    interaction_file
                    interaction_key
                    camera
                    frame_index
                    frame_position  ("first" | "middle" | "final")
                    segmentation_id
                    raw_object_name
        attrs:
            text_prompt
            raw_object_names  (JSON list of every raw name mapped to this key)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from policy_abstraction.utils.segmentation import ROBOT_SEGMENT_KEYWORDS

_DEFAULT_INTERACTIONS_DIR = "datasets/interactions"
_DEFAULT_OUTPUT = "datasets/all_objects.hdf5"
_DEFAULT_PREVIEW = "tmp/all_objects.png"

_CAMERAS = ("agentview", "eye_in_hand")
_FRAME_POSITIONS = ("first", "middle", "final")
_MIN_MASK_PIXELS = 50  # visibility threshold for a mask to count as "visible"


def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return (v / max(norm, eps)).astype(np.float32)


def _first_demo_key(data_grp: h5py.Group) -> str | None:
    demo_keys = [k for k in data_grp.keys() if k.startswith("demo_")]
    if not demo_keys:
        return None
    return sorted(demo_keys, key=lambda k: int(k.split("_")[1]))[0]


def _is_robot_name(name: str) -> bool:
    return any(kw in name for kw in ROBOT_SEGMENT_KEYWORDS)


def _bbox_from_mask(mask: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    H, W = mask.shape
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, H)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)
    return y0, x0, y1, x1


_DIGIT_RE = re.compile(r"\d+")


def sanitize_object_name(name: str) -> str:
    """Remove digits and replace underscores with spaces."""
    no_digits = _DIGIT_RE.sub("", name)
    return " ".join(no_digits.replace("_", " ").split())


def resolve_source_file(raw_path: str, datasets_root: Path) -> Path | None:
    """Re-root a recorded ``source_file`` under the local ``datasets_root``.

    Interaction attrs were produced on another host, so their absolute paths
    (e.g. ``/scratch/.../datasets/libero_object_new/foo.hdf5``) may not exist
    locally. We keep the trailing ``datasets/<dataset_dir>/<file>`` portion and
    resolve it against the current project root.
    """
    p = Path(raw_path)
    parts = p.parts
    if "datasets" in parts:
        idx = parts.index("datasets")
        candidate = datasets_root.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate
    # Fallback: search by basename under ``datasets_root/datasets``.
    for match in (datasets_root / "datasets").rglob(p.name):
        return match
    return None


def pick_frame_indices(start: int, end: int, demo_length: int) -> dict[str, int]:
    """Return frame indices clamped to ``[0, demo_length - 1]``."""
    lo = max(0, min(start, demo_length - 1))
    hi = max(0, min(end, demo_length - 1))
    if lo > hi:
        lo, hi = hi, lo
    mid = (lo + hi) // 2
    return {"first": lo, "middle": mid, "final": hi}


class ClipDinoEncoder:
    """CLIP (image + text) and DINOv2 (image) encoders."""

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch16",
        dino_model: str = "facebook/dinov2-base",
        device: str | None = None,
    ) -> None:
        from transformers import (
            AutoImageProcessor,
            CLIPModel,
            CLIPProcessor,
            CLIPTokenizer,
            Dinov2Model,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"loading CLIP={clip_model} on {device}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.clip = CLIPModel.from_pretrained(clip_model).to(device).eval()
        self.clip_model_name = clip_model

        logger.info(f"loading DINOv2={dino_model} on {device}")
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_model, use_fast=True
        )
        self.dino = Dinov2Model.from_pretrained(dino_model).to(device).eval()
        self.dino_model_name = dino_model

    @torch.no_grad()
    def encode_clip_image(self, image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image)
        inputs = self.clip_processor(images=pil, return_tensors="pt").to(self.device)
        feats = self.clip.get_image_features(**inputs).pooler_output[0].cpu().numpy()
        return _l2_normalize(feats)

    @torch.no_grad()
    def encode_clip_text(self, text: str) -> np.ndarray:
        inputs = self.clip_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.clip.get_text_features(**inputs).pooler_output[0].cpu().numpy()
        return _l2_normalize(feats)

    @torch.no_grad()
    def encode_dino_image(self, image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image)
        inputs = self.dino_processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.dino(pixel_values=pixel_values)
        feats = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        return _l2_normalize(feats)


def _flip_frame(arr: np.ndarray) -> np.ndarray:
    """LIBERO images are stored upside-down — flip both axes. Handles (H,W,*) arrays."""
    return arr[::-1, ::-1].copy()


def _get_or_create_object_group(
    out_file: h5py.File,
    sanitised: str,
    encoder: ClipDinoEncoder,
) -> h5py.Group:
    if sanitised in out_file:
        return out_file[sanitised]
    grp = out_file.create_group(sanitised)
    images_grp = grp.create_group("images")
    del images_grp  # created for its side-effect
    text_emb = encoder.encode_clip_text(sanitised)
    grp.create_dataset("clip_text_embedding", data=text_emb)
    grp.attrs["text_prompt"] = sanitised
    grp.attrs["raw_object_names"] = json.dumps([])
    return grp


def _register_raw_name(grp: h5py.Group, raw_name: str) -> None:
    existing = json.loads(grp.attrs.get("raw_object_names", "[]"))
    if raw_name not in existing:
        existing.append(raw_name)
        grp.attrs["raw_object_names"] = json.dumps(existing)


def _add_image_entry(
    images_grp: h5py.Group,
    *,
    crop: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    clip_emb: np.ndarray,
    dino_emb: np.ndarray,
    attrs: dict,
) -> None:
    idx = len(images_grp)
    entry = images_grp.create_group(str(idx))
    entry.create_dataset("image", data=crop, compression="gzip")
    entry.create_dataset("mask", data=mask, compression="gzip")
    entry.create_dataset("bbox", data=np.asarray(bbox, dtype=np.int32))
    entry.create_dataset("clip_embedding", data=clip_emb)
    entry.create_dataset("dino_embedding", data=dino_emb)
    for k, v in attrs.items():
        entry.attrs[k] = v


def process_interaction(
    *,
    interaction_grp: h5py.Group,
    interaction_key: str,
    interaction_file: str,
    source_demo_file: h5py.File,
    demo_key: str,
    out_file: h5py.File,
    encoder: ClipDinoEncoder,
    bbox_pad: int,
    max_images: int,
) -> int:
    attrs = interaction_grp.attrs
    raw_name = str(attrs["object_name"])
    if _is_robot_name(raw_name):
        return 0
    sanitised = sanitize_object_name(raw_name)
    if not sanitised:
        return 0
    seg_id = int(attrs["object_id"])
    start_ts = int(attrs["start_timestep"])
    end_ts = int(attrs["end_timestep"])

    data_grp = source_demo_file["data"]
    if demo_key not in data_grp:
        return 0
    obs = data_grp[demo_key]["obs"]

    # Any camera's RGB stream will have the correct demo length.
    demo_length = obs[f"{_CAMERAS[0]}_rgb"].shape[0]
    frame_indices = pick_frame_indices(start_ts, end_ts, demo_length)

    dataset_name = Path(interaction_file).stem.replace("_interactions", "")
    source_hdf5 = Path(str(attrs.get("source_file", ""))).name

    obj_grp = _get_or_create_object_group(out_file, sanitised, encoder)
    images_grp = obj_grp["images"]
    _register_raw_name(obj_grp, raw_name)

    added = 0
    for position in _FRAME_POSITIONS:
        if len(images_grp) >= max_images:
            break
        frame_idx = frame_indices[position]
        for camera in _CAMERAS:
            if len(images_grp) >= max_images:
                break
            rgb_key = f"{camera}_rgb"
            seg_key = f"{camera}_segmentation"
            if rgb_key not in obs or seg_key not in obs:
                continue

            rgb = _flip_frame(obs[rgb_key][frame_idx])  # (H, W, 3) uint8
            seg = obs[seg_key][frame_idx]
            seg = _flip_frame(seg[..., 0] if seg.ndim == 3 else seg)

            mask = seg == seg_id
            if int(mask.sum()) < _MIN_MASK_PIXELS:
                continue

            y0, x0, y1, x1 = _bbox_from_mask(mask, pad=bbox_pad)
            crop = rgb[y0:y1, x0:x1].copy()
            local_mask = mask[y0:y1, x0:x1]
            crop[~local_mask] = 0

            clip_emb = encoder.encode_clip_image(crop)
            dino_emb = encoder.encode_dino_image(crop)

            _add_image_entry(
                images_grp,
                crop=crop,
                mask=mask,
                bbox=(y0, x0, y1, x1),
                clip_emb=clip_emb,
                dino_emb=dino_emb,
                attrs={
                    "source_dataset": dataset_name,
                    "source_hdf5": source_hdf5,
                    "source_demo": demo_key,
                    "interaction_file": Path(interaction_file).name,
                    "interaction_key": interaction_key,
                    "camera": camera,
                    "frame_index": int(frame_idx),
                    "frame_position": position,
                    "segmentation_id": seg_id,
                    "raw_object_name": raw_name,
                },
            )
            added += 1
    return added


def extract_from_interaction_file(
    *,
    interaction_path: Path,
    datasets_root: Path,
    out_file: h5py.File,
    encoder: ClipDinoEncoder,
    bbox_pad: int,
    max_images: int,
) -> int:
    added_total = 0
    with h5py.File(interaction_path, "r") as f:
        if "interactions" not in f:
            logger.warning(
                f"no 'interactions' group in {interaction_path.name}, skipping"
            )
            return 0
        interactions = f["interactions"]

        # Pre-compute the "first demo" per source file so we can filter.
        source_to_first_demo: dict[str, str | None] = {}

        ikeys = sorted(interactions.keys(), key=lambda k: (len(k), k))
        for ikey in tqdm(ikeys, desc=interaction_path.name):
            attrs = interactions[ikey].attrs
            raw_src = str(attrs.get("source_file", ""))
            demo_key = str(attrs.get("demo_key", ""))
            if not raw_src or not demo_key:
                continue

            resolved = resolve_source_file(raw_src, datasets_root)
            if resolved is None:
                logger.warning(
                    f"[{interaction_path.name}:{ikey}] cannot resolve {raw_src}"
                )
                continue

            if raw_src not in source_to_first_demo:
                with h5py.File(resolved, "r") as src_f:
                    source_to_first_demo[raw_src] = _first_demo_key(src_f["data"])
            first_demo = source_to_first_demo[raw_src]
            if first_demo is None or demo_key != first_demo:
                continue

            with h5py.File(resolved, "r") as src_f:
                added_total += process_interaction(
                    interaction_grp=interactions[ikey],
                    interaction_key=ikey,
                    interaction_file=str(interaction_path),
                    source_demo_file=src_f,
                    demo_key=demo_key,
                    out_file=out_file,
                    encoder=encoder,
                    bbox_pad=bbox_pad,
                    max_images=max_images,
                )
    return added_total


def save_preview(
    hdf5_path: Path,
    preview_path: Path,
    tile_size: int = 120,
    max_images_per_object: int = 12,
) -> None:
    """Render a grid with every object's crops (rows = objects, cols = images)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(hdf5_path, "r") as f:
        names = sorted(f.keys())
        rows_data: list[tuple[str, list[np.ndarray]]] = []
        for n in names:
            if "images" not in f[n]:
                continue
            images_grp = f[n]["images"]
            idx_keys = sorted(images_grp.keys(), key=lambda k: int(k))[
                :max_images_per_object
            ]
            crops = [images_grp[k]["image"][:] for k in idx_keys]
            if crops:
                rows_data.append((n, crops))

    if not rows_data:
        logger.warning(f"no objects in {hdf5_path} — skipping preview")
        return

    n_rows = len(rows_data)
    n_cols = max(len(crops) for _, crops in rows_data)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * tile_size / 72, n_rows * tile_size / 72),
        dpi=100,
        squeeze=False,
    )

    for r, (name, crops) in enumerate(rows_data):
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(crops):
                ax.imshow(crops[c])
                if c == 0:
                    ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved preview grid ({n_rows} objects) to {preview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root containing the dataset subdirectories.",
    )
    parser.add_argument(
        "--interactions-dir",
        type=Path,
        default=None,
        help=f"Directory of interaction HDF5s (default: {_DEFAULT_INTERACTIONS_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output HDF5 path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help=f"Path for the grid preview PNG (default: {_DEFAULT_PREVIEW}). "
        "Pass an empty string to skip.",
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--bbox-pad", type=int, default=4)
    parser.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="Max image crops to store per sanitised object name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file.",
    )
    args = parser.parse_args()

    interactions_dir = args.interactions_dir or (
        args.datasets_root / _DEFAULT_INTERACTIONS_DIR
    )
    interactions_dir = Path(interactions_dir)
    out_path = args.output or (args.datasets_root / _DEFAULT_OUTPUT)
    out_path = Path(out_path)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists — rerun with --force to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    interaction_files = sorted(interactions_dir.glob("*.hdf5"))
    if not interaction_files:
        raise SystemExit(f"no *.hdf5 files found in {interactions_dir}")

    encoder = ClipDinoEncoder(clip_model=args.clip_model, dino_model=args.dino_model)

    total_added = 0
    with h5py.File(out_path, "w") as out_file:
        out_file.attrs["clip_model"] = encoder.clip_model_name
        out_file.attrs["dino_model"] = encoder.dino_model_name
        out_file.attrs["max_images_per_object"] = args.max_images

        for interaction_path in interaction_files:
            logger.info(f"=== {interaction_path.name} ===")
            total_added += extract_from_interaction_file(
                interaction_path=interaction_path,
                datasets_root=args.datasets_root,
                out_file=out_file,
                encoder=encoder,
                bbox_pad=args.bbox_pad,
                max_images=args.max_images,
            )

        out_file.attrs["num_images"] = total_added
        out_file.attrs["num_objects"] = len(out_file.keys())

    logger.info(
        f"wrote {total_added} image crops across {len(interaction_files)} interaction files "
        f"to {out_path}"
    )

    preview_path = (
        args.preview
        if args.preview is not None
        else (args.datasets_root / _DEFAULT_PREVIEW)
    )
    if str(preview_path):
        save_preview(
            out_path, Path(preview_path), max_images_per_object=args.max_images
        )


if __name__ == "__main__":
    main()

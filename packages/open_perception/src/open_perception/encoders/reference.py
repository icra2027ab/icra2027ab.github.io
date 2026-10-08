"""Reference bank: name -> L2-normalized embeddings (with optional source images).

Loadable from:
  * The canonical HDF5 layout produced by `scripts/extract_objects.py`
    (used by `sam3_clip_rerank_eval.py::_load_reference_bank`).
  * An in-memory dict of `{name: image | list[image]}` encoded on the fly with
    a supplied encoder. Useful when the user wants a custom set of references.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger

from ..utils.images import normalize_name

if TYPE_CHECKING:
    from .clip import ClipEncoder
    from .dino import DinoEncoder


class ReferenceBank:
    __slots__ = ("embeddings", "images", "space")

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        images: dict[str, list[np.ndarray]] | None = None,
        space: Literal["clip", "dino"] = "clip",
    ) -> None:
        self.embeddings = embeddings
        self.images = images or {}
        self.space = space

    def __contains__(self, name: str) -> bool:
        return name in self.embeddings

    @property
    def names(self) -> list[str]:
        return sorted(self.embeddings.keys())

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        embedding_key: str = "clip_embedding",
        space: Literal["clip", "dino"] | None = None,
    ) -> "ReferenceBank":
        import h5py

        if space is None:
            space = "dino" if "dino" in embedding_key else "clip"

        embs_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
        imgs_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
        with h5py.File(path, "r") as f:
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
            embeddings[name] = np.stack(embs, axis=0).astype(np.float32)
        images = {
            name: imgs_by_name[name] for name in embeddings if name in imgs_by_name
        }

        total = sum(v.shape[0] for v in embeddings.values())
        logger.info(
            f"[reference_bank] loaded {len(embeddings)} objects "
            f"({total} refs) from {path} key={embedding_key} space={space}"
        )
        return cls(embeddings=embeddings, images=images, space=space)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, np.ndarray | list[np.ndarray]],
        encoder: "ClipEncoder | DinoEncoder",
        space: Literal["clip", "dino"],
    ) -> "ReferenceBank":
        embeddings: dict[str, np.ndarray] = {}
        images: dict[str, list[np.ndarray]] = {}
        for name, value in data.items():
            imgs = [value] if isinstance(value, np.ndarray) else list(value)
            if not imgs:
                continue
            feats = encoder.encode_images(imgs)
            embeddings[name] = feats
            images[name] = imgs
        logger.info(
            f"[reference_bank] encoded {len(embeddings)} objects from dict in {space} space"
        )
        return cls(embeddings=embeddings, images=images, space=space)

    @classmethod
    def from_any(
        cls,
        source: Any,
        encoders: dict[str, Any],
        space: Literal["clip", "dino"] = "clip",
    ) -> "ReferenceBank":
        """Dispatch on source type — str (HDF5 path), dict, or existing bank."""
        if isinstance(source, ReferenceBank):
            return source
        if isinstance(source, str):
            key = "dino_embedding" if space == "dino" else "clip_embedding"
            return cls.from_hdf5(source, embedding_key=key, space=space)
        if isinstance(source, dict):
            if space not in encoders:
                raise ValueError(
                    f"encoder for space={space!r} not available; enable the corresponding "
                    "disambiguate stage"
                )
            return cls.from_dict(source, encoders[space], space=space)
        raise TypeError(f"cannot build ReferenceBank from {type(source).__name__}")

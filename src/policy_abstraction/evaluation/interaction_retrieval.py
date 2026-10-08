"""Interaction retrieval for the low-level guidance signal.

Loads an ``interactions.hdf5`` file (produced by
``scripts/ablations/extraction/extract_interactions.py``), builds a DINOv2 visual index over the
start frame of each interaction, and retrieves the most similar interaction
for a given live query frame.

Typical usage
-------------
.. code-block:: python

    retriever = InteractionRetriever(
        interactions_path="interactions.hdf5",
        encoder=DemoEncoder(),
        cache_path="interactions_embeddings.npz",
    )
    results = retriever.retrieve(query_frame, top_k=1)
    top = results[0]
    # top["source_file"], top["demo_key"], top["start_timestep"], ...
"""

from __future__ import annotations

import pathlib
from typing import Any

import h5py
import numpy as np
from loguru import logger


class InteractionIndex:
    """In-memory DINOv2 embedding index over interactions from ``interactions.hdf5``.

    For each interaction entry the visual embedding is taken from the start
    frame of the interaction loaded from the original source HDF5 demo.
    Embeddings are L2-normalised so cosine similarity is a plain dot product.

    Parameters
    ----------
    interactions_path:
        Path to the ``interactions.hdf5`` file produced by
        ``scripts/ablations/extraction/extract_interactions.py``.
    encoder:
        :class:`~policy_abstraction.retrieval.encoder.DemoEncoder` instance.
        Used only when building the index — not stored after the index is built.
    cache_path:
        Optional path to a ``.npz`` file for caching computed embeddings.
        If the file exists and was built from the same ``interactions.hdf5``,
        embeddings are loaded from cache instead of being recomputed.
    """

    def __init__(
        self,
        interactions_path: pathlib.Path | str,
        encoder: Any,
        cache_path: pathlib.Path | str | None = None,
    ) -> None:
        self._interactions_path = pathlib.Path(interactions_path)
        self._cache_path = pathlib.Path(cache_path) if cache_path else None

        self._visual_matrix: np.ndarray | None = None  # (N, 768) float32
        self._metadata: list[dict] = []

        self._build(encoder)

    @property
    def size(self) -> int:
        return len(self._metadata)

    @property
    def visual_matrix(self) -> np.ndarray:
        if self._visual_matrix is None:
            raise RuntimeError("InteractionIndex is empty.")
        return self._visual_matrix

    @property
    def metadata(self) -> list[dict]:
        return self._metadata

    def _build(self, encoder: Any) -> None:
        """Load interactions and compute (or restore from cache) embeddings."""
        if not self._interactions_path.exists():
            logger.warning(
                f"[InteractionIndex] interactions.hdf5 not found at {self._interactions_path}."
            )
            return

        # Load interaction metadata from HDF5
        meta_list: list[dict] = []
        with h5py.File(self._interactions_path, "r") as f:
            interactions_grp = f.get("interactions", f)
            for key in sorted(interactions_grp.keys(), key=lambda k: int(k)):
                grp = interactions_grp[key]
                attrs = dict(grp.attrs)
                meta_list.append(
                    {
                        "interaction_id": int(key),
                        "source_file": str(attrs.get("source_file", "")),
                        "demo_key": str(attrs.get("demo_key", "")),
                        "start_timestep": int(attrs.get("start_timestep", 0)),
                        "end_timestep": int(attrs.get("end_timestep", 0)),
                        "object_name": str(attrs.get("object_name", "")),
                        "via_object_name": str(attrs.get("via_object_name", "")),
                        "interaction_chain": str(attrs.get("interaction_chain", "")),
                        "interaction_order": int(attrs.get("interaction_order", 1)),
                    }
                )

        if not meta_list:
            logger.warning("[InteractionIndex] No interaction entries found.")
            return

        # Try loading from cache
        if self._cache_path is not None and self._cache_path.exists():
            try:
                cache = np.load(self._cache_path, allow_pickle=False)
                cached_n = int(cache["n_interactions"])
                if cached_n == len(meta_list):
                    self._visual_matrix = cache["visual_matrix"].astype(np.float32)
                    self._metadata = meta_list
                    logger.info(
                        f"[InteractionIndex] Loaded {len(meta_list)} interaction embeddings "
                        f"from cache {self._cache_path}."
                    )
                    return
                else:
                    logger.info(
                        f"[InteractionIndex] Cache has {cached_n} entries, "
                        f"interactions.hdf5 has {len(meta_list)} — rebuilding."
                    )
            except Exception as exc:
                logger.warning(
                    f"[InteractionIndex] Cache load failed: {exc}. Rebuilding."
                )

        # Build embeddings from start frames
        logger.info(
            f"[InteractionIndex] Building index for {len(meta_list)} interactions…"
        )
        embeddings: list[np.ndarray] = []
        valid_meta: list[dict] = []

        for entry in meta_list:
            emb = self._embed_interaction_start(entry, encoder)
            if emb is not None:
                embeddings.append(emb)
                valid_meta.append(entry)

        if not embeddings:
            logger.warning("[InteractionIndex] No embeddings computed.")
            return

        self._visual_matrix = np.stack(embeddings, axis=0).astype(np.float32)
        self._metadata = valid_meta

        # Save cache
        if self._cache_path is not None:
            try:
                np.savez(
                    self._cache_path,
                    visual_matrix=self._visual_matrix,
                    n_interactions=np.array(len(valid_meta)),
                )
                logger.info(
                    f"[InteractionIndex] Saved embedding cache to {self._cache_path}."
                )
            except Exception as exc:
                logger.warning(f"[InteractionIndex] Cache save failed: {exc}.")

        logger.info(f"[InteractionIndex] Index ready ({len(valid_meta)} interactions).")

    def _embed_interaction_start(self, entry: dict, encoder: Any) -> np.ndarray | None:
        """Load the interaction start frame from the source HDF5 and embed it."""
        source_file = entry["source_file"]
        demo_key = entry["demo_key"]
        start_t = entry["start_timestep"]

        if not source_file or not pathlib.Path(source_file).exists():
            logger.debug(f"[InteractionIndex] Source file not found: {source_file!r}")
            return None
        try:
            with h5py.File(source_file, "r") as f:
                frame = f["data"][demo_key]["obs"]["agentview_rgb"][start_t]  # type: ignore[index]
            # Apply LIBERO flip to match retrieval convention
            frame = np.ascontiguousarray(frame[::-1, ::-1, :])
            emb: np.ndarray = encoder.encode_frames([frame])
            return emb.astype(np.float32)
        except Exception as exc:
            logger.debug(
                f"[InteractionIndex] Embedding failed for {source_file}/{demo_key}@{start_t}: {exc}"
            )
            return None


class InteractionRetriever:
    """Retrieve the most similar interaction(s) for a live query frame.

    Uses the DINOv2 visual embedding of the query frame and computes cosine
    similarity against all indexed interaction start frames.

    Parameters
    ----------
    index:
        Pre-built :class:`InteractionIndex`.
    encoder:
        :class:`~policy_abstraction.retrieval.encoder.DemoEncoder` used to
        embed the query frame at retrieval time.
    """

    def __init__(self, index: InteractionIndex, encoder: Any) -> None:
        self.index = index
        self.encoder = encoder

    def retrieve(self, query_frame: np.ndarray, top_k: int = 1) -> list[dict]:
        """Return the ``top_k`` most visually similar interactions.

        Parameters
        ----------
        query_frame:
            ``(H, W, 3)`` uint8 RGB frame from the live episode, **already
            flipped** (``[::-1, ::-1, :]``).
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Ranked list (best first) of dicts with keys:
            ``rank``, ``interaction_id``, ``source_file``, ``demo_key``,
            ``start_timestep``, ``end_timestep``, ``object_name``,
            ``via_object_name``, ``interaction_chain``, ``interaction_order``,
            ``score``.
        """
        if self.index.size == 0:
            return []

        query_emb: np.ndarray = self.encoder.encode_frames([query_frame])  # (768,)
        scores: np.ndarray = self.index.visual_matrix @ query_emb  # (N,)

        top_k = min(top_k, self.index.size)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        for rank, idx in enumerate(top_indices):
            entry = dict(self.index.metadata[idx])
            entry["rank"] = rank + 1
            entry["score"] = float(scores[idx])
            results.append(entry)

        return results


def build_interaction_retrieval_components(args) -> InteractionRetriever | None:
    """Instantiate :class:`InteractionRetriever` from ``args``.

    Returns ``None`` when interaction retrieval is disabled or the interactions
    file is not found.

    Expected ``args`` attributes
    ----------------------------
    ``enable_interaction_retrieval``: bool
    ``interactions_path``: str — path to ``interactions.hdf5``
    ``dino_retrieval_model``: str — DINOv2 model name (shared with demo retriever)
    ``interaction_embeddings_cache``: str | None — optional .npz cache path
    """
    if not getattr(args, "enable_interaction_retrieval", False):
        return None

    interactions_path = pathlib.Path(getattr(args, "interactions_path", ""))
    if not interactions_path.exists():
        logger.warning(
            f"[InteractionRetrieval] interactions.hdf5 not found at {interactions_path}. "
            "Low-level interaction guidance disabled."
        )
        return None

    from policy_abstraction.retrieval.encoder import DemoEncoder

    encoder = DemoEncoder(
        dino_model=getattr(args, "dino_retrieval_model", "facebook/dinov2-base"),
        clip_model=getattr(
            args, "clip_retrieval_model", "openai/clip-vit-base-patch16"
        ),
    )

    cache_path = getattr(args, "interaction_embeddings_cache", None)
    index = InteractionIndex(
        interactions_path=interactions_path,
        encoder=encoder,
        cache_path=cache_path,
    )

    if index.size == 0:
        logger.warning(
            "[InteractionRetrieval] Index is empty. Low-level guidance disabled."
        )
        return None

    retriever = InteractionRetriever(index=index, encoder=encoder)
    logger.info(f"[InteractionRetrieval] Retriever ready ({index.size} interactions).")
    return retriever

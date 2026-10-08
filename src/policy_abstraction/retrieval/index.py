"""DatasetIndex: loads pre-computed demo embeddings into a searchable in-memory matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger


class DatasetIndex:
    """In-memory index of DINOv2 visual and CLIP text embeddings for demo retrieval.

    Discovers all ``*_embeddings.npz`` files under the given dataset directories,
    loads them, and stacks the embeddings into contiguous NumPy matrices for fast
    batch cosine-similarity queries (dot products on L2-normalised vectors).

    Parameters
    ----------
    dataset_dirs:
        One or more paths to dataset directories (e.g.
        ``"datasets/libero_spatial_new"``). All ``*_embeddings.npz`` files
        found recursively are indexed.
    auto_encode:
        Build the missing ``*_embeddings.npz`` caches instead of raising when
        a directory holds demos nobody has encoded yet. Off by default here
        so a bare ``DatasetIndex`` never writes to disk or loads a model as a
        side effect; ``evaluate.py`` turns it on (``--retrieval.auto-encode``).
    encoder_factory:
        Zero-argument callable returning a ``DemoEncoder``, used only when
        ``auto_encode`` actually has something to encode. Lets the caller
        share one encoder with the retriever instead of loading DINOv2 and
        CLIP twice.

    Examples
    --------
    >>> index = DatasetIndex(["datasets/libero_object_new"])
    >>> index.size
    250
    >>> index.metadata[0]
    {'source_kind': 'libero', 'source_path': '/abs/path/to/task.hdf5', 'demo_key': 'demo_0', 'task': '...', 'npz_path': '...'}
    """

    def __init__(
        self,
        dataset_dirs: list[Path | str],
        *,
        auto_encode: bool = False,
        encoder_factory=None,
    ) -> None:
        self._dataset_dirs = [Path(d) for d in dataset_dirs]
        self._auto_encode = bool(auto_encode)
        self._encoder_factory = encoder_factory
        self._visual_matrix: np.ndarray | None = None  # (N, 768) float32
        self._text_matrix: np.ndarray | None = None  # (N, 512) float32
        self._metadata: list[dict] = []
        self._build()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed demos."""
        return len(self._metadata)

    @property
    def metadata(self) -> list[dict]:
        """Per-demo metadata dicts (source_kind, source_path, demo_key, task, npz_path)."""
        return self._metadata

    @property
    def visual_matrix(self) -> np.ndarray:
        """``(N, 768)`` float32 visual embedding matrix."""
        if self._visual_matrix is None:
            raise RuntimeError("Index is empty — no embedding files found.")
        return self._visual_matrix

    @property
    def text_matrix(self) -> np.ndarray:
        """``(N, 512)`` float32 text embedding matrix."""
        if self._text_matrix is None:
            raise RuntimeError("Index is empty — no embedding files found.")
        return self._text_matrix

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_missing(self) -> list[Path]:
        """Build the caches for any unencoded source, then re-glob.

        Retrieval cannot read a dataset nobody encoded, and the encoding is
        mechanical — the same models, the same output path — so a run that is
        otherwise ready should not die on it. Returns the discovered npz files
        (empty when nothing encodable was found, which the caller reports).
        """
        from policy_abstraction.retrieval.encode import (
            encode_directories,
            sources_under,
        )

        pending = [d for d in self._dataset_dirs if sources_under(d)]
        if not pending:
            return []

        logger.info(
            f"[DatasetIndex] No embeddings found; encoding {len(pending)} "
            f"dataset dir(s) now: {[str(d) for d in pending]}"
        )
        if self._encoder_factory is None:
            from policy_abstraction.retrieval.encoder import DemoEncoder

            encoder = DemoEncoder()
        else:
            encoder = self._encoder_factory()
        encode_directories(pending, encoder)

        found: list[Path] = []
        for dataset_dir in self._dataset_dirs:
            found.extend(dataset_dir.glob("**/*_embeddings.npz"))
        return sorted(found, key=lambda q: str(q.resolve()))

    def _unencoded_message(self) -> str:
        """Say which dirs are unencoded and the exact command that fixes it."""
        from policy_abstraction.retrieval.encode import (
            encode_command_for,
            sources_under,
        )

        lines = [
            "[DatasetIndex] No *_embeddings.npz files found. Retrieval reads "
            "pre-computed embeddings, so every dataset has to be encoded once:",
            "",
        ]
        encodable: list[Path] = []
        for d in self._dataset_dirs:
            sources = sources_under(d)
            if sources:
                encodable.append(d)
                kinds = sorted({src.source_kind for src in sources})
                lines.append(
                    f"  {d} — {len(sources)} {'/'.join(kinds)} source(s), not encoded"
                )
            else:
                lines.append(
                    f"  {d} — no libero/calvin/robocasa/vla data found "
                    f"({'missing' if not Path(d).exists() else 'wrong path?'})"
                )
        lines += ["", "Encode them with:", "", encode_command_for(self._dataset_dirs)]
        if encodable:
            lines += [
                "",
                "…or let the run do it: --retrieval.auto-encode (on by default; "
                "this message means it was disabled or found nothing to encode).",
            ]
        lines += [
            "",
            "To run without retrieval entirely: --retrieval.no-enable "
            "(or --no-guide, which also turns off demo analysis and guidance).",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Discover and load all ``*_embeddings.npz`` files. Called once in ``__init__``."""
        missing = [d for d in self._dataset_dirs if not d.exists()]
        if missing:
            raise FileNotFoundError(
                "[DatasetIndex] These retrieval dataset directories do not "
                "exist:\n"
                + "\n".join(f"  {d}  (resolved: {d.resolve()})" for d in missing)
                + "\n\nThey come from --retrieval.dataset-dirs (the launchers "
                "set it from $RETRIEVAL_DIR), and paths are relative to the "
                "repo root. For LIBERO's own datasets, see "
                "docs/libero_pro_setup.md §6–§9."
            )

        npz_files: list[Path] = []
        for dataset_dir in self._dataset_dirs:
            npz_files.extend(dataset_dir.glob("**/*_embeddings.npz"))

        # sort globally for deterministic ordering
        npz_files = sorted(npz_files, key=lambda p: str(p.resolve()))

        if not npz_files:
            if self._auto_encode:
                npz_files = self._encode_missing()
            if not npz_files:
                raise FileNotFoundError(self._unencoded_message())

        visual_rows: list[np.ndarray] = []
        text_rows: list[np.ndarray] = []

        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as exc:
                print(f"[DatasetIndex] Warning: could not load {npz_path}: {exc}")
                continue

            visual_embs = data["visual_embeddings"]  # (D, 768)
            text_embs = data["text_embeddings"]  # (D, 512)
            demo_keys = data["demo_keys"]  # (D,) str

            # New schema (post-multi-env): source_kind + source_path. Legacy
            # libero caches stored hdf5_path + no source_kind — back-fill.
            if "source_kind" in data:
                source_kind = str(data["source_kind"].item())
                source_path = str(data["source_path"].item())
            else:
                source_kind = "libero"
                source_path = str(
                    data["hdf5_path"].item()
                    if data["hdf5_path"].ndim == 0
                    else data["hdf5_path"]
                )

            # ``task`` is per-cache (scalar) for libero, per-demo (array) for
            # calvin/robocasa. Normalise to a per-demo list.
            task_arr = data["task"]
            if task_arr.ndim == 0:
                tasks = [str(task_arr.item())] * visual_embs.shape[0]
            else:
                tasks = [str(t) for t in task_arr.tolist()]

            if visual_embs.shape[0] == 0:
                continue

            for i, demo_key in enumerate(demo_keys):
                visual_rows.append(visual_embs[i])
                text_rows.append(text_embs[i])
                self._metadata.append(
                    {
                        "source_kind": source_kind,
                        "source_path": source_path,
                        "demo_key": str(demo_key),
                        "task": tasks[i] if i < len(tasks) else tasks[0],
                        "npz_path": str(npz_path),
                    }
                )

        if visual_rows:
            self._visual_matrix = np.stack(visual_rows, axis=0).astype(np.float32)
            self._text_matrix = np.stack(text_rows, axis=0).astype(np.float32)
            print(
                f"[DatasetIndex] Indexed {self.size} demos from {len(npz_files)} files."
            )

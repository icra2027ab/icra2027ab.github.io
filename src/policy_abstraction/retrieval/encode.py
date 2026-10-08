"""Building the ``*_embeddings.npz`` caches retrieval reads.

The cache layout is a contract shared by three parties, so it lives here
rather than in the script that used to own it: ``scripts/encode_datasets.py``
writes caches, :class:`~policy_abstraction.retrieval.index.DatasetIndex` reads
them, and the index can now *build* a missing one on the spot rather than
failing a run that is otherwise ready to go.

One cache sits next to the source data it describes, named after the source
(``<name>_embeddings.npz``), and :func:`cache_path_for` is the single place
that mapping is written down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger
from tqdm import tqdm

from policy_abstraction.retrieval.demo_source import (
    DemoSource,
    build_source,
    detect_source_kind,
)

SOURCE_KINDS = ("libero", "calvin", "robocasa", "robocasa_hdf5", "vla")


def _cache_name(source: DemoSource) -> str:
    p = Path(source.source_path)
    if source.source_kind in ("libero", "robocasa_hdf5"):
        stem = p.stem
        if stem.endswith("_demo"):
            stem = stem[:-5]
        return f"{stem}_embeddings.npz"
    if source.source_kind == "calvin":
        try:
            scene_info = np.load(p / "scene_info.npy", allow_pickle=True).item()
            scene_name = next(iter(scene_info.keys()))
        except Exception:
            scene_name = p.name
        return f"{scene_name}_embeddings.npz"
    if source.source_kind == "vla":
        return f"{p.name}_embeddings.npz"
    if source.source_kind == "robocasa":
        # Walk up the lerobot tree to the named task dir
        # (atomic/<task>/<date>/lerobot/ → <task>).
        for parent in p.parents:
            if parent.parent.name in ("atomic", "composite"):
                return f"{parent.name}_embeddings.npz"
        return f"{p.parent.name}_embeddings.npz"
    raise ValueError(f"unknown source_kind {source.source_kind!r}")


def _cache_dir(source: DemoSource) -> Path:
    p = Path(source.source_path)
    if source.source_kind in ("libero", "robocasa_hdf5"):
        return p.parent
    return p


def cache_path_for(source: DemoSource) -> Path:
    """Where ``source``'s embedding cache belongs on disk."""
    return _cache_dir(source) / _cache_name(source)


def encode_source(
    source: DemoSource,
    encoder,
    out_path: Path,
    batch_size: int = 32,
) -> bool:
    """Encode every demo in ``source`` into ``out_path``.

    Returns True when a cache was written; False when the source held nothing
    to encode (an empty directory is not an error — it is reported and
    skipped, so encoding a tree of sources is not derailed by one dud).
    """
    refs = source.list_demos()
    if not refs:
        logger.warning(f"  [skip] no demos found under {source.source_path}")
        return False

    visual_embs: list[np.ndarray] = []
    text_embs: list[np.ndarray] = []
    demo_keys: list[str] = []
    tasks: list[str] = []

    label = Path(source.source_path).name
    for ref in tqdm(refs, desc=f"  {label}", leave=False):
        frames = list(source.load_frames(ref))
        if not frames:
            continue
        visual_embs.append(encoder.encode_frames(frames, batch_size=batch_size))
        task = source.task_for(ref)
        text_embs.append(encoder.encode_text(task))
        demo_keys.append(ref.demo_key)
        tasks.append(task)

    if not demo_keys:
        logger.warning(f"  [skip] all demos empty under {source.source_path}")
        return False

    visual_matrix = np.stack(visual_embs, axis=0).astype(np.float32)
    text_matrix = np.stack(text_embs, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        source_kind=source.source_kind,
        source_path=source.source_path,
        demo_keys=np.array(demo_keys),
        task=np.array(tasks),
        dino_model=encoder.dino_model_name,
        clip_model=encoder.clip_model_name,
        visual_embeddings=visual_matrix,
        text_embeddings=text_matrix,
    )
    logger.info(f"  Saved {out_path}  ({len(demo_keys)} demos)")
    return True


def discover_sources(
    source_kind: str | None,
    root: Path,
    discover_children: bool = False,
) -> list[DemoSource]:
    """Resolve a path into one or more :class:`DemoSource` instances."""
    if discover_children:
        if not root.is_dir():
            raise NotADirectoryError(
                f"--discover-children requires a directory: {root}"
            )
        out: list[DemoSource] = []
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            try:
                kind = source_kind or detect_source_kind(child)
                out.append(build_source(kind, child))
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(f"  [skip] {child.name}: {exc}")
        return out

    kind = source_kind or detect_source_kind(root)
    return [build_source(kind, root)]


def sources_under(directory: Path | str) -> list[DemoSource]:
    """Every encodable source under ``directory``, or ``[]`` if none is.

    Tries the directory itself first, then each immediate child, which covers
    both shapes retrieval is pointed at: a single folder of ``.vla`` rollouts,
    and a root whose children are each their own dataset. Never raises — a
    path that holds nothing recognisable is a fact for the caller to report,
    not an error here.
    """
    root = Path(directory)
    if not root.exists():
        return []
    try:
        return discover_sources(None, root)
    except (FileNotFoundError, ValueError, NotADirectoryError):
        pass
    if root.is_dir():
        try:
            return discover_sources(None, root, discover_children=True)
        except (FileNotFoundError, ValueError, NotADirectoryError):
            pass
    return []


def encode_command_for(directories: Iterable[Path | str]) -> str:
    """The exact ``scripts/encode_datasets.py`` invocation for these dirs.

    Emitted in error messages, so it has to be copy-pasteable as-is: one call
    per directory, with ``--discover-children`` when that is what it took to
    find the sources.
    """
    lines: list[str] = []
    for d in directories:
        root = Path(d)
        cmd = f"python scripts/encode_datasets.py --source-path {root}"
        try:
            discover_sources(None, root)
        except Exception:
            if sources_under(root):
                cmd += " --discover-children"
            else:
                # Nothing detectable: the kind is the thing the user has to
                # supply, so name the flag rather than pretend this will work.
                cmd += f" --source-kind <{'|'.join(SOURCE_KINDS)}>"
        lines.append(cmd)
    return "\n".join(f"    {line}" for line in lines)


def encode_directories(
    directories: Iterable[Path | str],
    encoder,
    *,
    batch_size: int = 32,
    force: bool = False,
) -> list[Path]:
    """Encode every source found under ``directories``; return caches written."""
    written: list[Path] = []
    for directory in directories:
        for source in sources_under(directory):
            out_path = cache_path_for(source)
            if out_path.exists() and not force:
                logger.info(f"[skip] {out_path}  (already encoded)")
                continue
            logger.info(f"=== {source.source_kind}: {source.source_path} ===")
            if encode_source(source, encoder, out_path, batch_size=batch_size):
                written.append(out_path)
    return written


__all__ = [
    "SOURCE_KINDS",
    "cache_path_for",
    "discover_sources",
    "encode_command_for",
    "encode_directories",
    "encode_source",
    "sources_under",
]

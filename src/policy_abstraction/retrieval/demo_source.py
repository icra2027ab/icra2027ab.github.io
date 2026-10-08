"""Format-agnostic demo source abstraction for retrieval.

Five concrete sources live in :mod:`policy_abstraction.retrieval.sources`:

* ``libero`` — per-task HDF5 files with ``data/demo_N/obs/agentview_rgb`` etc.
* ``calvin`` — NPZ-per-frame split directories with sparse language-annotated
  sub-episodes (``lang_annotations/auto_lang_ann.npy``).
* ``robocasa`` — LeRobot per-task directories with Parquet state and MP4
  per-camera video (RGB-only, no depth/seg/intrinsics).
* ``robocasa_hdf5`` — per-task HDF5 in LIBERO schema, produced by
  ``scripts/setup/collect_robocasa_demos.py``. Adds depth/seg/intrinsics needed
  by ``DemoAnalyzer`` and downstream guidance.
* ``vla`` — a directory of ``.vla`` rollouts recorded by ``evaluate.py``, one
  demo per file (RGB + state + action; no depth/seg, so retrieval only).

The encoder, retrieval index, and visualisation only need three operations
per source: enumerate demos, load all RGB frames for one demo (already
upright), and look up its language string.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np


class DemoRef(NamedTuple):
    """Identifier for one demo within a source.

    ``source_path`` plus ``demo_key`` round-trips through the embedding
    cache and the retrieval result dicts, and is the only thing the
    visualisation layer needs to load a preview frame.
    """

    source_path: str
    demo_key: str


@runtime_checkable
class DemoSource(Protocol):
    """Common interface every per-format adapter implements."""

    source_kind: str
    source_path: str

    def list_demos(self) -> list[DemoRef]: ...

    def load_frames(self, ref: DemoRef) -> np.ndarray:
        """Return ``(T, H, W, 3) uint8`` RGB frames already in upright orientation."""
        ...

    def load_first_frame(self, ref: DemoRef) -> np.ndarray | None:
        """Return ``(H, W, 3) uint8`` upright RGB, or ``None`` if unavailable."""
        ...

    def task_for(self, ref: DemoRef) -> str: ...


def detect_source_kind(path: str | Path) -> str:
    """Best-effort inference of the source format from the on-disk layout.

    Heuristics (checked in order):

    * file ending in ``.hdf5`` → ``"libero"``
    * directory containing ``ep_start_end_ids.npy`` → ``"calvin"``
    * directory whose ``meta/tasks.jsonl`` exists (LeRobot) → ``"robocasa"``
    * directory containing ``*.vla`` episodes → ``"vla"``
    * directory whose immediate children include any of the above markers →
      same kind (lets the caller point at a top-level dataset root)
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".hdf5":
        return "libero"
    if not p.exists():
        raise FileNotFoundError(f"source path does not exist: {p}")
    if (p / "ep_start_end_ids.npy").exists():
        return "calvin"
    if (p / "meta" / "tasks.jsonl").exists():
        return "robocasa"
    if any(p.glob("*.vla")):
        return "vla"
    if any(p.glob("*.hdf5")):
        return "libero"
    # Allow pointing at any ancestor of a robocasa lerobot tree (e.g. the
    # task dir, the dated subdir, or the atomic root). Look up to two
    # levels deep for ``meta/tasks.jsonl``.
    for marker in p.glob("**/meta/tasks.jsonl"):
        # Only accept matches that look like LeRobot (have data/ + videos/
        # siblings under the same parent as meta/).
        lerobot_root = marker.parent.parent
        if (lerobot_root / "data").is_dir() and (lerobot_root / "videos").is_dir():
            return "robocasa"
    raise ValueError(
        f"could not infer source kind for {p}; pass --source-kind explicitly"
    )


def build_source(kind: str, path: str | Path) -> DemoSource:
    """Construct the right :class:`DemoSource` for ``kind`` rooted at ``path``."""
    if kind == "libero":
        from policy_abstraction.retrieval.sources.libero_hdf5 import LiberoHDF5Source

        return LiberoHDF5Source(str(path))
    if kind == "calvin":
        from policy_abstraction.retrieval.sources.calvin_npz import CalvinNpzSource

        return CalvinNpzSource(str(path))
    if kind == "robocasa":
        from policy_abstraction.retrieval.sources.robocasa_lerobot import (
            RobocasaLeRobotSource,
        )

        return RobocasaLeRobotSource(str(path))
    if kind == "vla":
        from policy_abstraction.retrieval.sources.vla_dir import VlaDirSource

        return VlaDirSource(str(path))
    if kind == "robocasa_hdf5":
        from policy_abstraction.retrieval.sources.robocasa_hdf5 import (
            RobocasaHDF5Source,
        )

        return RobocasaHDF5Source(str(path))
    raise ValueError(f"unknown source kind {kind!r}")

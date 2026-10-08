"""LIBERO HDF5 demo source.

Wraps the existing libero per-task HDF5 layout used by the original
encoder (see ``scripts/encode_datasets.py``). Each demo lives at
``data/demo_<idx>/obs/agentview_rgb`` as a ``(T, H, W, 3) uint8`` array
stored upside-down — the loader applies the standard ``[::-1, ::-1]``
flip so callers always see upright frames.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from policy_abstraction.retrieval.demo_source import DemoRef


class LiberoHDF5Source:
    source_kind = "libero"

    def __init__(self, source_path: str) -> None:
        self.source_path = str(Path(source_path).resolve())

    def list_demos(self) -> list[DemoRef]:
        with h5py.File(self.source_path, "r") as f:
            keys = sorted(
                (k for k in f["data"].keys() if k.startswith("demo_")),
                key=lambda k: int(k.split("_")[1]),
            )
        return [DemoRef(self.source_path, k) for k in keys]

    def load_frames(self, ref: DemoRef) -> np.ndarray:
        with h5py.File(ref.source_path, "r") as f:
            raw = f["data"][ref.demo_key]["obs"]["agentview_rgb"][:]
        return np.ascontiguousarray(raw[:, ::-1, ::-1, :])

    def load_first_frame(self, ref: DemoRef) -> np.ndarray | None:
        try:
            with h5py.File(ref.source_path, "r") as f:
                raw = np.asarray(f["data"][ref.demo_key]["obs"]["agentview_rgb"][0])
            return np.ascontiguousarray(raw[::-1, ::-1, :])
        except Exception:
            return None

    def task_for(self, ref: DemoRef) -> str:
        stem = Path(ref.source_path).stem
        if stem.endswith("_demo"):
            stem = stem[:-5]
        return stem.replace("_", " ")

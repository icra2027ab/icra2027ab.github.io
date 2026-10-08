"""RoboCasa HDF5 demo source (LIBERO-schema).

Produced by ``scripts/setup/collect_robocasa_demos.py``, which replays trajectories
from the ``nvidia/RoboCasa-Cosmos-Policy`` LeRobot dataset inside a RoboCasa
sim and captures the full LIBERO-compatible observation set (RGB + depth +
instance segmentation + camera intrinsics/extrinsics + ee pose + gripper
state + actions). The schema is bit-for-bit identical to LIBERO's, so the
:class:`~policy_abstraction.retrieval.sources.libero_hdf5.LiberoHDF5Source`
load path applies verbatim — the only thing that differs is the value of
``source_kind`` (so the retriever / encoder can route per-source logic).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from policy_abstraction.retrieval.demo_source import DemoRef


class RobocasaHDF5Source:
    source_kind = "robocasa_hdf5"

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

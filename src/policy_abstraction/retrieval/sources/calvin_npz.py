"""CALVIN NPZ-per-frame demo source.

CALVIN stores one ``.npz`` per environment frame inside a split directory
(``training/``, ``validation/``). Episode boundaries are recorded in
``ep_start_end_ids.npy``; *language-annotated* sub-episodes — the unit we
treat as a "demo" for retrieval — live in
``lang_annotations/auto_lang_ann.npy`` as ``info.indx = [(start, end), …]``
parallel to ``language.task`` / ``language.ann``.

Each frame NPZ exposes ``rgb_static (200, 200, 3) uint8`` (upright already,
no flip needed). Sub-episodes are usually short (~50 frames), so loading
all frames into memory for embedding is fine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from policy_abstraction.retrieval.demo_source import DemoRef


class CalvinNpzSource:
    source_kind = "calvin"

    def __init__(self, source_path: str) -> None:
        self.source_path = str(Path(source_path).resolve())
        lang_path = Path(self.source_path) / "lang_annotations" / "auto_lang_ann.npy"
        if not lang_path.exists():
            raise FileNotFoundError(
                f"CALVIN split missing language annotations: {lang_path}"
            )
        self._lang = np.load(lang_path, allow_pickle=True).item()
        self._indx: list[tuple[int, int]] = [
            (int(s), int(e)) for s, e in self._lang["info"]["indx"]
        ]
        self._ann: list[str] = list(self._lang["language"]["ann"])
        if len(self._indx) != len(self._ann):
            raise ValueError(
                f"CALVIN annotation length mismatch at {self.source_path}: "
                f"{len(self._indx)} ranges vs {len(self._ann)} strings"
            )

    def list_demos(self) -> list[DemoRef]:
        return [
            DemoRef(self.source_path, f"ep_{s:07d}_{e:07d}") for (s, e) in self._indx
        ]

    def _range_for_key(self, demo_key: str) -> tuple[int, int]:
        # Format produced by list_demos: "ep_<start>_<end>" (zero-padded).
        parts = demo_key.split("_")
        if len(parts) != 3 or parts[0] != "ep":
            raise ValueError(f"unrecognised CALVIN demo key {demo_key!r}")
        return int(parts[1]), int(parts[2])

    def _frame_path(self, frame_idx: int) -> Path:
        return Path(self.source_path) / f"episode_{frame_idx:07d}.npz"

    def load_frames(self, ref: DemoRef) -> np.ndarray:
        start, end = self._range_for_key(ref.demo_key)
        frames: list[np.ndarray] = []
        for fi in range(start, end + 1):
            with np.load(self._frame_path(fi)) as d:
                frames.append(np.asarray(d["rgb_static"]))
        return np.stack(frames, axis=0)

    def load_first_frame(self, ref: DemoRef) -> np.ndarray | None:
        try:
            start, _ = self._range_for_key(ref.demo_key)
            with np.load(self._frame_path(start)) as d:
                return np.ascontiguousarray(np.asarray(d["rgb_static"]))
        except Exception:
            return None

    def task_for(self, ref: DemoRef) -> str:
        start, end = self._range_for_key(ref.demo_key)
        for i, (s, e) in enumerate(self._indx):
            if s == start and e == end:
                return self._ann[i]
        # Fallback — should not hit unless the user constructed a ref by hand.
        return f"calvin sub-episode {ref.demo_key}"

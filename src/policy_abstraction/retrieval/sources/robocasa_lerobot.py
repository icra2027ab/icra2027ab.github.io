"""RoboCasa (LeRobot) demo source.

Each task lives at ``<atomic_root>/<task>/<date>/lerobot/`` and exposes:

* ``data/chunk-000/episode_NNNNNN.parquet`` — per-step state, action,
  ``task_index``.
* ``videos/chunk-000/observation.images.<camera>/episode_NNNNNN.mp4`` —
  H.264 RGB streams, one MP4 per (episode, camera) pair.
* ``meta/tasks.jsonl`` — ``{"task_index": int, "task": str}`` per line.

We treat each Parquet file as one demo and embed the
``robot0_agentview_left`` MP4 as the visual stream (it is the closest
analogue to LIBERO's ``agentview``). Frames are upright as recorded — no
flip is applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq

from policy_abstraction.retrieval.demo_source import DemoRef

_PRIMARY_CAMERA = "robot0_agentview_left"


def _resolve_lerobot_root(path: str) -> Path:
    """Accept the ``lerobot/`` dir, its date-stamped parent, or the task root.

    Layout in the wild:
      ``atomic/<task>/<YYYYMMDD>/lerobot/{data,videos,meta}/``

    Any of those four directories is a valid source path. We search up to
    two levels deep for the LeRobot ``meta/tasks.jsonl`` marker.
    """
    p = Path(path).resolve()
    candidates: list[Path] = [p, p / "lerobot"]
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_dir():
                candidates.append(child / "lerobot")
                candidates.append(child)
    for candidate in candidates:
        if (candidate / "meta" / "tasks.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"could not find lerobot/meta/tasks.jsonl under {p}")


class RobocasaLeRobotSource:
    source_kind = "robocasa"

    def __init__(self, source_path: str) -> None:
        root = _resolve_lerobot_root(source_path)
        self.source_path = str(root)
        self._data_dir = root / "data" / "chunk-000"
        self._video_dir = (
            root / "videos" / "chunk-000" / f"observation.images.{_PRIMARY_CAMERA}"
        )
        self._tasks: dict[int, str] = {}
        with (root / "meta" / "tasks.jsonl").open() as f:
            for line in f:
                rec = json.loads(line)
                self._tasks[int(rec["task_index"])] = str(rec["task"])

    def _episode_id(self, demo_key: str) -> int:
        # demo_key format: "episode_NNNNNN"
        if not demo_key.startswith("episode_"):
            raise ValueError(f"unexpected RoboCasa demo key {demo_key!r}")
        return int(demo_key.split("_", 1)[1])

    def _parquet_path(self, episode_id: int) -> Path:
        return self._data_dir / f"episode_{episode_id:06d}.parquet"

    def _video_path(self, episode_id: int) -> Path:
        return self._video_dir / f"episode_{episode_id:06d}.mp4"

    def list_demos(self) -> list[DemoRef]:
        eps = sorted(self._data_dir.glob("episode_*.parquet"))
        keys = [p.stem for p in eps]
        return [DemoRef(self.source_path, k) for k in keys]

    def load_frames(self, ref: DemoRef) -> np.ndarray:
        eid = self._episode_id(ref.demo_key)
        mp4 = self._video_path(eid)
        # imageio.v3.imiter streams frames as np.ndarray (H, W, 3) uint8.
        return np.stack(list(iio.imiter(str(mp4))), axis=0)

    def load_first_frame(self, ref: DemoRef) -> np.ndarray | None:
        try:
            eid = self._episode_id(ref.demo_key)
            arr = iio.imread(str(self._video_path(eid)), index=0)
            return np.ascontiguousarray(np.asarray(arr))
        except Exception:
            return None

    def task_for(self, ref: DemoRef) -> str:
        eid = self._episode_id(ref.demo_key)
        # Read just the task_index column from the first row to keep this cheap.
        table = pq.read_table(
            str(self._parquet_path(eid)),
            columns=["task_index"],
        )
        idx = int(table["task_index"][0].as_py())
        return self._tasks.get(idx, f"robocasa task {idx}")

"""VLA rollout-directory demo source.

A directory of ``.vla`` episodes — as written by
:class:`~policy_abstraction.data.recorder.RobodmRecorder` during
``scripts/evaluate.py`` — is treated as a single dataset: every ``*.vla`` file
is one demo, keyed by its filename stem. This is what lets a folder of real
robot rollouts be retrieved from without first converting it to an HDF5
schema.

Each episode carries a JSON sidecar of the same stem (``episode_000.json``)
holding ``task_description`` and the serialised ``EnvSpec``. The sidecar is
required: the task string is the CLIP query for that demo, and the EnvSpec
names which camera is primary. Deriving either by guesswork would silently
poison the index, so a missing sidecar is an error rather than a fallback.

Frames are returned exactly as recorded. Real-robot recordings (UR5e) are
already upright; no flip is applied here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy_abstraction.retrieval.demo_source import DemoRef

# Legacy squeezed schema: before per-camera streams, the primary camera was
# always written to this key regardless of the env's real camera name.
_LEGACY_PRIMARY = "observation/agentview_image"

_IMAGE_PREFIX = "observation/"
_IMAGE_SUFFIX = "_image"

# Head-slice size for the first-frame preview. robodm decodes video streams in
# whole units, so a slice must span one before any frame comes back: slice(0,1)
# and slice(0,2) both yield an empty array, slice(0,8) yields 6 frames starting
# at the true frame 0 (verified against a full decode). Slicing costs ~0.2s and
# a few MB against ~14s and several GB for decoding a whole episode, which is
# what the k retrieval previews would otherwise pay each on episode start.
_HEAD_SLICE = 8


class VlaDirSource:
    """Directory of ``.vla`` rollouts, one demo per file."""

    source_kind = "vla"

    def __init__(self, source_path: str) -> None:
        root = Path(source_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"vla source must be a directory: {root}")
        if not any(root.glob("*.vla")):
            raise FileNotFoundError(f"no *.vla files under {root}")
        self.source_path = str(root)
        self._root = root
        # Populated by load_frames so a subsequent load_first_frame (the
        # retrieval preview) does not pay for a second full decode.
        self._first_frames: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Paths and sidecar
    # ------------------------------------------------------------------

    def _vla_path(self, demo_key: str) -> Path:
        path = self._root / f"{demo_key}.vla"
        if not path.exists():
            raise FileNotFoundError(f"no such episode: {path}")
        return path

    def _sidecar(self, demo_key: str) -> dict:
        path = self._vla_path(demo_key).with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(
                f"episode {demo_key!r} has no JSON sidecar at {path}. The "
                "sidecar carries the task string and EnvSpec this source "
                "needs; it is written alongside every .vla by RobodmRecorder."
            )
        with path.open() as fh:
            return json.load(fh)

    def _primary_stream_key(self, demo_key: str, available: list[str]) -> str:
        """Name of the feature stream holding the primary camera's RGB.

        Prefers the camera the episode's own EnvSpec marks primary, then the
        legacy squeezed key, then a sole image stream. Anything else is
        ambiguous and raises rather than picking a camera at random.
        """
        spec = self._sidecar(demo_key).get("env_spec") or {}
        for cam in spec.get("cameras", []):
            if cam.get("is_primary"):
                key = f"{_IMAGE_PREFIX}{cam['name']}{_IMAGE_SUFFIX}"
                if key in available:
                    return key

        if _LEGACY_PRIMARY in available:
            return _LEGACY_PRIMARY

        images = [
            k
            for k in available
            if k.startswith(_IMAGE_PREFIX) and k.endswith(_IMAGE_SUFFIX)
        ]
        if len(images) == 1:
            return images[0]
        raise KeyError(
            f"could not identify the primary camera stream for {demo_key!r}; "
            f"candidates={sorted(images)}. The episode's EnvSpec names no "
            "primary camera present in the file."
        )

    # ------------------------------------------------------------------
    # DemoSource interface
    # ------------------------------------------------------------------

    def list_demos(self) -> list[DemoRef]:
        return [
            DemoRef(self.source_path, p.stem) for p in sorted(self._root.glob("*.vla"))
        ]

    def load_frames(self, ref: DemoRef) -> np.ndarray:
        """Return ``(T, H, W, 3)`` uint8 RGB for the primary camera.

        robodm decodes every stream in the file, so peak memory is set by the
        whole episode (all cameras), not by the stream we keep.
        """
        import robodm

        traj = robodm.Trajectory(str(self._vla_path(ref.demo_key)), mode="r")
        try:
            data = traj.load()
        finally:
            traj.close()
        if not data:
            raise RuntimeError(f"no data decoded from {ref.demo_key}.vla")

        key = self._primary_stream_key(ref.demo_key, list(data))
        frames = np.asarray(data[key])
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"expected (T, H, W, 3) frames under {key!r}; got {frames.shape}"
            )
        if len(frames):
            self._first_frames[ref.demo_key] = np.ascontiguousarray(frames[0])
        return frames

    def load_first_frame(self, ref: DemoRef) -> np.ndarray | None:
        cached = self._first_frames.get(ref.demo_key)
        if cached is not None:
            return cached
        try:
            frames = self._load_head(ref)
            if frames is None or not len(frames):
                # Codec or stream layout the head slice cannot serve; pay for
                # the whole episode rather than return nothing.
                frames = self.load_frames(ref)
        except Exception:
            return None
        if not len(frames):
            return None
        first = np.ascontiguousarray(frames[0])
        self._first_frames[ref.demo_key] = first
        return first

    def _load_head(self, ref: DemoRef) -> np.ndarray | None:
        """Decode only the first :data:`_HEAD_SLICE` steps of an episode."""
        import robodm

        traj = robodm.Trajectory(str(self._vla_path(ref.demo_key)), mode="r")
        try:
            data = traj.load(data_slice=slice(0, _HEAD_SLICE))
        finally:
            traj.close()
        if not data:
            return None
        key = self._primary_stream_key(ref.demo_key, list(data))
        frames = np.asarray(data[key])
        return frames if frames.ndim == 4 else None

    def task_for(self, ref: DemoRef) -> str:
        task = self._sidecar(ref.demo_key).get("task_description")
        if not task:
            raise KeyError(
                f"episode {ref.demo_key!r} sidecar has no 'task_description'; "
                "it is the CLIP query for this demo."
            )
        return str(task)

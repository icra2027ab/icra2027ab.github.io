"""Tests for the ``vla`` demo source (a directory of .vla rollouts)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from policy_abstraction.retrieval.demo_source import (
    build_source,
    detect_source_kind,
)
from policy_abstraction.retrieval.sources.vla_dir import VlaDirSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sidecar(primary: str | None = "cam_front", extra: str = "cam_wrist") -> dict:
    """Episode metadata as RobodmRecorder writes it."""
    cameras = []
    for name in (primary, extra):
        if name is None:
            continue
        cameras.append({"name": name, "is_primary": name == primary})
    return {
        "task_description": "put all cables in the bag",
        "env_spec": {"cameras": cameras},
    }


def _write_episode(directory, stem: str, meta: dict | None = None) -> None:
    """Write a ``<stem>.json`` sidecar (no .vla payload) plus a stub .vla."""
    (directory / f"{stem}.vla").write_bytes(b"")
    if meta is not None:
        (directory / f"{stem}.json").write_text(json.dumps(meta))


def _write_real_vla(directory, stem: str, frames: dict[str, np.ndarray]) -> None:
    """Write a genuine robodm episode so load_frames can be round-tripped."""
    import robodm

    traj = robodm.Trajectory(str(directory / f"{stem}.vla"), mode="w")
    n = len(next(iter(frames.values())))
    for t in range(n):
        for key, arr in frames.items():
            traj.add(key, np.ascontiguousarray(arr[t]))
    traj.close()


# ---------------------------------------------------------------------------
# Detection and construction
# ---------------------------------------------------------------------------


class TestDetection:
    def test_directory_of_vla_files_detected(self, tmp_path):
        _write_episode(tmp_path, "episode_000", _sidecar())
        assert detect_source_kind(tmp_path) == "vla"

    def test_build_source_returns_vla_source(self, tmp_path):
        _write_episode(tmp_path, "episode_000", _sidecar())
        source = build_source("vla", tmp_path)
        assert isinstance(source, VlaDirSource)
        assert source.source_kind == "vla"

    def test_rejects_directory_without_episodes(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no \\*.vla files"):
            VlaDirSource(str(tmp_path))

    def test_rejects_a_file_path(self, tmp_path):
        target = tmp_path / "episode_000.vla"
        target.write_bytes(b"")
        with pytest.raises(NotADirectoryError):
            VlaDirSource(str(target))


# ---------------------------------------------------------------------------
# Demo listing and task lookup
# ---------------------------------------------------------------------------


class TestListingAndTasks:
    def test_lists_one_demo_per_file_sorted(self, tmp_path):
        for stem in ("episode_002", "episode_000", "episode_001"):
            _write_episode(tmp_path, stem, _sidecar())
        refs = VlaDirSource(str(tmp_path)).list_demos()
        assert [r.demo_key for r in refs] == [
            "episode_000",
            "episode_001",
            "episode_002",
        ]
        assert {r.source_path for r in refs} == {str(tmp_path.resolve())}

    def test_task_comes_from_sidecar(self, tmp_path):
        _write_episode(tmp_path, "episode_000", _sidecar())
        source = VlaDirSource(str(tmp_path))
        assert source.task_for(source.list_demos()[0]) == "put all cables in the bag"

    def test_missing_sidecar_raises(self, tmp_path):
        _write_episode(tmp_path, "episode_000", meta=None)
        source = VlaDirSource(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="no JSON sidecar"):
            source.task_for(source.list_demos()[0])

    def test_sidecar_without_task_raises(self, tmp_path):
        meta = _sidecar()
        del meta["task_description"]
        _write_episode(tmp_path, "episode_000", meta)
        source = VlaDirSource(str(tmp_path))
        with pytest.raises(KeyError, match="task_description"):
            source.task_for(source.list_demos()[0])


# ---------------------------------------------------------------------------
# Primary-camera resolution
# ---------------------------------------------------------------------------


class TestPrimaryStreamKey:
    def _source(self, tmp_path, meta):
        _write_episode(tmp_path, "episode_000", meta)
        return VlaDirSource(str(tmp_path))

    def test_prefers_env_spec_primary_camera(self, tmp_path):
        source = self._source(tmp_path, _sidecar())
        key = source._primary_stream_key(
            "episode_000",
            ["observation/cam_wrist_image", "observation/cam_front_image"],
        )
        assert key == "observation/cam_front_image"

    def test_falls_back_to_legacy_squeezed_key(self, tmp_path):
        # EnvSpec names a primary camera that has no stream in this file.
        source = self._source(tmp_path, _sidecar())
        key = source._primary_stream_key(
            "episode_000",
            ["observation/agentview_image", "observation/wrist_image"],
        )
        assert key == "observation/agentview_image"

    def test_falls_back_to_sole_image_stream(self, tmp_path):
        source = self._source(tmp_path, _sidecar(primary=None, extra="cam_wrist"))
        key = source._primary_stream_key(
            "episode_000", ["observation/only_image", "observation/state"]
        )
        assert key == "observation/only_image"

    def test_ambiguous_cameras_raise(self, tmp_path):
        source = self._source(tmp_path, _sidecar(primary=None, extra="cam_wrist"))
        with pytest.raises(KeyError, match="primary camera"):
            source._primary_stream_key(
                "episode_000", ["observation/a_image", "observation/b_image"]
            )


# ---------------------------------------------------------------------------
# Frame loading (real robodm round-trip)
# ---------------------------------------------------------------------------


class TestLoadFrames:
    @pytest.fixture
    def episode_dir(self, tmp_path):
        rng = np.random.default_rng(0)
        front = rng.integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)
        wrist = rng.integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)
        _write_real_vla(
            tmp_path,
            "episode_000",
            {
                "observation/cam_front_image": front,
                "observation/cam_wrist_image": wrist,
            },
        )
        (tmp_path / "episode_000.json").write_text(json.dumps(_sidecar()))
        return tmp_path, front

    def test_returns_primary_camera_stream(self, episode_dir):
        directory, front = episode_dir
        source = VlaDirSource(str(directory))
        frames = source.load_frames(source.list_demos()[0])
        assert frames.shape == front.shape
        assert frames.dtype == np.uint8

    def test_first_frame_matches_stream_head(self, episode_dir):
        directory, _ = episode_dir
        source = VlaDirSource(str(directory))
        ref = source.list_demos()[0]
        frames = source.load_frames(ref)
        np.testing.assert_array_equal(source.load_first_frame(ref), frames[0])

    def test_first_frame_without_prior_load(self, episode_dir):
        directory, _ = episode_dir
        source = VlaDirSource(str(directory))
        first = source.load_first_frame(source.list_demos()[0])
        assert first is not None and first.shape == (16, 16, 3)

"""Tests for policy_abstraction.retrieval.index.DatasetIndex."""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.retrieval.index import DatasetIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_npz(
    directory,
    name: str,
    n_demos: int,
    visual_dim: int = 768,
    text_dim: int = 512,
    source_path: str = "fake.hdf5",
    source_kind: str | None = None,
    task: str = "pick up the mug",
) -> None:
    """Write a synthetic *_embeddings.npz to *directory*.

    ``source_kind=None`` writes the legacy libero-only schema (``hdf5_path``,
    no ``source_kind``) that ``DatasetIndex`` back-fills; passing a kind writes
    the current multi-env schema.
    """
    rng = np.random.default_rng(0)
    path = directory / name
    if source_kind is None:
        source_fields = {"hdf5_path": np.array(source_path)}
    else:
        source_fields = {
            "source_kind": np.array(source_kind),
            "source_path": np.array(source_path),
        }
    np.savez(
        path,
        visual_embeddings=rng.random((n_demos, visual_dim), dtype=np.float64).astype(
            np.float32
        ),
        text_embeddings=rng.random((n_demos, text_dim), dtype=np.float64).astype(
            np.float32
        ),
        demo_keys=np.array([f"demo_{i}" for i in range(n_demos)]),
        task=np.array(task),
        **source_fields,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetIndex:
    """Tests for DatasetIndex."""

    def test_unencoded_directory_raises(self, tmp_path):
        """Directory that exists but holds no npz → actionable FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No \\*_embeddings.npz"):
            DatasetIndex([tmp_path])

    def test_the_unencoded_message_carries_a_runnable_command(self, tmp_path):
        """The whole point of the message: paste the command, be unblocked.

        A bare "run encode_datasets.py" costs a trip to the script's --help to
        find out which of five flags it wanted.
        """
        (tmp_path / "rollout_0.vla").touch()
        with pytest.raises(FileNotFoundError) as exc:
            DatasetIndex([tmp_path], auto_encode=False)
        msg = str(exc.value)
        assert f"--source-path {tmp_path}" in msg
        assert "1 vla source(s), not encoded" in msg
        assert "--retrieval.no-enable" in msg  # the way out, if they want it

    def test_the_unencoded_message_says_when_nothing_is_encodable(self, tmp_path):
        """An empty dir is a wrong path, not a missing encode — say so."""
        with pytest.raises(FileNotFoundError) as exc:
            DatasetIndex([tmp_path], auto_encode=False)
        assert "no libero/calvin/robocasa/vla data found" in str(exc.value)

    def test_auto_encode_builds_the_missing_cache(self, tmp_path, monkeypatch):
        """Discover → encode → re-glob → load, with only the model stubbed."""
        from policy_abstraction.retrieval import encode as encode_mod

        (tmp_path / "rollout_0.vla").touch()
        seen: list = []

        def _fake_encode_source(source, encoder, out_path, batch_size=32):
            seen.append((source.source_kind, encoder, out_path))
            _write_npz(out_path.parent, out_path.name, n_demos=2, source_kind="vla")
            return True

        monkeypatch.setattr(encode_mod, "encode_source", _fake_encode_source)
        sentinel = object()
        index = DatasetIndex(
            [tmp_path], auto_encode=True, encoder_factory=lambda: sentinel
        )

        assert index.size == 2
        kind, encoder, out_path = seen[0]
        assert kind == "vla"
        # The caller's encoder is reused, not a second DINOv2/CLIP load.
        assert encoder is sentinel
        assert out_path == tmp_path / f"{tmp_path.name}_embeddings.npz"

    def test_auto_encode_still_raises_when_there_is_nothing_to_encode(self, tmp_path):
        """No source under the path → encoding cannot rescue it; say so."""
        with pytest.raises(FileNotFoundError, match="no libero/calvin"):
            DatasetIndex([tmp_path], auto_encode=True)

    def test_auto_encode_is_off_by_default(self, tmp_path, monkeypatch):
        """A bare DatasetIndex must not write to disk or load a model."""
        from policy_abstraction.retrieval import encode as encode_mod

        (tmp_path / "rollout_0.vla").touch()

        def _boom(*a, **k):  # pragma: no cover - must not run
            raise AssertionError("auto-encode ran without being asked")

        monkeypatch.setattr(encode_mod, "encode_source", _boom)
        with pytest.raises(FileNotFoundError):
            DatasetIndex([tmp_path])

    def test_nonexistent_dir_raises(self, tmp_path):
        """Nonexistent directory → actionable FileNotFoundError."""
        fake_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="do not exist"):
            DatasetIndex([str(fake_dir)])

    def test_single_npz_loaded(self, tmp_path):
        """Single npz with 3 demos → size == 3."""
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=3)
        index = DatasetIndex([tmp_path])
        assert index.size == 3

    def test_metadata_keys_present(self, tmp_path):
        """Each metadata entry has the required keys."""
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=2)
        index = DatasetIndex([tmp_path])
        for meta in index.metadata:
            for key in ("source_kind", "source_path", "demo_key", "task", "npz_path"):
                assert key in meta, f"Missing metadata key: {key}"

    def test_visual_matrix_shape(self, tmp_path):
        """visual_matrix has shape (N, 768) float32."""
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=4)
        index = DatasetIndex([tmp_path])
        mat = index.visual_matrix
        assert mat.shape == (4, 768)
        assert mat.dtype == np.float32

    def test_text_matrix_shape(self, tmp_path):
        """text_matrix has shape (N, 512) float32."""
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=4)
        index = DatasetIndex([tmp_path])
        mat = index.text_matrix
        assert mat.shape == (4, 512)
        assert mat.dtype == np.float32

    def test_multiple_npz_files_stacked(self, tmp_path):
        """Two npz files with D1=3 and D2=5 demos → size == 8."""
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_npz(tmp_path, "a_embeddings.npz", n_demos=3)
        _write_npz(sub, "b_embeddings.npz", n_demos=5)
        index = DatasetIndex([tmp_path])
        assert index.size == 8

    def test_empty_npz_skipped(self, tmp_path):
        """npz with 0 demos is skipped (size == 0 after indexing)."""
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=0)
        index = DatasetIndex([tmp_path])
        assert index.size == 0

    def test_visual_matrix_raises_when_empty(self, tmp_path):
        """Accessing visual_matrix on an empty index raises RuntimeError."""
        # An npz that holds zero demos is the only way to build an empty index:
        # a directory with no npz at all is rejected up front by _build.
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=0)
        index = DatasetIndex([tmp_path])
        with pytest.raises(RuntimeError, match="empty"):
            _ = index.visual_matrix

    def test_text_matrix_raises_when_empty(self, tmp_path):
        """Accessing text_matrix on an empty index raises RuntimeError."""
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=0)
        index = DatasetIndex([tmp_path])
        with pytest.raises(RuntimeError, match="empty"):
            _ = index.text_matrix

    def test_demo_key_in_metadata(self, tmp_path):
        """demo_key in metadata matches the keys stored in the npz."""
        _write_npz(tmp_path, "task_embeddings.npz", n_demos=2)
        index = DatasetIndex([tmp_path])
        keys = [m["demo_key"] for m in index.metadata]
        assert "demo_0" in keys
        assert "demo_1" in keys

    def test_source_path_stored(self, tmp_path):
        """source_kind/source_path in metadata match what was written in the npz."""
        _write_npz(
            tmp_path,
            "task_embeddings.npz",
            n_demos=1,
            source_kind="calvin",
            source_path="/data/robot.npz",
        )
        index = DatasetIndex([tmp_path])
        assert index.metadata[0]["source_kind"] == "calvin"
        assert index.metadata[0]["source_path"] == "/data/robot.npz"

    def test_legacy_hdf5_path_backfilled(self, tmp_path):
        """A pre-multi-env npz (hdf5_path, no source_kind) back-fills to libero."""
        _write_npz(
            tmp_path, "task_embeddings.npz", n_demos=1, source_path="/data/robot.hdf5"
        )
        index = DatasetIndex([tmp_path])
        assert index.metadata[0]["source_kind"] == "libero"
        assert index.metadata[0]["source_path"] == "/data/robot.hdf5"

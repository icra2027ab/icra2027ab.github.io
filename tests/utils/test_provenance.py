"""Tests for `policy_abstraction.utils.provenance`."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from policy_abstraction.utils.provenance import (
    collect_provenance,
    write_provenance,
)


@dataclasses.dataclass
class _FakeArgs:
    seed: int = 7
    run: str = "demo"
    flag: bool = True


def test_collect_provenance_keys_present():
    prov = collect_provenance(_FakeArgs())
    for key in (
        "timestamp_utc",
        "hostname",
        "platform",
        "python_version",
        "git",
        "uv_lock_sha256",
        "args",
        "gpu",
        "checkpoints",
        "datasets",
    ):
        assert key in prov, f"missing key: {key}"


def test_collect_provenance_args_dataclass():
    prov = collect_provenance(_FakeArgs(seed=42, run="x", flag=False))
    assert prov["args"] == {"seed": 42, "run": "x", "flag": False}


def test_collect_provenance_args_namespace():
    import argparse

    ns = argparse.Namespace(seed=1, foo="bar")
    prov = collect_provenance(ns)
    assert prov["args"] == {"seed": 1, "foo": "bar"}


def test_collect_provenance_dataset_and_checkpoint_hashes(tmp_path: Path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"hello")
    dataset = tmp_path / "data.bin"
    dataset.write_bytes(b"abc")

    prov = collect_provenance(
        _FakeArgs(),
        checkpoint_paths=[ckpt],
        dataset_paths=[dataset],
    )
    assert len(prov["checkpoints"]) == 1
    assert prov["checkpoints"][0]["path"] == str(ckpt)
    assert prov["checkpoints"][0]["sha256"] is not None
    assert prov["checkpoints"][0]["size_bytes"] == 5

    assert prov["datasets"][0]["size_bytes"] == 3


def test_collect_provenance_extra_merged():
    prov = collect_provenance(_FakeArgs(), extra={"task": "replay", "n": 1})
    assert prov["task"] == "replay"
    assert prov["n"] == 1


def test_write_provenance_roundtrip(tmp_path: Path):
    prov = collect_provenance(_FakeArgs())
    out_path = write_provenance(tmp_path / "run", prov)
    assert out_path.name == "run_metadata.json"
    assert out_path.exists()

    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["args"] == prov["args"]
    assert loaded["timestamp_utc"] == prov["timestamp_utc"]


def test_write_provenance_creates_dir(tmp_path: Path):
    target = tmp_path / "deeply" / "nested" / "dir"
    write_provenance(target, collect_provenance(_FakeArgs()))
    assert (target / "run_metadata.json").exists()

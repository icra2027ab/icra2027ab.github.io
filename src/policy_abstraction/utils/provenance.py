"""Capture run provenance into a JSON file alongside run artifacts.

Every entrypoint should call :func:`collect_provenance` then
:func:`write_provenance` so each run directory is self-describing: git SHA,
``uv.lock`` hash, resolved CLI args, dataset hashes, checkpoint hashes, and
host/GPU info.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from policy_abstraction import PROJECT_ROOT


def _git_sha_and_dirty() -> tuple[str | None, bool | None]:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _file_sha256(path: str | Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _file_size(path: str | Path) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


def collect_provenance(
    args: Any,
    *,
    checkpoint_paths: list[str | Path] | None = None,
    dataset_paths: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable provenance dict for a single run.

    ``args`` may be a dataclass (e.g. a tyro-populated ``Args``) or any object
    with a ``__dict__`` (e.g. ``argparse.Namespace``).
    """
    sha, dirty = _git_sha_and_dirty()
    uv_lock = PROJECT_ROOT / "uv.lock"

    if dataclasses.is_dataclass(args) and not isinstance(args, type):
        args_dict = dataclasses.asdict(args)
    else:
        args_dict = dict(vars(args))

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git": {"sha": sha, "dirty": dirty},
        "uv_lock_sha256": _file_sha256(uv_lock),
        "args": args_dict,
        "gpu": _gpu_info(),
        "checkpoints": [
            {
                "path": str(p),
                "sha256": _file_sha256(p),
                "size_bytes": _file_size(p),
            }
            for p in (checkpoint_paths or [])
        ],
        "datasets": [
            {"path": str(p), "size_bytes": _file_size(p)} for p in (dataset_paths or [])
        ],
        **(extra or {}),
    }


def write_provenance(out_dir: str | Path, provenance: dict[str, Any]) -> Path:
    """Write ``run_metadata.json`` into ``out_dir``. Returns the written path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_metadata.json"
    with open(out_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info(f"Wrote run provenance to {out_path}")
    return out_path

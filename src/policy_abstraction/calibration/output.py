"""Writing the result where ``Ur5EnvServer`` reads it.

The file layout is the env's own (``docs/ur5.md``, *Camera calibration*)::

    {
      "base_frame": "base",
      "exterior_image_1_left": {"extrinsic": [[...]], "meta": {...}},
      "wrist_image_left": {"extrinsic": [[...]], "parent": "tcp", "meta": {...}}
    }

``extrinsic`` is always the camera's pose *in its parent frame*. For a fixed
camera the parent is the robot base — RTDE's ``base``, since the tracked point
came straight from ``getActualTCPPose()`` — and for a wrist camera it is the
TCP, flagged with ``"parent": "tcp"`` so the server composes it with the live
TCP pose on every ``get_camera_params`` call.

A previous file is merged, not replaced: recalibrating one camera keeps the
other's entry. If that older file was expressed in ``base_link`` (what the
URDF-based pipeline produced), its base-parented entries are rotated into
``base`` so the file stays in one frame — the correction is exact, a 180
degree turn about z, and the server would otherwise apply it to the new entry
too.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import shutil
from typing import Any, Optional

import numpy as np
from loguru import logger

from policy_abstraction.calibration.geometry import is_rigid

BASE_FRAME = "base"
TCP_PARENT = "tcp"
_NON_CAMERA_KEYS = {"base_frame"}


def base_link_to_base() -> np.ndarray:
    """Maps a pose expressed in URDF ``base_link`` into RTDE ``base``.

    The same matrix as ``Ur5EnvServer._base_frame_correction("base_link")``;
    duplicated rather than imported so this module stays free of the server's
    hardware imports.
    """
    T = np.eye(4)
    T[:3, :3] = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    return T


def camera_entry(
    T_parent_cam: np.ndarray,
    *,
    parent: str,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """One camera's entry. *parent* is ``"base"`` or ``"tcp"``."""
    T = np.asarray(T_parent_cam, dtype=np.float64).reshape(4, 4)
    if not is_rigid(T):
        raise ValueError("extrinsic is not a rigid transform")
    entry: dict[str, Any] = {"extrinsic": T.tolist()}
    if parent == TCP_PARENT:
        entry["parent"] = TCP_PARENT
    elif parent != BASE_FRAME:
        raise ValueError(f"parent must be 'base' or 'tcp', got {parent!r}")
    if meta:
        entry["meta"] = meta
    return entry


def _load_existing(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    if "convention" in raw or "T_cam_base" in raw or "T_base_cam" in raw:
        # A raw Kalib export names no camera, so there is nothing to merge.
        logger.warning(f"{path} is a raw Kalib export; it will be replaced.")
        return {}
    return raw


def _in_base_frame(existing: dict[str, Any], path: pathlib.Path) -> dict[str, Any]:
    """Return *existing* with every base-parented entry expressed in ``base``."""
    declared = existing.get("base_frame")
    cameras = {k: v for k, v in existing.items() if k not in _NON_CAMERA_KEYS}
    if not cameras:
        return {}
    if declared is None:
        raise ValueError(
            f"{path} has camera entries but declares no base_frame, so its "
            'extrinsics cannot be merged safely. Add \'"base_frame": "base"\' '
            'or \'"base_frame": "base_link"\' to it, or move it aside.'
        )
    if declared == BASE_FRAME:
        return cameras
    if declared != "base_link":
        raise ValueError(f"{path} declares unknown base_frame {declared!r}")
    C = base_link_to_base()
    out: dict[str, Any] = {}
    for name, entry in cameras.items():
        entry = dict(entry)
        if entry.get("parent", BASE_FRAME) == BASE_FRAME and "extrinsic" in entry:
            E = np.asarray(entry["extrinsic"], dtype=np.float64).reshape(4, 4)
            entry["extrinsic"] = (C @ E).tolist()
            logger.info(
                f"Rotated the existing {name!r} extrinsic from base_link into base."
            )
        out[name] = entry
    return out


def write_calibration(
    path: str | pathlib.Path,
    camera_name: str,
    entry: dict[str, Any],
    *,
    backup: bool = True,
) -> dict[str, Any]:
    """Merge *entry* for *camera_name* into the calibration file at *path*.

    Returns the file's final contents. When the file exists and *backup* is
    set, the previous version is copied to ``<path>.bak-<timestamp>`` first.
    """
    path = pathlib.Path(path)
    existing = _load_existing(path)
    cameras = _in_base_frame(existing, path) if existing else {}
    cameras[camera_name] = entry
    final: dict[str, Any] = {"base_frame": BASE_FRAME}
    for name in sorted(cameras):
        final[name] = cameras[name]

    if path.exists() and backup:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, bak)
        logger.info(f"Previous calibration kept at {bak}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(final, indent=2) + "\n")
    tmp.replace(path)
    logger.info(f"Wrote {camera_name!r} calibration to {path}")
    return final


__all__ = [
    "BASE_FRAME",
    "TCP_PARENT",
    "base_link_to_base",
    "camera_entry",
    "write_calibration",
]

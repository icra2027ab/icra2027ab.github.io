"""Convert a Kalib hand-eye result into the UR5 env's calibration file.

Kalib writes ``T_cam_base.json`` containing *both* directions plus a
``convention`` string. This repo's ``--env.calibration-path`` wants the
**camera-to-world** pose (world = the UR5e base frame), which is the opposite
of what the file's headline key is named — getting it backwards silently places
every reconstructed object somewhere else entirely, so the conversion is done
here once rather than by hand each time.

    python scripts/ur5_calibration_from_kalib.py \\
        --kalib-path ~/dev/Kalib/dataset/my_ur5e_dataset/Kalib/T_cam_base.json \\
        --camera exterior_image_1_left \\
        --out configs/ur5_camera_calibration.json

Intrinsics are intentionally omitted for RealSense cameras: the SDK reports the
intrinsics of the *aligned colour* stream, which is the frame the published
depth is registered to, and at whatever resolution you actually stream. A
calibration done at a different resolution (Kalib's default is 1280x720) would
be wrong for a 640x480 stream. Pass --include-intrinsic only if you calibrated
at the streaming resolution and want to override the factory values.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Optional

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    kalib_path: str
    """Kalib's ``T_cam_base.json``."""

    out: str
    """Where to write the env calibration JSON."""

    camera: str = "exterior_image_1_left"
    """EnvSpec camera name this extrinsic belongs to."""

    include_intrinsic: bool = False
    """Also copy an intrinsic matrix from the Kalib file, if it has one. Off by
    default — see the module docstring."""

    merge: Optional[str] = None
    """Existing calibration JSON to merge into (for multi-camera cells)."""


def camera_to_base(raw: dict) -> np.ndarray:
    """Extract the camera-to-base 4x4 from a Kalib result.

    Kalib states its convention as ``X_cam = T_cam_base @ X_base``, i.e. the
    key named ``T_cam_base`` maps base coordinates *into* the camera — it is
    the world-to-camera extrinsic. The camera-to-base pose we want is therefore
    ``T_base_cam``, which Kalib also stores. When only one is present, invert.
    """
    convention = str(raw.get("convention", ""))
    if "X_cam = T_cam_base" not in convention:
        raise SystemExit(
            f"Unrecognised Kalib convention {convention!r}. Refusing to guess "
            "which direction the matrix goes — verify it and extend this script."
        )
    if "T_base_cam" in raw:
        return np.asarray(raw["T_base_cam"], dtype=np.float64).reshape(4, 4)
    T_cam_base = np.asarray(raw["T_cam_base"], dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(T_cam_base)


def main(args: Args) -> None:
    raw = json.loads(pathlib.Path(args.kalib_path).read_text())
    E = camera_to_base(raw)

    R = E[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-6
    ):
        raise SystemExit(
            "Extracted extrinsic is not a rigid transform "
            f"(det={np.linalg.det(R):.6f}); the Kalib file looks malformed."
        )

    entry: dict = {"extrinsic": E.tolist()}
    if args.include_intrinsic:
        if "K" not in raw and "intrinsic" not in raw:
            raise SystemExit("--include-intrinsic set but the Kalib file has no K.")
        K = np.asarray(raw.get("K", raw.get("intrinsic")), dtype=np.float64).reshape(
            3, 3
        )
        entry["intrinsic"] = K.tolist()

    out_path = pathlib.Path(args.out)
    calib: dict = {}
    if args.merge:
        calib = json.loads(pathlib.Path(args.merge).read_text())
    calib[args.camera] = entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calib, indent=2) + "\n")

    meta = raw.get("meta", {})
    print(f"camera        : {args.camera}")
    print(f"camera origin in base frame: {np.round(E[:3, 3], 4).tolist()} m")
    print(
        f"intrinsic     : {'from file' if args.include_intrinsic else 'from the device (SDK)'}"
    )
    if meta:
        print(
            f"calibrated at : {meta.get('width')}x{meta.get('height')}, "
            f"reproj err {meta.get('avg_reprojection_err_px', float('nan')):.2f} px"
        )
    print(f"wrote         : {out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))

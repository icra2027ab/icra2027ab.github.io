"""Track objects across a video + reconstruct 3D pointclouds per frame.

Saves an orbiting 3D visualization that advances through tracked timesteps
while the camera circles the scene.

Uses per-frame depth from ``demo_multi_camera.npz`` so the unprojection
follows the moving objects. The camera is static here, but the same code
path works for a moving camera as long as you supply per-frame extrinsics.

    uv run python examples/07_track_and_reconstruct.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from open_perception import (
    CameraInfo,
    OpenPerception,
    PerceptionConfig,
    render_pointcloud_orbit_gif,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    multi = np.load(ASSETS / "demo" / "demo_multi_camera.npz")
    rgbs = multi["rgb_cam0"]
    depths = multi["depth_cam0"]
    K = multi["intrinsics_cam0"]
    Es = multi["extrinsics_cam0"]

    cfg = PerceptionConfig().order(
        ["sam3", "filter.overlap", "filter.erode", "reconstruct"]
    )
    cfg.update("reconstruct", flip_u=True)
    perc = OpenPerception(cfg)

    queries = ["mug", "plate"]
    for t, rgb in enumerate(rgbs):
        cam = CameraInfo(intrinsics=K, extrinsics=Es[t])
        if t == 0:
            perc.detect(queries, rgb, depth=depths[t], camera=cam)
        else:
            perc.track(rgb, depth=depths[t], camera=cam)

    print(f"tracked {len(perc.history)} frames")

    out_path = Path("tmp/07/track_reconstruct_orbit.gif")
    out_path.parent.mkdir(exist_ok=True)
    render_pointcloud_orbit_gif(perc.history, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""RGBD + camera intrinsics/extrinsics → per-detection world-frame point cloud.

Saves an orbiting 3D visualization of the reconstructed point clouds + camera
frustum as a GIF in ``tmp/``.

    uv run python examples/03_reconstruct.py
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
file_name = Path(__file__).stem


def main() -> None:
    frame = np.load(ASSETS / "demo" / "demo_rgbd_frame.npz")
    cam = CameraInfo(intrinsics=frame["intrinsics"], extrinsics=frame["extrinsics"])

    cfg = PerceptionConfig().order(
        [
            "sam3",
            "filter.overlap",
            "filter.size",
            "disambiguate.clip",
            "assign",
            "filter.erode",
            "reconstruct",
        ]
    )
    perc = OpenPerception(cfg)

    out = perc.detect(["mug", "plate"], frame["rgb"], depth=frame["depth"], camera=cam)

    for query, dets in out.detections.items():
        for i, det in enumerate(dets):
            pts = det.pointcloud
            assert pts is not None
            print(f"{query}[{i}]: {len(pts)} pts  centroid={pts.mean(axis=0)}")

    out_path = Path("tmp/03/reconstruct_orbit.gif")
    out_path.parent.mkdir(exist_ok=True)
    render_pointcloud_orbit_gif(perc.history, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

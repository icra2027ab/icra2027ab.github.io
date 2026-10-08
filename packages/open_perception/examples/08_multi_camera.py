"""Track + reconstruct from two cameras simultaneously, single instance.

A single :class:`OpenPerception` consumes both camera streams in one
``detect()``/``track()`` call — multi-view input is native, and the
:class:`TrackHistory` accumulator on ``perc.history`` collects everything
visualization helpers need.

    uv run python examples/08_multi_camera.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from open_perception import (
    CameraInfo,
    OpenPerception,
    PerceptionConfig,
    render_pointcloud_orbit_gif,
    render_segmentation_video,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    data = np.load(ASSETS / "demo" / "demo_multi_camera.npz")
    K0, K1 = data["intrinsics_cam0"], data["intrinsics_cam1"]
    E0_seq, E1_seq = data["extrinsics_cam0"], data["extrinsics_cam1"]

    cfg = PerceptionConfig().order(
        ["sam3", "filter.overlap", "filter.erode", "reconstruct"]
    )
    # demo_multi_camera.npz comes from robosuite/LIBERO, where rgb/depth are
    # 180°-rotated relative to the intrinsics — flip_u undoes the u-axis
    # mirror inside the unprojection so world points line up with extrinsics.
    cfg.update("reconstruct", flip_u=True)
    perc = OpenPerception(cfg)

    queries = ["mug", "plate"]
    n_frames = data["rgb_cam0"].shape[0]
    for t in range(n_frames):
        rgbs = [data["rgb_cam0"][t], data["rgb_cam1"][t]]
        depths = [data["depth_cam0"][t], data["depth_cam1"][t]]
        cams = [
            CameraInfo(intrinsics=K0, extrinsics=E0_seq[t], name="cam0_agentview"),
            CameraInfo(intrinsics=K1, extrinsics=E1_seq[t], name="cam1_eye_in_hand"),
        ]
        if t == 0:
            perc.detect(queries, rgbs, depth=depths, camera=cams)
        else:
            perc.track(rgbs, depth=depths, camera=cams)

    print(f"tracked {len(perc.history)} frames across two cameras")

    out_dir = Path("tmp") / "08"
    out_dir.mkdir(exist_ok=True)

    video_path = render_segmentation_video(
        perc.history, out_dir / "multi_camera_track.mp4"
    )
    print(f"wrote {video_path}")

    gif_path = render_pointcloud_orbit_gif(
        perc.history, out_dir / "multi_camera_orbit.gif"
    )
    print(f"wrote {gif_path}")


if __name__ == "__main__":
    main()

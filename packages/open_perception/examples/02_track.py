"""Track: detect on frame 0, propagate via SAM3 video.

uv run python examples/02_track.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import OpenPerception, render_segmentation_video

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    video_path = ASSETS / "videos" / "sample.mp4"
    frames = iio.imread(video_path)

    perc = OpenPerception()

    queries = ["towel", "can"]
    for i, frame in enumerate(frames):
        if i == 0:
            perc.detect(queries, frame)
        else:
            perc.track(frame)

    # Equivalent one-liner using the streaming helper:
    # for out in perc.stream(queries, frames):
    #    pass

    out_path = Path("tmp/02/track.mp4")
    render_segmentation_video(
        perc.history, out_path, show_id=True, show_fps=True, show_label=True
    )
    print(f"wrote {out_path}  ({len(perc.history)} frames)")


if __name__ == "__main__":
    main()

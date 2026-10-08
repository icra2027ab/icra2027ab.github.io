"""Streaming tracking with the SAM2 backend for faster per-frame propagation.

SAM2 cannot segment from text, so frame 0 still runs SAM3 for text-prompted
detection. Every subsequent frame is propagated by SAM2's lighter video
model, which is the common case for long streams.

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/sam2_tracker.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from open_perception import OpenPerception, PerceptionConfig, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


def main():
    data = np.load(ASSETS / "demo" / "demo_frames.npz")
    frames = data["rgb"]  # (T, H, W, 3) uint8
    print(f"loaded {len(frames)} frames of shape {frames.shape[1:]}")

    cfg = PerceptionConfig(tracker_backend="sam2")
    # Override `cfg.sam2_model` to pick a larger SAM2 variant if you need
    # higher mask quality — default is `facebook/sam2-hiera-tiny`.
    print(f"tracker backend: {cfg.tracker_backend}  ({cfg.sam2_model})")
    perc = OpenPerception(cfg)
    queries = ["white mug", "yellow and white mug", "plate"]

    overlays: list[np.ndarray] = []
    detect_ms = 0.0
    track_ms_per_frame: list[float] = []
    for i, out in enumerate(perc.stream(queries, frames)):
        n_total = sum(len(lst) for lst in out.detections.values())
        # `stream()` calls `detect()` for frame 0 and `track()` for the rest.
        frame_ms = sum(out.stats.timings_ms.values())
        if i == 0:
            detect_ms = frame_ms
        else:
            track_ms_per_frame.append(frame_ms)
        if i == 0 or (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"frame {i:3d}: {n_total} detections total  ({frame_ms:.1f} ms)")
        overlays.append(
            overlay_detections(frames[i], out, label=f"SAM2 tracked: {n_total}")
        )

    if track_ms_per_frame:
        mean_track = sum(track_ms_per_frame) / len(track_ms_per_frame)
        print(
            f"\ndetect (frame 0): {detect_ms:.1f} ms  |  "
            f"mean SAM2 track / frame: {mean_track:.1f} ms  over {len(track_ms_per_frame)} frames"
        )

    out_path = Path("tmp/open_perception_sam2_track.mp4")
    out_path.parent.mkdir(exist_ok=True)
    iio.imwrite(out_path, np.stack(overlays), fps=15, codec="libx264")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    _start = time.perf_counter()
    main()
    print(f"total: {time.perf_counter() - _start:.1f} s")

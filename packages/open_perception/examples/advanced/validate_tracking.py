"""Validate SAM3 video tracking on the extracted LIBERO demo.

Runs `OpenPerception.stream()` over the demo video and reports mean IoU
against the frame-0 GT masks (available for frame 0 only, so the first-frame
detection alone is checked) plus the tracker's per-frame stability.

Writes a side-by-side video:
    tmp/open_perception_tracking_validation.mp4

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/validate_tracking.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from open_perception import OpenPerception, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

# LIBERO object names -> prompts the user would actually type.
QUERY_BINDING = {
    "white mug": "porcelain_mug_1",
    "yellow and white mug": "white_yellow_mug_1",
    "red mug": "red_coffee_mug_1",
    "left plate": "plate_1",
    "right plate": "plate_2",
}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    data = np.load(ASSETS / "demo" / "demo_frames.npz")
    frames = data["rgb"]
    gt = np.load(ASSETS / "demo" / "demo_gt_masks_frame0.npz")

    print(f"video: {len(frames)} frames, resolution {frames.shape[1:]}")

    perc = OpenPerception()

    queries = list(QUERY_BINDING.keys())
    t0 = time.perf_counter()
    outputs = list(perc.stream(queries, frames))
    wall_s = time.perf_counter() - t0
    print(
        f"stream() completed in {wall_s:.2f}s  ({wall_s / len(frames) * 1000:.1f} ms/frame)"
    )

    # Frame-0 IoU vs GT — this measures the *detect* pipeline.
    print("\nFrame-0 IoU vs GT:")
    for query, gt_name in QUERY_BINDING.items():
        dets = outputs[0].detections.get(query, [])
        if gt_name not in gt.files or not dets:
            print(f"  {query:25s} — no GT or no detection")
            continue
        gt_mask = gt[gt_name]
        best = max(_iou(d.mask, gt_mask) for d in dets)
        print(f"  {query:25s} iou@frame0 = {best:.3f}  (n={len(dets)})")

    # Tracker stability — how often does each query drift into nothing?
    print("\nTracker stability (fraction of frames with at least 1 detection):")
    for query in queries:
        hits = sum(1 for o in outputs if o.detections.get(query))
        print(f"  {query:25s} {hits}/{len(outputs)}  ({hits / len(outputs):.0%})")

    # Colors ordered to match QUERY_BINDING: white mug, yellow/white mug,
    # red mug, left plate, right plate.
    palette = np.array(
        [
            [230, 60, 60],
            [230, 200, 60],
            [200, 60, 100],
            [60, 160, 230],
            [60, 200, 90],
        ],
        dtype=np.float32,
    )
    out_path = Path("tmp/open_perception_tracking_validation.mp4")
    out_path.parent.mkdir(exist_ok=True)
    iio.imwrite(
        out_path,
        np.stack(
            [
                overlay_detections(f, o, colors=palette)
                for f, o in zip(frames, outputs, strict=True)
            ]
        ),
        fps=15,
        codec="libx264",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

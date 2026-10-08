"""SAM2 as a drop-in segmentation stage (replacing SAM3).

SAM2 has no text-conditioning, so it segments either un-conditionally
("propose every mask in the image") or conditioned on a bounding box /
point prompt. The ``sam2`` stage exposes all three flavours via its
``mode`` parameter:

  - ``auto``  : every mask SAM2 proposes — bucketed under a single literal
                 key (default ``"object"``). Pair with disambiguate + assign
                 to attribute candidates back to the user's queries.
  - ``bbox``  : one mask per ``(x1, y1, x2, y2)`` prompt per query.
  - ``point`` : one mask per ``{"points": [...], "labels": [...]}`` group
                 per query.

Each section below builds a fresh ``OpenPerception`` so the configs stay
isolated. Run from the package root (``packages/open_perception``):

    uv run python examples/11_sam2_segment.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import OpenPerception, PerceptionConfig, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent / "assets"
QUERIES = ["mug", "plate"]


def _save(rgb, out, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, overlay_detections(rgb, out, show_label=True, show_score=True))
    print(f"  wrote {path}")


def run_auto(rgb) -> None:
    print("\n=== sam2 mode='auto' (propose all masks) ===")
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam2", "filter.overlap"])
    perc = OpenPerception(cfg)
    out = perc.detect(QUERIES, rgb)
    for q, dets in out.detections.items():
        print(f"  {q}: {len(dets)} mask(s)")
    _save(rgb, out, Path("tmp/11/sam2_auto.png"))


def run_bbox(rgb) -> None:
    print("\n=== sam2 mode='bbox' (one mask per box) ===")
    H, W = rgb.shape[:2]
    # Hand-picked boxes for `assets/demo/frame_00.png` (256x256). Replace
    # with detector outputs in real use; the stage cares only about the
    # final dict, not where it came from.
    bbox_prompts = {
        "mug": [(70, 120, 115, 175), (155, 120, 205, 175)],
        "plate": [(0, 145, 70, 200), (190, 145, 256, 200)],
    }
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam2"])
    cfg.update("sam2", mode="bbox", prompts=bbox_prompts)
    perc = OpenPerception(cfg)
    out = perc.detect(QUERIES, rgb)
    for q, dets in out.detections.items():
        print(f"  {q}: {len(dets)} mask(s)")
        for d in dets:
            print(f"    score={d.score:.3f} bbox={d.bbox.tolist()}")
    _save(rgb, out, Path("tmp/11/sam2_bbox.png"))


def run_point(rgb) -> None:
    print("\n=== sam2 mode='point' (one mask per point group) ===")
    point_prompts = {
        # One entry per object. Multi-point prompts go in the same list,
        # with `labels` flagging foreground (1) / background (0).
        "mug": [
            {"points": [(95, 150)], "labels": [1]},
            {"points": [(180, 150)], "labels": [1]},
        ],
        "plate": [
            {"points": [(30, 175)], "labels": [1]},
            {"points": [(225, 175)], "labels": [1]},
        ],
    }
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["sam2"])
    cfg.update("sam2", mode="point", prompts=point_prompts)
    perc = OpenPerception(cfg)
    out = perc.detect(QUERIES, rgb)
    for q, dets in out.detections.items():
        print(f"  {q}: {len(dets)} mask(s)")
        for d in dets:
            print(f"    score={d.score:.3f} bbox={d.bbox.tolist()}")
    _save(rgb, out, Path("tmp/11/sam2_point.png"))


def main() -> None:
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")
    print(f"loaded frame: {rgb.shape}")
    run_auto(rgb)
    run_bbox(rgb)
    run_point(rgb)


if __name__ == "__main__":
    main()

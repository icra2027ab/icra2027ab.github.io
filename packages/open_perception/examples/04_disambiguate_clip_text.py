"""Disambiguate similar instances with CLIP text embeddings.

Three mugs share the scene. Each query is matched to the best candidate
crop via CLIP text-image similarity — the default behaviour when no
reference bank is registered.

    uv run python examples/04_disambiguate_clip_text.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import InferenceOptions, OpenPerception, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent / "assets"
QUERIES = ["white mug", "yellow and white mug", "red mug"]


def main() -> None:
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")

    perc = OpenPerception()
    out = perc.detect(QUERIES, rgb, options=InferenceOptions(match_threshold=0.2))

    for q, dets in out.detections.items():
        print(f"{q}: scores={[round(d.score, 3) for d in dets]}")

    out_path = Path("tmp/04/disambiguate_clip_text.png")
    out_path.parent.mkdir(exist_ok=True)
    iio.imwrite(
        out_path, overlay_detections(rgb, out, show_label=True, show_score=True)
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

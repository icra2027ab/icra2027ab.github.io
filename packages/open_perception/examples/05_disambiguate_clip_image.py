"""Disambiguate similar instances with CLIP image embeddings.

Same three mugs as example 04 — this time each query is backed by a
bundled reference crop in ``assets/refs/``. CLIP scores candidate crops
against those reference images instead of the query text.

    uv run python examples/05_disambiguate_clip_image.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import InferenceOptions, OpenPerception, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent / "assets"
REFS = ASSETS / "demo" / "refs"
QUERIES = ["white mug", "yellow and white mug", "red mug"]


def main() -> None:
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")

    reference = {q: iio.imread(REFS / f"{q.replace(' ', '_')}.png") for q in QUERIES}
    perc = OpenPerception(reference=reference)

    out = perc.detect(QUERIES, rgb, options=InferenceOptions(match_threshold=0.2))
    for q, dets in out.detections.items():
        print(f"{q}: scores={[round(d.score, 3) for d in dets]}")

    out_path = Path("tmp/05/disambiguate_clip_image.png")
    out_path.parent.mkdir(exist_ok=True)
    iio.imwrite(
        out_path, overlay_detections(rgb, out, show_label=True, show_score=True)
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

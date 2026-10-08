"""Advanced: combine CLIP and DINO disambiguators, tune the assigner.

Shows three pipeline customizations:
  1. Enable DINO disambiguation in addition to CLIP (both scores combined).
  2. Tighten the overlap-filter IoU threshold via stage params.
  3. Switch the assigner to top-1 Hungarian for 1-mask-per-query semantics.

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/reorder_stages.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import InferenceOptions, OpenPerception, PerceptionConfig

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


def main():
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")

    cfg = PerceptionConfig()
    # DINO runs before CLIP; both scores get fused in the assigner.
    cfg.order(
        [
            "sam3",
            "filter.overlap",
            "filter.size",
            "disambiguate.dino",
            "disambiguate.clip",
            "assign",
        ]
    )
    # 1. DINO scored without a text fallback.
    cfg.update("disambiguate.dino", fallback_to_text=False)
    # 2. Looser overlap filter (drop only very redundant masks).
    cfg.update("filter.overlap", iou_threshold=0.85)
    # 3. Force 1-mask-per-query assignment with per-source weighting.
    cfg.update("assign", top_k=1, weights={"clip": 1.0, "dino": 0.7})

    perc = OpenPerception(cfg)
    print(cfg.summary)

    out = perc.detect(
        ["white mug", "yellow and white mug", "plate"],
        rgb,
        options=InferenceOptions(match_threshold=0.2),
    )
    for q, dets in out.detections.items():
        print(f"{q}: {[round(d.score, 3) for d in dets]}")

    print(f"\nmodel calls: {out.stats.model_calls}")
    print(f"candidates per view: {out.stats.n_candidates_per_view}")


if __name__ == "__main__":
    main()

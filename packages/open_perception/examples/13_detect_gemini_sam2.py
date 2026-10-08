"""Tiptop-style detection: Gemini proposes boxes, SAM2 segments them.

The ``detect.gemini`` stage asks a Gemini model to find the objects in the
scene and emit one bounding box per object. Those boxes land in
``ctx.box_prompts`` and the ``sam2`` stage (``mode='bbox'``) turns each into a
precise mask — no hand-picked boxes, no text-conditioned segmenter.

Two flavours:

  - open_vocab : Gemini enumerates the objects itself (optionally biased by a
                  ``task`` instruction). The proposed labels become the
                  detection queries, so call ``detect([], rgb)``.
  - grounded   : Gemini locates only the queries you pass in.

Requires the optional Gemini dependency and an API key:

    uv pip install 'open-perception[gemini]'
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
    uv run python examples/13_detect_gemini_sam2.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import OpenPerception, PerceptionConfig, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _save(rgb, out, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, overlay_detections(rgb, out, show_label=True, show_score=True))
    print(f"  wrote {path}")


def run_open_vocab(rgb) -> None:
    print("\n=== detect.gemini (open_vocab) -> sam2 bbox ===")
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["detect.gemini", "sam2", "filter.overlap"])
    cfg.update("detect.gemini", stage="open_vocab", task="pick up the mug")
    cfg.update("sam2", mode="bbox")
    perc = OpenPerception(cfg)
    # Empty queries — Gemini decides what's in the scene and the proposed
    # labels become the queries.
    out = perc.detect([], rgb)
    for q, dets in out.detections.items():
        print(f"  {q}: {len(dets)} mask(s)")
    _save(rgb, out, Path("tmp/13/gemini_open_vocab.png"))


def run_grounded(rgb) -> None:
    print("\n=== detect.gemini (grounded) -> sam2 bbox ===")
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["detect.gemini", "sam2", "filter.overlap"])
    cfg.update("detect.gemini", stage="grounded")
    cfg.update("sam2", mode="bbox")
    perc = OpenPerception(cfg)
    out = perc.detect(["mug", "plate"], rgb)
    for q, dets in out.detections.items():
        print(f"  {q}: {len(dets)} mask(s)")
    _save(rgb, out, Path("tmp/13/gemini_grounded.png"))


def main() -> None:
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")
    print(f"loaded frame: {rgb.shape}")
    run_open_vocab(rgb)
    run_grounded(rgb)


if __name__ == "__main__":
    main()

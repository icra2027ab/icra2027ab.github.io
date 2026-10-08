"""Detect and output the raw SAM3 masks — no disambiguation, no assignment.

Pure config-only — each resolved prompt becomes its own key in
``out.detections``. ``<object_names>`` expands to the query list, so e.g.
``prompts=["object", "<object_names>", "boat"]`` with queries
``["white mug", "yellow and white mug", "plate"]`` sends SAM3 five
prompts (``"object"``, the three queries, and ``"boat"``) and yields one
detection list per prompt. Literal prompts (``"object"``, ``"boat"``)
give the union-style negatives/extras alongside the per-query masks.

``filter.overlap`` in subtract mode resolves any pixel-level overlap
between the resulting masks.

Also renders the pipeline history as a GIF so you can see how each stage
changes the masks.

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/raw_sam3_masks.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import (
    OpenPerception,
    PerceptionConfig,
    overlay_detections,
    render_history_gif,
)

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
QUERIES = ["yellow and white mug", "plate"]


def main():
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")

    cfg = PerceptionConfig()
    cfg.order(["sam3", "filter.overlap"])

    perc = OpenPerception(cfg)
    # History capture is on by default; `out.history` holds one
    # PipelineSnapshot per stage, ready for `render_history_gif`.
    out = perc.detect(QUERIES, rgb)

    for query, dets in out.detections.items():
        print(f"{query}: {len(dets)} raw mask(s)")
        for i, det in enumerate(dets):
            print(
                f"  [{i}] score={det.score:.3f}  area={int(det.mask.sum())}px  bbox={det.bbox}"
            )

    out_dir = Path("tmp")
    out_dir.mkdir(exist_ok=True)

    vis = overlay_detections(rgb, out, show_id=False, show_label=True, show_score=True)
    img_path = out_dir / "open_perception_raw_sam3.png"
    iio.imwrite(img_path, vis)
    print(f"\nwrote {img_path}")

    gif_path = render_history_gif(
        rgb,
        out,
        out_dir / "open_perception_raw_sam3.gif",
        show_id=False,
        show_label=True,
        show_score=True,
    )
    print(f"wrote {gif_path} ({len(out.history)} stage snapshot(s))")

    print(f"pipeline stages run: {out.stats.stages_run}")
    print(f"timings (ms): {out.stats.timings_ms}")


if __name__ == "__main__":
    main()

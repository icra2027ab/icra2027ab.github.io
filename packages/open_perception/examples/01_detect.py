"""Open-vocabulary detection on a single image.

uv run python examples/01_detect.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from open_perception import OpenPerception, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")

    perc = OpenPerception()
    out = perc.detect(["mug", "plate"], rgb)

    for query, dets in out.detections.items():
        print(f"{query}: {len(dets)} instance(s)")

    out_path = Path("tmp/01/detect.png")
    out_path.parent.mkdir(exist_ok=True)
    iio.imwrite(out_path, overlay_detections(rgb, out, show_label=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

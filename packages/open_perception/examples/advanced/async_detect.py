"""Non-blocking detection: submit once, keep doing robot work, fetch when ready.

Shows the `AsyncOpenPerception` pattern for one-shot detection:

  1. `submit_detect(...)` returns immediately with a seq id
  2. A simulated control loop keeps ticking at ~100 Hz
  3. `latest()` is polled each tick until the submitted seq lands
  4. Meanwhile SAM3 + filters + disambiguation run on a worker thread

Contrast with `examples/01_detect.py`, where `perc.detect(...)` blocks the
caller for the full pipeline duration (hundreds of ms to seconds on GPU).

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/async_detect.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v3 as iio
from open_perception import AsyncOpenPerception, overlay_detections

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


def _simulated_control_tick() -> None:
    # Stand-in for whatever the robot's main loop does — IK, filtering, etc.
    # The important bit is that *this* thread keeps running while perception
    # works on its own.
    time.sleep(0.01)


def main() -> None:
    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")
    queries = ["white mug", "yellow and white mug", "plate"]

    with AsyncOpenPerception() as perc:
        # Fire off detection and immediately return control.
        t0 = time.perf_counter()
        seq = perc.submit_detect(queries, rgb)
        print(f"submitted detect (seq={seq}); main thread is now free")

        # Poll for the result while pretending to run a 100 Hz control loop.
        ticks = 0
        while True:
            got = perc.latest()
            if got is not None and got[0] == seq:
                break
            _simulated_control_tick()
            ticks += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        seq_out, out = got
        print(
            f"detection ready after {elapsed_ms:.0f} ms, "
            f"{ticks} control ticks ran in parallel "
            f"(worker busy={perc.busy}, processed={perc.n_processed})"
        )

        # Alternative to polling: block the main thread explicitly until the
        # next result arrives. Useful when you *do* want to wait.
        print(
            "\nsanity check — wait_next(after_seq=seq) immediately returns "
            "because seq already landed:"
        )
        again = perc.wait_next(after_seq=seq - 1, timeout=0.1)
        print(f"  {'hit' if again else 'miss'}: seq={again[0] if again else None}")

        for query, dets in out.detections.items():
            print(f"{query}: {len(dets)} instance(s)")
            for i, det in enumerate(dets):
                print(
                    f"  [{i}] score={det.score:.3f}  "
                    f"area={int(det.mask.sum())}px  bbox={det.bbox}"
                )

        out_dir = Path("tmp")
        out_dir.mkdir(exist_ok=True)
        vis = overlay_detections(
            rgb, out, show_id=True, show_label=True, show_score=True
        )
        img_path = out_dir / "open_perception_async_detect.png"
        iio.imwrite(img_path, vis)
        print(f"\nwrote {img_path}")


if __name__ == "__main__":
    main()

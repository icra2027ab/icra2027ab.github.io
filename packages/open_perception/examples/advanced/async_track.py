"""Non-blocking streaming tracking with drop-oldest backpressure.

The typical robotics pattern:

  - frame 0 runs detect (and seeds the SAM3 video tracker)
  - every subsequent frame gets submitted to track() at the control-loop
    rate (here ~60 Hz). The worker processes the *latest* submitted frame;
    frames submitted while it's busy are overwritten.

Contrast with `examples/02_track.py`, where `perc.stream(...)` drives the
frame cadence itself and blocks the caller on every frame.

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/async_track.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from open_perception import AsyncOpenPerception, overlay_detections, render_history_gif

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

# Simulated robot-control loop rate. Set higher than SAM3's per-frame
# throughput to exercise drop-oldest — otherwise the worker stays caught
# up and nothing ever gets dropped.
CONTROL_HZ = 15


def main() -> None:
    data = np.load(ASSETS / "demo" / "demo_frames.npz")
    frames = data["rgb"]  # (T, H, W, 3) uint8
    queries = ["white mug", "yellow and white mug", "plate"]
    print(f"loaded {len(frames)} frames of shape {frames.shape[1:]}")

    with AsyncOpenPerception() as perc:
        # Frame 0: detect, and wait once so the tracker session is seeded
        # before we start firing track submissions.
        t_detect = time.perf_counter()
        perc.submit_detect(queries, frames[0])
        got = perc.wait_next(timeout=30.0)
        assert got is not None, "detect never completed"
        seq0, first_out = got
        detect_ms = (time.perf_counter() - t_detect) * 1000
        print(
            f"frame   0: detect done in {detect_ms:.0f} ms — "
            f"{sum(len(lst) for lst in first_out.detections.values())} detections"
        )

        # Persist the pipeline history for the seed detect: initial SAM3
        # masks followed by each post-processing stage.
        history_gif_path = Path("tmp/open_perception_async_track_detect_history.gif")
        history_gif_path.parent.mkdir(exist_ok=True)
        render_history_gif(
            frames[0],
            first_out,
            history_gif_path,
            show_id=False,
            show_label=True,
            show_score=True,
        )
        print(f"wrote {history_gif_path} ({len(first_out.history)} stage snapshot(s))")

        # Render frame 0 from the detect output; subsequent frames reuse the
        # most recent finished track output (which may lag the current frame
        # by one SAM3 step under high load — that's the point of async).
        overlays: list[np.ndarray] = [
            overlay_detections(frames[0], first_out, label="frame 0 (detect)")
        ]
        last_shown_seq = seq0
        last_shown_out = first_out

        n_submitted = 0
        tick_period = 1.0 / CONTROL_HZ
        t_stream = time.perf_counter()
        for i, frame in enumerate(frames[1:], start=1):
            perc.submit_track(frame)
            n_submitted += 1

            # Grab whatever the worker has finished most recently.
            got = perc.latest()
            if got is not None and got[0] != last_shown_seq:
                last_shown_seq, last_shown_out = got
            n_det = sum(len(lst) for lst in last_shown_out.detections.values())
            overlays.append(
                overlay_detections(
                    frame,
                    last_shown_out,
                    label=f"frame {i:3d}  latest_seq={last_shown_seq}  "
                    f"n={n_det}  dropped={perc.n_dropped}",
                )
            )

            # Pace the main thread at CONTROL_HZ (robot loop stand-in).
            target = t_stream + (i * tick_period)
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

        # Drain any in-flight work so the final video reflects the last frame.
        tail = perc.wait_next(after_seq=last_shown_seq, timeout=5.0)
        if tail is not None:
            _, last_shown_out = tail
            overlays[-1] = overlay_detections(
                frames[-1], last_shown_out, label="final (drained)"
            )

        total_s = time.perf_counter() - t_stream
        print(
            f"\nsubmitted {n_submitted} track frames over {total_s:.2f}s "
            f"(~{n_submitted / max(total_s, 1e-6):.0f} Hz submit rate)"
        )
        print(
            f"worker processed {perc.n_processed} total submissions "
            f"(detect + track); dropped {perc.n_dropped} stale track submissions"
        )
        if perc.n_dropped == 0:
            print(
                "note: zero drops means SAM3 kept up with the control loop — "
                "try raising CONTROL_HZ to see drop-oldest in action"
            )

        out_path = Path("tmp/open_perception_async_track.mp4")
        out_path.parent.mkdir(exist_ok=True)
        iio.imwrite(out_path, np.stack(overlays), fps=CONTROL_HZ, codec="libx264")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Explicit two-stage pipeline: SAM3 text-prompted masks → SAM3 tracker.

Shows what the high-level `OpenPerception.detect()` + `track()` wires up
internally: frame 0 runs `Sam3Model` (text-prompted, `Sam3Wrapper.segment`),
and every subsequent frame is propagated by `Sam3TrackerVideoModel` — the
465M-param class-agnostic tracker that takes masks in and masks out, with
*no* text conditioning. The big 859M `Sam3VideoModel` that bundles detector
+ tracker + text encoder is never loaded.

Use this pattern when your initial masks come from somewhere else (an HDF5
dataset, a human annotator, a different detector) and you want the SAM3
tracker's memory-backed propagation on top.

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/decouple_detect_track.py
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from open_perception import overlay_masks
from open_perception.models.sam3 import Sam3Wrapper
from open_perception.tracking.sam3_tracker import Sam3VideoTracker

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
QUERIES = ["white mug", "yellow and white mug", "plate"]


def _cuda_sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(ASSETS / "demo" / "demo_frames.npz")
    frames = data["rgb"]  # (T, H, W, 3) uint8
    print(f"loaded {len(frames)} frames of shape {frames.shape[1:]}  on {device}")

    # --- Stage 1: text-prompted SAM3 detection on frame 0 only. ---
    sam3 = Sam3Wrapper("facebook/sam3", device=device)
    prompts = sam3.resolve_prompts(["<object_names>"], QUERIES)
    print(f"\n[detect] running Sam3Model on frame 0 with prompts: {prompts}")
    _cuda_sync(torch)
    t0 = time.perf_counter()
    masks, scores = sam3.segment(
        frames[0], prompts, score_threshold=0.15, mask_threshold=0.4
    )
    _cuda_sync(torch)
    detect_ms = (time.perf_counter() - t0) * 1000.0

    # Keep the top mask per query — simplest possible assignment, no CLIP.
    # Replace with OpenPerception's disambiguation stages for anything serious.
    if len(masks) == 0:
        print("[detect] no masks returned; abort")
        return
    top_k = min(len(QUERIES), len(masks))
    seed_masks = {i + 1: masks[i].astype(bool) for i in range(top_k)}
    seed_queries = {i + 1: QUERIES[i] for i in range(top_k)}
    print(
        f"[detect] {len(masks)} masks returned in {detect_ms:.1f} ms; "
        f"seeding tracker with top {top_k}: "
        f"{[(oid, seed_queries[oid], float(scores[oid - 1])) for oid in seed_masks]}"
    )

    # --- Stage 2: hand those masks to Sam3VideoTracker (no text, no detector). ---
    tracker = Sam3VideoTracker("facebook/sam3", device=device)
    print(
        f"[track] loaded Sam3TrackerVideoModel: "
        f"{sum(p.numel() for p in tracker.model.parameters()) / 1e6:.0f}M params, "
        f"dtype={tracker.dtype}"
    )

    _cuda_sync(torch)
    t0 = time.perf_counter()
    refined = tracker.initialize(frames[0], seed_masks)
    _cuda_sync(torch)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[track] initialize(): {init_ms:.1f} ms")

    # Warmup step, not counted.
    _cuda_sync(torch)
    tracker.step(frames[1])
    _cuda_sync(torch)

    def _label(masks: dict[int, np.ndarray]) -> str:
        return " | ".join(
            f"{oid}:{seed_queries.get(oid, '?')}" for oid in sorted(masks)
        )

    def _render(rgb: np.ndarray, masks: dict[int, np.ndarray]) -> np.ndarray:
        return overlay_masks(rgb, dict(sorted(masks.items())), label=_label(masks))

    # Steady-state timing.
    per_frame_ms: list[float] = []
    latest_masks = refined
    overlays: list[np.ndarray] = [_render(frames[0], refined)]
    # Frame 1 was the warmup step — repeat it so the overlay covers all frames.
    overlays.append(_render(frames[1], tracker.step(frames[1])))
    for i in range(2, len(frames)):
        _cuda_sync(torch)
        t0 = time.perf_counter()
        latest_masks = tracker.step(frames[i])
        _cuda_sync(torch)
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        overlays.append(_render(frames[i], latest_masks))

    mean_ms = sum(per_frame_ms) / len(per_frame_ms)
    print(
        f"\n[track] steady-state over {len(per_frame_ms)} frames: "
        f"mean={mean_ms:.1f} ms  →  {1000.0 / mean_ms:.1f} fps"
    )
    print(
        "(tracker runs Sam3TrackerVideoModel only — no text conditioning, "
        "no Sam3Model calls after frame 0)"
    )

    out_path = Path("tmp/open_perception_sam3_decoupled.mp4")
    out_path.parent.mkdir(exist_ok=True)
    iio.imwrite(out_path, np.stack(overlays), fps=15, codec="libx264")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

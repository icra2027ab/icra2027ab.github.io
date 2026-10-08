"""Benchmark SAM2 vs SAM3 (bundled) vs SAM3 decoupled mask-tracking.

Runs the three trackers over `assets/demo_frames.npz`, excludes the frame-0
`detect()` cost (SAM3-only open-vocabulary detection) and the first `track()`
call as warmup, and reports per-frame tracking latency for each backend.

Also writes a side-by-side overlay video so the three outputs can be compared
visually.

Backends:
    - ``sam3``            : `OpenPerception` with SAM3 video model (detector +
                            tracker bundled, text-conditioned every frame).
    - ``sam2``            : `OpenPerception` with SAM2 tracker.
    - ``sam3_decoupled``  : explicit two-stage pipeline from example 09 —
                            SAM3 text-prompted masks on frame 0 then the
                            class-agnostic SAM3 tracker propagates.

Run from the package root (``packages/open_perception``):

    uv run python examples/advanced/benchmark_sam2_vs_sam3.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from open_perception import OpenPerception, PerceptionConfig, overlay_masks
from open_perception.models.sam3 import Sam3Wrapper
from open_perception.tracking.sam3_tracker import Sam3VideoTracker

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
QUERIES = ["white mug", "yellow and white mug", "plate"]


def _cuda_sync() -> None:
    """No-op on CPU; forces GPU work to finish before timing on CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _overlay(
    rgb: np.ndarray, masks_by_query: dict[str, np.ndarray], title: str
) -> np.ndarray:
    """Blend masks onto ``rgb`` with colors keyed to QUERIES position.

    Passing masks in `QUERIES` order (with ``None`` for missing entries) keeps
    color assignments consistent across backends in the side-by-side video.
    """
    ordered = [masks_by_query.get(q) for q in QUERIES]
    return overlay_masks(rgb, ordered, label=title)


def _masks_from_output(output) -> dict[str, np.ndarray]:
    """Take the top detection per query from a `PerceptionOutput`."""
    out: dict[str, np.ndarray] = {}
    for query, dets in output.detections.items():
        if dets:
            out[query] = dets[0].mask.astype(bool)
    return out


def _masks_from_tracker(
    tracker_masks: dict[int, np.ndarray], queries_by_obj_id: dict[int, str]
) -> dict[str, np.ndarray]:
    return {
        queries_by_obj_id[oid]: mask.astype(bool)
        for oid, mask in tracker_masks.items()
        if oid in queries_by_obj_id
    }


def benchmark_openperception(
    backend: str, frames: np.ndarray
) -> tuple[dict[str, float], list[np.ndarray]]:
    cfg = PerceptionConfig(tracker_backend=backend)
    model_id = cfg.sam2_model if backend == "sam2" else cfg.sam3_model
    title = f"{backend.upper()} ({model_id})"
    print(f"\n=== {title} ===")
    perc = OpenPerception(cfg)

    # Frame 0: full detect + tracker session init. Excluded from timing.
    _cuda_sync()
    t0 = time.perf_counter()
    out0 = perc.detect(QUERIES, frames[0])
    _cuda_sync()
    detect_ms = (time.perf_counter() - t0) * 1000.0
    overlays = [_overlay(frames[0], _masks_from_output(out0), title)]

    # Frame 1: first track() call — includes CUDA kernel / graph warmup on GPU.
    # Report it separately so it doesn't skew the steady-state numbers.
    _cuda_sync()
    t0 = time.perf_counter()
    out1 = perc.track(frames[1])
    _cuda_sync()
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    overlays.append(_overlay(frames[1], _masks_from_output(out1), title))

    per_frame_ms: list[float] = []
    for i in range(2, len(frames)):
        _cuda_sync()
        t0 = time.perf_counter()
        outi = perc.track(frames[i])
        _cuda_sync()
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        overlays.append(_overlay(frames[i], _masks_from_output(outi), title))

    stats = _summarize(backend, detect_ms, warmup_ms, per_frame_ms)
    return stats, overlays


def benchmark_sam3_decoupled(
    frames: np.ndarray,
) -> tuple[dict[str, float], list[np.ndarray]]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    title = "SAM3 decoupled (detect→track)"
    print(f"\n=== {title} ===")

    sam3 = Sam3Wrapper("facebook/sam3", device=device)
    prompts = sam3.resolve_prompts(["<object_names>"], QUERIES)

    # Frame 0: text-prompted SAM3 detection (excluded from timing).
    _cuda_sync()
    t0 = time.perf_counter()
    masks, scores = sam3.segment(
        frames[0], prompts, score_threshold=0.15, mask_threshold=0.4
    )
    _cuda_sync()
    detect_ms = (time.perf_counter() - t0) * 1000.0

    if len(masks) == 0:
        raise RuntimeError("[sam3_decoupled] Sam3Wrapper returned no masks on frame 0")

    # Top mask per query — same simplification example 09 uses.
    top_k = min(len(QUERIES), len(masks))
    seed_masks = {i + 1: masks[i].astype(bool) for i in range(top_k)}
    queries_by_obj_id = {i + 1: QUERIES[i] for i in range(top_k)}
    print(
        f"  seeded tracker with top {top_k}: "
        f"{[(oid, queries_by_obj_id[oid], float(scores[oid - 1])) for oid in seed_masks]}"
    )

    tracker = Sam3VideoTracker("facebook/sam3", device=device)

    # initialize() is part of session setup, not per-frame tracking work.
    _cuda_sync()
    t0 = time.perf_counter()
    refined = tracker.initialize(frames[0], seed_masks)
    _cuda_sync()
    init_ms = (time.perf_counter() - t0) * 1000.0
    overlays = [
        _overlay(frames[0], _masks_from_tracker(refined, queries_by_obj_id), title)
    ]

    # Frame 1 warmup.
    _cuda_sync()
    t0 = time.perf_counter()
    warmup_out = tracker.step(frames[1])
    _cuda_sync()
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    overlays.append(
        _overlay(frames[1], _masks_from_tracker(warmup_out, queries_by_obj_id), title)
    )

    per_frame_ms: list[float] = []
    for i in range(2, len(frames)):
        _cuda_sync()
        t0 = time.perf_counter()
        step_out = tracker.step(frames[i])
        _cuda_sync()
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        overlays.append(
            _overlay(frames[i], _masks_from_tracker(step_out, queries_by_obj_id), title)
        )

    print(f"  initialize (excluded): {init_ms:7.1f} ms")
    stats = _summarize("sam3_decoupled", detect_ms, warmup_ms, per_frame_ms)
    return stats, overlays


def _summarize(
    backend: str, detect_ms: float, warmup_ms: float, per_frame_ms: list[float]
) -> dict[str, float]:
    mean_ms = statistics.mean(per_frame_ms)
    median_ms = statistics.median(per_frame_ms)
    p95_ms = _percentile(per_frame_ms, 95)
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    print(f"  detect (excluded):  {detect_ms:7.1f} ms")
    print(f"  track #1 warmup:    {warmup_ms:7.1f} ms  (excluded)")
    print(
        f"  track steady ({len(per_frame_ms)} frames):"
        f"  mean={mean_ms:6.1f} ms  median={median_ms:6.1f} ms"
        f"  p95={p95_ms:6.1f} ms  → {fps:5.1f} fps"
    )
    return {
        "backend": backend,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps": fps,
        "n_frames": len(per_frame_ms),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _save_side_by_side(
    overlays_per_backend: list[list[np.ndarray]], out_path: Path
) -> None:
    """Stack per-frame overlays from each backend horizontally and write an mp4."""
    n_frames = min(len(o) for o in overlays_per_backend)
    frames_out = [
        np.concatenate([o[i] for o in overlays_per_backend], axis=1)
        for i in range(n_frames)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames_out), fps=15, codec="libx264")
    print(
        f"\nwrote {out_path}  ({n_frames} frames, {frames_out[0].shape[1]}x{frames_out[0].shape[0]})"
    )


def main():
    data = np.load(ASSETS / "demo" / "demo_frames.npz")
    frames = data["rgb"]
    print(
        f"loaded {len(frames)} frames of shape {frames.shape[1:]}  "
        f"— excluding frame 0 (detect) and frame 1 (warmup)"
    )

    results = []
    overlays_per_backend: list[list[np.ndarray]] = []
    for backend in ("sam3", "sam2"):
        stats, overlays = benchmark_openperception(backend, frames)
        results.append(stats)
        overlays_per_backend.append(overlays)

    stats, overlays = benchmark_sam3_decoupled(frames)
    results.append(stats)
    overlays_per_backend.append(overlays)

    # Summary table + speedup.
    print("\n=== summary (steady-state, per-frame track) ===")
    print(
        f"{'backend':<18} {'mean ms':>10} {'median ms':>12} {'p95 ms':>10} {'fps':>8}"
    )
    for r in results:
        print(
            f"{r['backend']:<18} {r['mean_ms']:>10.1f} {r['median_ms']:>12.1f}"
            f" {r['p95_ms']:>10.1f} {r['fps']:>8.1f}"
        )
    by_name = {r["backend"]: r for r in results}
    sam3, sam2 = by_name["sam3"], by_name["sam2"]
    if sam2["mean_ms"] > 0:
        print(
            f"\nSAM2 speedup over SAM3 (mean): {sam3['mean_ms'] / sam2['mean_ms']:.2f}x"
        )
    if "sam3_decoupled" in by_name and by_name["sam3_decoupled"]["mean_ms"] > 0:
        dec = by_name["sam3_decoupled"]
        print(
            f"SAM3-decoupled speedup over SAM3 (mean): {sam3['mean_ms'] / dec['mean_ms']:.2f}x"
        )

    _save_side_by_side(
        overlays_per_backend, Path("tmp/open_perception_sam_benchmark_side_by_side.mp4")
    )


if __name__ == "__main__":
    main()

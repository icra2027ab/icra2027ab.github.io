"""Multi-camera fusion via the ``fuse_views`` pipeline stage.

Two cameras run through a single :class:`OpenPerception` instance with
``fuse_views`` configured in ``primary_guided`` mode — the main camera's
detections drive ``instance_id`` alignment in the secondary via 3D
point reprojection (occlusion-aware).

Writes two side-by-side mp4s under ``tmp/`` (independent vs.
primary-guided) so you can eyeball the difference: in independent mode
each camera mints its own ids, in primary-guided mode the same physical
object carries the same colour across cameras.

    uv run python examples/09_camera_fusion.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from open_perception import (
    CameraInfo,
    InferenceOptions,
    OpenPerception,
    PerceptionConfig,
    render_debug_image,
    render_history_gif,
    render_segmentation_video,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
REFS = ASSETS / "demo_2" / "refs"


def _build(
    mode: str, queries: list[str], debug_dir: Path | None = None
) -> OpenPerception:
    cfg = PerceptionConfig().order(
        [
            "sam2",
            "filter.size",
            "disambiguate.dino",
            "assign",
            "filter.overlap",
            "filter.erode",
            "reconstruct",
            "fuse_views",
        ]
    )
    # Per-frame mask propagation goes through SAM2 (much cheaper than the
    # SAM3 video model); initial frame-0 detection still uses SAM3 because
    # SAM2 cannot segment from text.

    score_weights = {"dino": 1.0, "segmentation": 0.0}
    cfg.tracker_backend = "sam2"
    # demo_multi_camera.npz comes from robosuite/LIBERO with rgb/depth stored
    # 180°-rotated relative to the intrinsics — same flip 08_multi_camera.py
    # uses. Without it, reconstruction mirrors the world points across u and
    # the secondary projection lands on the wrong pixels.
    cfg.update(
        "sam3", score_threshold=0.05, mask_threshold=0.4, prompts=["<object_names>"]
    )
    dino_params: dict = {"fallback_to_text": True}
    if debug_dir is not None:
        dino_params["debug_dir"] = str(debug_dir)
    cfg.update("disambiguate.dino", **dino_params)
    cfg.update("assign", weights=score_weights, top_k=1)

    cfg.update("filter.overlap", score_weights=score_weights)
    cfg.update("reconstruct", flip_u=True)
    cfg.update("fuse_views", mode=mode, main_camera=0, flip_u=True)

    reference = {q: iio.imread(REFS / f"{q.replace(' ', '_')}.png") for q in queries}
    perc = OpenPerception(cfg)
    perc.register_reference(reference, space="dino")
    return perc


def _run(perc: OpenPerception, queries: list[str], data) -> list[float]:
    """Run the demo and return the per-track-frame wall-clock latencies (s)."""
    Ks = [data["intrinsics_cam0"], data["intrinsics_cam1"]]
    Es = [data["extrinsics_cam0"], data["extrinsics_cam1"]]
    n_frames = data["rgb_cam0"].shape[0]
    opts = InferenceOptions(top_k_per_query=None, match_threshold=0.2)
    perc.reset()
    track_latencies: list[float] = []
    for t in range(n_frames):
        rgbs = [data["rgb_cam0"][t], data["rgb_cam1"][t]]
        depths = [data["depth_cam0"][t], data["depth_cam1"][t]]
        cams = [
            CameraInfo(intrinsics=K, extrinsics=E[t])
            for K, E in zip(Ks, Es, strict=True)
        ]
        if t == 0:
            perc.detect(queries, rgbs, depth=depths, camera=cams, options=opts)
        else:
            t0 = time.perf_counter()
            perc.track(rgbs, depth=depths, camera=cams, options=opts)
            track_latencies.append(time.perf_counter() - t0)
    return track_latencies


def _print_timings(
    label: str, perc: OpenPerception, track_latencies: list[float]
) -> None:
    """Median per-stage timings + end-to-end FPS over the tracked frames."""
    if not perc.history or len(perc.history) < 2:
        print(f"[{label}] no track frames recorded; skipping profile")
        return
    # Skip frame 0 (detect — runs the full pipeline including SAM3 detect),
    # report on the steady-state track frames only.
    per_stage: dict[str, list[float]] = {}
    n_track = 0
    for i, frame in enumerate(perc.history):
        if i == 0:
            continue
        n_track += 1
        for stage, ms in frame.stats.timings_ms.items():
            per_stage.setdefault(stage, []).append(ms)

    print(f"\n[{label}] per-stage timings over {n_track} track frames")
    print(f"  {'stage':<32} {'median ms':>10} {'p25 ms':>10} {'p75 ms':>10}")
    for stage in sorted(per_stage):
        xs = sorted(per_stage[stage])
        med = statistics.median(xs)
        p25 = xs[len(xs) // 4]
        p75 = xs[(len(xs) * 3) // 4]
        print(f"  {stage:<32} {med:>10.2f} {p25:>10.2f} {p75:>10.2f}")

    if track_latencies:
        med_lat = statistics.median(track_latencies)
        print(
            f"[{label}] end-to-end track(): median {med_lat * 1000:.1f} ms"
            f"  ({1.0 / med_lat:.2f} FPS)"
        )


def _log_alignment(label: str, perc: OpenPerception) -> None:
    print(f"\n[{label}] tracker state after run:")
    for query, states in sorted(perc._tracker_state.states.items()):
        per_view: dict[int, list[int]] = {}
        for s in states:
            per_view.setdefault(s.view_idx, []).append(s.instance_id)
        for view_idx, ids in sorted(per_view.items()):
            print(f"  {query!r} cam{view_idx}: ids={sorted(ids)}")


def main() -> None:
    data = np.load(ASSETS / "demo_2" / "demo_multi_camera.npz")
    queries = ["cream_cheese", "basket", "alphabet_soup"]
    out_dir = Path("tmp") / "10"
    out_dir.mkdir(exist_ok=True)

    mode = "primary_guided"
    debug_dir = out_dir / "disambiguate_dino"
    perc = _build(mode, queries, debug_dir=debug_dir)
    track_latencies = _run(perc, queries, data)
    _print_timings(mode, perc, track_latencies)
    path = render_segmentation_video(
        perc.history, out_dir / f"camera_fusion_{mode}.mp4", show_id=True, show_fps=True
    )
    print(f"wrote {path}")
    print(f"disambiguate.dino debug panels in {debug_dir}/")
    # Two history gifs per camera with complementary content:
    # - "detect" gif: frame 0 — every stage runs, shows the mask evolution
    #   through sam3 → filter.* → reconstruct → fuse_views. (projected_refs
    #   are empty here because instance_ids are seeded after fuse_views.)
    # - "track" gif: frame 1 — only runs_on_track stages fire, but
    #   fuse_views now has populated projected_refs (dotted footprints in
    #   the secondary view).
    rgb_keys = ("rgb_cam0", "rgb_cam1")
    for label, frame_idx in (("detect", 0), ("track", 1)):
        if frame_idx >= len(perc.history):
            continue
        frame = perc.history[frame_idx]
        for view_idx, rgb_key in enumerate(rgb_keys):
            view_snaps = [s for s in frame.stage_history if s.view_idx == view_idx]
            if not view_snaps:
                continue
            gif_path = render_history_gif(
                data[rgb_key][frame.frame_idx],
                view_snaps,
                out_dir / f"camera_fusion_{mode}_history_{label}_cam{view_idx}.gif",
                show_id=True,
                show_label=True,
            )
            print(f"wrote {gif_path}")
            debug_path = render_debug_image(
                data[rgb_key][frame.frame_idx],
                view_snaps,
                out_dir / f"camera_fusion_{mode}_debug_{label}_cam{view_idx}.png",
                config=perc.config,
                title=(f"{mode} — {label} frame {frame.frame_idx} cam{view_idx}"),
            )
            print(f"wrote {debug_path}")
    _log_alignment(mode, perc)


if __name__ == "__main__":
    main()

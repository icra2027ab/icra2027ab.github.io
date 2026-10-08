"""Compare collected RoboCasa HDF5 demos against the reference LeRobot MP4 videos.

Quantifies replay fidelity from ``scripts/setup/collect_robocasa_demos.py``.  Because
LeRobot recordings don't bundle the sim init state, even ``frame 0`` won't
match exactly (objects are placed by ``env.reset()``'s seed, not the recorder's
sim XML).  Trajectories then diverge further as the open-loop replay
accumulates control error.  This script reports:

* mean / max absolute pixel-diff per frame after resolution alignment,
* the first frame index whose mean-diff exceeds ``--divergence-threshold``,
* an aggregate score (mean mean-diff and SSIM mean over the first N frames),
* a side-by-side PNG strip (reference | collected | abs-diff) at evenly spaced
  sample frames so a human can eyeball whether the scene / camera / pose match.

Run from the project root venv (just needs ``imageio[ffmpeg]``, ``h5py``,
``numpy``, ``opencv-python``).  No GPU / robocasa venv needed.

    python scripts/debug/verify_robocasa_demos.py \\
        --hdf5-path data/robocasa/OpenSingleDoor_demo.hdf5 \\
        --lerobot-root $BASE_DATASETS_DIR/RoboCasa-Cosmos-Policy/atomic/OpenSingleDoor \\
        --out-dir tmp/robocasa_verify
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import h5py
import imageio.v3 as iio
import numpy as np
import tyro
from loguru import logger

_PRIMARY_CAMERA = "robot0_agentview_left"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """CLI arguments for :mod:`verify_robocasa_demos`."""

    hdf5_path: str
    """Path to the HDF5 file produced by ``collect_robocasa_demos.py``."""

    lerobot_root: str
    """LeRobot dataset root for the same task.  Same path you passed to
    ``collect_robocasa_demos.py --lerobot-root``."""

    out_dir: str = "tmp/robocasa_verify"
    """Where to write the per-demo side-by-side strips + a summary JSON."""

    max_frames: Optional[int] = None
    """Cap the number of frames compared per demo.  ``None`` = full episode."""

    divergence_threshold: float = 25.0
    """Mean abs pixel diff (uint8 RGB, per-channel) above which a frame counts
    as 'diverged'.  25.0 ≈ 10% of full scale; tune per scene."""

    num_sample_panels: int = 5
    """How many evenly-spaced frames to dump as side-by-side PNGs per demo."""


# ---------------------------------------------------------------------------
# LeRobot side
# ---------------------------------------------------------------------------


def _resolve_lerobot_root(path: str) -> Path:
    """Mirror :func:`retrieval.sources.robocasa_lerobot._resolve_lerobot_root`."""
    p = Path(path).resolve()
    candidates: list[Path] = [p, p / "lerobot"]
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_dir():
                candidates.append(child / "lerobot")
                candidates.append(child)
    for candidate in candidates:
        if (candidate / "meta" / "tasks.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"could not find lerobot/meta/tasks.jsonl under {p}")


def _load_lerobot_mp4(lerobot_root: Path, episode_id: int) -> np.ndarray:
    """Load all frames of one LeRobot MP4 as ``(T, H, W, 3) uint8`` upright RGB."""
    mp4 = (
        lerobot_root
        / "videos"
        / "chunk-000"
        / f"observation.images.{_PRIMARY_CAMERA}"
        / f"episode_{episode_id:06d}.mp4"
    )
    if not mp4.exists():
        raise FileNotFoundError(f"reference MP4 not found: {mp4}")
    return np.stack(list(iio.imiter(str(mp4))), axis=0)


# ---------------------------------------------------------------------------
# HDF5 side
# ---------------------------------------------------------------------------


def _load_hdf5_frames(hdf5_path: Path, demo_key: str) -> np.ndarray:
    """Return upright ``(T, H, W, 3) uint8`` for one demo.

    RoboCasa's MuJoCo offscreen render produces frames that are only
    **vertically** flipped (bottom-up OpenGL convention), so a ``[::-1, :]``
    flip recovers the upright orientation that matches the LeRobot MP4.

    Note: LIBERO's pipeline uses ``[::-1, ::-1]`` (vertical + horizontal),
    which would horizontally mirror RoboCasa frames and inflate the diff
    by ~50/255 per pixel.  Use the LIBERO convention only for LIBERO data.
    """
    with h5py.File(hdf5_path, "r") as f:
        raw = f["data"][demo_key]["obs"]["agentview_rgb"][:]
    return np.ascontiguousarray(raw[:, ::-1, :, :])


def _list_hdf5_demos(hdf5_path: Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted(
            (k for k in f["data"].keys() if k.startswith("demo_")),
            key=lambda k: int(k.split("_")[1]),
        )


def _demo_episode_id(hdf5_path: Path, demo_key: str) -> int:
    """Read the reference LeRobot episode id this demo was replayed from.

    ``collect_robocasa_demos.py`` writes ``episode_id`` as a per-demo attr.
    Falls back to ``int(demo_key.split('_')[1])`` if the attr is missing
    (older HDF5s).
    """
    with h5py.File(hdf5_path, "r") as f:
        grp = f["data"][demo_key]
        if "episode_id" in grp.attrs:
            return int(grp.attrs["episode_id"])
    return int(demo_key.split("_")[1])


# ---------------------------------------------------------------------------
# Per-frame comparison
# ---------------------------------------------------------------------------


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _abs_diff_stats(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    """Mean and max absolute pixel diff over RGB channels for one frame."""
    d = np.abs(ref.astype(np.int16) - got.astype(np.int16)).astype(np.float32)
    return float(d.mean()), float(d.max())


def _ssim_proxy(ref: np.ndarray, got: np.ndarray) -> float:
    """Cheap SSIM-like proxy: normalized cross-correlation of grayscale.

    Avoids a scikit-image dep.  Returns a value in roughly ``[-1, 1]`` where
    ``1`` means identical.
    """
    gx = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gy = cv2.cvtColor(got, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx -= gx.mean()
    gy -= gy.mean()
    denom = float(np.linalg.norm(gx) * np.linalg.norm(gy))
    if denom < 1e-6:
        return 0.0
    return float((gx * gy).sum() / denom)


# ---------------------------------------------------------------------------
# Per-demo verification
# ---------------------------------------------------------------------------


def _verify_demo(
    ref_frames: np.ndarray,
    got_frames: np.ndarray,
    *,
    threshold: float,
    max_frames: Optional[int],
) -> dict:
    """Compare two (T, H, W, 3) arrays frame-by-frame.  Returns a metrics dict."""
    T = min(ref_frames.shape[0], got_frames.shape[0])
    if max_frames is not None:
        T = min(T, max_frames)
    if T == 0:
        return {"num_frames_compared": 0}

    # Align resolution.  LeRobot MP4s are typically 224×224; collect_*.py uses
    # whatever ``--camera-height/width`` was passed (default 256×256).
    target_h = min(ref_frames.shape[1], got_frames.shape[1])
    target_w = min(ref_frames.shape[2], got_frames.shape[2])

    mean_diffs: list[float] = []
    max_diffs: list[float] = []
    ssims: list[float] = []
    divergence_t: Optional[int] = None
    for t in range(T):
        ref_t = _resize_to(ref_frames[t], target_h, target_w)
        got_t = _resize_to(got_frames[t], target_h, target_w)
        m, mx = _abs_diff_stats(ref_t, got_t)
        mean_diffs.append(m)
        max_diffs.append(mx)
        ssims.append(_ssim_proxy(ref_t, got_t))
        if divergence_t is None and m > threshold:
            divergence_t = t

    return {
        "num_frames_compared": T,
        "ref_resolution": list(ref_frames.shape[1:3]),
        "got_resolution": list(got_frames.shape[1:3]),
        "compare_resolution": [target_h, target_w],
        "first_frame_mean_diff": mean_diffs[0],
        "first_frame_max_diff": max_diffs[0],
        "first_frame_ssim": ssims[0],
        "overall_mean_diff_mean": float(np.mean(mean_diffs)),
        "overall_mean_diff_max": float(np.max(mean_diffs)),
        "overall_ssim_mean": float(np.mean(ssims)),
        "divergence_t": divergence_t,
        "divergence_threshold": threshold,
        "per_frame_mean_diff": mean_diffs,
        "per_frame_ssim": ssims,
    }


def _make_panel(
    ref_frames: np.ndarray,
    got_frames: np.ndarray,
    sample_indices: list[int],
    out_path: Path,
) -> None:
    """Vertically stacked rows of (reference | collected | abs-diff) panels."""
    h = min(ref_frames.shape[1], got_frames.shape[1])
    w = min(ref_frames.shape[2], got_frames.shape[2])
    rows: list[np.ndarray] = []
    for t in sample_indices:
        ref_t = _resize_to(ref_frames[t], h, w)
        got_t = _resize_to(got_frames[t], h, w)
        diff_t = np.abs(ref_t.astype(np.int16) - got_t.astype(np.int16))
        diff_t = np.clip(diff_t * 2, 0, 255).astype(np.uint8)  # ×2 for visibility

        # Annotate frame index in the top-left corner.
        label = f"t={t}"
        ref_t = ref_t.copy()
        cv2.putText(
            ref_t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        row = np.concatenate([ref_t, got_t, diff_t], axis=1)
        rows.append(row)

    panel = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 wants BGR.  RGB → BGR before write.
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    hdf5_path = Path(args.hdf5_path).resolve()
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")
    lerobot_root = _resolve_lerobot_root(args.lerobot_root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_keys = _list_hdf5_demos(hdf5_path)
    if not demo_keys:
        raise RuntimeError(f"No demos found under {hdf5_path}")
    logger.info(f"Verifying {len(demo_keys)} demo(s) from {hdf5_path.name}")

    per_demo_summary: dict[str, dict] = {}
    aggregate_mean_diffs: list[float] = []
    aggregate_ssims: list[float] = []
    aggregate_divergence_ts: list[int] = []

    for demo_key in demo_keys:
        episode_id = _demo_episode_id(hdf5_path, demo_key)
        try:
            ref_frames = _load_lerobot_mp4(lerobot_root, episode_id)
        except FileNotFoundError as exc:
            logger.warning(f"  {demo_key}: skipping ({exc})")
            per_demo_summary[demo_key] = {"error": str(exc)}
            continue
        got_frames = _load_hdf5_frames(hdf5_path, demo_key)

        metrics = _verify_demo(
            ref_frames,
            got_frames,
            threshold=args.divergence_threshold,
            max_frames=args.max_frames,
        )
        metrics["episode_id"] = episode_id
        metrics["ref_total_frames"] = int(ref_frames.shape[0])
        metrics["got_total_frames"] = int(got_frames.shape[0])

        # Side-by-side panel at evenly spaced sample frames.
        T = metrics["num_frames_compared"]
        if T > 0 and args.num_sample_panels > 0:
            n = min(args.num_sample_panels, T)
            samples = np.linspace(0, T - 1, n, dtype=int).tolist()
            panel_path = out_dir / f"{demo_key}_compare.png"
            _make_panel(ref_frames, got_frames, samples, panel_path)
            metrics["panel_path"] = str(panel_path)

        # Strip the heavy per-frame arrays out of the persistent summary so the
        # JSON stays human-readable; we still keep them for the logger output.
        per_frame_mean = metrics.pop("per_frame_mean_diff")
        per_frame_ssim = metrics.pop("per_frame_ssim")
        per_demo_summary[demo_key] = metrics

        aggregate_mean_diffs.append(metrics["overall_mean_diff_mean"])
        aggregate_ssims.append(metrics["overall_ssim_mean"])
        if metrics["divergence_t"] is not None:
            aggregate_divergence_ts.append(int(metrics["divergence_t"]))

        logger.info(
            f"  {demo_key} (ep {episode_id}, T={T}): "
            f"first-frame mean_diff={metrics['first_frame_mean_diff']:.1f} "
            f"ssim={metrics['first_frame_ssim']:+.3f} | "
            f"overall mean_diff={metrics['overall_mean_diff_mean']:.1f} "
            f"ssim={metrics['overall_ssim_mean']:+.3f} | "
            f"divergence_t={metrics['divergence_t']} "
            f"(threshold={args.divergence_threshold:.1f})"
        )
        # Spot-check: per-frame mean-diff at 25/50/75% to give a sense of drift.
        if T >= 4:
            for frac, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
                idx = int(frac * (T - 1))
                logger.info(
                    f"    t={idx} ({label}): mean_diff={per_frame_mean[idx]:.1f} "
                    f"ssim={per_frame_ssim[idx]:+.3f}"
                )

    summary = {
        "hdf5_path": str(hdf5_path),
        "lerobot_root": str(lerobot_root),
        "num_demos": len(per_demo_summary),
        "aggregate": {
            "first_frame_mean_diff_mean": (
                float(
                    np.mean(
                        [
                            m["first_frame_mean_diff"]
                            for m in per_demo_summary.values()
                            if "first_frame_mean_diff" in m
                        ]
                    )
                )
                if per_demo_summary
                else None
            ),
            "overall_mean_diff_mean": float(np.mean(aggregate_mean_diffs))
            if aggregate_mean_diffs
            else None,
            "overall_ssim_mean": float(np.mean(aggregate_ssims))
            if aggregate_ssims
            else None,
            "median_divergence_t": (
                float(np.median(aggregate_divergence_ts))
                if aggregate_divergence_ts
                else None
            ),
        },
        "per_demo": per_demo_summary,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary → {summary_path}")
    logger.info(f"Wrote {len(per_demo_summary)} side-by-side panels → {out_dir}")

    # Friendly interpretation hint.
    if summary["aggregate"]["overall_mean_diff_mean"] is None:
        return
    if summary["aggregate"]["overall_mean_diff_mean"] < 5.0:
        logger.success("Replay looks visually very close (mean abs diff < 5).")
    elif summary["aggregate"]["overall_mean_diff_mean"] < args.divergence_threshold:
        logger.info(
            "Replay is in the same ballpark as the reference, but with visible "
            "differences. Inspect the side-by-side panels."
        )
    else:
        logger.warning(
            "Replay diverges from the reference. Expected with no init-state "
            "restoration. Suitable for retrieval / first-frame seg analysis; "
            "not for policy-level imitation."
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

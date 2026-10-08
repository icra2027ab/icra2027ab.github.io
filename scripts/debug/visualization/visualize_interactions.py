"""Render a sample video of extracted gripper-object interactions.

Layout: For each demo, all its interactions are shown as stacked rows sharing
the same playback timeline.

Each row contains:
  • Left panel  – agentview RGB with projected finger-tip dots
  • Middle panel – eye-in-hand RGB with projected finger-tip dots
  • Right panel  – per-timestep distance plot (cursor at current frame)

Outside the interaction's time window the row is frozen and shown as opaque gray:
  • Before the window: frozen to the first frame of the interaction (gray)
  • After the window:  frozen to the last  frame of the interaction (gray)
  • Inside the window: live color frames

Usage:
    uv run tmp/visualize_interactions.py \
        --interactions tmp/test_interactions.hdf5 \
        --output tmp/sample_interactions.mp4 \
        --n-demos 6 --fps 20
"""

from __future__ import annotations

import argparse
import io
import random
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from policy_abstraction.utils.pcd import project_points_to_pixels
from policy_abstraction.utils.segmentation import colorize_segmentation

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_finger_dots(
    img: np.ndarray,
    left_tip: np.ndarray,
    right_tip: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
) -> np.ndarray:
    H, W = img.shape[:2]
    out = img.copy()
    tips = np.stack([left_tip, right_tip])
    pts, valid = project_points_to_pixels(tips, K, E, H, W, preprocessed=True)
    for i, (color, is_valid) in enumerate(
        zip([(0, 255, 0), (0, 0, 255)], valid, strict=False)
    ):
        if not is_valid:
            continue
        u, v = int(pts[i, 0]), int(pts[i, 1])
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(out, (u, v), 7, color, -1)
            cv2.circle(out, (u, v), 8, (0, 0, 0), 1)
    return out


def _draw_closest_point(
    img: np.ndarray,
    closest_3d: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    color: tuple = (255, 165, 0),
) -> np.ndarray:
    if np.any(np.isnan(closest_3d)):
        return img
    H, W = img.shape[:2]
    out = img.copy()
    pts, valid = project_points_to_pixels(
        closest_3d[None], K, E, H, W, preprocessed=True
    )
    if valid[0]:
        u, v = int(pts[0, 0]), int(pts[0, 1])
        if 0 <= u < W and 0 <= v < H:
            cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, 12, 2)
    return out


def _put_text(img, text, pos, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def _to_opaque_grayscale(img: np.ndarray, darken: float = 0.35) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if darken <= 0:
        return gray_rgb
    black = np.zeros_like(gray_rgb)
    return cv2.addWeighted(gray_rgb, 1.0 - darken, black, darken, 0.0)


def _overlay_seg_mask(
    img: np.ndarray,  # (H, W, 3) RGB uint8
    seg: np.ndarray,  # (H, W, 1) or (H, W) int32
    object_id: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend a colored highlight for *object_id* onto *img* (RGB)."""
    seg_2d = seg[..., 0] if seg.ndim == 3 else seg
    mask = seg_2d == object_id
    if not mask.any():
        return img
    # colorize_segmentation returns BGR — convert to RGB for blending
    seg_bgr = colorize_segmentation(seg_2d, {object_id}, bg_gray=0)
    seg_rgb = seg_bgr[:, :, ::-1]
    out = img.astype(np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + seg_rgb[mask].astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Distance plot
# ---------------------------------------------------------------------------


def _render_distance_plot(
    left_dists: np.ndarray,
    right_dists: np.ndarray,
    cursor_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    matplotlib.use("Agg")

    N = len(left_dists)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    xs = np.arange(N)
    left = np.where(np.isfinite(left_dists), left_dists, np.nan)
    right = np.where(np.isfinite(right_dists), right_dists, np.nan)

    ax.plot(xs, left * 100, color="#00ff88", linewidth=1.5, label="left finger")
    ax.plot(xs, right * 100, color="#ff4444", linewidth=1.5, label="right finger")
    ax.axhline(
        3.0,
        color="yellow",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="start (3 cm)",
    )
    ax.axhline(
        5.0,
        color="orange",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="stop (5 cm)",
    )
    ax.axvline(cursor_idx, color="white", linewidth=1.0, alpha=0.9)

    ax.set_xlim(0, N - 1)
    ymax = max(10, np.nanmax(np.concatenate([left, right])) * 100 * 1.15)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("timestep", color="white", fontsize=7)
    ax.set_ylabel("dist (cm)", color="white", fontsize=7)
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(
        fontsize=5,
        loc="upper right",
        facecolor="#1a1a2e",
        edgecolor="#555",
        labelcolor="white",
    )

    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Title card
# ---------------------------------------------------------------------------


def _make_title_card(
    text: str,
    width: int,
    height: int,
    fps: int,
    duration_s: float = 0.75,
) -> list[np.ndarray]:
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (25, 25, 50)
    lines = text.split("\n")
    y = height // 2 - (len(lines) - 1) * 20
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        x = max(8, (width - tw) // 2)
        _put_text(card, line, (x, y), scale=0.7)
        y += 36
    return [card] * max(1, int(fps * duration_s))


# ---------------------------------------------------------------------------
# Interaction row label overlay
# ---------------------------------------------------------------------------


def _add_row_label(
    row: np.ndarray, obj_name: str, start_t: int, end_t: int, state: str
) -> np.ndarray:
    """Prepend a solid header bar above the row."""
    W = row.shape[1]
    banner_h = 24
    header = np.zeros((banner_h, W, 3), dtype=np.uint8)
    header[:] = (20, 20, 40)

    state_color = (100, 220, 100) if state == "active" else (150, 150, 150)
    label = f"{obj_name}  [{start_t}, {end_t})  {state}"
    _put_text(header, label, (8, 16), scale=0.45, color=state_color)

    return np.concatenate([header, row], axis=0)


# ---------------------------------------------------------------------------
# Load raw interaction data from an open HDF5 group
# ---------------------------------------------------------------------------


def _read_interaction_raw(grp) -> dict:
    """Read all arrays from an interaction group into plain numpy dicts."""
    return {
        "attrs": dict(grp.attrs),
        "timesteps": grp["timesteps"][:],
        "left_finger_pos": grp["left_finger_pos"][:],
        "right_finger_pos": grp["right_finger_pos"][:],
        "left_finger_dist": grp["left_finger_dist"][:],
        "right_finger_dist": grp["right_finger_dist"][:],
        "left_closest_point": grp["left_closest_point"][:],
        "right_closest_point": grp["right_closest_point"][:],
    }


# ---------------------------------------------------------------------------
# Pre-render one interaction into color frames + frozen gray endpoints
# ---------------------------------------------------------------------------


def precompute_interaction(
    raw: dict,
    av_rgb: np.ndarray,  # (total_steps, H, W, 3)
    eih_rgb: np.ndarray,
    av_K: np.ndarray,  # (total_steps, 3, 3) or similar
    av_E: np.ndarray,
    eih_K: np.ndarray,
    eih_E: np.ndarray,
    panel_h: int,
    av_seg: np.ndarray | None = None,  # (total_steps, H, W, 1) int32
    eih_seg: np.ndarray | None = None,
) -> dict:
    """Return a dict ready for per-timestep lookup during frame assembly."""
    attrs = raw["attrs"]
    obj_name = attrs["object_name"]
    object_id = int(attrs["object_id"])
    start_t = int(attrs["start_timestep"])
    end_t = int(attrs["end_timestep"])
    timesteps = raw["timesteps"]
    left_pos = raw["left_finger_pos"]
    right_pos = raw["right_finger_pos"]
    left_dists = raw["left_finger_dist"]
    right_dists = raw["right_finger_dist"]
    left_cls = raw["left_closest_point"]
    right_cls = raw["right_closest_point"]
    N = len(timesteps)

    H_src, W_src = av_rgb.shape[1:3]
    scale = panel_h / H_src
    panel_w = int(W_src * scale)
    plot_w = panel_w

    print(f"    Pre-rendering {N} distance plots for '{obj_name}'…", end="", flush=True)
    dist_plots = [
        _render_distance_plot(left_dists, right_dists, i, plot_w, panel_h)
        for i in range(N)
    ]
    print(" done.")

    color_frames: list[np.ndarray] = []
    for i, t in enumerate(timesteps):
        av_f = cv2.resize(av_rgb[t], (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        eih_f = cv2.resize(eih_rgb[t], (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        # Segmentation overlay (resize mask to panel size before blending)
        if av_seg is not None:
            av_seg_t = cv2.resize(
                av_seg[t].squeeze().astype(np.int32),
                (panel_w, panel_h),
                interpolation=cv2.INTER_NEAREST,
            )
            av_f = _overlay_seg_mask(av_f, av_seg_t, object_id)
        if eih_seg is not None:
            eih_seg_t = cv2.resize(
                eih_seg[t].squeeze().astype(np.int32),
                (panel_w, panel_h),
                interpolation=cv2.INTER_NEAREST,
            )
            eih_f = _overlay_seg_mask(eih_f, eih_seg_t, object_id)

        av_f = _draw_finger_dots(av_f, left_pos[i], right_pos[i], av_K[t], av_E[t])
        eih_f = _draw_finger_dots(eih_f, left_pos[i], right_pos[i], eih_K[t], eih_E[t])

        ld = left_dists[i] if np.isfinite(left_dists[i]) else np.inf
        rd = right_dists[i] if np.isfinite(right_dists[i]) else np.inf
        if ld <= rd:
            av_f = _draw_closest_point(av_f, left_cls[i], av_K[t], av_E[t])
            eih_f = _draw_closest_point(eih_f, left_cls[i], eih_K[t], eih_E[t])
        else:
            av_f = _draw_closest_point(av_f, right_cls[i], av_K[t], av_E[t])
            eih_f = _draw_closest_point(eih_f, right_cls[i], eih_K[t], eih_E[t])

        min_d = min(ld, rd) if np.isfinite(min(ld, rd)) else float("nan")
        dist_str = f"{min_d * 100:.1f}cm" if np.isfinite(min_d) else "?"
        _put_text(av_f, f"t={t} d={dist_str}", (6, 18))
        _put_text(av_f, obj_name[:20], (6, 36))
        _put_text(eih_f, "eye-in-hand", (6, 18))

        row = np.concatenate([av_f, eih_f, dist_plots[i]], axis=1)
        color_frames.append(row)

    first_gray = _to_opaque_grayscale(color_frames[0])
    last_gray = _to_opaque_grayscale(color_frames[-1])
    t_to_idx = {int(t): i for i, t in enumerate(timesteps.tolist())}

    return {
        "obj_name": obj_name,
        "start_t": start_t,
        "end_t": end_t,
        "timesteps": timesteps,
        "color_frames": color_frames,
        "first_gray": first_gray,
        "last_gray": last_gray,
        "t_to_idx": t_to_idx,
        "panel_h": panel_h,
        "frame_w": panel_w * 2 + plot_w,
    }


# ---------------------------------------------------------------------------
# Assemble all rows for a demo into a full list of stacked frames
# ---------------------------------------------------------------------------


def build_demo_frames(
    interactions: list[dict],
    total_steps: int,
    panel_h: int,
    total_rows: int = 0,
) -> list[np.ndarray]:
    """
    For each global timestep t build one video frame by stacking all
    interaction rows vertically.  Rows outside their window are gray/frozen.
    Pads with blank rows up to *total_rows* so all demos share the same height.
    """
    frames: list[np.ndarray] = []
    n_rows = len(interactions)
    pad_rows = max(0, total_rows - n_rows)

    t_start = min(idata["start_t"] for idata in interactions)
    t_end = max(idata["end_t"] for idata in interactions)
    t_end = min(t_end, total_steps)

    for t in range(t_start, t_end):
        rows: list[np.ndarray] = []

        for idata in interactions:
            start_t = idata["start_t"]
            end_t = idata["end_t"]
            timesteps = idata["timesteps"]
            N = len(timesteps)

            if t < start_t:
                row = idata["first_gray"].copy()
                state = "waiting"
            elif t >= end_t:
                row = idata["last_gray"].copy()
                state = "done"
            else:
                idx = idata["t_to_idx"].get(t)
                if idx is None:
                    idx = int(np.searchsorted(timesteps, t, side="right") - 1)
                    idx = int(np.clip(idx, 0, N - 1))
                row = idata["color_frames"][idx].copy()
                state = "active"

            row = _add_row_label(row, idata["obj_name"], start_t, end_t, state)
            rows.append(row)

        # Global time stamp on the last active row
        if rows:
            h = rows[-1].shape[0]
            _put_text(
                rows[-1], f"demo t = {t}", (6, h - 8), scale=0.5, color=(200, 200, 255)
            )

        # Pad to consistent height across all demos
        if pad_rows > 0 and rows:
            row_h, row_w = rows[0].shape[:2]
            blank = np.zeros((row_h * pad_rows, row_w, 3), dtype=np.uint8)
            blank[:] = (15, 15, 30)
            rows.append(blank)

        frames.append(np.concatenate(rows, axis=0))

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render interactions per demo with a shared timeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interactions",
        "-i",
        type=Path,
        default=Path("datasets/interactions/libero_spatial_interactions.hdf5"),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tmp/sample_interactions.mp4"),
    )
    parser.add_argument(
        "--n-demos",
        type=int,
        default=6,
        help="Max number of demos to include (grouped by demo).",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--panel-height", type=int, default=256)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read the requested interaction records from the HDF5 file.
    # ------------------------------------------------------------------
    raw_by_demo: dict[tuple, list[dict]] = defaultdict(list)

    with h5py.File(args.interactions, "r") as f:
        total = int(f.attrs["total_interactions"])
        all_keys = sorted(f["interactions"].keys(), key=int)

        # Group all interaction keys by (source_file, demo_key)
        demo_to_keys: dict[tuple, list[str]] = defaultdict(list)
        for k in all_keys:
            grp = f["interactions"][k]
            demo_to_keys[(grp.attrs["source_file"], grp.attrs["demo_key"])].append(k)

        # Randomly pick demos (not individual interactions) until we have
        # at least n_demos interactions, so every demo shows all its rows.
        demo_list = list(demo_to_keys.keys())
        random.shuffle(demo_list)
        selected_keys: list[str] = []
        for demo in demo_list[: args.n_demos]:
            selected_keys.extend(demo_to_keys[demo])

        print(
            f"Loading {len(selected_keys)} / {total} interactions from {args.interactions}"
        )

        for k in selected_keys:
            grp = f["interactions"][k]
            raw = _read_interaction_raw(grp)
            raw["_key"] = k
            demo_id = (raw["attrs"]["source_file"], raw["attrs"]["demo_key"])
            raw_by_demo[demo_id].append(raw)

    # ------------------------------------------------------------------
    # 2. For each demo: load obs data, precompute, assemble, write.
    # ------------------------------------------------------------------
    max_rows = max(len(raws) for raws in raw_by_demo.values())

    with imageio.get_writer(
        str(args.output), fps=args.fps, macro_block_size=1
    ) as writer:
        for (src_file, demo_key), raws in raw_by_demo.items():
            print(f"\n=== Demo: {demo_key}  ({len(raws)} interactions) ===")

            with h5py.File(src_file, "r") as src:
                obs = src["data"][demo_key]["obs"]
                av_rgb = obs["agentview_rgb"][:][:, ::-1, ::-1, :]
                eih_rgb = obs["eye_in_hand_rgb"][:][:, ::-1, ::-1, :]
                av_K = obs["agentview_intrinsic"][:]
                av_E = obs["agentview_extrinsic"][:]
                eih_K = obs["eye_in_hand_intrinsic"][:]
                eih_E = obs["eye_in_hand_extrinsic"][:]
                total_steps = int(av_rgb.shape[0])
                # Segmentation masks — same spatial flip as RGB
                av_seg = (
                    obs["agentview_segmentation"][:][:, ::-1, ::-1, :]
                    if "agentview_segmentation" in obs
                    else None
                )
                eih_seg = (
                    obs["eye_in_hand_segmentation"][:][:, ::-1, ::-1, :]
                    if "eye_in_hand_segmentation" in obs
                    else None
                )

            interactions: list[dict] = []
            for raw in raws:
                attrs = raw["attrs"]
                print(
                    f"  [{raw['_key']}] {attrs['object_name']}  "
                    f"t=[{attrs['start_timestep']},{attrs['end_timestep']})"
                )
                idata = precompute_interaction(
                    raw,
                    av_rgb,
                    eih_rgb,
                    av_K,
                    av_E,
                    eih_K,
                    eih_E,
                    args.panel_height,
                    av_seg=av_seg,
                    eih_seg=eih_seg,
                )
                interactions.append(idata)

            t_start = min(r["attrs"]["start_timestep"] for r in raws)
            t_end = max(r["attrs"]["end_timestep"] for r in raws)
            print(
                f"  Building frames t=[{t_start},{t_end}) with {len(interactions)} rows each…",
                flush=True,
            )
            demo_frames = build_demo_frames(
                interactions, total_steps, args.panel_height, total_rows=max_rows
            )

            if not demo_frames:
                continue

            # Title card
            frame_w = demo_frames[0].shape[1]
            frame_h = demo_frames[0].shape[0]
            dataset_name = Path(src_file).stem
            title = f"{dataset_name}  |  {demo_key}\n{len(interactions)} interactions  |  {total_steps} steps"
            for fr in _make_title_card(title, frame_w, frame_h, args.fps):
                writer.append_data(fr)
            for fr in demo_frames:
                writer.append_data(fr)

    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

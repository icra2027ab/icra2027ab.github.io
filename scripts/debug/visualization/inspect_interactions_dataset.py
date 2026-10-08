"""Inspect extracted gripper-object interactions as a sample video.

For each demo that appears in the interaction file, the 3 longest interactions
are selected and stacked **vertically** into a single composite frame so they
play simultaneously.  Each row keeps the existing layout:

    agentview RGB | eye-in-hand RGB | distance plot

1st-order interactions (gripper ↔ object):
  Finger-tip dots (RGB):
    • left  finger → green dot  (0, 255, 0)
    • right finger → blue  dot  (0,   0, 255)
  Closest-point crosses on object surface:
    • closest to left  finger → green cross
    • closest to right finger → blue  cross
  Distance plot: left_finger_dist (blue) and right_finger_dist (green)

2nd-order interactions (gripper → primary → secondary):
  Mask overlays: primary object (cyan) and secondary object (yellow)
  Closest-point crosses: point on primary nearest secondary (cyan)
                         and point on secondary nearest primary (orange)
  Distance plot: gripper→primary dist (blue) and primary→secondary dist (orange)

Usage:
    uv run scripts/debug/visualization/inspect_interactions_dataset.py \\
        --interactions tmp/test_interactions.hdf5 \\
        --output tmp/sample_interactions.mp4 \\
        --fps 20
"""

from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from policy_abstraction.utils.pcd import project_points_to_pixels
from policy_abstraction.utils.segmentation import erode_segmentation

# ---------------------------------------------------------------------------
# Colour constants (RGB)
# ---------------------------------------------------------------------------

LEFT_DOT_COLOR = (0, 255, 0)  # green  – left finger tip dot
RIGHT_DOT_COLOR = (0, 0, 255)  # blue   – right finger tip dot
LEFT_CLOSEST_COLOR = (0, 255, 0)  # green  – closest point to left  finger
RIGHT_CLOSEST_COLOR = (0, 0, 255)  # blue   – closest point to right finger

LEFT_CHART_COLOR = "#4499ff"  # blue
RIGHT_CHART_COLOR = "#00cc44"  # green

# 2nd-order colours
PRIMARY_MASK_COLOR = (0, 220, 220)  # cyan   – primary object highlight
SECONDARY_MASK_COLOR = (255, 220, 0)  # yellow – secondary object highlight
CLOSEST_ON_PRIMARY_COLOR = (0, 200, 200)  # cyan   – cross on primary surface
CLOSEST_ON_SECOND_COLOR = (255, 140, 0)  # orange – cross on secondary surface
PRIMARY_CHART_COLOR = "#4499ff"  # blue   – gripper→primary dist
SECOND_CHART_COLOR = "#ff8800"  # orange – primary→secondary dist


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
    for color, is_valid, pt in zip(
        [LEFT_DOT_COLOR, RIGHT_DOT_COLOR], valid, pts, strict=False
    ):
        if not is_valid:
            continue
        u, v = int(pt[0]), int(pt[1])
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(out, (u, v), 7, color, -1)
            cv2.circle(out, (u, v), 8, (0, 0, 0), 1)
    return out


def _draw_closest_cross(
    img: np.ndarray,
    point_3d: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    color: tuple,
) -> np.ndarray:
    if np.any(np.isnan(point_3d)):
        return img
    H, W = img.shape[:2]
    out = img.copy()
    pts, valid = project_points_to_pixels(point_3d[None], K, E, H, W, preprocessed=True)
    if valid[0]:
        u, v = int(pts[0, 0]), int(pts[0, 1])
        if 0 <= u < W and 0 <= v < H:
            cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    return out


def _overlay_seg_mask(
    img: np.ndarray,  # (H, W, 3) RGB uint8
    seg: np.ndarray,  # (H, W) or (H, W, 1) int32 – already at display resolution
    object_id: int,
    color: tuple = (255, 220, 0),
    alpha: float = 0.40,
) -> np.ndarray:
    """Blend a semi-transparent highlight over pixels belonging to *object_id*."""
    seg2d = seg[..., 0] if seg.ndim == 3 else seg
    mask = erode_segmentation(seg2d) == object_id
    if not mask.any():
        return img
    overlay = img.copy()
    overlay[mask] = (
        (img[mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return overlay


def _put_text(
    img: np.ndarray,
    text: str,
    pos: tuple,
    scale: float = 0.50,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
) -> None:
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


# ---------------------------------------------------------------------------
# Distance plot
# ---------------------------------------------------------------------------


def _render_distance_plot(
    series_a: np.ndarray,
    label_a: str,
    color_a: str,
    series_b: np.ndarray,
    label_b: str,
    color_b: str,
    cursor_idx: int,
    width: int,
    height: int,
    hlines: list[tuple[float, str, str]] | None = None,
) -> np.ndarray:
    """Generic two-series distance plot.

    Args:
        hlines: list of (value_cm, color, label) tuples for horizontal threshold lines.
    """
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    N = len(series_a)
    xs = np.arange(N)

    a = np.where(np.isfinite(series_a), series_a, np.nan)
    b = np.where(np.isfinite(series_b), series_b, np.nan)

    ax.plot(xs, a * 100, color=color_a, linewidth=1.5, label=label_a)
    ax.plot(xs, b * 100, color=color_b, linewidth=1.5, label=label_b)

    if hlines:
        for val_cm, hcolor, hlabel in hlines:
            ax.axhline(
                val_cm,
                color=hcolor,
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
                label=hlabel,
            )

    ax.axvline(cursor_idx, color="white", linewidth=1.0, alpha=0.9)

    ax.set_xlim(0, max(N - 1, 1))
    valid = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    ymax = max(10.0, float(valid.max()) * 100 * 1.15) if len(valid) else 10.0
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
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Single-interaction row builder
# ---------------------------------------------------------------------------


def _load_interaction_data(grp) -> dict:
    attrs = dict(grp.attrs)
    order = int(attrs.get("interaction_order", 1))
    base = {
        "order": order,
        "source_file": attrs["source_file"],
        "demo_key": attrs["demo_key"],
        "object_name": attrs["object_name"],
        "object_id": int(attrs["object_id"]),
        "interaction_chain": attrs.get("interaction_chain", attrs["object_name"]),
        "start_t": int(attrs["start_timestep"]),
        "end_t": int(attrs["end_timestep"]),
        "timesteps": grp["timesteps"][:],
        "left_pos": grp["left_finger_pos"][:],
        "right_pos": grp["right_finger_pos"][:],
    }
    if order == 1:
        base.update(
            {
                "left_dists": grp["left_finger_dist"][:],
                "right_dists": grp["right_finger_dist"][:],
                "left_closest": grp["left_closest_point"][:],
                "right_closest": grp["right_closest_point"][:],
            }
        )
    else:
        base.update(
            {
                "via_object_id": int(attrs["via_object_id"]),
                "via_object_name": attrs["via_object_name"],
                "gripper_to_primary_dist": grp["gripper_to_primary_dist"][:],
                "primary_to_sec_dist": grp["primary_to_secondary_dist"][:],
                "closest_on_primary": grp["closest_on_primary"][:],
                "closest_on_secondary": grp["closest_on_secondary"][:],
            }
        )
    return base


def _build_row(
    ix: dict,  # interaction data dict
    frame_idx: int,  # index into the interaction's timestep array
    obs_cache: dict,  # pre-loaded obs arrays keyed by source_file+demo_key
    panel_h: int,
    panel_w: int,
    dist_plot_cache: list[np.ndarray],
) -> np.ndarray:
    """Render one horizontal row for frame_idx of a single interaction."""
    t = int(ix["timesteps"][frame_idx])
    cache = obs_cache[(ix["source_file"], ix["demo_key"])]

    av_frame = cv2.resize(
        cache["av_rgb"][t], (panel_w, panel_h), interpolation=cv2.INTER_AREA
    )
    eih_frame = cv2.resize(
        cache["eih_rgb"][t], (panel_w, panel_h), interpolation=cv2.INTER_AREA
    )

    av_seg_t = cv2.resize(
        cache["av_seg"][t].astype(np.int32),
        (panel_w, panel_h),
        interpolation=cv2.INTER_NEAREST,
    )
    eih_seg_t = cv2.resize(
        cache["eih_seg"][t].astype(np.int32),
        (panel_w, panel_h),
        interpolation=cv2.INTER_NEAREST,
    )

    if ix["order"] == 1:
        # Highlight the interacting object
        av_frame = _overlay_seg_mask(av_frame, av_seg_t, ix["object_id"])
        eih_frame = _overlay_seg_mask(eih_frame, eih_seg_t, ix["object_id"])

        # Finger-tip dots
        av_frame = _draw_finger_dots(
            av_frame,
            ix["left_pos"][frame_idx],
            ix["right_pos"][frame_idx],
            cache["av_K"][t],
            cache["av_E"][t],
        )
        eih_frame = _draw_finger_dots(
            eih_frame,
            ix["left_pos"][frame_idx],
            ix["right_pos"][frame_idx],
            cache["eih_K"][t],
            cache["eih_E"][t],
        )

        # Closest-point crosses (left→green, right→blue)
        av_frame = _draw_closest_cross(
            av_frame,
            ix["left_closest"][frame_idx],
            cache["av_K"][t],
            cache["av_E"][t],
            LEFT_CLOSEST_COLOR,
        )
        av_frame = _draw_closest_cross(
            av_frame,
            ix["right_closest"][frame_idx],
            cache["av_K"][t],
            cache["av_E"][t],
            RIGHT_CLOSEST_COLOR,
        )
        eih_frame = _draw_closest_cross(
            eih_frame,
            ix["left_closest"][frame_idx],
            cache["eih_K"][t],
            cache["eih_E"][t],
            LEFT_CLOSEST_COLOR,
        )
        eih_frame = _draw_closest_cross(
            eih_frame,
            ix["right_closest"][frame_idx],
            cache["eih_K"][t],
            cache["eih_E"][t],
            RIGHT_CLOSEST_COLOR,
        )

        ld = ix["left_dists"][frame_idx]
        rd = ix["right_dists"][frame_idx]
        min_d = min(
            ld if np.isfinite(ld) else np.inf, rd if np.isfinite(rd) else np.inf
        )
        dist_str = (
            f"grip→obj={min_d * 100:.1f}cm" if np.isfinite(min_d) else "grip→obj=?"
        )

    else:  # order == 2
        # Highlight primary (cyan) and secondary (yellow) objects
        av_frame = _overlay_seg_mask(
            av_frame,
            av_seg_t,
            ix["via_object_id"],
            color=PRIMARY_MASK_COLOR,
            alpha=0.35,
        )
        eih_frame = _overlay_seg_mask(
            eih_frame,
            eih_seg_t,
            ix["via_object_id"],
            color=PRIMARY_MASK_COLOR,
            alpha=0.35,
        )
        av_frame = _overlay_seg_mask(
            av_frame, av_seg_t, ix["object_id"], color=SECONDARY_MASK_COLOR, alpha=0.35
        )
        eih_frame = _overlay_seg_mask(
            eih_frame,
            eih_seg_t,
            ix["object_id"],
            color=SECONDARY_MASK_COLOR,
            alpha=0.35,
        )

        # Finger-tip dots (gripper is still the root of the chain)
        av_frame = _draw_finger_dots(
            av_frame,
            ix["left_pos"][frame_idx],
            ix["right_pos"][frame_idx],
            cache["av_K"][t],
            cache["av_E"][t],
        )
        eih_frame = _draw_finger_dots(
            eih_frame,
            ix["left_pos"][frame_idx],
            ix["right_pos"][frame_idx],
            cache["eih_K"][t],
            cache["eih_E"][t],
        )

        # Closest-point crosses: primary surface (cyan) and secondary surface (orange)
        av_frame = _draw_closest_cross(
            av_frame,
            ix["closest_on_primary"][frame_idx],
            cache["av_K"][t],
            cache["av_E"][t],
            CLOSEST_ON_PRIMARY_COLOR,
        )
        av_frame = _draw_closest_cross(
            av_frame,
            ix["closest_on_secondary"][frame_idx],
            cache["av_K"][t],
            cache["av_E"][t],
            CLOSEST_ON_SECOND_COLOR,
        )
        eih_frame = _draw_closest_cross(
            eih_frame,
            ix["closest_on_primary"][frame_idx],
            cache["eih_K"][t],
            cache["eih_E"][t],
            CLOSEST_ON_PRIMARY_COLOR,
        )
        eih_frame = _draw_closest_cross(
            eih_frame,
            ix["closest_on_secondary"][frame_idx],
            cache["eih_K"][t],
            cache["eih_E"][t],
            CLOSEST_ON_SECOND_COLOR,
        )

        gp = ix["gripper_to_primary_dist"][frame_idx]
        ps = ix["primary_to_sec_dist"][frame_idx]
        gp_str = f"{gp * 100:.1f}" if np.isfinite(gp) else "?"
        ps_str = f"{ps * 100:.1f}" if np.isfinite(ps) else "?"
        dist_str = f"grip→pri={gp_str}cm  pri→sec={ps_str}cm"

    _put_text(av_frame, f"t={t}  {dist_str}", (6, 18))
    _put_text(eih_frame, "eye-in-hand", (6, 18))

    row = np.concatenate([av_frame, eih_frame, dist_plot_cache[frame_idx]], axis=1)

    # Banner: show interaction chain + demo info
    label_h = 28
    banner = np.zeros((label_h, row.shape[1], 3), dtype=np.uint8)
    order_tag = "[2nd]" if ix["order"] == 2 else "[1st]"
    banner[:] = (55, 30, 30) if ix["order"] == 2 else (30, 40, 55)
    label = f"{order_tag} {ix['interaction_chain']}  |  {ix['demo_key']}  |  t=[{ix['start_t']}, {ix['end_t']})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    x = max(6, (banner.shape[1] - tw) // 2)
    y = (label_h + th) // 2
    banner_color = (255, 160, 80) if ix["order"] == 2 else (255, 220, 80)
    _put_text(banner, label, (x, y), scale=0.50, color=banner_color)

    return np.concatenate([banner, row], axis=0)


LABEL_BANNER_H = 28  # must match the label_h constant in _build_row


def _blank_row(panel_h: int, row_w: int) -> np.ndarray:
    total_h = LABEL_BANNER_H + panel_h
    blank = np.zeros((total_h, row_w, 3), dtype=np.uint8)
    _put_text(blank, "interaction ended", (row_w // 4, total_h // 2))
    return blank


# ---------------------------------------------------------------------------
# Title card
# ---------------------------------------------------------------------------


def _make_title_card(
    text: str,
    width: int,
    height: int,
    fps: int,
    duration_s: float = 0.6,
) -> list[np.ndarray]:
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (25, 25, 50)
    lines = text.split("\n")
    y = height // 2 - (len(lines) - 1) * 22
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        x = max(8, (width - tw) // 2)
        _put_text(card, line, (x, y), scale=0.7)
        y += 40
    return [card] * max(1, int(fps * duration_s))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect interaction dataset: 3 longest interactions per demo, stacked vertically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interactions", "-i", type=Path, default=Path("tmp/test_interactions.hdf5")
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("tmp/sample_interactions.mp4")
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--panel-height", type=int, default=256)
    parser.add_argument(
        "--top-k", type=int, default=10, help="Longest interactions per demo to stack."
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all interactions and group by (source_file, demo_key)
    # ------------------------------------------------------------------
    demo_groups: dict[tuple, list[dict]] = defaultdict(list)

    with h5py.File(args.interactions, "r") as f:
        for k in sorted(f["interactions"].keys(), key=int):
            ix = _load_interaction_data(f["interactions"][k])
            demo_groups[(ix["source_file"], ix["demo_key"])].append(ix)

    # Sort each group by length descending, keep top-k
    for key in demo_groups:
        demo_groups[key].sort(key=lambda x: len(x["timesteps"]), reverse=True)
        demo_groups[key] = demo_groups[key][: args.top_k]

    # ------------------------------------------------------------------
    # Pre-load obs arrays per (source_file, demo_key)
    # ------------------------------------------------------------------
    print("Pre-loading observations…")
    obs_cache: dict[tuple, dict] = {}
    for (src_file, demo_key), _ in demo_groups.items():
        if (src_file, demo_key) in obs_cache:
            continue
        print(f"  {Path(src_file).name} / {demo_key}")
        with h5py.File(src_file, "r") as f:
            obs = f["data"][demo_key]["obs"]
            obs_cache[(src_file, demo_key)] = {
                "av_rgb": obs["agentview_rgb"][:][:, ::-1, ::-1, :],
                "eih_rgb": obs["eye_in_hand_rgb"][:][:, ::-1, ::-1, :],
                "av_seg": obs["agentview_segmentation"][:][:, ::-1, ::-1],
                "eih_seg": obs["eye_in_hand_segmentation"][:][:, ::-1, ::-1],
                "av_K": obs["agentview_intrinsic"][:],
                "av_E": obs["agentview_extrinsic"][:],
                "eih_K": obs["eye_in_hand_intrinsic"][:],
                "eih_E": obs["eye_in_hand_extrinsic"][:],
            }

    # ------------------------------------------------------------------
    # Compute layout dimensions from the first loaded frame
    # ------------------------------------------------------------------
    first_cache = next(iter(obs_cache.values()))
    H_src, W_src = first_cache["av_rgb"].shape[1:3]
    scale = args.panel_height / H_src
    panel_w = int(W_src * scale)
    plot_w = panel_w  # same width as one camera panel
    row_w = panel_w * 2 + plot_w
    row_h = args.panel_height + LABEL_BANNER_H  # camera panels + name banner

    # Fixed canvas height across all groups (use max n_rows so frames are uniform)
    max_n_rows = max(len(ixs) for ixs in demo_groups.values())
    canvas_h = row_h * max_n_rows
    canvas_w = row_w

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    print(f"\nRendering {len(demo_groups)} demo group(s) → {args.output}")

    with imageio.get_writer(
        str(args.output), fps=args.fps, macro_block_size=1
    ) as writer:
        for (src_file, demo_key), interactions in demo_groups.items():
            n_rows = len(interactions)
            clip_h = canvas_h
            clip_w = canvas_w

            title_text = (
                f"Demo: {demo_key}\n"
                f"{Path(src_file).stem}\n"
                f"Top-{n_rows} longest interactions"
            )
            for card in _make_title_card(title_text, clip_w, clip_h, args.fps):
                writer.append_data(card)

            # Pre-render distance plots for each interaction
            print(f"\n  [{demo_key}] Pre-rendering distance plots…")
            dist_plot_caches: list[list[np.ndarray]] = []
            for ix in interactions:
                N = len(ix["timesteps"])
                plots = []
                if ix["order"] == 1:
                    hlines = [
                        (5.0, "yellow", "start 5 cm"),
                        (10.0, "orange", "stop 10 cm"),
                    ]
                    for i in range(N):
                        plots.append(
                            _render_distance_plot(
                                ix["left_dists"],
                                "left (cm)",
                                LEFT_CHART_COLOR,
                                ix["right_dists"],
                                "right (cm)",
                                RIGHT_CHART_COLOR,
                                i,
                                plot_w,
                                args.panel_height,
                                hlines=hlines,
                            )
                        )
                else:
                    hlines = [
                        (5.0, "yellow", "start 5 cm"),
                        (10.0, "orange", "stop 10 cm"),
                    ]
                    for i in range(N):
                        plots.append(
                            _render_distance_plot(
                                ix["gripper_to_primary_dist"],
                                "grip→primary (cm)",
                                PRIMARY_CHART_COLOR,
                                ix["primary_to_sec_dist"],
                                "primary→sec (cm)",
                                SECOND_CHART_COLOR,
                                i,
                                plot_w,
                                args.panel_height,
                                hlines=hlines,
                            )
                        )
                dist_plot_caches.append(plots)
                print(f"    {ix['interaction_chain']}: {N} frames rendered")

            # Build composite frames aligned to the global timestep range
            ts_to_fi = [
                {int(ix["timesteps"][fi]): fi for fi in range(len(ix["timesteps"]))}
                for ix in interactions
            ]
            global_start = min(ix["start_t"] for ix in interactions)
            global_end = max(ix["end_t"] for ix in interactions)
            total_frames = global_end - global_start
            print(
                f"  Compositing t=[{global_start}, {global_end}) ({total_frames} frames)…",
                end="",
                flush=True,
            )

            for t in range(global_start, global_end):
                rows: list[np.ndarray] = []
                for ix, dist_plots, lookup in zip(
                    interactions, dist_plot_caches, ts_to_fi, strict=False
                ):
                    if t in lookup:
                        fi = lookup[t]
                        row = _build_row(
                            ix, fi, obs_cache, args.panel_height, panel_w, dist_plots
                        )
                    else:
                        row = _blank_row(args.panel_height, row_w)
                    rows.append(row)

                composite = np.concatenate(rows, axis=0)
                # Pad to canvas_h if this group has fewer rows than max_n_rows
                if composite.shape[0] < canvas_h:
                    pad = np.zeros(
                        (canvas_h - composite.shape[0], canvas_w, 3), dtype=np.uint8
                    )
                    composite = np.concatenate([composite, pad], axis=0)
                writer.append_data(composite)

            print(" done.")

    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

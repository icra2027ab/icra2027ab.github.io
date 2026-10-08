"""Alignment multiplot: live rollout and reference overview.

Produces a 4-subplot image each timestep using OpenCV for fast rendering
(~10-30ms vs ~300ms with matplotlib):

1. **Live Rollout** (top): Live distance measurements plotted as they
   accumulate.  Selected reference chunks are overlaid (dimmer, start marker).
   Background painted by the active sub-trajectory color.
   Vertical time bar at the current live timestep.

2. **Reference Overview**: Full reference distance features vs
   reference time with std band.  Background colored by sub-trajectory.
   Currently selected chunks are highlighted per feature (thicker, brighter,
   markers).  A vertical time bar shows ``aligned_t``.

3. **Temporal Guidance Factors**: Per-pair guidance weights over time.

4. **HL Relevances** (bottom): Per-object relevance scores over time.

The rendered image is returned as a ``[H, W, 3]`` uint8 numpy array suitable
for logging to Rerun and appending to the output video frame.
"""

from __future__ import annotations

import colorsys
import hashlib

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour helpers (matching RerunVisualizer._series_color_for_label)
# ---------------------------------------------------------------------------


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Deterministic high-saturation BGR tuple in [0, 255] for *label*."""
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    sat = 0.72 + 0.18 * (digest[1] / 255.0)
    val = 0.88 + 0.10 * (digest[2] / 255.0)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for cv2


def _dim_color(c: tuple[int, int, int], factor: float = 0.55) -> tuple[int, int, int]:
    """Return a dimmed version of *c* (BGR)."""
    return (int(c[0] * factor), int(c[1] * factor), int(c[2] * factor))


def _subtraj_color_bgr(idx: int) -> tuple[int, int, int]:
    """Background colour for sub-trajectory *idx* (BGR, pastel)."""
    hue = (idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.20, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Subplot region helper
# ---------------------------------------------------------------------------


class _SubplotRegion:
    """Defines a rectangular pixel region within the canvas and handles
    data-to-pixel coordinate mapping."""

    __slots__ = ("x0", "y0", "w", "h", "x_min", "x_max", "y_min", "y_max")

    def __init__(self, x0: int, y0: int, w: int, h: int) -> None:
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0

    def set_xlim(self, lo: float, hi: float) -> None:
        self.x_min = lo
        self.x_max = hi if hi != lo else lo + 1.0

    def set_ylim(self, lo: float, hi: float) -> None:
        self.y_min = lo
        self.y_max = hi if hi != lo else lo + 1.0

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        """Map data coords to canvas pixel coords."""
        px = self.x0 + int((x - self.x_min) / (self.x_max - self.x_min) * self.w)
        # y is inverted (pixel y increases downward)
        py = (
            self.y0
            + self.h
            - int((y - self.y_min) / (self.y_max - self.y_min) * self.h)
        )
        return (px, py)

    def to_px_array(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorized data-to-pixel conversion. Returns [N, 2] int32 (x, y)."""
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        px_f = self.x0 + (xs - self.x_min) / x_range * self.w
        py_f = self.y0 + self.h - (ys - self.y_min) / y_range * self.h
        px = np.nan_to_num(px_f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
        py = np.nan_to_num(py_f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
        return np.column_stack([px, py])

    def fill_band_x(
        self,
        img: np.ndarray,
        x_start: float,
        x_end: float,
        color: tuple[int, int, int],
        alpha: float = 0.35,
    ) -> None:
        """Draw a vertical background band (x range) with alpha blending."""
        px0 = max(
            self.x0,
            self.x0 + int((x_start - self.x_min) / (self.x_max - self.x_min) * self.w),
        )
        px1 = min(
            self.x0 + self.w,
            self.x0 + int((x_end - self.x_min) / (self.x_max - self.x_min) * self.w),
        )
        if px1 <= px0:
            return
        roi = img[self.y0 : self.y0 + self.h, px0:px1]
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
    alpha: float = 1.0,
) -> None:
    """Draw a polyline. If alpha < 1, blend onto img."""
    if len(pts) < 2:
        return
    pts_cv = pts.reshape(-1, 1, 2).astype(np.int32)
    if alpha >= 0.99:
        cv2.polylines(
            img,
            [pts_cv],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        # Only blend in the bounding box region for speed
        x_coords, y_coords = pts[:, 0], pts[:, 1]
        pad = thickness + 1
        bx0 = max(0, int(x_coords.min()) - pad)
        by0 = max(0, int(y_coords.min()) - pad)
        bx1 = min(img.shape[1], int(x_coords.max()) + pad)
        by1 = min(img.shape[0], int(y_coords.max()) + pad)
        if bx1 <= bx0 or by1 <= by0:
            return
        roi = img[by0:by1, bx0:bx1]
        overlay = roi.copy()
        shifted_pts = pts.copy()
        shifted_pts[:, 0] -= bx0
        shifted_pts[:, 1] -= by0
        pts_cv = shifted_pts.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(
            overlay,
            [pts_cv],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_rotated_text(
    img: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    font: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Draw *text* rotated 90° counter-clockwise, centred at (*cx*, *cy*)."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    # Render text into a small horizontal canvas, then rotate
    pad = 2
    tmp = np.full((th + 2 * pad, tw + 2 * pad, 3), 255, dtype=np.uint8)
    cv2.putText(tmp, text, (pad, th + pad), font, scale, color, 1, cv2.LINE_AA)
    # Rotate 90° CCW
    rotated = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rh, rw = rotated.shape[:2]
    # Compute placement so the rotated text is centred at (cx, cy)
    x0 = cx - rw // 2
    y0 = cy - rh // 2
    # Clip to image bounds
    sx = max(0, -x0)
    sy = max(0, -y0)
    ex = min(rw, img.shape[1] - x0)
    ey = min(rh, img.shape[0] - y0)
    if ex <= sx or ey <= sy:
        return
    # White (255) pixels in the rotated image are background — use them as mask
    roi = img[y0 + sy : y0 + ey, x0 + sx : x0 + ex]
    src = rotated[sy:ey, sx:ex]
    mask = np.any(src < 250, axis=2)
    roi[mask] = src[mask]


# ---------------------------------------------------------------------------
# AlignmentPlotter
# ---------------------------------------------------------------------------


class AlignmentPlotter:
    """Renders a 4-subplot alignment multiplot, accumulating state over time.

    Parameters
    ----------
    ref_traj:
        ``[T_ref, n_pairs]`` float32 reference distance trajectory.
    pair_labels:
        Ordered list of pair label strings (length ``n_pairs``).
    subtrajectories:
        List of ``(start, end)`` index pairs (end exclusive) defining
        sub-trajectory segments in the reference trajectory.
    pair_subtraj_idx:
        Per-pair sub-trajectory index (length ``n_pairs``).  ``None`` entries
        mean the feature is relevant to all sub-trajectories.
    ref_std:
        ``[T_ref, n_pairs]`` optional per-pair uncertainty from multi-demo
        alignment.
    fig_width:
        Figure width in inches (used with dpi to compute pixel width).
    fig_height:
        Figure height in inches (used with dpi to compute pixel height).
    dpi:
        Render resolution.
    """

    # Layout constants
    _MARGIN_LEFT = 100
    _MARGIN_RIGHT = 20
    _MARGIN_TOP = 14
    _MARGIN_BOTTOM = 24
    _GAP = 36  # vertical gap between subplots
    _TITLE_H = 24  # space for title text above each subplot
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

    def __init__(
        self,
        ref_traj: np.ndarray,
        pair_labels: list[str],
        subtrajectories: list[tuple[int, int]],
        pair_subtraj_idx: list[int | None] | None = None,
        ref_std: np.ndarray | None = None,
        fig_width: float = 10,
        fig_height: float = 5,
        dpi: int = 100,
    ) -> None:
        self._ref_traj = ref_traj.astype(np.float32)
        self._ref_std = ref_std.astype(np.float32) if ref_std is not None else None
        self._pair_labels = list(pair_labels)
        self._subtrajectories = list(subtrajectories)
        self._n_pairs = len(pair_labels)
        self._T_ref = ref_traj.shape[0]

        # Canvas dimensions
        self._img_w = int(fig_width * dpi)
        self._img_h = int(fig_height * 1.7 * dpi)

        # Per-pair sub-trajectory assignment (None = all)
        self._pair_subtraj_idx: list[int | None] = (
            list(pair_subtraj_idx)
            if pair_subtraj_idx is not None
            else [None] * self._n_pairs
        )

        # Build subtraj → feature indices mapping
        self._subtraj_feature_map: dict[int, list[int]] = {}
        for si in range(len(subtrajectories)):
            self._subtraj_feature_map[si] = [
                i
                for i, st in enumerate(self._pair_subtraj_idx)
                if st is None or st == si
            ]

        # Per-label colour cache (BGR)
        self._colors = {lbl: _color_for_label(lbl) for lbl in pair_labels}

        # Pre-compute subplot regions (height_ratios=[2, 2, 1, 1])
        plot_w = self._img_w - self._MARGIN_LEFT - self._MARGIN_RIGHT
        total_plot_h = (
            self._img_h
            - self._MARGIN_TOP
            - self._MARGIN_BOTTOM
            - 3 * self._GAP
            - 4 * self._TITLE_H
        )
        # Heights: 2:2:1:1 ratio
        unit = total_plot_h / 6.0
        heights = [int(2 * unit), int(2 * unit), int(unit), int(unit)]
        # Adjust last to absorb rounding
        heights[-1] = total_plot_h - sum(heights[:-1])

        self._regions: list[_SubplotRegion] = []
        y_cursor = self._MARGIN_TOP
        for h in heights:
            y_cursor += self._TITLE_H
            r = _SubplotRegion(self._MARGIN_LEFT, y_cursor, plot_w, h)
            self._regions.append(r)
            y_cursor += h + self._GAP

        # Accumulated live data
        self._live_history: list[np.ndarray] = []
        self._subtraj_history: list[int] = []
        self._chunk_history: list[tuple[int, np.ndarray, int, int]] = []
        self._current_chunk: tuple[np.ndarray, int, int] | None = None
        self._guidance_factor_history: list[np.ndarray] = []
        self._relevance_history: list[dict[str, float]] = []
        self._relevance_objects: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        timestep: int,
        live_distances: dict[str, float],
        aligned_t: int,
        current_subtraj_idx: int | None,
        chunk_size: int | None = None,
        aligned_ts: np.ndarray | None = None,
        is_replan: bool = False,
        guidance_weights: np.ndarray | None = None,
        relevances: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Record one timestep and return the rendered multiplot image.

        Returns
        -------
        np.ndarray
            ``[H, W, 3]`` uint8 RGB image.
        """
        # --- Accumulate live data ---
        vec = np.full(self._n_pairs, np.nan, dtype=np.float32)
        for i, lbl in enumerate(self._pair_labels):
            if lbl in live_distances:
                vec[i] = live_distances[lbl]
        while len(self._live_history) <= timestep:
            self._live_history.append(np.full(self._n_pairs, np.nan, dtype=np.float32))
        self._live_history[timestep] = vec

        while len(self._subtraj_history) <= timestep:
            self._subtraj_history.append(
                current_subtraj_idx if current_subtraj_idx is not None else 0
            )
        self._subtraj_history[timestep] = (
            current_subtraj_idx if current_subtraj_idx is not None else 0
        )

        # --- Accumulate temporal guidance factors ---
        gf_vec = np.full(self._n_pairs, np.nan, dtype=np.float32)
        if guidance_weights is not None and len(guidance_weights) == self._n_pairs:
            gf_vec = np.asarray(guidance_weights, dtype=np.float32)
        while len(self._guidance_factor_history) <= timestep:
            self._guidance_factor_history.append(
                np.full(self._n_pairs, np.nan, dtype=np.float32)
            )
        self._guidance_factor_history[timestep] = gf_vec

        # --- Accumulate per-object relevances ---
        rel = dict(relevances) if relevances else {}
        while len(self._relevance_history) <= timestep:
            self._relevance_history.append({})
        self._relevance_history[timestep] = rel
        for name in rel:
            if name not in self._relevance_objects:
                self._relevance_objects.append(name)

        # --- Update chunk info on replan ---
        if is_replan and chunk_size is not None:
            ref_starts = (
                aligned_ts.copy()
                if aligned_ts is not None
                else np.full(self._n_pairs, aligned_t, dtype=np.int32)
            )
            _si = current_subtraj_idx if current_subtraj_idx is not None else 0
            self._chunk_history.append((timestep, ref_starts.copy(), chunk_size, _si))
            self._current_chunk = (ref_starts.copy(), chunk_size, _si)

        return self._render(timestep, aligned_t)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, timestep: int, aligned_t: int) -> np.ndarray:
        """Render the 4-subplot figure and return as uint8 RGB array."""
        # White background
        img = np.full((self._img_h, self._img_w, 3), 255, dtype=np.uint8)

        n_live = len(self._live_history)
        x_end = float(max(self._T_ref, n_live, 1))

        r_live, r_ref, r_gf, r_rel = self._regions

        # Compute y-limits for distance plots
        all_vals = []
        if n_live > 0:
            arr = np.array(self._live_history[:n_live])
            valid = arr[~np.isnan(arr)]
            if len(valid):
                all_vals.append(valid)
        ref_valid = self._ref_traj[~np.isnan(self._ref_traj)]
        if len(ref_valid):
            all_vals.append(ref_valid)
        if all_vals:
            combined = np.concatenate(all_vals)
            y_lo = float(combined.min()) - 0.02
            y_hi = float(combined.max()) + 0.02
        else:
            y_lo, y_hi = -0.05, 1.05

        for r in (r_live, r_ref):
            r.set_xlim(0, x_end)
            r.set_ylim(y_lo, y_hi)
        r_gf.set_xlim(0, x_end)
        r_gf.set_ylim(-0.05, 1.1)
        r_rel.set_xlim(0, x_end)
        r_rel.set_ylim(-0.05, 1.1)

        # Draw borders and titles
        titles = [
            "Live Rollout",
            "Reference Distance Features",
            "Temporal Guidance Factors",
            "HL Relevances",
        ]
        for r, title in zip(self._regions, titles, strict=True):
            # Border
            cv2.rectangle(
                img, (r.x0, r.y0), (r.x0 + r.w, r.y0 + r.h), (180, 180, 180), 1
            )
            # Title
            cv2.putText(
                img,
                title,
                (r.x0, r.y0 - 4),
                self._FONT,
                0.65,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw y-axis labels (rotated 90° CCW)
        for r, ylabel in zip(
            self._regions, ["distance", "distance", "factor", "relevance"], strict=True
        ):
            _draw_rotated_text(
                img, ylabel, 10, r.y0 + r.h // 2, self._FONT_SMALL, 1.1, (80, 80, 80)
            )

        # Draw tick marks on y-axis
        for r in self._regions:
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                val = r.y_min + frac * (r.y_max - r.y_min)
                _, py = r.to_px(r.x_min, val)
                py = np.clip(py, r.y0, r.y0 + r.h)
                cv2.line(img, (r.x0 - 4, py), (r.x0, py), (150, 150, 150), 1)
                txt = f"{val:.2f}"
                cv2.putText(
                    img,
                    txt,
                    (r.x0 - 68, py + 4),
                    self._FONT_SMALL,
                    1.0,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA,
                )

        # Subplot contents
        self._draw_live_rollout(img, r_live, timestep, x_end)
        self._draw_ref_overview(img, r_ref, aligned_t, x_end)
        self._draw_guidance_factors(img, r_gf, timestep, x_end)
        self._draw_relevances(img, r_rel, timestep, x_end)

        # Draw legend at the top-right
        self._draw_legend(img, r_ref)

        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _draw_subtraj_bands(self, img: np.ndarray, r: _SubplotRegion) -> None:
        """Draw sub-trajectory background bands."""
        n_hist = len(self._subtraj_history)
        if n_hist == 0:
            return
        block_start = 0
        block_si = self._subtraj_history[0]
        for t_idx in range(1, n_hist):
            si = self._subtraj_history[t_idx]
            if si != block_si:
                r.fill_band_x(
                    img, block_start, t_idx, _subtraj_color_bgr(block_si), 0.30
                )
                block_start = t_idx
                block_si = si
        r.fill_band_x(img, block_start, n_hist, _subtraj_color_bgr(block_si), 0.30)

    def _draw_vline(
        self,
        img: np.ndarray,
        r: _SubplotRegion,
        x_val: float,
        color: tuple[int, int, int] = (160, 160, 160),
    ) -> None:
        """Draw a dashed vertical line at data x coordinate."""
        px, _ = r.to_px(x_val, 0)
        px = np.clip(px, r.x0, r.x0 + r.w)
        # Dashed: draw segments
        for y in range(r.y0, r.y0 + r.h, 6):
            y_end = min(y + 3, r.y0 + r.h)
            cv2.line(img, (px, y), (px, y_end), color, 1)

    # ------------------------------------------------------------------
    # Subplot 1: Live rollout
    # ------------------------------------------------------------------

    def _draw_live_rollout(
        self, img: np.ndarray, r: _SubplotRegion, timestep: int, x_end: float
    ) -> None:
        n_live = len(self._live_history)
        if n_live == 0:
            return

        # Background bands
        self._draw_subtraj_bands(img, r)

        # Reference chunks (accumulated)
        for live_t, ref_starts, cs, chunk_si in self._chunk_history:
            allowed = self._subtraj_feature_map.get(chunk_si, [])
            for i in allowed:
                lbl = self._pair_labels[i]
                c = _dim_color(self._colors[lbl], 0.55)
                rs = int(ref_starts[i])
                re = min(rs + cs, self._T_ref)
                seg_len = re - rs
                if seg_len <= 0:
                    continue
                xs = np.arange(live_t, live_t + seg_len, dtype=np.float32)
                ys = self._ref_traj[rs:re, i]
                pts = r.to_px_array(xs, ys)
                _draw_polyline(img, pts, c, thickness=1, alpha=0.55)

        # Live distance lines
        live_arr = np.array(self._live_history[:n_live])
        n_hist = len(self._subtraj_history)
        subtraj_arr = (
            np.array(self._subtraj_history[:n_hist], dtype=np.int32)
            if n_hist > 0
            else None
        )
        t_range = np.arange(n_live, dtype=np.float32)

        for i, lbl in enumerate(self._pair_labels):
            c = self._colors[lbl]
            fi_subtraj = self._pair_subtraj_idx[i]
            vals = live_arr[:, i].copy()
            if fi_subtraj is not None and subtraj_arr is not None:
                n_mask = min(n_live, n_hist)
                vals[:n_mask][subtraj_arr[:n_mask] != fi_subtraj] = np.nan

            # Draw contiguous non-NaN segments
            self._draw_line_segments(img, r, t_range, vals, c, thickness=2)

        # Time bar
        self._draw_vline(img, r, float(timestep))

    # ------------------------------------------------------------------
    # Subplot 2: Reference overview
    # ------------------------------------------------------------------

    def _draw_ref_overview(
        self, img: np.ndarray, r: _SubplotRegion, aligned_t: int, x_end: float
    ) -> None:
        t_axis = np.arange(self._T_ref, dtype=np.float32)

        # Background bands for sub-trajectories with features
        for si, (s, e) in enumerate(self._subtrajectories):
            r.fill_band_x(img, s, e, _subtraj_color_bgr(si), 0.35)

            feat_indices = self._subtraj_feature_map.get(si, [])
            xs = t_axis[s:e]
            for i in feat_indices:
                lbl = self._pair_labels[i]
                c = self._colors[lbl]
                ys = self._ref_traj[s:e, i]
                pts = r.to_px_array(xs, ys)
                _draw_polyline(img, pts, c, thickness=1, alpha=0.5)

                # Std band
                if self._ref_std is not None:
                    std = self._ref_std[s:e, i]
                    self._draw_std_band(img, r, xs, ys, std, c, alpha=0.12)

        # Highlight current chunk
        if self._current_chunk is not None:
            ref_starts, cs, chunk_si = self._current_chunk
            allowed = self._subtraj_feature_map.get(chunk_si, [])
            for i in allowed:
                lbl = self._pair_labels[i]
                c = self._colors[lbl]
                rs = int(ref_starts[i])
                re = min(rs + cs, self._T_ref)
                if re <= rs:
                    continue
                xs = np.arange(rs, re, dtype=np.float32)
                ys = self._ref_traj[rs:re, i]
                pts = r.to_px_array(xs, ys)
                _draw_polyline(img, pts, c, thickness=3)
                # Start/end markers
                if len(pts) > 0:
                    cv2.circle(img, tuple(pts[0]), 4, c, -1, cv2.LINE_AA)
                    cv2.rectangle(
                        img,
                        (pts[-1][0] - 2, pts[-1][1] - 2),
                        (pts[-1][0] + 2, pts[-1][1] + 2),
                        c,
                        -1,
                    )

        # Time bar
        self._draw_vline(img, r, float(aligned_t))

    # ------------------------------------------------------------------
    # Subplot 3: Temporal guidance factors
    # ------------------------------------------------------------------

    def _draw_guidance_factors(
        self, img: np.ndarray, r: _SubplotRegion, timestep: int, x_end: float
    ) -> None:
        n_gf = len(self._guidance_factor_history)
        if n_gf == 0:
            return

        self._draw_subtraj_bands(img, r)

        gf_arr = np.array(self._guidance_factor_history[:n_gf])
        n_hist = len(self._subtraj_history)
        subtraj_arr = (
            np.array(self._subtraj_history[:n_hist], dtype=np.int32)
            if n_hist > 0
            else None
        )
        t_range = np.arange(n_gf, dtype=np.float32)

        for i, lbl in enumerate(self._pair_labels):
            c = self._colors[lbl]
            fi_subtraj = self._pair_subtraj_idx[i]
            vals = gf_arr[:, i].copy()
            if fi_subtraj is not None and subtraj_arr is not None:
                n_mask = min(n_gf, n_hist)
                vals[:n_mask][subtraj_arr[:n_mask] != fi_subtraj] = np.nan
            self._draw_line_segments(img, r, t_range, vals, c, thickness=2)

        self._draw_vline(img, r, float(timestep))

    # ------------------------------------------------------------------
    # Subplot 4: Per-object relevances
    # ------------------------------------------------------------------

    def _draw_relevances(
        self, img: np.ndarray, r: _SubplotRegion, timestep: int, x_end: float
    ) -> None:
        n_rel = len(self._relevance_history)
        if n_rel == 0 or not self._relevance_objects:
            return

        self._draw_subtraj_bands(img, r)

        t_range = np.arange(n_rel, dtype=np.float32)
        history = self._relevance_history

        for obj_name in self._relevance_objects:
            c = _color_for_label(obj_name)
            vals = np.array(
                [d.get(obj_name, np.nan) for d in history[:n_rel]],
                dtype=np.float32,
            )
            self._draw_line_segments(img, r, t_range, vals, c, thickness=2)

        # Relevance legend (small, inside the subplot)
        y_leg = r.y0 + 16
        for obj_name in self._relevance_objects:
            c = _color_for_label(obj_name)
            cv2.rectangle(
                img,
                (r.x0 + r.w - 170, y_leg - 10),
                (r.x0 + r.w - 156, y_leg + 2),
                c,
                -1,
            )
            cv2.putText(
                img,
                obj_name,
                (r.x0 + r.w - 152, y_leg + 2),
                self._FONT_SMALL,
                1.0,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y_leg += 20

        self._draw_vline(img, r, float(timestep))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_line_segments(
        self,
        img: np.ndarray,
        r: _SubplotRegion,
        xs: np.ndarray,
        ys: np.ndarray,
        color: tuple[int, int, int],
        thickness: int = 1,
        alpha: float = 0.9,
    ) -> None:
        """Draw line, skipping NaN gaps (contiguous segments only)."""
        valid = ~np.isnan(ys)
        if not np.any(valid):
            return

        # Find contiguous runs of valid data
        diff = np.diff(valid.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if valid[0]:
            starts = np.concatenate([[0], starts])
        if valid[-1]:
            ends = np.concatenate([ends, [len(ys)]])

        for s, e in zip(starts, ends, strict=True):
            if e - s < 2:
                continue
            pts = r.to_px_array(xs[s:e], ys[s:e])
            _draw_polyline(img, pts, color, thickness=thickness, alpha=alpha)

    def _draw_std_band(
        self,
        img: np.ndarray,
        r: _SubplotRegion,
        xs: np.ndarray,
        ys: np.ndarray,
        std: np.ndarray,
        color: tuple[int, int, int],
        alpha: float = 0.12,
    ) -> None:
        """Draw a filled uncertainty band (mean +/- std)."""
        upper = ys + std
        lower = ys - std
        pts_upper = r.to_px_array(xs, upper)
        pts_lower = r.to_px_array(xs, lower)
        # Form a closed polygon: upper forward, lower reversed
        polygon = np.concatenate([pts_upper, pts_lower[::-1]], axis=0)
        if len(polygon) < 3:
            return
        # Compute bounding box for efficient blending
        px_min = max(0, polygon[:, 0].min() - 1)
        py_min = max(0, polygon[:, 1].min() - 1)
        px_max = min(img.shape[1], polygon[:, 0].max() + 1)
        py_max = min(img.shape[0], polygon[:, 1].max() + 1)
        if px_max <= px_min or py_max <= py_min:
            return
        roi = img[py_min:py_max, px_min:px_max]
        overlay = roi.copy()
        shifted = polygon.copy()
        shifted[:, 0] -= px_min
        shifted[:, 1] -= py_min
        cv2.fillPoly(overlay, [shifted.astype(np.int32)], color)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)

    def _draw_legend(self, img: np.ndarray, r: _SubplotRegion) -> None:
        """Draw a color legend for the pair labels."""
        x_start = r.x0 + r.w - 10
        y_start = r.y0 + 14

        # Measure max label width to compute box size
        max_w = 0
        entries = []
        for lbl in self._pair_labels:
            c = self._colors[lbl]
            (tw, th), _ = cv2.getTextSize(lbl, self._FONT, 0.55, 1)
            max_w = max(max_w, tw)
            entries.append((lbl, c, tw, th))

        box_w = max_w + 30
        box_h = len(entries) * 22 + 10
        bx0 = x_start - box_w
        by0 = y_start - 14

        # Semi-transparent background
        roi = img[
            max(0, by0) : min(img.shape[0], by0 + box_h),
            max(0, bx0) : min(img.shape[1], bx0 + box_w),
        ]
        if roi.size > 0:
            overlay = np.full_like(roi, (255, 255, 255), dtype=np.uint8)
            cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, dst=roi)

        y = y_start
        for lbl, c, _tw, _th in entries:
            # Color swatch
            cv2.rectangle(img, (bx0 + 4, y - 8), (bx0 + 16, y + 4), c, -1)
            # Label text
            cv2.putText(
                img, lbl, (bx0 + 22, y + 4), self._FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )
            y += 22

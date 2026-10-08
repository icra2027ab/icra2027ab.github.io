"""Overlay helpers for detections, tracked masks, and labels.

These are the small visualization utilities that every example script
reimplements: alpha-blend masks onto an RGB frame with a color cycle, and
optionally draw a text label with a black outline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .config import PerceptionConfig
    from .types import (
        CameraInfo,
        Detection,
        PerceptionOutput,
        PipelineSnapshot,
        ProjectedReference,
        TrackedFrame,
        TrackHistory,
    )


DEFAULT_COLORS = np.array(
    [[230, 60, 60], [60, 200, 90], [60, 100, 230], [230, 200, 60]],
    dtype=np.float32,
)


def overlay_masks(
    rgb: np.ndarray,
    masks: Iterable[np.ndarray | None] | Mapping[object, np.ndarray | None],
    *,
    annotations: Iterable[str | None] | Mapping[object, str | None] | None = None,
    colors: np.ndarray | None = None,
    alpha: float = 0.5,
    label: str | None = None,
) -> np.ndarray:
    """Alpha-blend boolean masks onto ``rgb`` using a color cycle.

    ``masks`` may be any iterable of bool arrays, or a mapping whose values
    are bool arrays (keys are ignored — call sites that want ordered output
    should pass ``dict(sorted(...))``). Entries that are ``None`` or all-False
    are skipped *without* shifting the color index, so callers can align
    colors to a fixed position (e.g. a query list) by padding with ``None``.

    ``annotations`` is an optional per-mask text overlay. Pass an iterable
    aligned to ``masks`` (same length) or, if ``masks`` is a mapping, another
    mapping keyed the same way. Empty/``None`` entries are skipped. Text is
    drawn at each mask's bounding-box top-left.

    ``label`` is a single frame-level caption drawn at ``(8, 8)``.
    """
    palette = DEFAULT_COLORS if colors is None else np.asarray(colors, dtype=np.float32)
    if isinstance(masks, Mapping):
        # keys and values from the same mapping are aligned.
        keys = list(masks.keys())
        mask_list = list(masks.values())
    else:
        keys = None
        mask_list = list(masks)

    ann_list = _resolve_annotations(annotations, mask_list, keys)

    vis = rgb.astype(np.float32).copy()
    text_items: list[tuple[str, tuple[int, int]]] = []
    for idx, (mask, ann) in enumerate(zip(mask_list, ann_list, strict=True)):
        if mask is None or not mask.any():
            continue
        color = palette[idx % len(palette)]
        vis[mask] = (1.0 - alpha) * vis[mask] + alpha * color
        if ann:
            anchor = _mask_anchor(mask)
            if anchor is not None:
                text_items.append((ann, anchor))
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    if label is not None:
        text_items.append((label, (8, 8)))
    if text_items:
        vis = _draw_texts(vis, text_items)
    return vis


def overlay_detections(
    rgb: np.ndarray,
    output: "PerceptionOutput",
    *,
    colors: np.ndarray | None = None,
    alpha: float = 0.5,
    label: str | None = None,
    show_id: bool = False,
    show_label: bool = False,
    show_score: bool = False,
) -> np.ndarray:
    """Overlay every :class:`Detection` in ``output`` — one color per query.

    Per-detection annotations can be enabled with the ``show_*`` flags:
        - ``show_id``    — ``det.instance_id`` (skipped if ``None``)
        - ``show_label`` — the query string
        - ``show_score`` — ``det.score`` formatted to 2 decimals

    Enabled fields are joined with spaces; id is wrapped in ``[]``. Text is
    drawn at each mask's bounding-box top-left. ``label`` is still the
    frame-level caption at ``(8, 8)``.
    """
    palette = DEFAULT_COLORS if colors is None else np.asarray(colors, dtype=np.float32)
    vis = rgb.astype(np.float32).copy()
    text_items: list[tuple[str, tuple[int, int]]] = []
    for idx, dets in enumerate(output.detections.values()):
        color = palette[idx % len(palette)]
        for det in dets:
            vis[det.mask] = (1.0 - alpha) * vis[det.mask] + alpha * color
            ann = _format_detection_annotation(
                det, show_id=show_id, show_label=show_label, show_score=show_score
            )
            if ann:
                anchor = _mask_anchor(det.mask)
                if anchor is not None:
                    text_items.append((ann, anchor))
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    if label is not None:
        text_items.append((label, (8, 8)))
    if text_items:
        vis = _draw_texts(vis, text_items)
    return vis


def draw_label(
    img: np.ndarray,
    text: str,
    *,
    xy: tuple[int, int] = (8, 8),
    fill: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Return a copy of ``img`` with ``text`` drawn at ``xy`` (outlined)."""
    return _draw_texts(img, [(text, xy)], fill=fill, outline=outline)


def render_history_gif(
    rgb: np.ndarray,
    history: "list[PipelineSnapshot] | PerceptionOutput",
    path: str | Path,
    *,
    duration: int = 800,
    colors: np.ndarray | None = None,
    alpha: float = 0.5,
    include_input: bool = True,
    loop: int = 0,
    show_id: bool = False,
    show_label: bool = False,
    show_score: bool = False,
) -> Path:
    """Write a GIF showing the mask state after each pipeline stage.

    Each frame is labelled with the stage that just ran (e.g. ``"sam3"``,
    ``"filter.overlap"``). Enable capture by running the pipeline with
    ``InferenceOptions(record_history=True)``.

    Parameters
    ----------
    rgb:
        The image the pipeline was run on — used as the backdrop for every
        frame so viewers can see what the masks are attached to.
    history:
        Either the snapshot list or a ``PerceptionOutput`` (``.history`` is
        read from it).
    path:
        Destination GIF path. Created parent dirs are not made automatically.
    duration:
        Milliseconds per frame (GIFs via Pillow use ms).
    include_input:
        Prepend a plain "input" frame before any masks appear.
    loop:
        Passed through to imageio; ``0`` means loop forever.
    show_id, show_label, show_score:
        Per-detection annotations drawn at each mask's bounding-box top-left,
        matching ``overlay_detections``. ``instance_id`` is wrapped in ``[]``;
        ``score`` is formatted to 2 decimals. Snapshots taken before
        ``assign`` populates ``instance_id`` simply skip ``show_id``.
    """
    import imageio.v3 as iio

    from .types import PerceptionOutput

    snapshots = (
        history.history if isinstance(history, PerceptionOutput) else list(history)
    )
    query_colors = _query_color_map(snapshots, colors)

    frames: list[np.ndarray] = []
    if include_input:
        frames.append(draw_label(rgb, "input"))
    for snap in snapshots:
        frames.append(
            _render_snapshot_frame(
                rgb,
                snap,
                alpha=alpha,
                query_colors=query_colors,
                show_id=show_id,
                show_label=show_label,
                show_score=show_score,
            )
        )

    arr = np.stack(frames, axis=0)
    out_path = Path(path)
    iio.imwrite(str(out_path), arr, extension=".gif", loop=loop, duration=duration)
    return out_path


def _query_color_map(
    snapshots: "list[PipelineSnapshot]", colors: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Assign each query a stable color so it stays the same across frames."""
    palette = DEFAULT_COLORS if colors is None else np.asarray(colors, dtype=np.float32)
    queries: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for q in snap.detections.keys():
            if q not in seen:
                seen.add(q)
                queries.append(q)
    return {q: palette[i % len(palette)] for i, q in enumerate(queries)}


def _render_snapshot_frame(
    rgb: np.ndarray,
    snap: "PipelineSnapshot",
    *,
    alpha: float,
    query_colors: dict[str, np.ndarray],
    show_id: bool = False,
    show_label: bool = False,
    show_score: bool = False,
    show_stage_label: bool = True,
) -> np.ndarray:
    vis = rgb.astype(np.float32).copy()
    text_items: list[tuple[str, tuple[int, int]]] = []
    painted = False
    if snap.detections:
        for q, dets in snap.detections.items():
            color = query_colors[q] if q in query_colors else DEFAULT_COLORS[0]
            for det in dets:
                mask = det.mask
                if mask is None or not mask.any():
                    continue
                vis[mask] = (1.0 - alpha) * vis[mask] + alpha * color
                painted = True
                ann = _format_detection_annotation(
                    det,
                    show_id=show_id,
                    show_label=show_label,
                    show_score=show_score,
                )
                if ann:
                    anchor = _mask_anchor(mask)
                    if anchor is not None:
                        text_items.append((ann, anchor))
    if not painted and snap.instances is not None and snap.instances.shape[0] > 0:
        palette = (
            np.stack(list(query_colors.values())) if query_colors else DEFAULT_COLORS
        )
        for i in range(snap.instances.shape[0]):
            m = snap.instances[i]
            if not m.any():
                continue
            color = palette[i % len(palette)]
            vis[m] = (1.0 - alpha) * vis[m] + alpha * color
        painted = True
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    # Geometric proposals (boxes/points) — the output of detector stages like
    # detect.gemini, which emit no masks. Without this, those stages' tiles
    # render as bare RGB even though they DID produce intermediate output.
    if not painted and (
        getattr(snap, "box_prompts", None) or getattr(snap, "point_prompts", None)
    ):
        vis = _overlay_geometric_prompts(
            vis,
            getattr(snap, "box_prompts", {}) or {},
            getattr(snap, "point_prompts", {}) or {},
            query_colors,
            show_label=show_label,
        )
    if snap.projected_refs:
        vis = _overlay_projected_refs(vis, snap.projected_refs, query_colors)
    if show_stage_label:
        text_items.append((snap.stage, (8, 8)))
    return _draw_texts(vis, text_items)


def _overlay_geometric_prompts(
    vis: np.ndarray,
    box_prompts: dict[str, np.ndarray],
    point_prompts: dict[str, np.ndarray],
    query_colors: dict[str, np.ndarray],
    *,
    show_label: bool = True,
) -> np.ndarray:
    """Draw box/point proposals (one colour per query) onto ``vis``."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil)

    def _color(query: str) -> tuple[int, int, int]:
        base = query_colors.get(query) if isinstance(query_colors, dict) else None
        col = base if base is not None else DEFAULT_COLORS[0]
        return tuple(int(c) for c in col[:3])

    for query, boxes in (box_prompts or {}).items():
        col = _color(query)
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
            draw.rectangle(
                [float(x1), float(y1), float(x2), float(y2)], outline=col, width=2
            )
            if show_label:
                draw.text((float(x1) + 2, max(0.0, float(y1) - 10)), query, fill=col)
    r = 4
    for query, pts in (point_prompts or {}).items():
        col = _color(query)
        for x, y in np.asarray(pts, dtype=np.float32).reshape(-1, 2):
            draw.ellipse(
                [float(x) - r, float(y) - r, float(x) + r, float(y) + r],
                outline=col,
                width=2,
            )
            draw.ellipse(
                [float(x) - 1, float(y) - 1, float(x) + 1, float(y) + 1], fill=col
            )
            if show_label:
                draw.text((float(x) + r + 2, float(y) - 6), query, fill=col)
    return np.asarray(pil)


def _overlay_projected_refs(
    vis: np.ndarray,
    refs: "Sequence[ProjectedReference]",
    query_colors: dict[str, np.ndarray],
) -> np.ndarray:
    """Splat fuse_views projected reference points + footprint outlines.

    Aligned (``matched=True``) refs are drawn in the query's colour;
    seeded refs are drawn dimmer and outlined with a dashed-equivalent
    (just the dilation ring) so callers can tell at a glance which
    fusion outcome happened on this frame.
    """
    from scipy.ndimage import binary_dilation

    out = vis.copy()
    H, W = out.shape[:2]
    for ref in refs:
        base = query_colors.get(ref.query) if isinstance(query_colors, dict) else None
        color = (base if base is not None else DEFAULT_COLORS[0]).astype(np.float32)
        if not ref.matched:
            color = color * 0.55  # dim seeded refs
        ring = binary_dilation(ref.footprint, iterations=2) & ~ref.footprint
        if ring.any():
            out_f = out.astype(np.float32)
            out_f[ring] = 0.5 * out_f[ring] + 0.5 * color
            out = np.clip(out_f, 0, 255).astype(np.uint8)
        if ref.uv.shape[0] == 0:
            continue
        radius = 2
        col_u8 = color.astype(np.uint8)
        for u, v in ref.uv:
            u_i, v_i = int(u), int(v)
            y0, y1 = max(0, v_i - radius), min(H, v_i + radius + 1)
            x0, x1 = max(0, u_i - radius), min(W, u_i + radius + 1)
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] = col_u8
    return out


def _draw_texts(
    img: np.ndarray,
    items: Iterable[tuple[str, tuple[int, int]]],
    *,
    fill: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Draw each ``(text, (x, y))`` pair on a copy of ``img`` with an outline."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for text, (x, y) in items:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, fill=outline)
        draw.text((x, y), text, fill=fill)
    return np.asarray(pil)


def _mask_anchor(mask: np.ndarray) -> tuple[int, int] | None:
    """Top-left ``(x, y)`` of the mask's bounding box, or ``None`` if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min())


def _format_detection_annotation(
    det: "Detection", *, show_id: bool, show_label: bool, show_score: bool
) -> str | None:
    parts: list[str] = []
    if show_id and det.instance_id is not None:
        parts.append(f"[{det.instance_id}]")
    if show_label:
        parts.append(det.query)
    if show_score:
        parts.append(f"{det.score:.2f}")
    return " ".join(parts) if parts else None


def render_pointcloud_orbit_gif(
    pointclouds: "Sequence[np.ndarray] | Sequence[Sequence[np.ndarray]] | Sequence[PerceptionOutput] | TrackHistory",
    path: str | Path,
    *,
    cameras: "CameraInfo | Sequence[CameraInfo | None] | Sequence[Sequence[CameraInfo | None]] | None" = None,
    labels: Sequence[str] | None = None,
    colors: np.ndarray | None = None,
    n_frames: int | None = None,
    duration: int = 80,
    image_size: tuple[int, int] = (480, 480),
    elev_deg: float = 20.0,
    orbit_turns: float = 1.0,
    point_size: float = 4.0,
    bg: tuple[int, int, int] = (18, 18, 26),
    loop: int = 0,
    draw_frustum: bool = True,
) -> Path:
    """Render an orbiting 3D view of one or more point clouds as a GIF.

    ``pointclouds`` can be one of:
      - a ``TrackHistory`` from ``OpenPerception.history`` — the simplest
        path: per-frame point clouds, cameras and labels are pulled
        directly off the history so callers don't have to hand-build
        them.
      - a list of ``(Ni, 3)`` arrays — a single-timestep scene (orbits only).
      - a list of such lists — a temporal sequence; the GIF advances through
        timesteps while the camera orbits, so both structure and motion are
        visible.
      - a list of ``PerceptionOutput`` — one snapshot per step with
        ``det.pointcloud`` populated (run with ``"reconstruct"`` in the
        pipeline order).

    ``cameras`` are drawn as small frustums in world frame (skipped if
    ``extrinsics`` is ``None``). Pass a flat list for static cameras, or a
    list-of-lists (one inner list per timestep) for moving cameras — the
    frustums then follow the per-frame poses. ``labels`` supplies a
    per-object legend aligned to the color palette. ``n_frames`` defaults
    to ``max(120, 2 * len(timesteps))`` so longer videos play back at a
    reasonable rate. Axis limits and box aspect are fixed across the GIF
    from the global bounding box so rotations stay stable and distances
    are preserved (equal aspect per axis).

    Rendering is done with matplotlib (Agg backend). World frame is assumed
    to be Z-up.
    """
    from .types import TrackHistory

    if isinstance(pointclouds, TrackHistory):
        if cameras is None:
            cameras = pointclouds.cameras_per_frame()
        if labels is None:
            labels = pointclouds.labels_by_query_view()
        pointclouds = pointclouds.pointclouds_by_query_view()

    if n_frames is None:
        n_frames = max(120, 2 * len(pointclouds))
    import imageio.v3 as iio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    timesteps, query_index = _normalize_pc_timesteps(pointclouds)
    palette_255 = (
        DEFAULT_COLORS if colors is None else np.asarray(colors, dtype=np.float32)
    )
    palette = np.clip(palette_255.astype(np.float32) / 255.0, 0.0, 1.0)
    bg_f: tuple[float, float, float] = (
        float(bg[0]) / 255.0,
        float(bg[1]) / 255.0,
        float(bg[2]) / 255.0,
    )
    cams_per_ts = _normalize_cameras_per_timestep(cameras, len(timesteps))

    all_pts = _concat_nonempty([pc for ts in timesteps for pc in ts])
    if all_pts.size == 0:
        raise ValueError("render_pointcloud_orbit_gif: all point clouds are empty")
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    max_span = float((mx - mn).max())
    frustum_segments_per_ts: list[list[tuple[np.ndarray, np.ndarray]]] = (
        [_camera_frustum_segments(cs, max_span * 0.12) for cs in cams_per_ts]
        if draw_frustum
        else [[] for _ in cams_per_ts]
    )
    all_frustum_pts = [
        p for segs in frustum_segments_per_ts for seg in segs for p in seg
    ]
    if all_frustum_pts:
        frustum_pts = np.array(all_frustum_pts, dtype=np.float32)
        mn = np.minimum(mn, frustum_pts.min(axis=0))
        mx = np.maximum(mx, frustum_pts.max(axis=0))
    max_span = float((mx - mn).max())
    pad = max(max_span * 0.1, 1e-3)
    xlim = (float(mn[0] - pad), float(mx[0] + pad))
    ylim = (float(mn[1] - pad), float(mx[1] + pad))
    zlim = (float(mn[2] - pad), float(mx[2] + pad))
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    legend = _legend_items(query_index, labels, palette_255)

    W, H = image_size
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg_f)
    ax = fig.add_subplot(111, projection="3d")

    frames_out: list[np.ndarray] = []
    n_ts = len(timesteps)
    for i in range(n_frames):
        ax.cla()
        ax.set_facecolor(bg_f)
        ax.set_axis_off()

        ts_idx = int(i * n_ts / n_frames) if n_ts > 1 else 0
        for obj_idx, pts in enumerate(timesteps[ts_idx]):
            if pts is None or pts.size == 0:
                continue
            col = palette[obj_idx % len(palette)]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],  # pyright: ignore[reportArgumentType]
                s=float(point_size) ** 2,  # pyright: ignore[reportArgumentType]
                c=[col],
                marker=".",
                depthshade=True,
                linewidths=0.0,
            )

        ts_segments = frustum_segments_per_ts[ts_idx]
        if ts_segments:
            ax.add_collection3d(
                Line3DCollection(
                    [(tuple(a), tuple(b)) for a, b in ts_segments],
                    colors=(0.9, 0.9, 0.9, 0.85),
                    linewidths=1.0,
                )
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_box_aspect((xr, yr, zr))
        azim = 360.0 * orbit_turns * i / max(n_frames, 1)
        ax.view_init(elev=elev_deg, azim=azim)

        if legend:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    markerfacecolor=(
                        float(col[0]) / 255.0,
                        float(col[1]) / 255.0,
                        float(col[2]) / 255.0,
                    ),
                    markeredgecolor="none",
                    markersize=8,
                    label=name,
                )
                for name, col in legend
            ]
            leg = ax.legend(
                handles=handles,
                loc="upper left",
                fontsize=9,
                framealpha=0.35,
                facecolor=bg_f,
                edgecolor="none",
                labelcolor="white",
            )
            for text in leg.get_texts():
                text.set_color("white")
        if n_ts > 1:
            ax.text2D(
                0.98,
                0.97,
                f"t={ts_idx}",
                transform=ax.transAxes,
                color="white",
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
            )

        fig.canvas.draw()
        # Agg canvas exposes buffer_rgba(); stubs don't advertise it.
        buf = fig.canvas.buffer_rgba  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        rgba = np.asarray(buf())
        frames_out.append(rgba[..., :3].copy())

    plt.close(fig)

    out_path = Path(path)
    iio.imwrite(
        str(out_path),
        np.stack(frames_out, axis=0),
        extension=".gif",
        loop=loop,
        duration=duration,
    )
    return out_path


def render_segmentation_video(
    history: "TrackHistory",
    path: str | Path,
    *,
    layout: "Literal['side_by_side', 'grid', 'single']" = "side_by_side",
    view: int | None = None,
    fps: int = 15,
    codec: str = "libx264",
    show_id: bool = False,
    show_label: bool = False,
    show_score: bool = False,
    show_fps: bool = False,
    colors: np.ndarray | None = None,
    alpha: float = 0.5,
) -> Path:
    """Write a per-frame segmentation overlay video from a ``TrackHistory``.

    Replaces the boilerplate every multi-camera example used to repeat:
    bucket detections by view, render ``overlay_detections`` per frame
    per view, ``np.concatenate`` along axis 2, ``imageio.imwrite``.

    Parameters
    ----------
    history:
        The accumulator on ``OpenPerception.history`` (or any equivalent
        ``TrackHistory``).
    path:
        Output file. Extension drives the container — use ``.mp4`` for
        H.264 and ``.gif`` for animated GIF (``codec`` and ``fps`` are
        ignored for GIF).
    layout:
        ``"side_by_side"`` concatenates per-view overlays horizontally
        (sensible for 2-camera demos). ``"grid"`` packs them into a
        roughly-square grid. ``"single"`` requires ``view`` and renders
        only that view.
    view:
        Index of the view to render when ``layout='single'``.
    show_id, show_label, show_score:
        Forwarded to ``overlay_detections``.
    show_fps:
        Draw the achieved FPS at the top-left of each frame, computed from
        the wall-clock interval between successive ``TrackedFrame.timestamp``
        values (``time.monotonic`` set when the frame was recorded). The
        first frame inherits the second frame's interval; if the history has
        only one frame, ``-- FPS`` is drawn.

    Returns the output ``Path``.
    """
    import imageio.v3 as iio

    from .types import PerceptionOutput

    if len(history) == 0:
        raise ValueError("render_segmentation_video: history is empty")
    n_views = history.n_views_effective

    if layout == "single":
        if view is None:
            raise ValueError("layout='single' requires view=<index>")
        if not (0 <= view < n_views):
            raise ValueError(f"view={view} out of range for {n_views} views")

    palette = DEFAULT_COLORS if colors is None else np.asarray(colors, dtype=np.float32)

    def _overlay(frame, view_idx: int) -> np.ndarray:
        view_dets = {
            q: [d for d in lst if d.view_idx == view_idx]
            for q, lst in frame.detections.items()
        }
        stub = PerceptionOutput(detections=view_dets, stats=frame.stats)
        return overlay_detections(
            frame.rgbs[view_idx],
            stub,
            colors=palette,
            alpha=alpha,
            show_id=show_id,
            show_label=show_label,
            show_score=show_score,
        )

    fps_per_frame: list[str] | None = None
    if show_fps:
        timestamps = [f.timestamp for f in history.frames]
        deltas: list[float | None] = [None] * len(timestamps)
        for i in range(1, len(timestamps)):
            deltas[i] = timestamps[i] - timestamps[i - 1]
        if len(timestamps) > 1:
            deltas[0] = deltas[1]  # frame 0 inherits frame 1's interval
        fps_per_frame = [
            f"{1.0 / dt:.1f} FPS" if dt and dt > 0 else "-- FPS" for dt in deltas
        ]

    composed: list[np.ndarray] = []
    for i, frame in enumerate(history.frames):
        if layout == "single":
            assert view is not None  # for type-checkers
            img = _overlay(frame, view)
        else:
            per_view = [_overlay(frame, v) for v in range(n_views)]
            if layout == "side_by_side":
                img = np.concatenate(per_view, axis=1)
            elif layout == "grid":
                img = _grid_compose(per_view)
            else:
                raise ValueError(f"unknown layout: {layout!r}")
        if fps_per_frame is not None:
            img = _draw_texts(img, [(fps_per_frame[i], (8, 8))])
        composed.append(img)

    video = np.stack(composed, axis=0)
    out_path = Path(path)
    if out_path.suffix.lower() == ".gif":
        iio.imwrite(str(out_path), video, extension=".gif", duration=int(1000 / fps))
    else:
        iio.imwrite(str(out_path), video, fps=fps, codec=codec)
    return out_path


def render_disambiguation_panel(
    crops: Sequence[np.ndarray],
    queries: Sequence[str],
    references: Mapping[str, Sequence[np.ndarray]],
    scores: np.ndarray,
    path: str | Path,
    *,
    sources: Sequence[str] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    cell_in: float = 1.4,
) -> Path:
    """Save a grid PNG of crops × references annotated with similarity scores.

    Layout::

        [empty] | ref(q0) | ref(q1) | ...
        crop_0  | s00     | s01     | ...
        crop_1  | s10     | s11     | ...

    Each score cell is colour-coded by ``score`` magnitude (``cmap``) with the
    numeric value drawn on top. ``sources`` (per-query: ``"image"``/``"text"``/
    ``"missing"``) is appended to each column header so the caller can see
    which queries hit the reference bank vs. fell back to text.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    M = len(crops)
    P = len(queries)
    if scores.shape != (M, P):
        raise ValueError(
            f"scores shape {scores.shape} does not match (crops={M}, queries={P})"
        )

    n_rows = max(M, 1) + 1
    n_cols = P + 1
    fig_w = cell_in * n_cols
    fig_h = cell_in * n_rows
    if title:
        fig_h += 0.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=11)

    finite = scores[np.isfinite(scores)]
    vmax = float(finite.max()) if finite.size else 1.0
    vmin = float(finite.min()) if finite.size else 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)

    axes[0, 0].axis("off")
    for j, q in enumerate(queries):
        ax = axes[0, j + 1]
        refs = list(references.get(q, []))
        if refs:
            ax.imshow(refs[0])
            if len(refs) > 1:
                ax.text(
                    0.98,
                    0.02,
                    f"+{len(refs) - 1}",
                    transform=ax.transAxes,
                    color="white",
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5),
                )
        else:
            ax.text(0.5, 0.5, "no ref", ha="center", va="center", fontsize=9)
            ax.set_facecolor("#eeeeee")
        src_tag = ""
        if sources is not None and j < len(sources) and sources[j] != "image":
            src_tag = f" [{sources[j]}]"
        ax.set_title(f"{q}{src_tag}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    if M == 0:
        ax = axes[1, 0]
        ax.text(0.5, 0.5, "no crops", ha="center", va="center", fontsize=10)
        ax.axis("off")
        for j in range(P):
            axes[1, j + 1].axis("off")
    else:
        for i in range(M):
            ax_crop = axes[i + 1, 0]
            ax_crop.imshow(crops[i])
            ax_crop.set_title(f"crop #{i}", fontsize=9)
            ax_crop.set_xticks([])
            ax_crop.set_yticks([])
            row = scores[i]
            best_j = int(np.argmax(row)) if row.size else -1
            for j in range(P):
                ax = axes[i + 1, j + 1]
                s = float(scores[i, j])
                ax.imshow(np.full((1, 1), s), cmap=cmap_obj, norm=norm, aspect="auto")
                rgba = cmap_obj(norm(s))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if lum < 0.55 else "black"
                ax.text(
                    0.5,
                    0.5,
                    f"{s:.3f}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=11,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    if j == best_j:
                        spine.set_edgecolor("#d62728")
                        spine.set_linewidth(2.0)
                    else:
                        spine.set_edgecolor("#cccccc")
                        spine.set_linewidth(0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_debug_image(
    rgb: np.ndarray,
    snapshots: "list[PipelineSnapshot] | TrackedFrame",
    path: str | Path,
    *,
    view_idx: int | None = None,
    config: "PerceptionConfig | None" = None,
    title: str | None = None,
    column_in: float = 3.2,
    alpha: float = 0.5,
) -> Path:
    """Per-stage debug panel — one column per pipeline snapshot.

    Each column renders, top to bottom:
      1. Stage name + the stage's hyperparameter overrides from ``config``.
      2. The raw ``instances`` pool, one colour per candidate index, annotated
         with ``#i`` and the SAM3 segmentation score (when ``mask_scores``
         is populated).
      3. The labelled ``detections`` overlay, one colour per query, with
         ``[id] query score`` annotations.
      4. A text block listing every detection's ``[id] query: score``,
         per-instance ``mask_score``, and the (N_instances × N_queries)
         per-source matrix from ``ctx.scores`` (e.g. ``scores.dino``,
         ``scores.clip``, ``scores.clip_text_fallback``) so you can see
         the raw similarity that drove each label decision.

    Use this to inspect how masks, labels and scores evolve from
    ``sam3`` → filters → ``disambiguate.*`` → ``assign`` → ``reconstruct``
    → ``fuse_views``: e.g. spotting a stage that drops a candidate, or a
    query whose label flips between disambiguate and assign.

    Parameters
    ----------
    rgb:
        The image the pipeline ran on. Shared across columns as the
        backdrop for both the instance and detection overlays.
    snapshots:
        Either a list of :class:`PipelineSnapshot` (e.g.
        ``frame.stage_history``) or a :class:`TrackedFrame`. When a frame
        is passed and ``view_idx`` is omitted, every view's snapshots are
        rendered in a single panel — pass ``view_idx`` to slice to one view.
    config:
        Optional :class:`PerceptionConfig`; only used to print each
        stage's user-set ``_params`` overrides in the header. Pass
        ``perc.config`` from your :class:`OpenPerception` instance.
    column_in:
        Column width in inches. Total figure width = ``column_in * n_cols``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    from .types import TrackedFrame

    if isinstance(snapshots, TrackedFrame):
        snaps = list(snapshots.stage_history)
    else:
        snaps = list(snapshots)
    if view_idx is not None:
        snaps = [s for s in snaps if s.view_idx == view_idx]
    if not snaps:
        raise ValueError("render_debug_image: no snapshots to render")

    query_colors = _query_color_map(snaps, None)
    n_cols = len(snaps)

    # Approximate monospace columns that fit one column at a given font size.
    # Empirically ~1 char per (font_pt / 11) inches at dpi=120; tuned to the
    # column_in default so neither params (fs=8) nor detail block (fs=7)
    # overflow into the neighbouring column.
    wrap_params = max(20, int(column_in * 11))
    wrap_detail = max(24, int(column_in * 13))

    # Bottom row hosts a per-detection mini-grid (crop + all metrics),
    # one DETECTION PER SUB-ROW (crop on the left, metrics on the right).
    # Total row height grows with the worst-case detection count, and each
    # sub-row needs to fit the longest metrics block.
    detection_metric_blocks: list[list[list[str]]] = [
        [
            _format_detection_metrics(det, snap, q)
            for q, dets in snap.detections.items()
            for det in dets
        ]
        for snap in snaps
    ]
    max_dets = max(
        (len(col) for col in detection_metric_blocks),
        default=0,
    )
    max_metric_lines = max(
        (len(block) for col in detection_metric_blocks for block in col),
        default=0,
    )
    # 0.18 in per metric line accounts for matplotlib's ~1.2× linespacing at
    # fontsize=7 and the inner padding gridspec adds around each sub-cell.
    # The +0.5 floor keeps very short cards (e.g. no scores yet) from
    # collapsing down to nothing.
    per_det_h = max(1.4, 0.18 * max_metric_lines + 0.5)
    detail_h = max(3.2, per_det_h * max(max_dets, 1))

    height_ratios = [1.6, 3.2, 3.2, detail_h]
    fig_w = column_in * n_cols
    fig_h = column_in * (sum(height_ratios) / 2.5)
    if title:
        fig_h += 0.4

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=11)

    gs = gridspec.GridSpec(
        4,
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.3,
        wspace=0.05,
    )

    for c, snap in enumerate(snaps):
        base_name = snap.stage.split("#")[0]
        params = config._params.get(base_name, {}) if config is not None else {}
        n_inst = 0 if snap.instances is None else int(snap.instances.shape[0])
        n_det = sum(len(v) for v in snap.detections.values())

        ax_h = fig.add_subplot(gs[0, c])
        ax_h.axis("off")
        header = (
            f"{snap.stage}\n"
            f"view={snap.view_idx}  frame={snap.frame_idx}  "
            f"inst={n_inst}  det={n_det}"
        )
        ax_h.text(
            0.0,
            1.0,
            header,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            transform=ax_h.transAxes,
        )
        param_lines: list[str] = []
        for k, v in params.items():
            param_lines.extend(_wrap_kv(k, v, wrap_params))
        if not param_lines:
            param_lines = ["(no overrides)"]
        ax_h.text(
            0.0,
            0.55,
            "\n".join(param_lines),
            fontsize=8,
            family="monospace",
            va="top",
            ha="left",
            transform=ax_h.transAxes,
            color="#444444",
        )

        ax_i = fig.add_subplot(gs[1, c])
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        if n_inst:
            anns = []
            seg = snap.mask_scores
            for i in range(n_inst):
                if seg is not None and i < seg.shape[0]:
                    anns.append(f"#{i} {float(seg[i]):.2f}")
                else:
                    anns.append(f"#{i}")
            inst_vis = overlay_masks(
                rgb,
                [snap.instances[i] for i in range(n_inst)],  # type: ignore[index]
                annotations=anns,
                alpha=alpha,
            )
        else:
            inst_vis = draw_label(rgb, "(no instances)")
        ax_i.imshow(inst_vis)
        ax_i.set_title("instances", fontsize=9)

        ax_d = fig.add_subplot(gs[2, c])
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        if n_det:
            det_vis = _render_snapshot_frame(
                rgb,
                snap,
                alpha=alpha,
                query_colors=query_colors,
                show_id=True,
                show_label=True,
                show_score=True,
                show_stage_label=False,
            )
        else:
            det_vis = draw_label(rgb, "(no detections)")
        ax_d.imshow(det_vis)
        ax_d.set_title("detections", fontsize=9)

        flat_dets = [(q, det) for q, dets in snap.detections.items() for det in dets]
        if not flat_dets:
            ax_t = fig.add_subplot(gs[3, c])
            ax_t.axis("off")
            ax_t.text(
                0.5,
                0.5,
                "(no detections)",
                ha="center",
                va="center",
                fontsize=8,
                color="#777777",
                transform=ax_t.transAxes,
            )
        else:
            sub = gridspec.GridSpecFromSubplotSpec(
                len(flat_dets),
                2,
                subplot_spec=gs[3, c],
                width_ratios=[1.0, 1.7],
                hspace=0.2,
                wspace=0.05,
            )
            metrics_blocks = detection_metric_blocks[c]
            for j, ((q, det), metric_lines) in enumerate(
                zip(flat_dets, metrics_blocks, strict=True)
            ):
                ax_crop = fig.add_subplot(sub[j, 0])
                ax_crop.set_xticks([])
                ax_crop.set_yticks([])
                crop = _crop_from_mask(rgb, det.mask)
                if crop is not None:
                    ax_crop.imshow(crop)
                else:
                    ax_crop.text(
                        0.5,
                        0.5,
                        "(empty)",
                        ha="center",
                        va="center",
                        fontsize=7,
                        transform=ax_crop.transAxes,
                    )
                color = (
                    query_colors.get(q, DEFAULT_COLORS[0]).astype(np.float32) / 255.0
                )
                for spine in ax_crop.spines.values():
                    spine.set_edgecolor(
                        (float(color[0]), float(color[1]), float(color[2]))
                    )
                    spine.set_linewidth(2.0)

                ax_m = fig.add_subplot(sub[j, 1])
                ax_m.axis("off")
                ax_m.text(
                    0.0,
                    1.0,
                    "\n".join(metric_lines),
                    fontsize=7,
                    family="monospace",
                    va="top",
                    ha="left",
                    transform=ax_m.transAxes,
                )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _short_repr(value: object, max_len: int = 60) -> str:
    s = repr(value)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _wrap_kv(key: str, value: object, width: int) -> list[str]:
    """Render ``key=value`` then hard-wrap so each line fits ``width`` chars.

    Continuation lines are indented two spaces so wrapped values stay
    visually grouped under their key.
    """
    import textwrap

    text = f"{key}={value!r}"
    if len(text) <= width:
        return [text]
    return textwrap.wrap(
        text,
        width=width,
        subsequent_indent="  ",
        break_long_words=True,
        break_on_hyphens=False,
    ) or [text]


def _crop_from_mask(
    rgb: np.ndarray, mask: np.ndarray, *, pad: int = 4, dim_bg: float = 0.4
) -> np.ndarray | None:
    """Return ``rgb`` cropped to ``mask``'s bounding box with the bg dimmed.

    ``None`` if the mask is empty. ``dim_bg`` darkens pixels outside the
    mask so the actual instance pops out — set to ``1.0`` to disable.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = rgb.shape[:2]
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    crop = rgb[y0:y1, x0:x1].copy()
    sub_mask = mask[y0:y1, x0:x1]
    if dim_bg < 1.0 and not sub_mask.all():
        f = crop.astype(np.float32)
        f[~sub_mask] *= float(dim_bg)
        crop = np.clip(f, 0, 255).astype(np.uint8)
    return crop


def _format_detection_metrics(
    det: "Detection",
    snap: "PipelineSnapshot",
    query: str,
    *,
    name_width: int = 12,
) -> list[str]:
    """All metrics tied to one detection: id, query, score, src, seg, scores.*[i].

    The ``query``-matching column in each source matrix is starred (``*``)
    and the column with the highest value is carat-tagged (``^``) so the
    reader can spot at a glance whether the assigned label is also the
    column where the score is highest. ``name_width`` truncates long query
    names so each metric line fits within the per-detection card.
    """
    lines: list[str] = []
    id_str = f"[{det.instance_id}]" if det.instance_id is not None else "[-]"
    lines.append(f"{id_str} {_truncate(query, name_width + 4)}")
    lines.append(f"score={det.score:.3f}")
    si = det.source_index
    if si is not None:
        lines.append(f"src#{si}")
    if (
        si is not None
        and snap.mask_scores is not None
        and si < snap.mask_scores.shape[0]
    ):
        lines.append(f"seg={float(snap.mask_scores[si]):.3f}")
    if si is not None and snap.scores:
        for source, mat in snap.scores.items():
            arr = np.asarray(mat)
            if arr.ndim != 2 or si >= arr.shape[0]:
                continue
            queries = snap.queries or [f"q{j}" for j in range(arr.shape[1])]
            best_j = int(np.argmax(arr[si])) if arr.shape[1] > 0 else -1
            lines.append(f"{source}:")
            for j, qname in enumerate(queries):
                v = float(arr[si, j])
                qshow = _truncate(qname, name_width)
                tags = ""
                if qname == query:
                    tags += "*"
                if j == best_j:
                    tags += "^"
                tag_str = f" {tags}" if tags else ""
                lines.append(f" {qshow}={v:+.2f}{tag_str}")
    return lines


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _grid_compose(views: list[np.ndarray]) -> np.ndarray:
    """Pack views into the smallest near-square grid; pad missing cells with zeros."""
    n = len(views)
    if n == 1:
        return views[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w, c = views[0].shape
    canvas = np.zeros((rows * h, cols * w, c), dtype=views[0].dtype)
    for i, v in enumerate(views):
        r, col = divmod(i, cols)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = v
    return canvas


def _normalize_pc_timesteps(
    pointclouds,
) -> tuple[list[list[np.ndarray]], list[str]]:
    """Return ``(timesteps, query_index)`` in the canonical shape.

    ``query_index[k]`` is the label (or empty string) for the k-th object slot
    across all timesteps — used for the optional legend. For ``PerceptionOutput``
    inputs we pick it up from the detection dict keys; otherwise it's empty.
    """
    from .types import PerceptionOutput

    seq = list(pointclouds)
    if not seq:
        return [], []

    # Case A: list of PerceptionOutput (temporal, queried)
    if isinstance(seq[0], PerceptionOutput):
        query_order: list[str] = []
        seen: set[str] = set()
        for out in seq:
            for q in out.detections.keys():
                if q not in seen:
                    seen.add(q)
                    query_order.append(q)
        timesteps: list[list[np.ndarray]] = []
        for out in seq:
            step: list[np.ndarray] = []
            for q in query_order:
                pts_for_q: list[np.ndarray] = []
                for det in out.detections.get(q, []):
                    if det.pointcloud is not None and det.pointcloud.size > 0:
                        pts_for_q.append(det.pointcloud)
                step.append(
                    np.concatenate(pts_for_q, axis=0)
                    if pts_for_q
                    else np.zeros((0, 3), dtype=np.float32)
                )
            timesteps.append(step)
        return timesteps, query_order

    first = seq[0]
    # Case B: list[list[ndarray]] — explicit temporal
    if isinstance(first, (list, tuple)) and (
        not isinstance(first, np.ndarray)  # defensive
    ):
        timesteps = [[np.asarray(p, dtype=np.float32) for p in step] for step in seq]
        return timesteps, []

    # Case C: list[ndarray] — static single-timestep
    step = [np.asarray(p, dtype=np.float32) for p in seq]
    return [step], []


def _normalize_cameras(cameras) -> list:
    from .types import CameraInfo

    if cameras is None:
        return []
    if isinstance(cameras, CameraInfo):
        return [cameras]
    return [c for c in cameras if c is not None]


def _normalize_cameras_per_timestep(cameras, n_ts: int) -> list[list]:
    """Return ``n_ts`` per-timestep camera lists.

    A flat list (or single ``CameraInfo``) is treated as static and broadcast
    to every timestep. A list-of-lists is taken as already temporal — its
    length must match ``n_ts``.
    """
    from .types import CameraInfo

    if cameras is None:
        return [[] for _ in range(n_ts)]
    if isinstance(cameras, CameraInfo):
        return [[cameras] for _ in range(n_ts)]
    seq = list(cameras)
    if not seq:
        return [[] for _ in range(n_ts)]
    if isinstance(seq[0], (list, tuple)) and not isinstance(seq[0], CameraInfo):
        if len(seq) != n_ts:
            raise ValueError(
                f"cameras has {len(seq)} timesteps but pointclouds has {n_ts}"
            )
        return [[c for c in step if c is not None] for step in seq]
    flat = [c for c in seq if c is not None]
    return [flat for _ in range(n_ts)]


def _concat_nonempty(arrays: Iterable[np.ndarray]) -> np.ndarray:
    non_empty = [a for a in arrays if a is not None and a.size > 0]
    if not non_empty:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([a.astype(np.float32, copy=False) for a in non_empty], axis=0)


def _camera_frustum_segments(
    cameras, scale: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for cam in cameras:
        if cam is None or cam.extrinsics is None:
            continue
        E = np.asarray(cam.extrinsics, dtype=np.float32)
        R = E[:3, :3]
        t = E[:3, 3]
        K = np.asarray(cam.intrinsics, dtype=np.float32)
        fx_c, fy_c = float(K[0, 0]), float(K[1, 1])
        cx_c, cy_c = float(K[0, 2]), float(K[1, 2])
        d = scale
        hx = cx_c / fx_c * d
        hy = cy_c / fy_c * d
        corners_cam = np.array(
            [[-hx, -hy, d], [hx, -hy, d], [hx, hy, d], [-hx, hy, d]],
            dtype=np.float32,
        )
        corners_world = corners_cam @ R.T + t
        apex = t
        for c in corners_world:
            segments.append((apex, c))
        for i in range(4):
            segments.append((corners_world[i], corners_world[(i + 1) % 4]))
    return segments


def _legend_items(
    query_index: list[str],
    labels: Sequence[str] | None,
    palette: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    names = list(labels) if labels is not None else query_index
    return [(n, palette[i % len(palette)]) for i, n in enumerate(names) if n]


def _resolve_annotations(
    annotations: Iterable[str | None] | Mapping[object, str | None] | None,
    mask_list: list,
    keys: list | None,
) -> list[str | None]:
    if annotations is None:
        return [None] * len(mask_list)
    if isinstance(annotations, Mapping):
        if keys is None:
            raise ValueError("annotations mapping requires masks to also be a mapping")
        result: list[str | None] = []
        for k in keys:
            v = annotations.get(k)
            result.append(v if isinstance(v, str) else None)
        return result
    ann_list = list(annotations)
    if len(ann_list) != len(mask_list):
        raise ValueError(
            f"annotations has {len(ann_list)} entries but masks has {len(mask_list)}"
        )
    return ann_list

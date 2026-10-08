"""The exterior view with the perception masks and the distance pairs drawn on.

What an audience sees when guidance is on: each tracked object filled in its
own colour with its name beside it, and each distance pair as a line between
the entities it measures — in the *pair's* colour, the same one the 3-D scene
and the trend plot use for it, with the live reading and the demo's reading
at the aligned time on a small chip at the midpoint.

Everything is OpenCV on a numpy frame, so the same picture can be written to a
file in a test. Projection reuses :mod:`policy_abstraction.visualization.drawing`
and the banner/badge helpers of :class:`~policy_abstraction.experiments.overlay.OverlayCanvas`.
"""

from __future__ import annotations

import colorsys
from typing import Optional, Sequence

import cv2
import numpy as np

from policy_abstraction.demo.session import LiveView
from policy_abstraction.experiments.overlay import (
    COLOR_LABEL,
    OverlayCanvas,
    to_drawable,
)
from policy_abstraction.visualization.drawing import (
    draw_pair,
    entity_to_pixel,
    interpolate_error_color,
)

# RGB. The states the banner can be in.
COLOR_GUIDED = (60, 235, 120)
COLOR_UNGUIDED = (90, 170, 255)
COLOR_IDLE = (200, 200, 210)
COLOR_STOPPING = (255, 190, 60)
COLOR_UNAVAILABLE = (255, 120, 90)
_CHIP_BG = (20, 20, 26)
_GREY = (150, 150, 158)


def object_color(index: int) -> tuple[int, int, int]:
    """A distinct, saturated RGB per object index. Objects are not pairs, so
    they get their own palette; it is chosen to read against the table."""
    hue = (0.12 + index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def object_colors(names: dict[int, str]) -> dict[int, tuple[int, int, int]]:
    """Stable colour per instance id: assigned in id order, so an object keeps
    its colour from one frame to the next."""
    return {sid: object_color(i) for i, sid in enumerate(sorted(names))}


def draw_masks(
    img: np.ndarray,
    seg: np.ndarray,
    names: dict[int, str],
    colors: dict[int, tuple[int, int, int]],
    *,
    alpha: float = 0.38,
    label: bool = True,
) -> None:
    """Fill each instance of *seg* that has a name, outline it, caption it.

    In place. *seg* is an ``[H, W]`` integer instance map (0 = background)
    at the frame's resolution; a map of another size is resized nearest-
    neighbour first.
    """
    if seg is None or not names:
        return
    seg_arr = np.squeeze(np.asarray(seg))
    if seg_arr.ndim != 2:
        return
    H, W = img.shape[:2]
    if seg_arr.shape != (H, W):
        seg_arr = cv2.resize(
            seg_arr.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST
        )
    fill = img.copy()
    any_mask = np.zeros((H, W), dtype=bool)
    contours_by_id: dict[int, list] = {}
    for sid, _name in names.items():
        mask = seg_arr == sid
        if not mask.any():
            continue
        color = colors.get(sid, _GREY)
        fill[mask] = color
        any_mask |= mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_by_id[sid] = contours
    if not any_mask.any():
        return
    blended = cv2.addWeighted(fill, alpha, img, 1.0 - alpha, 0)
    img[any_mask] = blended[any_mask]
    font_unit = max(0.4, min(1.0, W / 1100.0))
    for sid, contours in contours_by_id.items():
        color = colors.get(sid, _GREY)
        cv2.drawContours(img, contours, -1, color, 2, cv2.LINE_AA)
        if not label:
            continue
        ys, xs = np.where(seg_arr == sid)
        x, y = int(xs.min()), int(ys.min())
        _chip(
            img,
            to_drawable(names[sid]),
            (x, max(y - 6, 0)),
            color,
            font_unit,
            anchor="below" if y < 24 else "above",
        )


def _chip(
    img: np.ndarray,
    text: str,
    at: tuple[int, int],
    color: tuple[int, int, int],
    font_unit: float,
    *,
    anchor: str = "above",
    taken: Optional[list[tuple[int, int, int, int]]] = None,
) -> tuple[int, int, int, int]:
    """A small dark chip with *text* in *color*, its bottom-left (``above``)
    or top-left (``below``) at *at*. With *taken*, the chip is nudged down
    until it overlaps none of those boxes, and its box is appended."""
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.55 * font_unit + 0.15
    thickness = 1 if size < 0.8 else 2
    (tw, th), base = cv2.getTextSize(text, font, size, thickness)
    pad = max(3, int(5 * font_unit))
    w, h = tw + 2 * pad, th + base + 2 * pad
    x = int(min(max(at[0], 0), max(W - w, 0)))
    y = int(at[1] - h) if anchor == "above" else int(at[1])
    y = min(max(y, 0), max(H - h, 0))
    if taken is not None:
        for _ in range(12):
            box = (x, y, x + w, y + h)
            if not any(_overlap(box, other) for other in taken):
                break
            y = min(y + h + 2, max(H - h, 0))
        taken.append((x, y, x + w, y + h))
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), _CHIP_BG, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.putText(
        img,
        text,
        (x + pad, y + pad + th),
        font,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return (x, y, x + w, y + h)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_distance_pairs(
    img: np.ndarray,
    result,
    colors: dict[str, tuple[int, int, int]],
    *,
    seg: Optional[np.ndarray],
    K: np.ndarray,
    E: np.ndarray,
    ee_pos: np.ndarray,
    preprocessed: bool = False,
    readouts: bool = True,
) -> int:
    """Draw every pair of *result* onto *img* in its own colour.

    Pairs of a sub-trajectory other than the current one are drawn faint, so
    the eye lands on what guidance is acting on now. Returns how many pairs
    had both endpoints in view.
    """
    pairs = result.pairs or []
    if not pairs:
        return 0
    H, W = img.shape[:2]
    K3 = np.asarray(K, dtype=np.float64)
    K3 = K3[:3, :3] if K3.shape == (4, 4) else K3
    E = np.asarray(E, dtype=np.float64)
    seg_arr = (
        np.squeeze(np.asarray(seg, dtype=np.int32))
        if seg is not None
        else np.zeros((H, W), dtype=np.int32)
    )
    if seg_arr.shape != (H, W):
        seg_arr = cv2.resize(seg_arr, (W, H), interpolation=cv2.INTER_NEAREST)
    entity_3d = result.entity_centroids or {}
    fingers = {k: entity_3d[k] for k in ("gripper_l", "gripper_r") if k in entity_3d}
    entities = {e for p in pairs for e in (p.from_entity, p.to_entity)}
    entity_2d = {
        e: entity_to_pixel(
            e,
            seg_arr,
            ee_pos,
            K3,
            E,
            H,
            W,
            fingers or None,
            preprocessed=preprocessed,
        )
        for e in entities
    }
    current = result.current_subtraj_idx

    def _active(pair) -> bool:
        return (
            current is None or pair.subtraj_idx is None or pair.subtraj_idx == current
        )

    faint = [p for p in pairs if not _active(p)]
    active = [p for p in pairs if _active(p)]
    if faint:
        layer = img.copy()
        for pair in faint:
            draw_pair(
                layer,
                pair,
                entity_2d,
                colors.get(pair.label, _GREY),
                K3,
                E,
                H,
                W,
                entity_3d,
                preprocessed=preprocessed,
            )
        cv2.addWeighted(layer, 0.25, img, 0.75, 0, img)
    drawn = 0
    font_unit = max(0.4, min(1.0, W / 1100.0))
    taken: list[tuple[int, int, int, int]] = []
    for pair in active:
        color = colors.get(pair.label, _GREY)
        a, b = entity_2d.get(pair.from_entity), entity_2d.get(pair.to_entity)
        if a is None or b is None:
            continue
        draw_pair(
            img,
            pair,
            entity_2d,
            color,
            K3,
            E,
            H,
            W,
            entity_3d,
            preprocessed=preprocessed,
        )
        drawn += 1
        if not readouts:
            continue
        live = result.live_distances.get(pair.label)
        ref = result.ref_distances.get(pair.label)
        if live is None:
            continue
        text = f"{live:.2f}"
        if ref is not None and np.isfinite(ref):
            text += f" / {ref:.2f} m"
        else:
            text += " m"
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        box = _chip(img, text, (mid[0] + 6, mid[1] - 4), color, font_unit, taken=taken)
        err = result.distance_errors.get(pair.label)
        if err is not None and np.isfinite(err):
            # A dot in the error colour: green on the demo, red far from it.
            cv2.circle(
                img,
                (box[0] - 6, (box[1] + box[3]) // 2),
                4,
                interpolate_error_color(float(err)),
                -1,
                cv2.LINE_AA,
            )
    return drawn


def render_live_view(
    view: LiveView,
    *,
    show_overlays: bool,
    banner: Optional[str] = None,
    banner_color: tuple[int, int, int] = COLOR_IDLE,
    badges: Sequence[str] = (),
    preprocessed: bool = False,
    masks: bool = True,
    readouts: bool = True,
    heatmap=None,
    badge_start_line: int = 0,
) -> np.ndarray:
    """The exterior frame of *view*, annotated for the audience.

    With *show_overlays* and an analysed view: object masks, distance pairs
    with their readouts. With *heatmap* (a
    :class:`~policy_abstraction.demo.heatmap.TrainingHeatmap`): the training
    grasps' density on the table, under everything else. Always: *badges*
    top-left and *banner* along the bottom. Returns a new ``[H, W, 3]`` uint8
    RGB array; ``view.image`` is never modified. Raises ``ValueError`` when
    the view has no frame yet.
    """
    if view.image is None:
        raise ValueError("no camera frame in this view yet")
    img = np.ascontiguousarray(np.asarray(view.image, dtype=np.uint8)).copy()
    notes: list[str] = list(badges)
    if heatmap is not None:
        if view.intrinsic is None or view.extrinsic is None:
            notes.append("no camera calibration — training heat map not drawn")
        else:
            try:
                heatmap.blend(
                    img, view.intrinsic, view.extrinsic, preprocessed=preprocessed
                )
            except Exception as exc:  # a footprint outside the view, say
                notes.append(f"training heat map not drawn: {exc}")
    if show_overlays and view.analysed:
        result = view.abstraction
        if masks and view.seg is not None and view.names:
            referenced = {
                e for p in (result.pairs or []) for e in (p.from_entity, p.to_entity)
            }
            names = {
                sid: name
                for sid, name in view.names.items()
                if not referenced or sid in referenced
            } or dict(view.names)
            draw_masks(img, view.seg, names, object_colors(view.names))
        if view.intrinsic is not None and view.extrinsic is not None:
            ee = view.ee_pos
            if ee is None:
                notes.append("no end-effector position — pairs not drawn")
            else:
                draw_distance_pairs(
                    img,
                    result,
                    view.colors,
                    seg=view.seg,
                    K=view.intrinsic,
                    E=view.extrinsic,
                    ee_pos=np.asarray(ee, dtype=np.float64),
                    preprocessed=preprocessed,
                    readouts=readouts,
                )
        else:
            notes.append("no camera calibration — pairs not drawn")
    canvas = OverlayCanvas(img, preprocessed=preprocessed)
    for line, note in enumerate(notes, start=max(0, int(badge_start_line))):
        canvas.draw_badge(note, line=line, color=COLOR_LABEL)
    if banner:
        canvas.draw_banner(banner, color=banner_color)
    return canvas.image


__all__ = [
    "COLOR_GUIDED",
    "COLOR_IDLE",
    "COLOR_STOPPING",
    "COLOR_UNAVAILABLE",
    "COLOR_UNGUIDED",
    "draw_distance_pairs",
    "draw_masks",
    "object_color",
    "object_colors",
    "render_live_view",
]

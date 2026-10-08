"""Pure stateless drawing utilities shared by VisualizationHandler and annotation.py.

All functions operate on numpy arrays passed as arguments and have no side-effects
beyond modifying the *img* array in-place (where applicable).

Color convention: RGB (matching Rerun display convention).
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from policy_abstraction.utils.pcd import project_points_to_pixels
from policy_abstraction.vlm.types import DistancePair


def interpolate_error_color(err: float | None) -> tuple[int, int, int]:
    """Map distance error to a smoothly interpolated RGB color.

    Anchor points:
    - ``0.00`` -> green ``(0, 200, 0)``
    - ``0.02`` -> orange ``(255, 165, 0)``
    - ``0.05`` -> red ``(220, 0, 0)``

    Returns gray ``(128, 128, 128)`` for ``None``.
    """
    if err is None:
        return (128, 128, 128)

    anchors: list[tuple[float, tuple[int, int, int]]] = [
        (0.00, (0, 200, 0)),
        (0.02, (255, 165, 0)),
        (0.05, (220, 0, 0)),
    ]

    e = abs(float(err))
    if e >= anchors[-1][0]:
        return anchors[-1][1]

    for (x0, c0), (x1, c1) in zip(anchors[:-1], anchors[1:], strict=False):
        if e <= x1:
            t = (e - x0) / (x1 - x0) if x1 > x0 else 1.0
            return (
                int(round((1.0 - t) * c0[0] + t * c1[0])),
                int(round((1.0 - t) * c0[1] + t * c1[1])),
                int(round((1.0 - t) * c0[2] + t * c1[2])),
            )

    return anchors[-1][1]


def entity_to_pixel(
    entity: int | str,
    seg_arr: np.ndarray,
    ee_pos: np.ndarray,
    K3: np.ndarray,
    E: np.ndarray,
    H: int,
    W: int,
    finger_positions: dict[str, np.ndarray] | None = None,
    *,
    preprocessed: bool = True,
) -> tuple[int, int] | None:
    """Project an entity (seg-id or robot entity) to pixel coordinates.

    Args:
        preprocessed: Whether the frame being drawn on has its u axis mirrored
            relative to *K3* — see
            :attr:`~policy_abstraction.envs.spec.CameraSpec.pixels_mirrored`
            for a live frame, or the demo episode's ``frames_preprocessed`` for
            a reference one. The default matches the simulators, which is what
            every caller assumed before real cameras arrived.

    Returns ``None`` when the entity is not visible or the projection falls
    outside the image.
    """
    if entity == "ee":
        pts_2d, valid = project_points_to_pixels(
            ee_pos[np.newaxis].astype(np.float64),
            K3,
            E,
            H,
            W,
            preprocessed=preprocessed,
        )
        return (int(pts_2d[0, 0]), int(pts_2d[0, 1])) if valid[0] else None
    if isinstance(entity, str) and entity in ("gripper_l", "gripper_r"):
        if finger_positions is None:
            return None
        pos = finger_positions.get(entity)
        if pos is None:
            return None
        pts_2d, valid = project_points_to_pixels(
            pos[np.newaxis].astype(np.float64),
            K3,
            E,
            H,
            W,
            preprocessed=preprocessed,
        )
        return (int(pts_2d[0, 0]), int(pts_2d[0, 1])) if valid[0] else None
    if isinstance(entity, str):
        return None
    mask = seg_arr == entity
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (int(xs.mean()), int(ys.mean()))


def draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_px: int = 8,
    gap_px: int = 5,
) -> None:
    """Draw a dashed line from *pt1* to *pt2* on *img* (in-place)."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    total = float(np.sqrt(dx * dx + dy * dy))
    if total < 1.0:
        return
    segment = dash_px + gap_px
    n_seg = int(total / segment) + 1
    for i in range(n_seg):
        t0 = i * segment / total
        t1 = min((i * segment + dash_px) / total, 1.0)
        p0 = (int(pt1[0] + t0 * dx), int(pt1[1] + t0 * dy))
        p1 = (int(pt1[0] + t1 * dx), int(pt1[1] + t1 * dy))
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def compute_entity_centroids_3d(
    pairs: list[DistancePair],
    pcd: np.ndarray,
    ee_pos: np.ndarray,
    finger_positions: dict[str, np.ndarray] | None = None,
) -> dict[int | str, np.ndarray]:
    """Return world-frame 3-D centroids for every entity that appears in *pairs*."""
    entities: set[int | str] = {e for p in pairs for e in (p.from_entity, p.to_entity)}
    result: dict[int | str, np.ndarray] = {}
    for e in entities:
        if e == "ee":
            result[e] = np.asarray(ee_pos, dtype=np.float64).ravel()[:3]
        elif e in ("gripper_l", "gripper_r") and finger_positions is not None:
            pos = finger_positions.get(str(e))
            if pos is not None:
                result[e] = np.asarray(pos, dtype=np.float64).ravel()[:3]
        elif isinstance(e, int):
            pts = pcd[pcd[:, 6] == e, :3]
            if pts.shape[0] > 0:
                result[e] = pts.mean(axis=0).astype(np.float64)
    return result


def compute_centroids_2d(
    seg_mask: np.ndarray,
    ids: list[int] | set[int],
) -> dict[int, tuple[int, int]]:
    """Return ``{id: (u, v)}`` pixel centroid of each id's seg-mask region.

    Args:
        seg_mask: ``[H, W]`` or ``[H, W, 1]`` integer mask.
        ids:      Segment IDs to compute centroids for. IDs not present in the
                  mask are silently omitted from the result.

    Returns:
        Mapping from segment ID to (u, v) image pixel coordinates.
    """
    arr = np.asarray(seg_mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    out: dict[int, tuple[int, int]] = {}
    for sid in ids:
        ys, xs = np.where(arr == int(sid))
        if xs.size == 0:
            continue
        out[int(sid)] = (int(xs.mean()), int(ys.mean()))
    return out


def _draw_pair(
    img: np.ndarray,
    pair: DistancePair,
    entity_2d: dict[int | str, tuple[int, int] | None],
    color: tuple[int, int, int],
    K3: np.ndarray,
    E: np.ndarray,
    H: int,
    W: int,
    entity_3d: dict[int | str, np.ndarray] | None,
    *,
    preprocessed: bool = True,
) -> None:
    """Draw a single distance pair onto *img* (in-place) with the given color."""
    pt_a = entity_2d[pair.from_entity]
    pt_b = entity_2d[pair.to_entity]
    if pt_a is None or pt_b is None:
        return

    drew_3d = False
    if entity_3d is not None and (pair.axis is not None or pair.plane is not None):
        from_3d = entity_3d.get(pair.from_entity)
        to_3d = entity_3d.get(pair.to_entity)
        if from_3d is not None and to_3d is not None:
            delta = to_3d - from_3d

            if pair.axis is not None:
                ax = np.asarray(pair.axis, dtype=np.float64)
                ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
                inter_3d = from_3d + float(np.dot(delta, ax)) * ax
            else:
                n = np.asarray(pair.plane, dtype=np.float64)
                n = n / max(float(np.linalg.norm(n)), 1e-9)
                in_plane = delta - float(np.dot(delta, n)) * n
                inter_3d = from_3d + in_plane

            pts_world = np.stack([from_3d, inter_3d, to_3d])
            pts_px, valid = project_points_to_pixels(
                pts_world, K3, E, H, W, preprocessed=preprocessed
            )

            if valid[0] and valid[1]:
                pa = (int(pts_px[0, 0]), int(pts_px[0, 1]))
                pi = (int(pts_px[1, 0]), int(pts_px[1, 1]))
                cv2.line(img, pa, pi, color, 2, cv2.LINE_AA)
                drew_3d = True

            if valid[1] and valid[2]:
                pi = (int(pts_px[1, 0]), int(pts_px[1, 1]))
                pb = (int(pts_px[2, 0]), int(pts_px[2, 1]))
                draw_dashed_line(img, pi, pb, color, thickness=1)
                drew_3d = True

            if drew_3d:
                if valid[0]:
                    cv2.circle(
                        img, (int(pts_px[0, 0]), int(pts_px[0, 1])), 4, color, -1
                    )
                if valid[2]:
                    cv2.circle(
                        img, (int(pts_px[2, 0]), int(pts_px[2, 1])), 4, color, -1
                    )

    if not drew_3d:
        cv2.line(img, pt_a, pt_b, color, 2)
        cv2.circle(img, pt_a, 5, color, -1)
        cv2.circle(img, pt_b, 5, color, -1)


draw_pair = _draw_pair
"""Public name for :func:`_draw_pair`, for callers that pick their own colour
per pair (the demo overlay colours by pair, not by error)."""


def draw_pairs_on_image(
    img: np.ndarray,
    pairs: list[DistancePair],
    seg_arr: np.ndarray,
    K3: np.ndarray,
    E: np.ndarray,
    ee_pos: np.ndarray,
    H: int,
    W: int,
    distance_errors: dict[str, float],
    live_distances: dict[str, float] | None = None,
    entity_3d: dict[int | str, np.ndarray] | None = None,
    current_subtraj_idx: int | None = None,
    *,
    preprocessed: bool = True,
) -> None:
    """Draw pair lines and error-colored markers onto *img* (in-place).

    When *entity_3d* world-frame centroids are supplied, axis and plane
    pairs receive a richer two-segment overlay:

    * **Axis pair**: solid line from ``from_entity`` → intersection with the
      perpendicular plane through ``to_entity``; dashed line from the
      intersection → ``to_entity``.
    * **Plane pair**: solid line from ``from_entity`` → in-plane projection
      of ``to_entity`` (the measured component); dashed line from the
      projection → ``to_entity`` (the normal residual).

    All 3-D points are projected to pixel space before drawing.
    Falls back to a plain straight line when *entity_3d* is absent.

    When *current_subtraj_idx* is provided, pairs whose ``subtraj_idx`` does
    not match it are drawn almost transparently (alpha ≈ 0.15) so attention
    focuses on the pairs relevant to the active sub-trajectory.
    """
    # Precompute pixel positions for all unique entities. Pull gripper
    # finger positions out of entity_3d so entity_to_pixel can project them.
    finger_positions: dict[str, np.ndarray] | None = None
    if entity_3d is not None:
        fp = {k: entity_3d[k] for k in ("gripper_l", "gripper_r") if k in entity_3d}
        if fp:
            finger_positions = fp

    entities: set[int | str] = {e for p in pairs for e in (p.from_entity, p.to_entity)}
    entity_2d: dict[int | str, tuple[int, int] | None] = {
        e: entity_to_pixel(
            e,
            seg_arr,
            ee_pos,
            K3,
            E,
            H,
            W,
            finger_positions,
            preprocessed=preprocessed,
        )
        for e in entities
    }

    def _is_relevant(pair: DistancePair) -> bool:
        if current_subtraj_idx is None or pair.subtraj_idx is None:
            return True
        return pair.subtraj_idx == current_subtraj_idx

    relevant = [p for p in pairs if _is_relevant(p)]
    irrelevant = [p for p in pairs if not _is_relevant(p)]

    # Draw irrelevant pairs onto a scratch layer then blend at low opacity
    if irrelevant:
        overlay = img.copy()
        for pair in irrelevant:
            err = distance_errors.get(pair.label)
            color = interpolate_error_color(err)
            _draw_pair(
                overlay,
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
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Draw relevant pairs at full opacity
    for pair in relevant:
        err = distance_errors.get(pair.label)
        color = interpolate_error_color(err)
        _draw_pair(
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


def subtraj_hue(idx: int) -> float:
    """Golden-ratio hue in ``[0, 1)`` for sub-trajectory *idx* (0 is red)."""
    return (idx * 0.618033988749895) % 1.0


def subtraj_color(idx: int, *, bright: bool) -> tuple[int, int, int]:
    """Return an RGB color for sub-trajectory *idx* using golden-ratio hues."""
    hue = subtraj_hue(idx)
    s, v = (0.85, 0.95) if bright else (0.35, 0.75)
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def subtraj_tone(
    subtraj_idx: int | None, tone: int, n_tones: int
) -> tuple[int, int, int]:
    """The *tone*-th of *n_tones* shades of sub-trajectory *subtraj_idx*'s hue.

    Every distance pair of one sub-trajectory shares that sub-trajectory's hue
    (reds for the first, blues for the second, ...) and the pairs within it
    are told apart by saturation, from fully saturated down to a pale tint.
    Pairs with no sub-trajectory get greys.
    """
    if subtraj_idx is None:
        v = 0.85 - 0.4 * (tone / max(n_tones - 1, 1))
        g = int(v * 255)
        return (g, g, g)
    sat = 0.95 - 0.55 * (tone / max(n_tones - 1, 1))
    r, g, b = colorsys.hsv_to_rgb(subtraj_hue(subtraj_idx), sat, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def pair_colors(pairs: list[DistancePair]) -> dict[str, tuple[int, int, int]]:
    """One RGB per pair label: its sub-trajectory's hue, in a tone that tells
    it from the other pairs of that sub-trajectory.

    The single source of the pair palette. The 3-D dimension lines, the Rerun
    distance series, the live camera overlay and the demo's trend plot all
    colour a pair through this, so the same line is the same colour wherever
    it appears. Pairs without a sub-trajectory come out grey.
    """
    by_subtraj: dict[int | None, list[str]] = {}
    for pair in pairs:
        by_subtraj.setdefault(pair.subtraj_idx, []).append(pair.label)
    colors: dict[str, tuple[int, int, int]] = {}
    for subtraj_idx, labels in by_subtraj.items():
        for tone, label in enumerate(labels):
            colors[label] = subtraj_tone(subtraj_idx, tone, len(labels))
    return colors


def draw_subtraj_overlay(
    img: np.ndarray,
    subtrajectories: list[tuple[int, int]],
    ref_ee_2d: np.ndarray | None,
    ref_ee_valid: np.ndarray | None,
    current_subtraj_idx: int | None,
) -> np.ndarray:
    """Draw EE trajectory paths per sub-trajectory onto *img* (in-place).

    Args:
        img:                Image to draw on (modified in-place).
        subtrajectories:    List of ``(start, end)`` index pairs.
        ref_ee_2d:          ``[T, 2]`` float32 projected pixel coordinates.
        ref_ee_valid:       ``[T]`` bool validity mask.
        current_subtraj_idx: Index of the currently active sub-trajectory.

    Returns:
        The modified *img*.
    """
    if ref_ee_2d is None or ref_ee_valid is None or not subtrajectories:
        return img

    for i, (start, end) in enumerate(subtrajectories):
        is_current = i == current_subtraj_idx
        color = subtraj_color(i, bright=is_current)
        thickness = 2 if is_current else 1
        pts = ref_ee_2d[start:end]
        valid = ref_ee_valid[start:end]
        prev_pt = None
        for j in range(len(pts)):
            if not valid[j]:
                prev_pt = None
                continue
            pt = (int(pts[j, 0]), int(pts[j, 1]))
            if prev_pt is not None:
                cv2.line(img, prev_pt, pt, color, thickness, cv2.LINE_AA)
            prev_pt = pt

    return img

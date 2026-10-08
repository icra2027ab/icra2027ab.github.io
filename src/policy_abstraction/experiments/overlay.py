"""Drawing the staging target onto a live camera frame.

:class:`OverlayCanvas` is a small drawing surface that knows one thing the
factors do not: how a world-frame point becomes a pixel. Factors call
``draw_grid`` / ``highlight_cell`` / ``draw_orientation_arrow`` and never touch a
camera matrix.

Two conventions are load-bearing and easy to get backwards:

* **Projection.** ``project_points_to_pixels`` takes a *camera-to-world*
  extrinsic and mirrors the u axis when ``preprocessed=True``. That mirror
  compensates for the ``[::-1, ::-1]`` the pipeline applies to frames from
  cameras whose ``CameraSpec.image_transform`` is ``flip_vh`` (every simulator
  adapter). Cameras that publish upright frames — the real UR5e — must not
  mirror. :func:`preprocessed_for` derives it from the spec, so the same code
  overlays a LIBERO scene and a real workcell. Getting it wrong mirrors the grid
  about the image centre, which looks plausible and is completely wrong.
* **Colour order.** Everything here is RGB, matching the raw observation images
  and :mod:`policy_abstraction.visualization.drawing`. OpenCV does not care as
  long as nobody swaps midway.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from policy_abstraction.envs.spec import CameraSpec
from policy_abstraction.experiments.grid import CellId, TableGrid
from policy_abstraction.utils.pcd import project_points_to_pixels

# RGB.
COLOR_GRID = (90, 190, 255)
COLOR_TARGET = (60, 235, 90)
COLOR_TARGET_FILL = (60, 235, 90)
COLOR_LABEL = (245, 245, 245)
COLOR_ARROW = (255, 190, 60)
COLOR_BADGE_BG = (25, 25, 30)
COLOR_REGION = (150, 150, 200)
COLOR_DISTRACTOR = (255, 190, 60)


# OpenCV ships only Hershey fonts, which are ASCII: anything else draws as
# "?". Substituting is better than dropping — an operator reading "45deg" knows
# what it means, "45?" is noise.
_ASCII_SUBSTITUTIONS = {
    "\u00b0": "deg",
    "\u2014": "-",
    "\u2013": "-",
    "\u2026": "...",
    "\u2192": "->",
    "\u00d7": "x",
    "\u2713": "OK",
    "\u2717": "X",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def to_drawable(text: str) -> str:
    """Fold *text* to something the Hershey fonts can actually render."""
    for src, dst in _ASCII_SUBSTITUTIONS.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Axis-aligned ``(x, y, w, h)`` intersection test."""
    return (
        a[0] < b[0] + b[2]
        and b[0] < a[0] + a[2]
        and a[1] < b[1] + b[3]
        and b[1] < a[1] + a[3]
    )


def preprocessed_for(camera: CameraSpec | None) -> bool:
    """Whether pixels must be u-mirrored for *camera*'s displayed frames.

    True exactly when the pipeline pre-flips the frame. The rule itself lives
    on :attr:`~policy_abstraction.envs.spec.CameraSpec.pixels_mirrored` — this
    only adds the "no camera at all" default, which keeps the historical
    simulator behaviour rather than silently changing geometry.
    """
    if camera is None:
        return True
    return camera.pixels_mirrored


class OverlayCanvas:
    """An image being annotated, plus the camera that produced it.

    Args:
        image: ``[H, W, 3]`` uint8 RGB frame. Copied, never mutated in place.
        intrinsic: ``[3, 3]`` pinhole K, or ``None`` when the camera is
            uncalibrated — every 3-D drawing call then becomes a no-op and
            :attr:`projectable` is False, so a caller can say so in the UI
            instead of silently showing an unannotated view.
        extrinsic: ``[4, 4]`` camera-to-world pose.
        preprocessed: See :func:`preprocessed_for`.
        scale: Draw at this multiple of the source resolution. A 256x256 sim
            frame is unreadable at 1:1 on a workcell monitor; the grid and text
            are scaled with it so lines stay hairline-thin rather than blocky.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        intrinsic: Optional[np.ndarray] = None,
        extrinsic: Optional[np.ndarray] = None,
        preprocessed: bool = True,
        scale: float = 1.0,
    ) -> None:
        img = np.ascontiguousarray(np.asarray(image))
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected [H, W, 3] RGB image, got shape {img.shape}")
        self._src_h, self._src_w = img.shape[:2]
        self.scale = float(scale)
        if self.scale != 1.0:
            img = cv2.resize(
                img,
                (int(self._src_w * self.scale), int(self._src_h * self.scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        self.image = img.astype(np.uint8, copy=True)
        self._K = None if intrinsic is None else np.asarray(intrinsic, dtype=np.float64)
        self._E = None if extrinsic is None else np.asarray(extrinsic, dtype=np.float64)
        self._preprocessed = bool(preprocessed)
        self._anchors: dict[str, tuple[TableGrid, CellId]] = {}
        self._object_anchors: dict[str, np.ndarray] = {}
        # Where each object's caption was actually placed, and the boxes
        # already taken, so captions do not land on top of each other.
        self._object_labels: dict[str, tuple[int, int]] = {}
        self._label_boxes: list[tuple[int, int, int, int]] = []
        # Text is sized against the *output* width, not the upscale factor. A
        # 256 px sim frame drawn at 3x is 768 px wide; text at 3x the 1x size
        # would overflow it. Tying it to the canvas keeps a caption legible at
        # any resolution and always fitting.
        self._font_unit = max(0.34, min(1.1, self.image.shape[1] / 1100.0))

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    @property
    def projectable(self) -> bool:
        """Whether world-frame drawing is possible (both matrices present)."""
        return self._K is not None and self._E is not None

    @property
    def line_width(self) -> int:
        return max(1, int(round(self.scale)))

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World ``[N, 3]`` to canvas ``[N, 2]`` pixels plus a validity mask.

        Points are projected at the *source* resolution — which is the
        resolution the intrinsics describe — and scaled afterwards. Projecting
        with a scaled K would be equivalent only if the principal point scaled
        too, which is the kind of thing that is right on a centred camera and
        silently wrong on any other.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        # Both matrices, or neither: `projectable` is exactly this test,
        # but re-binding them locally is what lets a type checker see it.
        K, E = self._K, self._E
        if K is None or E is None:
            return np.zeros((len(pts), 2), dtype=np.int32), np.zeros(
                len(pts), dtype=bool
            )
        px, valid = project_points_to_pixels(
            pts,
            K,
            E,
            self._src_h,
            self._src_w,
            preprocessed=self._preprocessed,
        )
        if self.scale != 1.0:
            px = np.round(px.astype(np.float64) * self.scale).astype(np.int32)
        return px, valid

    # ------------------------------------------------------------------
    # Anchors — lets a non-positional factor draw relative to a positional one
    # ------------------------------------------------------------------

    def register_anchor(self, factor_name: str, grid: TableGrid, cell: CellId) -> None:
        self._anchors[factor_name] = (grid, cell)

    def resolve_anchor(
        self, name: Optional[str], assignments: dict[str, str]
    ) -> Optional[tuple[TableGrid, CellId]]:
        """The (grid, cell) an orientation/appearance factor should draw at.

        A named anchor wins; otherwise the first registered one. ``None`` when
        the experiment has no positional factor at all, in which case the caller
        skips the annotation rather than guessing a location.
        """
        if name is not None:
            return self._anchors.get(name)
        return next(iter(self._anchors.values()), None)

    def register_object(self, object_id: str, position) -> None:
        """Remember where an object was drawn, so another factor can point at it."""
        self._object_anchors[object_id] = np.asarray(position, dtype=np.float64)

    def object_anchor(self, object_id: str) -> Optional[np.ndarray]:
        return self._object_anchors.get(object_id)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _polyline(self, points_3d: np.ndarray, color, thickness: int) -> None:
        """Draw a world-frame polyline, skipping the parts behind the camera."""
        px, valid = self.project(points_3d)
        run: list[np.ndarray] = []
        for point, ok in zip(px, valid, strict=True):
            if ok:
                run.append(point)
                continue
            if len(run) >= 2:
                cv2.polylines(
                    self.image,
                    [np.asarray(run, dtype=np.int32).reshape(-1, 1, 2)],
                    False,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            run = []
        if len(run) >= 2:
            cv2.polylines(
                self.image,
                [np.asarray(run, dtype=np.int32).reshape(-1, 1, 2)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def draw_grid(
        self,
        grid: TableGrid,
        *,
        color=COLOR_GRID,
        labels: bool = True,
        cells: Optional[Sequence[CellId]] = None,
    ) -> None:
        """Draw every grid line, and optionally a faint label in each cell.

        With *cells*, only those cells are outlined — for a factor that
        sweeps part of the grid, so the report shows the part it measured.
        """
        if not self.projectable:
            return
        if cells is None:
            for line in grid.grid_lines():
                self._polyline(line, color, self.line_width)
            shown: list[CellId] = list(grid.cells())
        else:
            shown = list(cells)
            for cell in shown:
                px, valid = self.project(grid.cell_corners(cell))
                if valid.all():
                    cv2.polylines(
                        self.image,
                        [px.reshape(-1, 1, 2)],
                        True,
                        color,
                        self.line_width,
                        cv2.LINE_AA,
                    )
        if not labels or not shown:
            return
        centers = np.stack([grid.cell_center(c) for c in shown])
        px, valid = self.project(centers)
        for cell, point, ok in zip(shown, px, valid, strict=True):
            if ok:
                self._text(grid.label(cell), point, color, centred=True, scale=0.35)

    def shade_cell(
        self,
        grid: TableGrid,
        cell: CellId,
        color,
        *,
        alpha: float = 0.45,
        outline: bool = True,
    ) -> None:
        """Fill *cell*'s footprint with *color* at *alpha*.

        Used by the report's projected success map, where every cell is
        tinted by its own rate. Kept separate from :meth:`highlight_cell`,
        which is the operator's single "stage it here" marker.
        """
        if not self.projectable:
            return
        px, valid = self.project(grid.cell_corners(cell))
        if not valid.all():
            return
        poly = px.reshape(-1, 1, 2)
        overlay = self.image.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, alpha, self.image, 1.0 - alpha, 0, self.image)
        if outline:
            cv2.polylines(self.image, [poly], True, color, self.line_width, cv2.LINE_AA)

    def cell_center_px(
        self, grid: TableGrid, cell: CellId
    ) -> Optional[tuple[int, int]]:
        """Pixel of *cell*'s lifted centre, or ``None`` if it is not visible."""
        px, valid = self.project(grid.cell_center(cell, lifted=True)[None, :])
        return (int(px[0][0]), int(px[0][1])) if valid[0] else None

    def highlight_cell(
        self,
        grid: TableGrid,
        cell: CellId,
        *,
        label: Optional[str] = None,
        color=COLOR_TARGET,
    ) -> None:
        """Mark *cell* as the place to stage the object.

        Draws a filled translucent quad plus a crosshair at the cell's lifted
        centre — the point an object's centroid should end up at — so the
        operator has both a footprint and a point to aim at.
        """
        if not self.projectable:
            return
        self.register_anchor(grid.name, grid, cell)
        corners_px, corner_valid = self.project(grid.cell_corners(cell))
        if corner_valid.all():
            overlay = self.image.copy()
            cv2.fillPoly(overlay, [corners_px.reshape(-1, 1, 2)], color)
            cv2.addWeighted(overlay, 0.22, self.image, 0.78, 0, self.image)
            cv2.polylines(
                self.image,
                [corners_px.reshape(-1, 1, 2)],
                True,
                color,
                self.line_width + 1,
                cv2.LINE_AA,
            )
        center_px, center_valid = self.project(
            grid.cell_center(cell, lifted=True)[None, :]
        )
        if center_valid[0]:
            cv2.drawMarker(
                self.image,
                tuple(int(v) for v in center_px[0]),
                color,
                cv2.MARKER_CROSS,
                int(18 * self.scale),
                self.line_width + 1,
                cv2.LINE_AA,
            )
            if label:
                self._text(
                    label,
                    center_px[0] + np.array([0, int(-16 * self.scale)]),
                    color,
                    centred=True,
                    scale=0.6,
                    outline=True,
                )

    def draw_orientation_arrow(
        self,
        grid: TableGrid,
        cell: CellId,
        angle_deg: float,
        *,
        color=COLOR_ARROW,
    ) -> None:
        """Draw an in-plane arrow through *cell* at *angle_deg* from ``u_axis``.

        The arrow is a 3-D segment lying in the grid plane, so it foreshortens
        with the table instead of being drawn flat in image space — which would
        show the operator an angle the object cannot be placed at.
        """
        if not self.projectable:
            return
        theta = np.deg2rad(angle_deg)
        direction = np.cos(theta) * np.asarray(grid.u_axis) + np.sin(
            theta
        ) * np.asarray(grid.v_axis)
        center = grid.cell_center(cell, lifted=True)
        half = 0.40 * min(grid.cell_size_u, grid.cell_size_v)
        ends = np.stack([center - half * direction, center + half * direction])
        px, valid = self.project(ends)
        if valid.all():
            cv2.arrowedLine(
                self.image,
                tuple(int(v) for v in px[0]),
                tuple(int(v) for v in px[1]),
                color,
                self.line_width + 1,
                cv2.LINE_AA,
                tipLength=0.3,
            )

    def _place_label(
        self, text: str, anchor: np.ndarray, *, scale: float
    ) -> tuple[int, int]:
        """A spot near *anchor* for a centred caption that no other caption owns.

        Objects ten centimetres apart on the table can be twenty pixels apart
        under perspective, and two captions in the same place name neither
        object. The caption is nudged downward in steps until it is clear, which
        keeps it attached to its own marker — moving it far enough to be
        unambiguous would just make it ambiguous in a new way.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = scale * 2.0 * self._font_unit
        (tw, th), _ = cv2.getTextSize(
            to_drawable(text), font, size, self._thickness(size)
        )
        step = th + max(2, int(3 * self._font_unit))
        x, y = int(anchor[0]), int(anchor[1])
        for _ in range(6):
            box = (x - tw // 2, y - th // 2, tw, th)
            if not any(_boxes_overlap(box, taken) for taken in self._label_boxes):
                break
            y += step
        self._label_boxes.append((x - tw // 2, y - th // 2, tw, th))
        return x, y

    def draw_region(self, region: TableGrid, *, color=COLOR_REGION) -> None:
        """Outline the patch of table objects may be placed in.

        Only the boundary: the interior is where the object markers go, and a
        full grid drawn under them would read as "put it in a cell", which is
        the other experiment.
        """
        if not self.projectable:
            return
        self._polyline(region.outline(), color, self.line_width)

    def mark_object(
        self,
        object_id: str,
        position: Sequence[float],
        *,
        label: str = "",
        color=COLOR_DISTRACTOR,
    ) -> None:
        """Draw a labelled cross where one object goes, and remember the spot.

        Every object in the scene gets one of these, target and distractor
        alike — the operator stages the whole scene, and a marker only for the
        target would leave the rest of it unspecified.
        """
        self.register_object(object_id, position)
        if not self.projectable:
            return
        px, valid = self.project(np.asarray(position, dtype=np.float64)[None, :])
        if not valid[0]:
            return
        centre = tuple(int(v) for v in px[0])
        cv2.drawMarker(
            self.image,
            centre,
            color,
            cv2.MARKER_TILTED_CROSS,
            int(16 * self.scale),
            self.line_width + 1,
            cv2.LINE_AA,
        )
        if label:
            at = self._place_label(
                label, px[0] + np.array([0, int(-14 * self.scale)]), scale=0.5
            )
            self._object_labels[object_id] = at
            self._text(label, at, color, centred=True, scale=0.5, outline=True)

    def highlight_object(
        self,
        object_id: str,
        *,
        label: str = "",
        color=COLOR_TARGET,
    ) -> None:
        """Ring the object the prompt names, at the spot :meth:`mark_object` set.

        A no-op when the object was never marked, which is the honest answer:
        the factor that stages the scene has not drawn yet, or the object is not
        in this layout, and inventing a ring somewhere would be worse than none.
        """
        position = self.object_anchor(object_id)
        if position is None or not self.projectable:
            return
        px, valid = self.project(position[None, :])
        if not valid[0]:
            return
        centre = tuple(int(v) for v in px[0])
        for radius in (int(16 * self.scale), int(22 * self.scale)):
            cv2.circle(
                self.image, centre, radius, color, self.line_width + 1, cv2.LINE_AA
            )
        if label:
            # Over the caption :meth:`mark_object` already placed, not beside
            # it: recolouring the one name is what says "this one", where a
            # second copy of it would just be more text near a crowded marker.
            at = self._object_labels.get(object_id)
            if at is not None:
                self._text(label, at, color, centred=True, scale=0.5, outline=True)

    def draw_point(self, position: Sequence[float], *, color, radius: int = 5) -> None:
        """Mark a single world-frame point (e.g. where the object actually is)."""
        px, valid = self.project(np.asarray(position, dtype=np.float64)[None, :])
        if valid[0]:
            cv2.circle(
                self.image,
                tuple(int(v) for v in px[0]),
                int(radius * self.scale),
                color,
                -1,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # 2-D chrome
    # ------------------------------------------------------------------

    def _thickness(self, size: float) -> int:
        return max(1, int(round(size * 2.0)))

    def fit_text(self, text: str, size: float, max_width: int) -> str:
        """Ellipsise *text* so it fits *max_width* pixels at *size*."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = self._thickness(size)
        text = to_drawable(text)
        if cv2.getTextSize(text, font, size, thickness)[0][0] <= max_width:
            return text
        for cut in range(len(text) - 1, 0, -1):
            candidate = text[:cut] + "..."
            if cv2.getTextSize(candidate, font, size, thickness)[0][0] <= max_width:
                return candidate
        return ""

    def text_size(self, text: str, *, scale: float = 0.5) -> tuple[int, int]:
        """``(width, height)`` in canvas pixels of *text* as :meth:`_text` draws it."""
        size = scale * 2.0 * self._font_unit
        (tw, th), _ = cv2.getTextSize(
            to_drawable(text), cv2.FONT_HERSHEY_SIMPLEX, size, self._thickness(size)
        )
        return int(tw), int(th)

    def _text(
        self,
        text: str,
        at,
        color,
        *,
        centred: bool = False,
        scale: float = 0.5,
        outline: bool = False,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = to_drawable(text)
        size = scale * 2.0 * self._font_unit
        thickness = self._thickness(size)
        (tw, th), _ = cv2.getTextSize(text, font, size, thickness)
        x, y = int(at[0]), int(at[1])
        if centred:
            x -= tw // 2
            y += th // 2
        if outline:
            cv2.putText(
                self.image,
                text,
                (x, y),
                font,
                size,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
        cv2.putText(self.image, text, (x, y), font, size, color, thickness, cv2.LINE_AA)

    def draw_badge(self, text: str, *, color=COLOR_LABEL, line: int = 0) -> None:
        """A caption in the top-left corner, stacked downward by *line*."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = self._font_unit
        thickness = self._thickness(size)
        pad = max(4, int(6 * self._font_unit))
        text = self.fit_text(text, size, self.image.shape[1] - 4 * pad)
        (tw, th), base = cv2.getTextSize(text, font, size, thickness)
        top = pad + line * (th + base + pad)
        cv2.rectangle(
            self.image,
            (pad, top),
            (pad + tw + 2 * pad, top + th + base + pad),
            COLOR_BADGE_BG,
            -1,
        )
        cv2.putText(
            self.image,
            text,
            (pad * 2, top + th + pad // 2),
            font,
            size,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_banner(self, text: str, *, color) -> None:
        """A full-width strip along the bottom — the current rollout's state."""
        h, w = self.image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 1.15 * self._font_unit
        thickness = self._thickness(size)
        text = self.fit_text(text, size, w - int(16 * self._font_unit))
        (tw, th), base = cv2.getTextSize(text, font, size, thickness)
        strip = int(th + base + 12 * self._font_unit)
        overlay = self.image.copy()
        cv2.rectangle(overlay, (0, h - strip), (w, h), COLOR_BADGE_BG, -1)
        cv2.addWeighted(overlay, 0.75, self.image, 0.25, 0, self.image)
        cv2.putText(
            self.image,
            text,
            ((w - tw) // 2, h - strip // 2 + th // 2),
            font,
            size,
            color,
            thickness,
            cv2.LINE_AA,
        )


def render_staging_view(
    image: np.ndarray,
    factors: Sequence,
    assignments: dict[str, str],
    *,
    intrinsic: Optional[np.ndarray] = None,
    extrinsic: Optional[np.ndarray] = None,
    camera: Optional[CameraSpec] = None,
    scale: float = 1.0,
    banner: Optional[str] = None,
    banner_color=COLOR_TARGET,
    preprocessed: Optional[bool] = None,
    notes: Sequence[str] = (),
) -> np.ndarray:
    """Annotate *image* with every factor's target for *assignments*.

    Positional factors are drawn first so later factors can anchor to the cell
    they selected. *notes* are extra badge lines under the factors' captions —
    what the window wants the operator to know about the picture itself, such
    as that a reference frame is ghosted into it.
    """
    canvas = OverlayCanvas(
        image,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        preprocessed=(
            preprocessed_for(camera) if preprocessed is None else bool(preprocessed)
        ),
        scale=scale,
    )

    ordered = sorted(factors, key=lambda f: f.draw_priority)
    badge_line = 0
    for factor in ordered:
        level_id = assignments.get(factor.name)
        if level_id is None:
            continue
        factor.annotate(canvas, level_id, assignments)
        # With a projection the marks on the image *are* the instruction, and
        # one badge per factor keeps the scene visible. Without one there is
        # nothing drawn to read, so the full staging text — coordinates and all
        # — is the only thing the operator has.
        captions = (
            [factor.instruction(level_id)]
            if canvas.projectable
            else factor.staging_lines(level_id, assignments)
        )
        for caption in captions:
            canvas.draw_badge(caption, line=badge_line)
            badge_line += 1
    if not canvas.projectable:
        canvas.draw_badge(
            "no camera calibration — overlay unavailable", line=badge_line
        )
        badge_line += 1
    for note in notes:
        canvas.draw_badge(note, line=badge_line)
        badge_line += 1
    if banner:
        canvas.draw_banner(banner, color=banner_color)
    return canvas.image


__all__ = [
    "COLOR_ARROW",
    "COLOR_DISTRACTOR",
    "COLOR_GRID",
    "COLOR_REGION",
    "COLOR_TARGET",
    "OverlayCanvas",
    "preprocessed_for",
    "to_drawable",
    "render_staging_view",
]

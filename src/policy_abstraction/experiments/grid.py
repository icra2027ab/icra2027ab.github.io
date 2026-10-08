"""A rectangular grid laid on a plane in the robot's world frame.

The grid is the instrument that makes a positional distribution shift
*measurable*: every rollout stages its object at a named cell, so a success rate
can be attributed to a position rather than to "roughly over there".

Geometry lives in 3-D, in the same world frame the env server reports object and
end-effector positions in (the robot base frame, for the UR5e). Projection to
the camera view is a separate concern — see
:mod:`policy_abstraction.experiments.overlay` — so the same grid drives the
operator's overlay, the results plots, and the metric position recorded
alongside each rollout.

Cells are addressed as ``(row, col)`` and labelled for humans as ``A1``, ``B3``,
… — rows are letters, columns are 1-based numbers, which is what an operator can
read off an overlay and repeat back without transcription errors. A grid that
was *extended* after the fact can number its columns from somewhere else
(:attr:`TableGrid.first_col`), so a column added in front of ``A1`` becomes
``A0`` and every cell already collected under keeps its name.
"""

from __future__ import annotations

import dataclasses
import string
from typing import Iterator, Sequence

import numpy as np

# ``(row, col)``, both 0-based.
CellId = tuple[int, int]


def cell_label(row: int, col: int, first_col: int = 1) -> str:
    """Human-facing name for a cell: row letter + column number.

    Column ``0`` is numbered *first_col* — ``1`` by default, so the corner
    cell is ``A1``. Rows past ``Z`` wrap to ``AA``, ``AB``, … so a large grid
    stays labelled rather than silently colliding.
    """
    if row < 0 or col < 0:
        raise ValueError(f"cell indices must be non-negative, got ({row}, {col})")
    if first_col < 0:
        raise ValueError(f"first_col must be non-negative, got {first_col}")
    letters = ""
    n = row
    while True:
        letters = string.ascii_uppercase[n % 26] + letters
        n = n // 26 - 1
        if n < 0:
            break
    return f"{letters}{col + first_col}"


def parse_cell_label(label: str, first_col: int = 1) -> CellId:
    """Inverse of :func:`cell_label`. Raises ``ValueError`` on a malformed label.

    *first_col* must match the one the label was written with: ``"B0"`` is
    malformed on an ordinary grid and the first column of row ``B`` on a grid
    numbered from ``0``.
    """
    text = label.strip().upper()
    split = 0
    while split < len(text) and text[split].isalpha():
        split += 1
    letters, digits = text[:split], text[split:]
    if not letters or not digits.isdigit():
        raise ValueError(f"malformed cell label {label!r}; expected e.g. 'B3'")
    row = 0
    for ch in letters:
        row = row * 26 + (ord(ch) - ord("A") + 1)
    col = int(digits) - first_col
    if col < 0:
        raise ValueError(f"column in {label!r} must be >= {first_col}")
    return row - 1, col


def _unit(vec: Sequence[float], what: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{what} must be a 3-vector, got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise ValueError(f"{what} must be non-zero")
    return arr / norm


@dataclasses.dataclass(frozen=True)
class TableGrid:
    """A ``rows x cols`` grid of cells on a plane in world coordinates.

    Attributes:
        origin: World-frame position of the grid's outer corner — the corner of
            cell ``(0, 0)`` furthest from every other cell. Cell centres are
            offset half a cell from it.
        u_axis: Direction of increasing **column**. Normalised on construction.
        v_axis: Direction of increasing **row**. Orthogonalised against
            ``u_axis`` and normalised on construction, so a hand-written config
            can give approximate axes and still get a square grid.
        cell_size_u: Cell width along ``u_axis``, in metres.
        cell_size_v: Cell height along ``v_axis``, in metres.
        rows: Number of rows.
        cols: Number of columns.
        marker_height: Offset along the plane normal, in metres, at which the
            per-cell target marker is drawn. Set this to roughly half the
            object's height so the marker sits where the object's centroid will
            be rather than sunk into the table.
        up_hint: Disambiguates which way :attr:`normal` points. ``u x v`` is a
            normal of the plane, but whether it points up out of the table or
            down into it depends on the order the two axes were written in —
            and a ``marker_height`` applied the wrong way sinks every target
            marker into the tabletop. The normal is flipped, if needed, to have
            a non-negative dot product with this vector. The world-frame "up" of
            a horizontal work surface is the default; set it to the surface's
            outward direction for a grid on a wall or an incline.
        name: Identifies the grid when a config declares more than one (e.g. one
            for the cable and one for the box).
        first_col: The number the first column is labelled with; ``1`` by
            default. Set it to ``0`` when a grid is widened by one column *in
            front of* its first one after data has been collected: the new
            column is ``A0``, ``B0``, … and every existing label still names
            the same cell, so the merged results stay comparable. Widening
            by more than one column is not a rename this can express; extend
            behind the last column instead, where labels just continue.
    """

    origin: tuple[float, float, float]
    u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    cell_size_u: float = 0.10
    cell_size_v: float = 0.10
    rows: int = 3
    cols: int = 3
    marker_height: float = 0.0
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0)
    name: str = "table"
    first_col: int = 1

    def __post_init__(self) -> None:
        if self.first_col not in (0, 1):
            raise ValueError(
                f"first_col must be 0 or 1, got {self.first_col}: columns are "
                "numbered from 1, or from 0 for a grid widened by one column "
                "in front of its first"
            )
        if self.rows < 1 or self.cols < 1:
            raise ValueError(
                f"grid must have at least one row and column, "
                f"got rows={self.rows}, cols={self.cols}"
            )
        if self.cell_size_u <= 0 or self.cell_size_v <= 0:
            raise ValueError(
                f"cell sizes must be positive, got "
                f"({self.cell_size_u}, {self.cell_size_v})"
            )
        origin = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        if origin.shape != (3,):
            raise ValueError(f"origin must be a 3-vector, got shape {origin.shape}")

        u = _unit(self.u_axis, "u_axis")
        v = _unit(self.v_axis, "v_axis")
        # Gram-Schmidt: a config author gives the two table edges by eye, and
        # they are rarely exactly perpendicular. Squaring them here means the
        # cell an operator is shown is the cell whose centre gets logged.
        v_orth = v - float(np.dot(v, u)) * u
        if float(np.linalg.norm(v_orth)) < 1e-6:
            raise ValueError(
                "u_axis and v_axis are parallel; they must span the grid plane"
            )
        v = v_orth / float(np.linalg.norm(v_orth))

        hint = _unit(self.up_hint, "up_hint")
        if abs(float(np.dot(hint, np.cross(u, v)))) < 1e-6:
            raise ValueError(
                "up_hint lies in the grid plane, so it cannot orient the "
                "normal. Point it out of the plane (world +z for a tabletop)."
            )

        object.__setattr__(self, "origin", tuple(float(x) for x in origin))
        object.__setattr__(self, "up_hint", tuple(float(x) for x in hint))
        object.__setattr__(self, "u_axis", tuple(float(x) for x in u))
        object.__setattr__(self, "v_axis", tuple(float(x) for x in v))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def normal(self) -> np.ndarray:
        """Unit normal of the grid plane, oriented by :attr:`up_hint`.

        ``u x v`` up to a sign; the sign is chosen so the normal points the
        same way as ``up_hint`` (world +z by default). That makes a positive
        :attr:`marker_height` always lift *off* the surface, whichever order the
        two in-plane axes happen to be written in.
        """
        n = np.cross(np.asarray(self.u_axis), np.asarray(self.v_axis))
        hint = np.asarray(self.up_hint, dtype=np.float64)
        if float(np.dot(n, hint)) < 0:
            n = -n
        return n

    @property
    def num_cells(self) -> int:
        return self.rows * self.cols

    @property
    def extent_u(self) -> float:
        """Width of the whole grid along ``u_axis``, in metres."""
        return self.cols * self.cell_size_u

    @property
    def extent_v(self) -> float:
        """Height of the whole grid along ``v_axis``, in metres."""
        return self.rows * self.cell_size_v

    def point_at(self, u: float, v: float, *, lifted: bool = False) -> np.ndarray:
        """World point *u* metres along ``u_axis`` and *v* along ``v_axis``.

        The metric twin of :meth:`corner_point`, which counts in cells. Used
        where a position inside the rectangle is continuous rather than snapped
        to a cell — a free placement region, for instance.

        Args:
            lifted: Offset by :attr:`marker_height` along the plane normal, as
                :meth:`cell_center` does.
        """
        point = (
            np.asarray(self.origin, dtype=np.float64)
            + float(u) * np.asarray(self.u_axis)
            + float(v) * np.asarray(self.v_axis)
        )
        if lifted and self.marker_height:
            point = point + self.marker_height * self.normal
        return point

    def outline(self, *, samples: int = 5) -> np.ndarray:
        """The grid's outer boundary as a closed ``[4*(samples-1)+1, 3]`` polyline.

        Sampled along each edge for the same reason :meth:`grid_lines` is: an
        edge can leave the frame partway along, and a sampled polyline degrades
        to the visible part instead of vanishing.
        """
        if samples < 2:
            raise ValueError(f"samples must be >= 2, got {samples}")
        corners = [
            (0.0, 0.0),
            (0.0, self.cols),
            (self.rows, self.cols),
            (self.rows, 0.0),
        ]
        points: list[np.ndarray] = []
        for i in range(4):
            r0, c0 = corners[i]
            r1, c1 = corners[(i + 1) % 4]
            ts = np.linspace(0.0, 1.0, samples)[:-1]
            points.extend(
                self.corner_point(r0 + t * (r1 - r0), c0 + t * (c1 - c0)) for t in ts
            )
        points.append(self.corner_point(*corners[0]))
        return np.stack(points)

    def cells(self) -> Iterator[CellId]:
        """Every cell in row-major order."""
        for r in range(self.rows):
            for c in range(self.cols):
                yield (r, c)

    def contains(self, cell: CellId) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _check(self, cell: CellId) -> CellId:
        if not self.contains(cell):
            raise KeyError(
                f"cell {cell} outside {self.rows}x{self.cols} grid {self.name!r}"
            )
        return cell

    def corner_point(self, row: float, col: float) -> np.ndarray:
        """World point at fractional grid coordinates, measured from ``origin``.

        ``(0, 0)`` is the grid corner and ``(rows, cols)`` the opposite one, so
        cell ``(r, c)``'s centre is ``corner_point(r + 0.5, c + 0.5)``.
        """
        return (
            np.asarray(self.origin, dtype=np.float64)
            + col * self.cell_size_u * np.asarray(self.u_axis)
            + row * self.cell_size_v * np.asarray(self.v_axis)
        )

    def cell_center(self, cell: CellId, *, lifted: bool = False) -> np.ndarray:
        """World-frame centre of *cell*.

        Args:
            lifted: Offset the point by :attr:`marker_height` along the plane
                normal — where the object's centroid sits, rather than where it
                touches the table.
        """
        r, c = self._check(cell)
        point = self.corner_point(r + 0.5, c + 0.5)
        if lifted and self.marker_height:
            point = point + self.marker_height * self.normal
        return point

    def cell_corners(self, cell: CellId) -> np.ndarray:
        """``[4, 3]`` world-frame corners of *cell*, counter-clockwise in-plane."""
        r, c = self._check(cell)
        return np.stack(
            [
                self.corner_point(r, c),
                self.corner_point(r, c + 1),
                self.corner_point(r + 1, c + 1),
                self.corner_point(r + 1, c),
            ]
        )

    def cell_of(self, point: Sequence[float]) -> CellId | None:
        """Which cell *point* projects into, or ``None`` if it is outside.

        The point is projected onto the grid plane first, so an object sitting
        above the table still resolves to the cell beneath it.
        """
        rel = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(self.origin)
        col = float(np.dot(rel, np.asarray(self.u_axis))) / self.cell_size_u
        row = float(np.dot(rel, np.asarray(self.v_axis))) / self.cell_size_v
        cell = (int(np.floor(row)), int(np.floor(col)))
        return cell if self.contains(cell) else None

    def grid_lines(self, *, samples: int = 5) -> list[np.ndarray]:
        """Every grid line as an ``[samples, 3]`` polyline of world points.

        The lines are straight in 3-D and a pinhole camera maps them to straight
        segments, so two points would suffice geometrically. They are sampled
        instead because a line can leave the frame — or pass behind the camera —
        partway along, and a sampled polyline degrades to the visible part
        rather than vanishing.
        """
        if samples < 2:
            raise ValueError(f"samples must be >= 2, got {samples}")
        lines: list[np.ndarray] = []
        for r in range(self.rows + 1):
            ts = np.linspace(0.0, self.cols, samples)
            lines.append(np.stack([self.corner_point(r, t) for t in ts]))
        for c in range(self.cols + 1):
            ts = np.linspace(0.0, self.rows, samples)
            lines.append(np.stack([self.corner_point(t, c) for t in ts]))
        return lines

    def label(self, cell: CellId) -> str:
        return cell_label(*self._check(cell), first_col=self.first_col)

    def parse_label(self, label: str) -> CellId:
        """Inverse of :meth:`label`, numbering columns as this grid does.

        Raises ``ValueError`` on a malformed label; the cell is *not* checked
        against the grid's extent (see :meth:`contains`).
        """
        return parse_cell_label(label, first_col=self.first_col)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "origin": list(self.origin),
            "u_axis": list(self.u_axis),
            "v_axis": list(self.v_axis),
            "cell_size_u": self.cell_size_u,
            "cell_size_v": self.cell_size_v,
            "rows": self.rows,
            "cols": self.cols,
            "marker_height": self.marker_height,
            "up_hint": list(self.up_hint),
            "first_col": self.first_col,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TableGrid":
        return cls(
            origin=tuple(data["origin"]),
            u_axis=tuple(data.get("u_axis", (1.0, 0.0, 0.0))),
            v_axis=tuple(data.get("v_axis", (0.0, 1.0, 0.0))),
            cell_size_u=float(data.get("cell_size_u", 0.10)),
            cell_size_v=float(data.get("cell_size_v", 0.10)),
            rows=int(data.get("rows", 3)),
            cols=int(data.get("cols", 3)),
            marker_height=float(data.get("marker_height", 0.0)),
            up_hint=tuple(data.get("up_hint", (0.0, 0.0, 1.0))),
            name=str(data.get("name", "table")),
            first_col=int(data.get("first_col", 1)),
        )


__all__ = ["CellId", "TableGrid", "cell_label", "parse_cell_label"]

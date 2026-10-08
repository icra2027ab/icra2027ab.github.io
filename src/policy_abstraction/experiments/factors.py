"""The axes an experiment varies, and how each one is shown to the operator.

A *factor* is one thing that changes between rollouts — where the cable sits,
which way it points, which box is on the table. A *level* is one setting of it.
An experiment is the cross product of its factors' levels, repeated.

Everything that is specific to a factor lives behind this interface:

* what its levels are (``levels``),
* how to tell the operator to stage one (``instruction``),
* how to draw it on the live camera view (``annotate``),
* and, for positional factors, where it is in the world (``world_position``),
  which is what gets logged next to the outcome.

Two further hooks matter once an experiment has more than one moving part:

* ``prompt_value`` lets a factor fill a placeholder in the task template, which
  is how the disambiguation experiment changes what the policy is *told*
  between rollouts while the scene stays put.
* ``link`` lets a factor resolve a sibling it names in its config, once, before
  the plan is built — :class:`TargetObjectFactor` takes its levels from the
  :class:`ObjectLayoutFactor` that stages the objects, so the object list is
  written in one place.

Adding an experiment therefore means adding a class here and naming it in a YAML
file — no change to the session, the GUI, the store or the report.
:class:`GridPositionFactor`, :class:`OrientationFactor`,
:class:`AppearanceFactor`, :class:`ObjectLayoutFactor` and
:class:`TargetObjectFactor` are all implemented, so the experiments built out of
them are configuration rather than code.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Optional, Sequence

import numpy as np

from policy_abstraction.experiments.grid import CellId, TableGrid
from policy_abstraction.experiments.layout import (
    DEFAULT_MARGIN,
    DEFAULT_MIN_SEPARATION,
    SceneObject,
    layout_label,
    parse_objects,
    sample_layouts,
)


@dataclasses.dataclass(frozen=True)
class FactorLevel:
    """One setting of one factor."""

    id: str
    """Stable, filename-safe identifier. Written into records and paths, so it
    must not change once an experiment has started."""
    label: str
    """What the operator reads."""
    payload: dict = dataclasses.field(default_factory=dict)
    """Factor-specific data (a cell, an angle, a variant name)."""

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "payload": dict(self.payload)}


class Factor(abc.ABC):
    """One experimental axis.

    Subclasses are registered in :data:`FACTOR_TYPES` under a ``kind`` string so
    an experiment config can name them.
    """

    kind: str = "factor"

    draw_priority: int = 1
    """Order this factor is drawn in, low first. ``0`` is for factors that lay
    down the geometry others sit on — a grid, a placement region — and ``1`` for
    the marks that go on top of it. Explicit rather than inferred, because "does
    this factor have a world position" is a different question from "must this
    factor be drawn first" and they only *usually* have the same answer."""

    def __init__(self, name: str) -> None:
        self.name = name

    def link(self, factors: Sequence["Factor"]) -> None:  # noqa: B027 — optional
        """Resolve references to sibling factors, once, before planning.

        Called by :class:`~policy_abstraction.experiments.config.ExperimentConfig`
        with every factor in the experiment, including this one. A factor that
        names another in its config (``layout: object_layout``) resolves it here
        rather than at draw time, so a typo is an error when the config is
        loaded rather than a silently missing annotation an hour into a session.

        Levels may depend on what is resolved here: linking happens before the
        plan is built or validated.
        """

    @property
    @abc.abstractmethod
    def levels(self) -> tuple[FactorLevel, ...]:
        """Every setting this factor can take, in the order they are swept."""

    def has_level(self, level_id: str) -> bool:
        return any(lv.id == level_id for lv in self.levels)

    def level(self, level_id: str) -> FactorLevel:
        for lv in self.levels:
            if lv.id == level_id:
                return lv
        raise KeyError(
            f"factor {self.name!r} has no level {level_id!r}; "
            f"known: {[lv.id for lv in self.levels]}"
        )

    def instruction(self, level_id: str) -> str:
        """One line telling the operator how to stage this level."""
        return f"{self.name}: {self.level(level_id).label}"

    def staging_lines(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> list[str]:
        """Every line the operator needs to stage this level.

        One line for most factors — :meth:`instruction` — but a factor that
        stages several objects at once has one line per object, and those lines
        are the whole instruction when the cell is uncalibrated and there is no
        overlay to read positions off.
        """
        return [self.instruction(level_id)]

    def prompt_value(self, level_id: str) -> Optional[str]:
        """What this level substitutes into the task template, if anything.

        ``None`` — the default — means this factor does not touch the prompt.
        See :meth:`~policy_abstraction.experiments.config.ExperimentConfig.task_for`.
        """
        return None

    def world_position(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> Optional[np.ndarray]:
        """Where in the world this level puts the object, if it is positional.

        ``None`` for non-positional factors (orientation, appearance).

        *assignments* is the full ``{factor: level}`` for the rollout, for a
        factor whose position depends on another's choice — the target object's
        position is wherever the layout factor put it. Callers that have no
        assignments to hand may omit it, and a factor that needs them returns
        ``None`` rather than guessing.
        """
        return None

    def recorded_positions(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> dict[str, list[float]]:
        """World-frame positions this factor wants written into the record.

        Keyed by a name that is unique across the experiment — the factor's own
        name by default. A factor staging several objects returns one entry per
        object, so the analysis never has to re-derive geometry from a config
        that may since have been edited.
        """
        pos = self.world_position(level_id, assignments)
        return {} if pos is None else {self.name: [float(v) for v in pos]}

    def measurement_queries(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> dict[str, str]:
        """What to ask the perception pipeline for, to *measure* each position.

        ``{key: object query}`` with the same keys as :meth:`recorded_positions`,
        so a measured position lands beside the nominal one it checks. Empty
        for a factor that stages nothing positional.
        """
        return {}

    def geometry_lines(self) -> list[str]:
        """Coordinates worth printing during ``--calibrate``, if any.

        This is what an operator tapes to the cell wall: the world-frame points
        the overlay is claiming, in metres, so they can be checked against a
        tape measure rather than against a rendering.
        """
        return []

    def annotate(  # noqa: B027 — optional hook; a factor with no
        # geometry (appearance) deliberately draws nothing.
        self,
        canvas,
        level_id: str,
        assignments: dict[str, str],
    ) -> None:
        """Draw this level's target onto *canvas*.

        Args:
            canvas: An :class:`~policy_abstraction.experiments.overlay.OverlayCanvas`
                — the image plus the projection it was built with.
            level_id: The level being staged.
            assignments: The full ``{factor: level}`` for this rollout, so a
                factor can position itself relative to another (an orientation
                arrow is drawn *in* the cell the position factor selected).
        """

    @abc.abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: dict) -> "Factor": ...


# ---------------------------------------------------------------------------
# Position on a grid
# ---------------------------------------------------------------------------


class GridPositionFactor(Factor):
    """Where an object is placed, as a cell on a :class:`TableGrid`.

    Levels are the grid's cells, labelled ``A1``-style. ``cells`` restricts the
    sweep to a subset — useful when part of the grid is out of the arm's reach
    or occluded, and much better than shrinking the grid, which would renumber
    every remaining cell and break comparison with an earlier experiment.
    """

    kind = "grid_position"
    draw_priority = 0

    def __init__(
        self,
        name: str,
        grid: TableGrid,
        cells: Optional[Sequence[CellId | str]] = None,
        object: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.grid = grid
        self.object = (object or "").strip() or _object_from_factor_name(name)
        """What the perception pipeline is asked to find when the staged
        position is measured back from a recording — ``"cable"`` for a factor
        called ``cable_position``. Set ``object:`` in the YAML when the noun
        the detector needs is not the one the factor is named after."""
        if cells is None:
            selected: list[CellId] = list(grid.cells())
        else:
            selected = [
                grid.parse_label(c) if isinstance(c, str) else (int(c[0]), int(c[1]))
                for c in cells
            ]
            for cell in selected:
                if not grid.contains(cell):
                    raise ValueError(
                        f"factor {name!r}: cell {grid.label(cell) if grid.contains(cell) else cell} "
                        f"is outside the {grid.rows}x{grid.cols} grid"
                    )
        if not selected:
            raise ValueError(f"factor {name!r} has no cells")
        self.cells: tuple[CellId, ...] = tuple(selected)

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        return tuple(
            FactorLevel(
                id=self.grid.label(cell),
                label=self.grid.label(cell),
                payload={"row": cell[0], "col": cell[1]},
            )
            for cell in self.cells
        )

    def cell(self, level_id: str) -> CellId:
        lv = self.level(level_id)
        return int(lv.payload["row"]), int(lv.payload["col"])

    def instruction(self, level_id: str) -> str:
        return f"Place the {self.name} in cell {level_id}"

    def world_position(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> np.ndarray:
        return self.grid.cell_center(self.cell(level_id), lifted=True)

    def measurement_queries(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> dict[str, str]:
        return {self.name: self.object}

    def geometry_lines(self) -> list[str]:
        lines = [f"grid {self.grid.name!r}: {self.grid.rows}x{self.grid.cols}"]
        for cell in self.cells:
            centre = self.grid.cell_center(cell, lifted=True)
            lines.append(
                f"  {self.grid.label(cell):>4}  "
                f"[{centre[0]:+.3f}, {centre[1]:+.3f}, {centre[2]:+.3f}]"
            )
        return lines

    def annotate(self, canvas, level_id: str, assignments: dict[str, str]) -> None:
        canvas.draw_grid(self.grid)
        canvas.highlight_cell(self.grid, self.cell(level_id), label=level_id)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "grid": self.grid.to_dict(),
            "cells": [self.grid.label(c) for c in self.cells],
            "object": self.object,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridPositionFactor":
        return cls(
            name=data["name"],
            grid=TableGrid.from_dict(data["grid"]),
            cells=data.get("cells"),
            object=data.get("object"),
        )


def _object_from_factor_name(name: str) -> str:
    """``cable_position`` -> ``cable``: the noun a positional factor is named after."""
    stem = name.strip().lower()
    for suffix in ("_position", "_pos", "_location", "_placement"):
        if stem.endswith(suffix) and len(stem) > len(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ").strip() or name


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


class OrientationFactor(Factor):
    """Which way the object points, as a yaw angle in the grid plane.

    Angles are degrees measured from the anchor grid's ``u_axis``, rotating
    towards ``v_axis``. Anchoring to the grid rather than to the world means the
    instruction ("point it along 90°") reads the same whichever way the table is
    turned relative to the robot base.
    """

    kind = "orientation"

    def __init__(
        self,
        name: str,
        angles_deg: Sequence[float],
        anchor: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        if not len(angles_deg):
            raise ValueError(f"factor {name!r} needs at least one angle")
        self.angles_deg = tuple(float(a) for a in angles_deg)
        self.anchor = anchor
        """Name of the :class:`GridPositionFactor` whose selected cell the arrow
        is drawn in. ``None`` draws at the centre of the first grid available."""

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        return tuple(
            FactorLevel(
                id=f"{int(round(a)):03d}deg",
                label=f"{a:g}°",
                payload={"angle_deg": a},
            )
            for a in self.angles_deg
        )

    def angle(self, level_id: str) -> float:
        return float(self.level(level_id).payload["angle_deg"])

    def instruction(self, level_id: str) -> str:
        return f"Orient the {self.name} at {self.level(level_id).label}"

    def annotate(self, canvas, level_id: str, assignments: dict[str, str]) -> None:
        anchor = canvas.resolve_anchor(self.anchor, assignments)
        if anchor is None:
            return
        grid, cell = anchor
        canvas.draw_orientation_arrow(grid, cell, self.angle(level_id))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "angles_deg": list(self.angles_deg),
            "anchor": self.anchor,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OrientationFactor":
        return cls(
            name=data["name"],
            angles_deg=data["angles_deg"],
            anchor=data.get("anchor"),
        )


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------


class AppearanceFactor(Factor):
    """Which physical variant of the object is on the table.

    Purely nominal — the operator swaps the object by hand — so it has no
    geometry. It still earns a factor because it must be swept, instructed,
    displayed and logged exactly like the others.
    """

    kind = "appearance"

    def __init__(
        self,
        name: str,
        variants: Sequence[str],
        descriptions: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(name)
        if not len(variants):
            raise ValueError(f"factor {name!r} needs at least one variant")
        self.variants = tuple(str(v) for v in variants)
        self.descriptions = dict(descriptions or {})

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        return tuple(
            FactorLevel(
                id=v,
                label=self.descriptions.get(v, v.replace("_", " ")),
                payload={"variant": v},
            )
            for v in self.variants
        )

    def instruction(self, level_id: str) -> str:
        # Underscores are the factor's id, not the operator's words.
        return f"Use the {self.level(level_id).label} {self.name.replace('_', ' ')}"

    # No annotate(): an appearance has no geometry to draw. The instruction
    # badge the overlay adds for every factor already says which variant to
    # fit, and drawing a second one would just duplicate it.

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "variants": list(self.variants),
            "descriptions": dict(self.descriptions),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppearanceFactor":
        return cls(
            name=data["name"],
            variants=data["variants"],
            descriptions=data.get("descriptions"),
        )


# ---------------------------------------------------------------------------
# Several objects on the table at once
# ---------------------------------------------------------------------------


class ObjectLayoutFactor(Factor):
    """Where a whole set of named objects goes, as one of several layouts.

    A level is a *scene*: one position per object, sampled inside a region of
    the table and kept at least ``min_separation`` apart. Re-sampling per layout
    is what makes the companion :class:`TargetObjectFactor` measure anything —
    with the objects nailed to fixed spots, "went to the right object" and "went
    to the same place as always" are the same observation.

    The layouts are frozen at construction from ``seed``, so they are identical
    across machines and across both arms of a baseline/guided comparison, and
    they are written into the plan the store keeps.
    """

    kind = "object_layout"
    draw_priority = 0

    def __init__(
        self,
        name: str,
        region: TableGrid,
        objects: Sequence[str | dict | SceneObject],
        *,
        layouts: int = 4,
        min_separation: float = DEFAULT_MIN_SEPARATION,
        margin: float = DEFAULT_MARGIN,
        seed: int = 0,
    ) -> None:
        super().__init__(name)
        self.region = region
        self.objects = parse_objects(objects)
        self.num_layouts = int(layouts)
        self.min_separation = float(min_separation)
        self.margin = float(margin)
        self.seed = int(seed)
        self._layouts = sample_layouts(
            region,
            self.objects,
            count=self.num_layouts,
            min_separation=self.min_separation,
            margin=self.margin,
            seed=self.seed,
        )

    # -- objects -------------------------------------------------------

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(o.id for o in self.objects)

    def object(self, object_id: str) -> SceneObject:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        raise KeyError(
            f"factor {self.name!r} has no object {object_id!r}; "
            f"known: {list(self.object_ids)}"
        )

    def positions(self, level_id: str) -> dict[str, np.ndarray]:
        """``{object_id: [x, y, z]}`` for one layout."""
        return self._layouts[self._layout_index(level_id)]

    def position_of(self, level_id: str, object_id: str) -> np.ndarray:
        return self.positions(level_id)[self.object(object_id).id]

    def _layout_index(self, level_id: str) -> int:
        return int(self.level(level_id).payload["index"])

    # -- factor interface ----------------------------------------------

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        return tuple(
            FactorLevel(
                id=layout_label(i),
                label=layout_label(i),
                payload={
                    "index": i,
                    "positions": {
                        k: [float(v) for v in pos] for k, pos in layout.items()
                    },
                },
            )
            for i, layout in enumerate(self._layouts)
        )

    def instruction(self, level_id: str) -> str:
        return (
            f"Layout {level_id}: place {len(self.objects)} objects on the "
            f"marked crosses"
        )

    def staging_lines(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> list[str]:
        """The instruction, then one line per object with its base-frame point.

        The coordinates are there for the uncalibrated case: with no overlay the
        crosses do not exist, and a number an operator can measure to is the
        difference between staging the scene and guessing at it.
        """
        lines = [self.instruction(level_id)]
        for obj in self.objects:
            pos = self.position_of(level_id, obj.id)
            lines.append(
                f"  {obj.label}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
            )
        return lines

    def measurement_queries(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> dict[str, str]:
        return {f"{self.name}.{obj.id}": obj.label for obj in self.objects}

    def recorded_positions(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> dict[str, list[float]]:
        """One entry per object, keyed ``<factor>.<object>``.

        Every object's position is recorded, not just the target's: where the
        *distractors* were is the covariate a disambiguation failure has to be
        explained against.
        """
        return {
            f"{self.name}.{obj_id}": [float(v) for v in pos]
            for obj_id, pos in self.positions(level_id).items()
        }

    def geometry_lines(self) -> list[str]:
        lines = [
            f"region {self.region.name!r}: "
            f"{self.region.extent_u * 100:.0f} x {self.region.extent_v * 100:.0f} cm, "
            f"{self.num_layouts} layouts, "
            f">= {self.min_separation * 100:.0f} cm apart"
        ]
        for level in self.levels:
            lines.append(f"  layout {level.id}")
            for obj in self.objects:
                pos = self.position_of(level.id, obj.id)
                lines.append(
                    f"    {obj.label:>18}  "
                    f"[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
                )
        return lines

    def annotate(self, canvas, level_id: str, assignments: dict[str, str]) -> None:
        canvas.draw_region(self.region)
        for obj in self.objects:
            position = self.position_of(level_id, obj.id)
            canvas.mark_object(obj.id, position, label=obj.label)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "region": self.region.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "layouts": self.num_layouts,
            "min_separation": self.min_separation,
            "margin": self.margin,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectLayoutFactor":
        region = data.get("region") or data.get("grid")
        if region is None:
            raise ValueError(
                f"factor {data.get('name')!r} needs a `region:` block — the "
                "patch of table the objects are scattered over."
            )
        return cls(
            name=data["name"],
            region=TableGrid.from_dict(region),
            objects=data["objects"],
            layouts=int(data.get("layouts", 4)),
            min_separation=float(data.get("min_separation", DEFAULT_MIN_SEPARATION)),
            margin=float(data.get("margin", DEFAULT_MARGIN)),
            seed=int(data.get("seed", 0)),
        )


# ---------------------------------------------------------------------------
# Which object the prompt names
# ---------------------------------------------------------------------------


class TargetObjectFactor(Factor):
    """Which of the staged objects the policy is told to pick up.

    The scene is identical across this factor's levels — same objects, same
    places — and only the instruction changes. That is the experiment: any
    difference in success rate between levels is the policy resolving (or
    failing to resolve) the reference in the prompt, with the scene held fixed.

    Its levels come from the layout factor it names, so the object list is
    written once. It contributes ``{<name>}`` to the task template, which is how
    the prompt actually changes per rollout — see
    :meth:`~policy_abstraction.experiments.config.ExperimentConfig.task_for`.
    """

    kind = "target_object"

    def __init__(self, name: str, layout: str) -> None:
        super().__init__(name)
        self.layout = layout
        """Name of the :class:`ObjectLayoutFactor` that stages the objects."""
        self._layout_factor: Optional[ObjectLayoutFactor] = None

    def link(self, factors: Sequence[Factor]) -> None:
        for factor in factors:
            if factor.name != self.layout:
                continue
            if not isinstance(factor, ObjectLayoutFactor):
                raise ValueError(
                    f"factor {self.name!r} points `layout:` at {self.layout!r}, "
                    f"which is a {factor.kind!r} factor. It must name an "
                    f"{ObjectLayoutFactor.kind!r} factor — the one that puts the "
                    "objects on the table."
                )
            self._layout_factor = factor
            return
        raise ValueError(
            f"factor {self.name!r} points `layout:` at {self.layout!r}, and no "
            f"factor by that name exists; known: {[f.name for f in factors]}"
        )

    @property
    def layout_factor(self) -> ObjectLayoutFactor:
        if self._layout_factor is None:
            raise RuntimeError(
                f"factor {self.name!r} has not been linked to its layout factor. "
                "Build it through ExperimentConfig, which links every factor "
                "before the plan is built."
            )
        return self._layout_factor

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        return tuple(
            FactorLevel(
                id=obj.id,
                label=obj.label,
                payload={"prompt_name": obj.prompt_name},
            )
            for obj in self.layout_factor.objects
        )

    def instruction(self, level_id: str) -> str:
        return f"TARGET: the {self.level(level_id).label} — the others are distractors"

    def prompt_value(self, level_id: str) -> str:
        return str(self.level(level_id).payload["prompt_name"])

    def world_position(
        self, level_id: str, assignments: Optional[dict[str, str]] = None
    ) -> Optional[np.ndarray]:
        """Where the target is — which is wherever the chosen layout put it."""
        layout = self.layout_factor
        layout_level = (assignments or {}).get(layout.name)
        if layout_level is None:
            return None
        return layout.position_of(layout_level, level_id)

    def annotate(self, canvas, level_id: str, assignments: dict[str, str]) -> None:
        canvas.highlight_object(level_id, label=self.level(level_id).label)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "layout": self.layout}

    @classmethod
    def from_dict(cls, data: dict) -> "TargetObjectFactor":
        return cls(name=data["name"], layout=data["layout"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FACTOR_TYPES: dict[str, type[Factor]] = {
    GridPositionFactor.kind: GridPositionFactor,
    OrientationFactor.kind: OrientationFactor,
    AppearanceFactor.kind: AppearanceFactor,
    ObjectLayoutFactor.kind: ObjectLayoutFactor,
    TargetObjectFactor.kind: TargetObjectFactor,
}


def register_factor(cls: type[Factor]) -> type[Factor]:
    """Register a new factor type so configs can name it. Usable as a decorator."""
    FACTOR_TYPES[cls.kind] = cls
    return cls


def factor_from_dict(data: dict) -> Factor:
    kind = data.get("kind")
    if kind not in FACTOR_TYPES:
        raise ValueError(
            f"unknown factor kind {kind!r}; known: {sorted(FACTOR_TYPES)}. "
            "Register new types with experiments.factors.register_factor."
        )
    return FACTOR_TYPES[kind].from_dict(data)


__all__ = [
    "FACTOR_TYPES",
    "AppearanceFactor",
    "Factor",
    "FactorLevel",
    "GridPositionFactor",
    "ObjectLayoutFactor",
    "OrientationFactor",
    "TargetObjectFactor",
    "factor_from_dict",
    "register_factor",
]

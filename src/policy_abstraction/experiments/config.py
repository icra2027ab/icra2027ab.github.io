"""The YAML file that defines an experiment.

One file describes the grid, the factors, how many repeats, and the per-rollout
limits. It is written once — by hand or by ``--calibrate`` — and then copied into
the experiment directory, so the config a set of results was collected under
travels with the results and cannot drift away from them.

A minimal positional-shift experiment::

    name: cable_in_box_position
    task: "Put the cable in the box."
    repeats: 3
    max_steps: 600
    timeout_s: 90
    factors:
      - kind: grid_position
        name: cable_position
        grid:
          origin: [0.35, -0.20, 0.02]
          u_axis: [0, 1, 0]
          v_axis: [1, 0, 0]
          cell_size_u: 0.10
          cell_size_v: 0.10
          rows: 3
          cols: 4
          marker_height: 0.015

Adding the follow-up experiments is adding a stanza. Orientation::

      - kind: orientation
        name: cable_orientation
        anchor: cable_position
        angles_deg: [0, 45, 90, 135]

Appearance::

      - kind: appearance
        name: cable_appearance
        variants: [black_usb, white_hdmi, coiled_grey]
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Optional, Sequence

import yaml
from loguru import logger

from policy_abstraction.experiments.factors import (
    Factor,
    ObjectLayoutFactor,
    factor_from_dict,
)
from policy_abstraction.experiments.grid import TableGrid
from policy_abstraction.experiments.plan import (
    PlanGrouping,
    PlannedRollout,
    PlanOrder,
    build_plan,
    validate_factors,
)


@dataclasses.dataclass
class ExperimentConfig:
    """Everything about *what* to run, independent of which policy runs it."""

    name: str
    """Experiment identity. Also the directory name, so keep it path-safe."""
    task: str = ""
    """Natural-language instruction. Empty means "use whatever the env server
    publishes for the selected task", which is the right default in simulation
    and the wrong one on hardware, where the prompt is part of the checkpoint's
    contract."""
    factors: list[Factor] = dataclasses.field(default_factory=list)
    repeats: int = 1
    """How many times each condition is run."""
    order: PlanOrder = "systematic"
    grouping: PlanGrouping = "condition"
    """``condition`` runs all repeats of a cell back to back; ``sweep`` runs
    the whole grid once per repeat. See :mod:`policy_abstraction.experiments.plan`."""
    seed: int = 0

    # -- per-rollout limits ------------------------------------------------
    max_steps: Optional[int] = None
    """Step cap. ``None`` uses the env server's own per-task horizon."""
    timeout_s: Optional[float] = None
    """Wall-clock cap. Exceeding it ends the rollout and records it as a
    ``TIMEOUT`` failure on its own; the operator is then asked what the arm
    was doing, not whether it failed. ``None`` disables
    the wall-clock cap and leaves ``max_steps`` as the only bound."""

    # -- presentation ------------------------------------------------------
    camera: Optional[str] = None
    """Which camera the operator stages against. ``None`` uses the env's
    primary camera, which is the one the policy sees."""
    view_scale: float = 2.0
    """Upscale factor for the operator's view. Sim frames are 256 px."""

    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("experiment name must be non-empty")
        # Before validation: a factor may take its levels from a sibling it
        # only resolves here, and a factor with no levels is what validation
        # rejects.
        for factor in self.factors:
            factor.link(self.factors)
        validate_factors(self.factors)
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")

    # ------------------------------------------------------------------

    def build_plan(self) -> tuple[PlannedRollout, ...]:
        return build_plan(
            self.factors,
            repeats=self.repeats,
            order=self.order,
            grouping=self.grouping,
            seed=self.seed,
        )

    def factor(self, name: str) -> Factor:
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(f"no factor {name!r}; known: {[f.name for f in self.factors]}")

    @property
    def grids(self) -> list[TableGrid]:
        """Every grid or region any factor references, for plotting."""
        seen: dict[str, TableGrid] = {}
        for f in self.factors:
            for attr in ("grid", "region"):
                surface = getattr(f, attr, None)
                if isinstance(surface, TableGrid):
                    seen.setdefault(f.name, surface)
        return list(seen.values())

    def positional_factors(self) -> list[Factor]:
        """Factors that place an object somewhere — the ones plots can map."""
        return [
            f
            for f in self.factors
            if f.levels and f.world_position(f.levels[0].id) is not None
        ]

    def measurement_queries(self, assignments: dict[str, str]) -> dict[str, str]:
        """``{position key: object query}`` for every positional factor staged."""
        out: dict[str, str] = {}
        for f in self.factors:
            level_id = assignments.get(f.name)
            if level_id is None:
                continue
            out.update(f.measurement_queries(level_id, assignments))
        return out

    def target_positions(self, assignments: dict[str, str]) -> dict[str, list[float]]:
        """Every world-frame position the staged scene involves, under *assignments*.

        One entry per positional factor for the simple cases, and one per object
        for a factor that stages several at once. Passed the whole assignment
        dict, so a factor whose position depends on another's choice — the
        target object is wherever the layout put it — can answer.
        """
        out: dict[str, list[float]] = {}
        for f in self.factors:
            level_id = assignments.get(f.name)
            if level_id is None:
                continue
            out.update(f.recorded_positions(level_id, assignments))
        return out

    # ------------------------------------------------------------------
    # Named objects
    # ------------------------------------------------------------------

    def scene_objects(self) -> list[tuple[str, str]]:
        """``(id, label)`` for every named object the scene stages at once.

        Empty for an experiment with no object-layout factor — a positional
        experiment stages one thing, so there is nothing to confuse it with,
        and the operator is never asked which object the policy went for.
        """
        out: list[tuple[str, str]] = []
        for factor in self.factors:
            if isinstance(factor, ObjectLayoutFactor):
                out += [(obj.id, obj.label) for obj in factor.objects]
        return out

    def target_factor(self) -> Optional[Factor]:
        """The factor whose level names the object the prompt asks for.

        ``None`` when nothing names one — including a layout experiment that
        stages the scene and always asks for the same thing, which is a
        legitimate configuration with no reference to resolve.
        """
        for factor in self.factors:
            if factor.kind == "target_object":
                return factor
        return None

    # ------------------------------------------------------------------
    # The prompt
    # ------------------------------------------------------------------

    def task_for(self, assignments: dict[str, str]) -> str:
        """The instruction the policy is given for this condition.

        :attr:`task` is a template: a factor that contributes a
        :meth:`~policy_abstraction.experiments.factors.Factor.prompt_value` fills
        the ``{factor_name}`` placeholder named after it. That is how the
        disambiguation experiment changes the prompt per rollout —
        ``"Put the {target_object} in the bag."`` with a ``target_object``
        factor becomes ``"Put the red mug in the bag."`` — while an experiment
        whose task has no placeholders gets its task back unchanged.

        A template naming something no factor supplies is left as written and
        logged, rather than raising: the run is already staged and mid-session,
        and a literal prompt is a recoverable mistake where a crash is not.
        """
        template = self.task
        if not template:
            return template
        fields = {}
        for f in self.factors:
            level_id = assignments.get(f.name)
            if level_id is None:
                continue
            value = f.prompt_value(level_id)
            if value is not None:
                fields[f.name] = value
        if not fields:
            return template
        try:
            return template.format(**fields)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning(
                f"Task template {template!r} could not be filled ({exc}); "
                f"sending it to the policy verbatim. Placeholders available: "
                f"{sorted(fields)}."
            )
            return template

    def with_objects(self, objects: Sequence[Any]) -> "ExperimentConfig":
        """A copy whose object-layout factors stage *objects* instead.

        For trying a different set, or a different *number*, of objects without
        editing the YAML. Everything downstream follows: the layouts are
        re-sampled for the new count, and a linked ``target_object`` factor
        takes its levels from the new list.

        This changes the plan, so an experiment already collecting under the old
        list will refuse to resume — which is correct, and why the flag that
        reaches here says so.
        """
        data = self.to_dict()
        staged = [
            f for f in data["factors"] if f.get("kind") == ObjectLayoutFactor.kind
        ]
        if not staged:
            raise ValueError(
                f"experiment {self.name!r} has no {ObjectLayoutFactor.kind!r} "
                "factor, so there is no object list to replace."
            )
        for factor in staged:
            factor["objects"] = list(objects)
        return ExperimentConfig.from_dict(data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "repeats": self.repeats,
            "order": self.order,
            "grouping": self.grouping,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "timeout_s": self.timeout_s,
            "camera": self.camera,
            "view_scale": self.view_scale,
            "notes": self.notes,
            "factors": [f.to_dict() for f in self.factors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        factors: Sequence[dict] = data.get("factors") or []
        return cls(
            name=str(data["name"]),
            task=str(data.get("task", "")),
            factors=[factor_from_dict(f) for f in factors],
            repeats=int(data.get("repeats", 1)),
            order=data.get("order", "systematic"),
            grouping=data.get("grouping", "condition"),
            seed=int(data.get("seed", 0)),
            max_steps=data.get("max_steps"),
            timeout_s=data.get("timeout_s"),
            camera=data.get("camera"),
            view_scale=float(data.get("view_scale", 2.0)),
            notes=str(data.get("notes", "")),
        )

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=None),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ExperimentConfig":
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"experiment config not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a YAML mapping")
        return cls.from_dict(data)


__all__ = ["ExperimentConfig"]

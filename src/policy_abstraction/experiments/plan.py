"""The ordered list of conditions an experiment will stage.

The plan is fixed when the experiment is created and stored with it. That is
what makes resume meaningful: rollout 17 is the same condition today as it was
yesterday, and two policy settings run against the *same* sequence of
placements, so their success rates are comparable cell by cell rather than only
in aggregate.

Repeats are grouped one of two ways, chosen by ``grouping``:

* ``condition`` (the default): every repeat of a condition runs back to back
  (``A1, A1, A1, A2, A2, A2, …``). The object is staged once per cell and
  left there, which is the fastest to operate and the least error-prone —
  the operator is not walking a cable across the table between every
  rollout.
* ``sweep``: the full cross product is swept once, then again, then again
  (``A1, A2, …, A1, A2, …``). An experiment stopped halfway through then
  still has balanced coverage of every cell, at the cost of a move between
  every rollout.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Literal, Sequence

import numpy as np

from policy_abstraction.experiments.factors import Factor

PlanOrder = Literal["systematic", "random"]
PlanGrouping = Literal["condition", "sweep"]


@dataclasses.dataclass(frozen=True)
class PlannedRollout:
    """One condition to stage, and its position in the plan."""

    index: int
    """0-based position in the plan. The stable key used by resume."""
    assignments: dict[str, str]
    """``{factor_name: level_id}``."""
    repeat: int
    """Which pass over the cross product this rollout belongs to, 0-based."""

    def describe(self, factors: Sequence[Factor]) -> str:
        """One-line human summary, e.g. ``cable_position=B3, box=black``."""
        by_name = {f.name: f for f in factors}
        parts = []
        for name, level_id in self.assignments.items():
            factor = by_name.get(name)
            label = factor.level(level_id).label if factor is not None else level_id
            parts.append(f"{name}={label}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "assignments": dict(self.assignments),
            "repeat": self.repeat,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlannedRollout":
        return cls(
            index=int(data["index"]),
            assignments=dict(data["assignments"]),
            repeat=int(data.get("repeat", 0)),
        )


def validate_factors(factors: Sequence[Factor]) -> None:
    """Reject factor sets that cannot be planned against.

    Names key the assignment dict, so duplicates would silently drop a factor.
    That failure surfaces far downstream — as a level id being looked up on the
    wrong factor — so it is caught here instead.
    """
    if not factors:
        raise ValueError("an experiment needs at least one factor")
    names = [f.name for f in factors]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"factor names must be unique; duplicated: {duplicates}. "
            "Name them for the axis, not the object (e.g. 'cable_position' and "
            "'cable_orientation' rather than two factors both called 'cable')."
        )
    for factor in factors:
        if not factor.levels:
            raise ValueError(f"factor {factor.name!r} has no levels")
        level_ids = [lv.id for lv in factor.levels]
        if len(set(level_ids)) != len(level_ids):
            raise ValueError(f"factor {factor.name!r} has duplicate level ids")


def build_plan(
    factors: Sequence[Factor],
    *,
    repeats: int = 1,
    order: PlanOrder = "systematic",
    grouping: PlanGrouping = "condition",
    seed: int = 0,
) -> tuple[PlannedRollout, ...]:
    """Enumerate the conditions to run.

    Args:
        factors: The experiment's axes. Every combination of their levels is a
            condition.
        repeats: How many times each condition is run.
        order: ``systematic`` keeps the cross product in factor order — the
            operator moves the object a short, predictable distance between
            conditions, which is faster to stage and less error-prone.
            ``random`` shuffles the conditions, which guards against an
            ordering confound (arm warm-up, lighting drift, operator fatigue)
            at the cost of longer object moves. With ``sweep`` grouping each
            sweep is shuffled separately.
        grouping: ``condition`` runs every repeat of a condition back to back
            before moving to the next; ``sweep`` runs the full cross product
            once per repeat. See the module docstring for the trade-off.
        seed: Seeds the shuffle. Two policy settings run with the same seed get
            the same order, which is the point.

    Returns:
        The plan, indexed ``0..n-1``.
    """
    validate_factors(factors)
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if order not in ("systematic", "random"):
        raise ValueError(f"order must be 'systematic' or 'random', got {order!r}")
    if grouping not in ("condition", "sweep"):
        raise ValueError(f"grouping must be 'condition' or 'sweep', got {grouping!r}")

    combinations = [
        dict(zip([f.name for f in factors], level_ids, strict=True))
        for level_ids in itertools.product(
            *[[lv.id for lv in f.levels] for f in factors]
        )
    ]

    rng = np.random.default_rng(seed)
    plan: list[PlannedRollout] = []

    def append(assignments: dict[str, str], repeat: int) -> None:
        plan.append(
            PlannedRollout(
                index=len(plan), assignments=dict(assignments), repeat=repeat
            )
        )

    if grouping == "condition":
        block = list(combinations)
        if order == "random":
            rng.shuffle(block)  # type: ignore[arg-type]
        for assignments in block:
            for repeat in range(repeats):
                append(assignments, repeat)
        return tuple(plan)

    for repeat in range(repeats):
        block = list(combinations)
        if order == "random":
            # Shuffle within the sweep, not across the whole plan: coverage
            # stays balanced after every completed pass either way.
            rng.shuffle(block)  # type: ignore[arg-type]
        for assignments in block:
            append(assignments, repeat)
    return tuple(plan)


__all__ = [
    "PlanGrouping",
    "PlanOrder",
    "PlannedRollout",
    "build_plan",
    "validate_factors",
]

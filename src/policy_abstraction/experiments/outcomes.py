"""What happened in one rollout, and how it is written down.

A rollout ends for exactly one reason, and the reason is worth more than a
boolean: an afternoon of runs that all fail is not informative until you can say
*how* they failed, and whether the how correlates with where the object was.
:class:`FailureMode` is that taxonomy; :class:`RolloutRecord` is the row it lands
in.

The record is deliberately flat and JSON-native. It is appended to a JSONL file
one line at a time, so an interrupted session loses at most the rollout that was
in flight, and a half-finished experiment is still readable by anything that can
read JSON.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any, Iterable, Optional, Union


class Outcome(str, enum.Enum):
    """The verdict on a rollout."""

    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"
    """Discarded — a mis-staged scene, an operator slip, a robot fault. Kept in
    the log for provenance but excluded from every rate the report computes."""

    @property
    def counts_toward_rate(self) -> bool:
        return self is not Outcome.ABORTED


class FailureMode(str, enum.Enum):
    """Why a failed rollout failed. ``NONE`` on success."""

    NONE = "none"
    CATASTROPHIC = "catastrophic"
    """Operator hit the emergency stop: the arm was about to hit something, the
    object left the workspace, the motion was unsafe. Tracked separately from
    ordinary task failure because it is a safety signal, not a capability one."""
    GRASP_MISS = "grasp_miss"
    """Reached the object but closed on nothing, or grasped and immediately lost it."""
    DROPPED = "dropped"
    """Grasped and carried, but released before or outside the target."""
    WRONG_TARGET = "wrong_target"
    """Acted confidently on the wrong object or the wrong destination."""
    NO_MOTION = "no_motion"
    """Never made a meaningful attempt — stalled, drifted, or froze."""
    COLLISION = "collision"
    """Struck the scene or the fixture hard enough to disturb it."""
    TIMEOUT = "timeout"
    """Ran out of step budget or wall-clock without finishing. Recorded
    automatically; the operator never has to select it."""
    OTHER = "other"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


# Modes an operator picks from in the GUI, in the order they are offered.
# TIMEOUT is excluded: it is derived, never chosen.
OPERATOR_FAILURE_MODES: tuple[FailureMode, ...] = (
    FailureMode.GRASP_MISS,
    FailureMode.DROPPED,
    FailureMode.WRONG_TARGET,
    FailureMode.NO_MOTION,
    FailureMode.COLLISION,
    FailureMode.CATASTROPHIC,
    FailureMode.OTHER,
)


FailureModes = Union[FailureMode, Iterable[FailureMode]]
"""What every API that takes a failure classification accepts: one mode, or
any number of them. A rollout rarely fails for exactly one reason — it can
grasp the wrong object *and* drop it on the way — and forcing the operator to
pick one throws the second observation away."""


def normalize_failure_modes(modes: FailureModes | None) -> tuple[FailureMode, ...]:
    """The modes an operator selected, in the order the taxonomy offers them.

    Accepts a single mode or any iterable of them. ``NONE`` is dropped — it is
    the absence of a classification, not a member of one — and duplicates are
    collapsed, so ``(DROPPED, DROPPED)`` is one observation rather than two.
    The order follows :data:`FailureMode` rather than the order given, so two
    operators ticking the same boxes in a different sequence record the same
    thing.
    """
    if modes is None:
        return ()
    if isinstance(modes, FailureMode):
        modes = (modes,)
    chosen = {FailureMode(m) for m in modes} - {FailureMode.NONE}
    return tuple(m for m in FailureMode if m in chosen)


def failure_modes_label(modes: Iterable[FailureMode]) -> str:
    """``"Grasp Miss + Dropped"``; ``""`` when nothing was selected."""
    return " + ".join(m.label for m in normalize_failure_modes(modes))


NO_OBJECT = "(none)"
"""``RolloutRecord.actual_target`` for "it never committed to any of them".

A distinct value rather than an empty string, because "the operator watched and
says it went for nothing" and "nobody was asked" are different observations and
the confusion matrix counts them in different columns. The parentheses keep it
clear of any object id: ids derived from labels are ``[a-z0-9_]`` only.
"""


ActualTargets = Union[str, Iterable[str]]
"""What every API that takes "which object it went for" accepts: one object id,
or any number of them. A policy that cannot resolve the reference does not
always commit to a single wrong object — it can reach for the black cable,
hesitate, and go for the red one — and each of those is a confusion worth
writing down."""


def normalize_actual_targets(targets: ActualTargets | None) -> tuple[str, ...]:
    """The objects an operator said it went for, in the order they gave them.

    Accepts a single id or any iterable of them. Empty strings are dropped
    and duplicates collapsed, keeping the first occurrence — the order is the
    operator's, since "black then red" and "red then black" are different
    observations. :data:`NO_OBJECT` is exclusive: it means nothing was
    reached, so it is dropped as soon as any object is named alongside it.
    """
    if targets is None:
        return ()
    if isinstance(targets, str):
        targets = (targets,)
    seen: list[str] = []
    for t in targets:
        t = str(t)
        if t and t not in seen:
            seen.append(t)
    if len(seen) > 1 and NO_OBJECT in seen:
        seen.remove(NO_OBJECT)
    return tuple(seen)


class StopReason(str, enum.Enum):
    """What ended the rollout's action loop, independent of the verdict."""

    OPERATOR = "operator"
    """The operator pressed stop — success or failure hotkey."""
    ENV_DONE = "env_done"
    """The environment reported ``done`` (a simulator's own success check)."""
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    """Wall-clock budget exhausted."""
    ERROR = "error"
    """The episode raised. The record keeps the message."""


@dataclasses.dataclass
class RolloutRecord:
    """One row of the experiment: what was staged, what happened, what it cost."""

    index: int
    """Position in the plan. Stable across resume — the key used to tell which
    planned rollouts still need running."""
    assignments: dict[str, str]
    """``{factor_name: level_id}`` — the condition this rollout staged."""
    outcome: Outcome
    failure_mode: FailureMode = FailureMode.NONE
    """The primary failure mode: the first of :attr:`failure_modes` in
    taxonomy order, or ``NONE``. Kept as its own column so a record written
    before several modes could be selected reads exactly as it did, and so
    anything that wants *one* answer per rollout has one."""
    failure_modes: list[FailureMode] = dataclasses.field(default_factory=list)
    """Every failure mode the operator selected. Empty on a success or an
    aborted rollout. Usually one entry; several when the rollout failed in
    more than one way, e.g. went for the wrong object and then dropped it.
    Always consistent with :attr:`failure_mode` — construct with either and
    the other follows; amend through :meth:`set_failure_modes`."""
    stop_reason: StopReason = StopReason.OPERATOR

    # -- automatically captured -------------------------------------------
    steps: int = 0
    inferences: int = 0
    duration_s: float = 0.0
    """Wall clock from the operator pressing start to the verdict being
    recorded. See :attr:`warmup` before averaging these."""
    warmup: bool = False
    """This was the first rollout of its session to reach the policy, so its
    :attr:`duration_s` includes the server's warm-up — the first inference
    against a freshly loaded checkpoint costs far more than the rest, and on
    hardware that is minutes, not milliseconds. Kept and reported, but excluded
    from every completion-time statistic: one warm-up in a plan of thirty-six
    moves the mean by more than the effect these experiments are measuring."""
    timed_out: bool = False
    started_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float = 0.0

    # -- provenance --------------------------------------------------------
    episode_path: Optional[str] = None
    """The ``.vla`` this rollout was recorded to, relative to the experiment
    directory."""
    video_path: Optional[str] = None
    target_positions: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    """``{factor_name: [x, y, z]}`` world-frame target for each positional
    factor. Recorded so the analysis does not have to re-derive geometry from
    the config, which may since have been edited."""
    observed_positions: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    """``{key: [x, y, z]}`` where the object *actually* was, measured from the
    rollout's first frame by the perception pipeline — keyed like
    :attr:`target_positions`, so the two are directly comparable. Empty until
    ``scripts/experiments/measure_positions.py`` has run over the experiment;
    the report then draws these instead of the nominal cell centres."""
    task: str = ""
    """The instruction the policy was actually given. Constant across a
    positional experiment and the whole independent variable in a
    disambiguation one, where the template is filled per rollout — so it is
    recorded rather than reconstructed from the config after the fact."""
    actual_target: str = ""
    """Which object the policy actually went for, when the scene held several.

    The whole point of a disambiguation experiment is *which* object was
    reached, not merely whether it was the right one, and that is a judgement
    only the operator watching can make — so it is asked for alongside the
    failure mode and recorded here. A success sets it to the object the prompt
    named, since that is what a success means.

    :data:`NO_OBJECT` when the operator says it committed to nothing. Empty
    when the experiment stages no named objects (a positional experiment has
    nothing to confuse), and on rollouts collected before this was asked for,
    where the report counts it as unrecorded rather than guessing.

    The first of :attr:`actual_targets` when several were reached. Kept as
    its own column so anything that wants *one* answer per rollout has one,
    and so a record written before several could be given reads as it did."""
    actual_targets: list[str] = dataclasses.field(default_factory=list)
    """Every object the operator says it went for, in the order it went for
    them. Usually one entry; several when the policy reached for more than one
    object before the rollout was stopped. Always consistent with
    :attr:`actual_target` — construct with either and the other follows;
    amend through :meth:`set_actual_targets`."""
    notes: str = ""
    error: Optional[str] = None
    policy_setting: str = ""
    """Which arm of the experiment this rollout belongs to (e.g. ``baseline`` or
    ``guided``). Redundant with the experiment config for a single-setting
    experiment, and essential when several are merged for comparison."""

    def __post_init__(self) -> None:
        # Either field may have been given; the list wins when it says
        # anything, otherwise it is derived from the single mode. This is what
        # keeps a record built the old way (`failure_mode=GRASP_MISS`) and one
        # built the new way (`failure_modes=[GRASP_MISS, DROPPED]`) equivalent.
        modes = normalize_failure_modes(self.failure_modes)
        if not modes:
            modes = normalize_failure_modes(self.failure_mode)
        object.__setattr__(self, "failure_modes", list(modes))
        object.__setattr__(
            self, "failure_mode", modes[0] if modes else FailureMode.NONE
        )
        # The acted-on object(s) follow the same rule: the list wins when it
        # says anything, otherwise it is derived from the single column.
        targets = normalize_actual_targets(self.actual_targets)
        if not targets:
            targets = normalize_actual_targets(self.actual_target)
        object.__setattr__(self, "actual_targets", list(targets))
        object.__setattr__(self, "actual_target", targets[0] if targets else "")
        # From here on the paired fields are kept in step by __setattr__. Not
        # before: the generated __init__ assigns them one at a time, and the
        # second one's default would otherwise wipe out the first.
        self.__dict__["_synced"] = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Assigning either failure field updates the other.

        ``record.failure_mode = DROPPED`` makes the list ``[DROPPED]``, and
        ``record.failure_modes = [...]`` re-derives the primary — so code that
        predates several modes per rollout, and only knows the single column,
        still produces a consistent record. ``actual_target`` and
        ``actual_targets`` are paired the same way.
        """
        synced = self.__dict__.get("_synced")
        if name in ("failure_mode", "failure_modes") and synced:
            modes = normalize_failure_modes(value)
            object.__setattr__(self, "failure_modes", list(modes))
            object.__setattr__(
                self, "failure_mode", modes[0] if modes else FailureMode.NONE
            )
            return
        if name in ("actual_target", "actual_targets") and synced:
            targets = normalize_actual_targets(value)
            object.__setattr__(self, "actual_targets", list(targets))
            object.__setattr__(self, "actual_target", targets[0] if targets else "")
            return
        object.__setattr__(self, name, value)

    @property
    def succeeded(self) -> bool:
        return self.outcome is Outcome.SUCCESS

    def set_failure_modes(self, modes: FailureModes | None) -> None:
        """Replace the classification: one mode, several, or ``None`` to clear."""
        self.failure_modes = list(normalize_failure_modes(modes))

    @property
    def failure_label(self) -> str:
        """Every selected mode, joined for display; ``""`` on a success."""
        return failure_modes_label(self.failure_modes)

    def set_actual_targets(self, targets: ActualTargets | None) -> None:
        """Replace what it went for: one object, several, or ``None`` to clear."""
        self.actual_targets = list(normalize_actual_targets(targets))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["outcome"] = self.outcome.value
        d["failure_mode"] = self.failure_mode.value
        d["failure_modes"] = [m.value for m in self.failure_modes]
        d["stop_reason"] = self.stop_reason.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        # Drop unknown keys rather than raising: a record written by a newer
        # version of this code must still load in an older one, otherwise a
        # half-migrated experiment directory becomes unreadable.
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["outcome"] = Outcome(kwargs.get("outcome", Outcome.ABORTED.value))
        kwargs["failure_mode"] = FailureMode(
            kwargs.get("failure_mode", FailureMode.NONE.value)
        )
        # Rows written before several modes could be selected carry only the
        # single column; __post_init__ derives the list from it.
        kwargs["failure_modes"] = [
            FailureMode(m) for m in kwargs.get("failure_modes") or ()
        ]
        kwargs["stop_reason"] = StopReason(
            kwargs.get("stop_reason", StopReason.OPERATOR.value)
        )
        # Likewise rows written before several objects could be named carry
        # only `actual_target`; __post_init__ derives the list from it.
        kwargs["actual_targets"] = [str(t) for t in kwargs.get("actual_targets") or ()]
        return cls(**kwargs)


__all__ = [
    "NO_OBJECT",
    "OPERATOR_FAILURE_MODES",
    "ActualTargets",
    "FailureMode",
    "FailureModes",
    "Outcome",
    "RolloutRecord",
    "StopReason",
    "failure_modes_label",
    "normalize_actual_targets",
    "normalize_failure_modes",
]

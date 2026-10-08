"""One definition per operator action, used for both the button and the hotkey.

The requirement is that every quick action is reachable from the GUI *and* from
the keyboard. The way that requirement usually rots is by being implemented
twice — a button wired to one handler, a shortcut wired to another — and drifting
until one of them does the wrong thing to a moving robot.

So an :class:`Action` is declared once, and :func:`build_actions` produces the
whole set. The window renders each as a button labelled with its own shortcut,
and installs the same ``QAction`` as an application-wide shortcut. The enabled
state is computed from one predicate, so a button that is greyed out has a hotkey
that does nothing, and vice versa.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence

from policy_abstraction.experiments.outcomes import (
    FailureMode,
    FailureModes,
    Outcome,
    StopReason,
    normalize_failure_modes,
)
from policy_abstraction.experiments.session import ExperimentSession, SessionState


@dataclasses.dataclass(frozen=True)
class Action:
    """A single operator action."""

    id: str
    label: str
    shortcut: str
    """Qt key sequence, e.g. ``"Space"``, ``"S"``, ``"Ctrl+Q"``."""
    tooltip: str
    callback: Callable[[], None]
    enabled_when: Callable[[ExperimentSession], bool]
    style: str = "default"
    """``default`` / ``primary`` / ``success`` / ``danger`` — how the window
    colours the button. Purely presentational."""
    confirm: Optional[str] = None
    """When set, the window asks this question before firing. Reserved for
    actions that throw work away; never used for a stop, which must be instant."""

    @property
    def button_text(self) -> str:
        return f"{self.label}  [{self.shortcut}]"


def _idle(session: ExperimentSession) -> bool:
    return session.state is SessionState.STAGING and session.current is not None


def _running(session: ExperimentSession) -> bool:
    return session.state is SessionState.RUNNING


def build_actions(
    session: ExperimentSession,
    *,
    on_start: Callable[[], None],
    on_success: Callable[[], None],
    on_catastrophic: Callable[[], None],
    on_failure: Callable[[], None],
    on_skip: Callable[[], None],
    on_quit: Callable[[], None],
    on_report: Callable[[], None],
    on_settings: Optional[Callable[[], None]] = None,
    on_history: Optional[Callable[[], None]] = None,
    on_ghost: Optional[Callable[[], None]] = None,
) -> tuple[Action, ...]:
    """The standard action set for a collection session.

    Ordered as they appear in the toolbar: start on the left, the stops in the
    middle where the hand rests, the destructive ones at the far right.

    *on_settings* is optional because editing the experiment needs a reviser,
    and a window built without one (a test, a viewer) should not offer a button
    that cannot work. *on_history* is optional on the same principle, and
    *on_ghost* because only an experiment created from a reference has
    reference frames to show.
    """
    ghost: tuple[Action, ...] = ()
    if on_ghost is not None:
        ghost = (
            Action(
                id="ghost",
                label="Reference ghost",
                shortcut="G",
                tooltip=(
                    "Cycle how much of the reference experiment's staged scene "
                    "shows through the live view: off, faint, strong, reference only. "
                    "Only drawn while staging."
                ),
                callback=on_ghost,
                enabled_when=lambda s: s.reference is not None,
            ),
        )
    history: tuple[Action, ...] = ()
    if on_history is not None:
        history = (
            Action(
                id="history",
                label="History",
                # Ctrl, like Settings and for the same reason: it opens a modal
                # dialog and it rewrites recorded verdicts. The single letters
                # are reserved for the things done with the arm moving.
                shortcut="Ctrl+H",
                tooltip=(
                    "The rollouts collected so far, with their verdicts "
                    "editable. Only available between rollouts."
                ),
                callback=on_history,
                enabled_when=lambda s: not s.is_running,
            ),
        )
    settings: tuple[Action, ...] = ()
    if on_settings is not None:
        settings = (
            Action(
                id="settings",
                label="Settings",
                # Deliberately not a bare letter. Every other action here is
                # one keystroke because the operator's hand is on the keyboard
                # and the arm is moving; this one opens a modal dialog and can
                # end in deleting results, so it asks for the reach.
                shortcut="Ctrl+E",
                tooltip=(
                    "Edit the grid, the repeats and the limits without "
                    "restarting. Only available between rollouts."
                ),
                callback=on_settings,
                enabled_when=lambda s: not s.is_running,
            ),
        )
    return (
        Action(
            id="start",
            label="Start rollout",
            shortcut="Space",
            tooltip=(
                "Home the robot and run the policy on the staged scene. "
                "Only available while no rollout is in progress."
            ),
            callback=on_start,
            enabled_when=_idle,
            style="primary",
        ),
        Action(
            id="stop_success",
            label="Stop — success",
            shortcut="S",
            tooltip="End the rollout, record it as a success, and return home.",
            callback=on_success,
            enabled_when=_running,
            style="success",
        ),
        Action(
            id="stop_failure",
            label="Stop — failure",
            shortcut="F",
            tooltip=(
                "End the rollout and return home, then pick which kind of "
                "failure it was."
            ),
            callback=on_failure,
            enabled_when=_running,
            style="danger",
        ),
        Action(
            id="stop_catastrophic",
            label="STOP — catastrophic",
            shortcut="C",
            tooltip=(
                "Emergency stop: end the rollout immediately, record a "
                "catastrophic failure, and return home. Asks nothing."
            ),
            callback=on_catastrophic,
            enabled_when=_running,
            style="danger",
        ),
        Action(
            id="skip",
            label="Skip condition",
            shortcut="K",
            tooltip=(
                "Discard this condition as mis-staged. It is logged as aborted "
                "and offered again later, so the plan still gets its data."
            ),
            callback=on_skip,
            enabled_when=_idle,
        ),
        *ghost,
        *history,
        *settings,
        Action(
            id="report",
            label="Report",
            shortcut="R",
            tooltip="Regenerate the summary and plots from what is collected so far.",
            callback=on_report,
            enabled_when=lambda s: not s.is_running,
        ),
        Action(
            id="quit",
            label="Interrupt experiment",
            shortcut="Esc",
            tooltip=(
                "Stop any running rollout, park the robot, write the report, "
                "and close. The experiment can be resumed by name."
            ),
            callback=on_quit,
            enabled_when=lambda s: True,
            style="danger",
        ),
    )


# The stop verdicts, as (outcome, failure mode, stop reason) triples. Kept here
# rather than inline in the window so the mapping from a button to a recorded
# verdict is in one readable place.
STOP_SUCCESS = (Outcome.SUCCESS, (), StopReason.OPERATOR)
STOP_CATASTROPHIC = (
    Outcome.FAILURE,
    (FailureMode.CATASTROPHIC,),
    StopReason.OPERATOR,
)


def stop_failure(
    modes: FailureModes,
) -> tuple[Outcome, tuple[FailureMode, ...], StopReason]:
    """The stop tuple for an operator-classified failure, one mode or several."""
    return (Outcome.FAILURE, normalize_failure_modes(modes), StopReason.OPERATOR)


def shortcut_help(actions: Sequence[Action]) -> str:
    """A one-line cheat sheet, for the status bar."""
    return "   ".join(f"{a.shortcut}: {a.label}" for a in actions)


__all__ = [
    "STOP_CATASTROPHIC",
    "STOP_SUCCESS",
    "Action",
    "build_actions",
    "shortcut_help",
    "stop_failure",
]

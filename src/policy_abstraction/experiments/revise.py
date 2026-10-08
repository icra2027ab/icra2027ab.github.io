"""Changing an experiment's definition after the experiment already exists.

The plan is frozen on disk on purpose — that is what makes resume mean
something (see :mod:`policy_abstraction.experiments.store`). But an operator at
the cell routinely discovers that the grid should have been 4x3, or that three
repeats is one too many for the afternoon they have. Editing the YAML and
restarting costs another checkpoint load, so the GUI lets them edit the
experiment in place instead.

What that edit *means* depends on what is on disk, and the rules are all here so
the GUI can stay a form:

=================================  ========================================
The edit does not change the plan  Applied in place. Nothing is at risk:
(task, timeout, view scale, …)     the same conditions in the same order.
The plan changes, nothing          Applied in place. The plan is rewritten;
collected yet                      there is nothing to be inconsistent with.
The name changes                   A different experiment. Created, or
                                   resumed if that name already holds the
                                   same plan.
The plan changes and rollouts      :class:`ExperimentConflict`. The operator
have been recorded                 must choose: a new experiment under a new
                                   name, or destroy what is there.
=================================  ========================================

Only that last case is destructive, and it is never taken implicitly — the
caller has to come back with ``destroy=True`` after asking.
"""

from __future__ import annotations

import dataclasses
import enum
import pathlib
import shutil
import time
from typing import Callable, Optional

from loguru import logger

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.store import ExperimentStore, _safe, experiment_root


class RevisionAction(enum.Enum):
    """What applying an edit would do to the directory it targets."""

    UNCHANGED = "unchanged"
    """The edit is a no-op: same name, same config, same plan."""
    IN_PLACE = "in_place"
    """Same directory, and nothing recorded there is invalidated."""
    RESUME = "resume"
    """A different directory that already holds this exact plan."""
    CREATE = "create"
    """A different directory that does not exist yet."""
    DESTRUCTIVE = "destructive"
    """The target holds rollout records that the new plan would orphan."""

    @property
    def needs_choice(self) -> bool:
        return self is RevisionAction.DESTRUCTIVE


class ExperimentConflict(RuntimeError):
    """Raised when applying an edit would orphan recorded rollouts.

    Carries the :class:`Revision` so the caller can put the numbers in front of
    the operator rather than re-deriving them.
    """

    def __init__(self, revision: "Revision") -> None:
        super().__init__(revision.describe())
        self.revision = revision


@dataclasses.dataclass(frozen=True)
class Revision:
    """What one edit would mean, decided but not yet applied."""

    name: str
    """Experiment (directory) name the edit targets."""
    root: pathlib.Path
    action: RevisionAction
    plan_changed: bool
    name_changed: bool
    config_changed: bool
    old_total: int
    """Rollouts in the plan currently on disk at *root*, 0 if there is none."""
    new_total: int
    """Rollouts the edited config plans."""
    records: int = 0
    """Rows in the target's ``rollouts.jsonl`` — everything at risk."""
    scored: int = 0
    """How many of those count toward a rate (i.e. are not aborted)."""

    @property
    def needs_choice(self) -> bool:
        return self.action.needs_choice

    def describe(self) -> str:
        if self.action is RevisionAction.UNCHANGED:
            return "Nothing changed."
        if self.action is RevisionAction.DESTRUCTIVE:
            return (
                f"{self.root} holds {self.records} rollout record"
                f"{'' if self.records == 1 else 's'} "
                f"({self.scored} scored) collected under a plan of "
                f"{self.old_total} rollouts. The edited settings plan "
                f"{self.new_total}, so resuming would mix conditions."
            )
        if self.action is RevisionAction.RESUME:
            return (
                f"{self.root} already holds this exact plan "
                f"({self.new_total} rollouts); it will be resumed."
            )
        if self.action is RevisionAction.CREATE:
            return f"A new experiment at {self.root} ({self.new_total} rollouts)."
        if self.plan_changed:
            return (
                f"The plan is rewritten in place: {self.old_total} rollouts "
                f"-> {self.new_total}. Nothing is recorded there yet."
            )
        return f"Applied in place; the {self.new_total}-rollout plan is unchanged."


@dataclasses.dataclass(frozen=True)
class AppliedRevision:
    """The result of :meth:`ExperimentReviser.apply`."""

    revision: Revision
    store: ExperimentStore
    config: ExperimentConfig
    name: str
    config_path: Optional[pathlib.Path]
    """Where the edited config was written as a standalone YAML, if anywhere.
    The authoritative copy always lives in the store as ``experiment.yaml``."""
    destroyed: bool = False

    def describe(self) -> str:
        lines = [self.revision.describe()]
        if self.destroyed:
            lines.append(f"The previous results at {self.store.root} were deleted.")
        if self.config_path is not None:
            lines.append(f"Settings written to {self.config_path}")
        return "\n".join(lines)


def plan_signature(config: ExperimentConfig) -> list[dict]:
    """The comparable form of a config's plan.

    Two configs with equal signatures stage the same conditions in the same
    order, so results collected under either are poolable.
    """
    return [p.to_dict() for p in config.build_plan()]


class ExperimentReviser:
    """Applies edited settings to the experiment a session is running.

    Holds the little bit of context the GUI does not have — which evaluation
    run the experiment lives under, and where the operator's YAML came from —
    so the window can hand it a config and a name and get back a store.
    """

    def __init__(
        self,
        *,
        run: str,
        name: str,
        config: ExperimentConfig,
        store: ExperimentStore,
        base: str | pathlib.Path = "data",
        source_config: Optional[str | pathlib.Path] = None,
        metadata: Optional[dict] = None,
        on_applied: Optional[Callable[["AppliedRevision"], None]] = None,
    ) -> None:
        self.run = run
        self.base = pathlib.Path(base)
        self.name = name
        self.config = config
        self.store = store
        self.source_config = pathlib.Path(source_config) if source_config else None
        self.metadata = dict(metadata or {})
        self.on_applied = on_applied
        """Called after a successful :meth:`apply`, for whatever the caller has
        to re-point at the new directory — the log sink, say. Never called on
        the path that raises."""

    # ------------------------------------------------------------------

    def root_for(self, name: str) -> pathlib.Path:
        return experiment_root(self.run, name, base=self.base)

    def config_dir(self) -> pathlib.Path:
        """Where a revised config is written as a standalone YAML."""
        if self.source_config is not None:
            return self.source_config.parent
        return pathlib.Path("configs") / "experiments"

    def suggest_name(self, name: str) -> str:
        """A near-by experiment name whose directory does not exist yet.

        ``cable_position_baseline`` -> ``cable_position_baseline_v2``, then
        ``_v3``. Deliberately a suffix rather than a timestamp: the operator has
        to read it back off the screen and type it again tomorrow to resume.
        """
        stem = name
        start = 2
        if "_v" in name:
            head, _, tail = name.rpartition("_v")
            if tail.isdigit() and head:
                stem, start = head, int(tail) + 1
        for i in range(start, start + 500):
            candidate = f"{stem}_v{i}"
            if not self.root_for(candidate).exists():
                return candidate
        return f"{stem}_v{start}"

    # ------------------------------------------------------------------

    def inspect(self, config: ExperimentConfig, name: str) -> Revision:
        """Decide what applying *config* under *name* would do. Touches nothing."""
        name = (name or "").strip() or self.name
        root = self.root_for(name)
        target = ExperimentStore(root)
        new_plan = plan_signature(config)

        name_changed = root != self.store.root
        config_changed = config.to_dict() != self.config.to_dict()

        if target.exists:
            old_plan = [p.to_dict() for p in target.load_plan()]
        else:
            old_plan = []
        plan_changed = bool(old_plan) and old_plan != new_plan
        records = target.record_count()
        scored = len(target.results()) if records else 0

        if not target.exists:
            action = RevisionAction.CREATE
        elif plan_changed and records:
            action = RevisionAction.DESTRUCTIVE
        elif plan_changed:
            action = (
                RevisionAction.IN_PLACE if not name_changed else RevisionAction.CREATE
            )
        elif name_changed:
            action = RevisionAction.RESUME
        elif config_changed:
            action = RevisionAction.IN_PLACE
        else:
            action = RevisionAction.UNCHANGED

        return Revision(
            name=name,
            root=root,
            action=action,
            plan_changed=plan_changed,
            name_changed=name_changed,
            config_changed=config_changed,
            old_total=len(old_plan),
            new_total=len(new_plan),
            records=records,
            scored=scored,
        )

    # ------------------------------------------------------------------

    def apply(
        self,
        config: ExperimentConfig,
        name: str,
        *,
        destroy: bool = False,
        write_config_file: bool = True,
    ) -> AppliedRevision:
        """Point the experiment at *config* under *name*.

        Args:
            destroy: Permit deleting recorded rollouts that the new plan would
                orphan. Without it that case raises rather than guessing.
            write_config_file: Also drop the edited settings next to the
                operator's original YAML, so the same experiment can be
                relaunched from the shell later.

        Raises:
            ExperimentConflict: The target holds records the new plan orphans
                and *destroy* was not given.
        """
        revision = self.inspect(config, name)
        if revision.needs_choice and not destroy:
            raise ExperimentConflict(revision)

        root = revision.root
        destroyed = False
        if revision.action is RevisionAction.DESTRUCTIVE:
            logger.warning(f"Deleting the experiment at {root} on operator request")
            shutil.rmtree(root)
            destroyed = True
        elif revision.plan_changed:
            # No records to orphan, so the frozen plan is simply wrong rather
            # than load-bearing. Drop it and let `open` freeze the new one;
            # leaving it would trip the store's own mismatch guard.
            logger.info(f"Rewriting the plan at {root} (nothing collected yet)")
            ExperimentStore(root).plan_path.unlink(missing_ok=True)

        previous = self.store
        store = ExperimentStore.open(
            root,
            config,
            metadata={**self.metadata, "experiment_name": revision.name},
        )
        if store.root != previous.root and previous.exists:
            # The old directory is left untouched, which is the whole appeal of
            # the rename -- but the launcher's default name still points at it,
            # so it is given a forwarding address rather than being left to
            # look like the current experiment on the next launch.
            self._leave_forwarding_address(previous, revision.name)
        # `open` only writes experiment.yaml when it creates the directory, so
        # an in-place edit has to refresh it -- otherwise the config travelling
        # with the results would be the one they were *not* collected under.
        config.save(store.config_path)

        config_path: Optional[pathlib.Path] = None
        # Nothing changed: writing a YAML anyway would leave a file behind for
        # an operator who opened the dialog, looked, and pressed OK.
        if write_config_file and revision.action is not RevisionAction.UNCHANGED:
            config_path = self._write_config_file(config, revision.name)

        self.name = revision.name
        self.config = config
        self.store = store
        applied = AppliedRevision(
            revision=revision,
            store=store,
            config=config,
            name=revision.name,
            config_path=config_path,
            destroyed=destroyed,
        )
        if self.on_applied is not None:
            try:
                self.on_applied(applied)
            except Exception as exc:  # never lose an applied revision to a hook
                logger.warning(f"post-revision hook raised: {exc}")
        return applied

    @staticmethod
    def _leave_forwarding_address(previous: ExperimentStore, successor: str) -> None:
        meta = previous.load_metadata()
        meta["superseded_by"] = successor
        meta["superseded_at"] = time.time()
        try:
            previous.save_metadata(meta)
        except OSError as exc:
            logger.warning(f"Could not note the rename in {previous.root}: {exc}")

    def _write_config_file(
        self, config: ExperimentConfig, name: str
    ) -> Optional[pathlib.Path]:
        """Write the edited settings as ``<config dir>/<name>.yaml``.

        Named for the experiment, not for the config, so re-editing the same
        experiment updates one file instead of accumulating a pile of them. The
        operator's original YAML is never overwritten — that file may well be
        the source for the *other* arm of the comparison, which is running
        against the plan it already has.
        """
        directory = self.config_dir()
        path = directory / f"{_safe(name)}.yaml"
        if self.source_config is not None and path.resolve() == (
            self.source_config.resolve()
        ):
            path = directory / f"{_safe(name)}_edited.yaml"
        try:
            return config.save(path)
        except OSError as exc:
            logger.warning(f"Could not write {path}: {exc}")
            return None


__all__ = [
    "AppliedRevision",
    "ExperimentConflict",
    "ExperimentReviser",
    "Revision",
    "RevisionAction",
    "plan_signature",
]

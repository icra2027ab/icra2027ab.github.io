"""On-disk layout for one experiment, and the resume logic built on it.

Layout, under ``data/<run>/experiments/<name>/``::

    experiment.yaml     the config this experiment was defined by
    plan.json           the frozen condition list — resume's source of truth
    rollouts.jsonl      one line per finished rollout, append-only
    metadata.json       policy setting, args snapshot, timestamps
    episodes/*.vla      the recordings, written by the existing RobodmRecorder
    videos/*.mp4        replay videos, written by EpisodeRunner.save_video
    report/             summary.json, report.md, plots
    evaluation.log      loguru sink, plus the servers' stdout/stderr

``rollouts.jsonl`` is append-only and flushed per line on purpose. A GUI session
can be killed at any moment — the operator hits the e-stop and then Ctrl-C — and
an append-only log loses at most the rollout that was in flight. Rewriting a
whole summary file after every rollout would risk truncating the lot.

The plan is written once and never regenerated. If the config changes mid-
experiment, :meth:`ExperimentStore.open` says so rather than quietly renumbering
conditions under already-collected results.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import time
from typing import Iterator, Optional

from loguru import logger

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord
from policy_abstraction.experiments.plan import PlannedRollout


class ExperimentExistsError(RuntimeError):
    """Raised when an experiment directory exists and the caller must choose."""


@dataclasses.dataclass
class ExperimentStatus:
    """A glance at how far along an experiment is."""

    total: int
    completed: int
    successes: int
    failures: int
    aborted: int

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    @property
    def scored(self) -> int:
        """Rollouts that count toward a rate (everything but aborted)."""
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return self.successes / self.scored if self.scored else 0.0

    def __str__(self) -> str:
        return (
            f"{self.completed}/{self.total} rollouts "
            f"({self.successes}S / {self.failures}F"
            + (f" / {self.aborted} aborted" if self.aborted else "")
            + f", {self.success_rate:.0%} success)"
        )


class ExperimentStore:
    """Reads and writes one experiment's directory."""

    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root)

    # -- paths ---------------------------------------------------------

    @property
    def config_path(self) -> pathlib.Path:
        return self.root / "experiment.yaml"

    @property
    def plan_path(self) -> pathlib.Path:
        return self.root / "plan.json"

    @property
    def rollouts_path(self) -> pathlib.Path:
        return self.root / "rollouts.jsonl"

    @property
    def metadata_path(self) -> pathlib.Path:
        return self.root / "metadata.json"

    @property
    def episodes_dir(self) -> pathlib.Path:
        return self.root / "episodes"

    @property
    def videos_dir(self) -> pathlib.Path:
        return self.root / "videos"

    @property
    def report_dir(self) -> pathlib.Path:
        return self.root / "report"

    @property
    def log_path(self) -> pathlib.Path:
        return self.root / "evaluation.log"

    @property
    def reference_frame_path(self) -> pathlib.Path:
        return self.root / "reference_frame.png"

    @property
    def reference_camera_path(self) -> pathlib.Path:
        return self.root / "reference_camera.json"

    @property
    def observed_positions_path(self) -> pathlib.Path:
        """Measured object positions, one entry per rollout index.

        A sidecar rather than superseding rows in ``rollouts.jsonl``: the
        measurement is derived from the recording after the fact and can be
        redone with a better detector, and re-appending every row to carry it
        would turn each re-run into a spurious "correction" of every verdict.
        """
        return self.root / "observed_positions.json"

    # -- reference view ---------------------------------------

    def save_reference_view(
        self,
        image,
        *,
        intrinsic=None,
        extrinsic=None,
        camera_name: str = "",
        preprocessed: bool = True,
        replace: bool = False,
    ) -> None:
        """Store one staging frame and the camera that produced it.

        The report projects the per-cell results back onto this frame, so
        the results are read against the actual scene rather than against
        an abstract square. It is written once, on the first
        :meth:`~policy_abstraction.experiments.session.ExperimentSession.prepare_env`
        of the experiment: a later frame would show the objects wherever
        the last rollout left them, and — worse — a camera that had been
        nudged mid-session would silently re-project every earlier
        rollout through the new pose.

        Args:
            replace: Overwrite a frame already there — for choosing, after
                the fact, a recorded frame that shows the scene better. The
                first frame ever saved is kept beside it as
                ``reference_frame.original.png`` so the choice is reversible.
        """
        if self.reference_frame_path.exists():
            if not replace:
                return
            original = self.reference_frame_path.with_suffix(".original.png")
            if not original.exists():
                self.reference_frame_path.replace(original)
                camera_original = self.reference_camera_path.with_suffix(
                    ".original.json"
                )
                if self.reference_camera_path.exists():
                    self.reference_camera_path.replace(camera_original)
        try:
            import imageio.v3 as iio

            self.root.mkdir(parents=True, exist_ok=True)
            iio.imwrite(self.reference_frame_path, image)
        except Exception as exc:
            logger.warning(f"Could not save the reference frame: {exc}")
            return
        self._write_json(
            self.reference_camera_path,
            {
                "camera_name": camera_name,
                "preprocessed": bool(preprocessed),
                "intrinsic": (None if intrinsic is None else _as_list(intrinsic)),
                "extrinsic": (None if extrinsic is None else _as_list(extrinsic)),
            },
        )
        logger.info(f"Saved the reference staging view to {self.reference_frame_path}")

    def load_reference_view(self):
        """``(image, camera_dict)``, or ``(None, {})`` when none was saved."""
        if not self.reference_frame_path.exists():
            return None, {}
        try:
            import imageio.v3 as iio

            image = iio.imread(self.reference_frame_path)
        except Exception as exc:
            logger.warning(f"Could not read the reference frame: {exc}")
            return None, {}
        camera: dict = {}
        if self.reference_camera_path.exists():
            try:
                with open(self.reference_camera_path, "r", encoding="utf-8") as fh:
                    camera = json.load(fh)
            except json.JSONDecodeError:
                logger.warning(f"{self.reference_camera_path} is unreadable")
        return image, camera

    @property
    def exists(self) -> bool:
        return self.plan_path.exists()

    # -- lifecycle -----------------------------------------------------

    @classmethod
    def open(
        cls,
        root: str | pathlib.Path,
        config: ExperimentConfig,
        *,
        overwrite: bool = False,
        archive: bool = True,
        metadata: Optional[dict] = None,
    ) -> "ExperimentStore":
        """Create or resume the experiment at *root*.

        Args:
            overwrite: Discard any existing results and start fresh. Without it
                an existing experiment is *resumed*, which is the default
                because losing an afternoon of rollouts to a mistyped name
                should not be one keystroke away.
            archive: Rename the discarded results aside rather than deleting
                them. The default, and what ``--overwrite`` on the command line
                does. The GUI's settings editor passes ``False`` — but only
                after the operator has confirmed the deletion twice, which a
                command-line flag cannot do.

        Raises:
            ExperimentExistsError: When resuming an experiment whose stored plan
                disagrees with *config* — appending to it would mix conditions.
        """
        store = cls(root)
        if store.exists and overwrite:
            logger.warning(f"Overwriting existing experiment at {store.root}")
            if archive:
                store._archive_existing()
            else:
                store._delete_existing()
        store.root.mkdir(parents=True, exist_ok=True)
        store.episodes_dir.mkdir(exist_ok=True)
        store.videos_dir.mkdir(exist_ok=True)
        store.report_dir.mkdir(exist_ok=True)

        plan = config.build_plan()
        if store.plan_path.exists():
            stored = store.load_plan()
            if [p.to_dict() for p in stored] != [p.to_dict() for p in plan]:
                raise ExperimentExistsError(
                    f"{store.root} holds a plan of {len(stored)} rollouts that "
                    f"does not match the {len(plan)} the current config "
                    "generates. Resuming would append rollouts collected under "
                    "different conditions to the same results. Either restore "
                    f"the config in {store.config_path}, or pass --overwrite to "
                    "start this experiment over (the old results are moved "
                    "aside, not deleted)."
                )
            logger.info(f"Resuming experiment at {store.root}")
        else:
            store.save_plan(plan)
            config.save(store.config_path)
            logger.info(
                f"Created experiment at {store.root} ({len(plan)} rollouts planned)"
            )

        meta = store.load_metadata()
        meta.setdefault("created_at", time.time())
        meta["updated_at"] = time.time()
        if metadata:
            meta.update(metadata)
        store.save_metadata(meta)
        return store

    def _archive_existing(self) -> None:
        """Move the current results aside instead of deleting them.

        ``--overwrite`` is what an operator types when an experiment went wrong,
        and "went wrong" is exactly when the old data turns out to have been
        worth keeping. The rename costs nothing and is reversible.
        """
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archived = self.root.with_name(f"{self.root.name}.superseded_{stamp}")
        self.root.rename(archived)
        logger.warning(f"Previous results moved to {archived}")

    def _delete_existing(self) -> None:
        """Delete the current results outright. No undo.

        Only ever reached from the settings editor, where the operator has
        confirmed twice and typed the experiment name back. Everywhere else
        overwriting archives instead.
        """
        shutil.rmtree(self.root)
        logger.warning(f"Deleted the previous results at {self.root}")

    # -- plan ----------------------------------------------------------

    def save_plan(self, plan: tuple[PlannedRollout, ...]) -> None:
        self._write_json(self.plan_path, [p.to_dict() for p in plan])

    def load_plan(self) -> tuple[PlannedRollout, ...]:
        with open(self.plan_path, "r", encoding="utf-8") as fh:
            return tuple(PlannedRollout.from_dict(d) for d in json.load(fh))

    def load_config(self) -> ExperimentConfig:
        return ExperimentConfig.load(self.config_path)

    # -- metadata ------------------------------------------------------

    def load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning(f"{self.metadata_path} is corrupt; starting fresh metadata")
            return {}

    def save_metadata(self, metadata: dict) -> None:
        self._write_json(self.metadata_path, metadata)

    # -- rollout records -----------------------------------------------

    def append(self, record: RolloutRecord) -> None:
        """Append one finished rollout. Durable before the call returns."""
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.rollouts_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")
            fh.flush()

    def record_count(self) -> int:
        """How many rollout rows are on disk, scored or aborted.

        Counted without parsing: this answers "is there anything here to lose"
        for the settings editor, which asks it on every keystroke that changes
        the plan.
        """
        if not self.rollouts_path.exists():
            return 0
        with open(self.rollouts_path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    # -- measured positions --------------------------------------------

    def load_observed_positions(self) -> dict:
        """The sidecar written by :func:`~policy_abstraction.experiments.measure.measure_experiment`.

        ``{"rollouts": {"<index>": {"<key>": {"position": [x, y, z], ...}}}, ...}``,
        or an empty dict when nothing has been measured.
        """
        if not self.observed_positions_path.exists():
            return {}
        try:
            with open(self.observed_positions_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.warning(f"{self.observed_positions_path} is unreadable ({exc})")
            return {}
        return data if isinstance(data, dict) else {}

    def save_observed_positions(self, data: dict) -> None:
        self._write_json(self.observed_positions_path, data)

    def attach_observed_positions(
        self, records: list[RolloutRecord]
    ) -> list[RolloutRecord]:
        """Fill ``observed_positions`` on *records* from the sidecar, in place.

        Returns the same list, so it can be used inline. A record with no
        measurement keeps an empty dict, which is what the plots test for.
        """
        by_index = self.load_observed_positions().get("rollouts") or {}
        for record in records:
            entry = by_index.get(str(record.index))
            if not entry:
                continue
            record.observed_positions = {
                key: [float(v) for v in item["position"]]
                for key, item in entry.items()
                if isinstance(item, dict) and item.get("position") is not None
            }
        return records

    def records(self) -> list[RolloutRecord]:
        """Every recorded rollout, in the order it was collected.

        A truncated final line — the signature of a kill mid-write — is dropped
        with a warning rather than raising, so one lost rollout does not make
        the whole experiment unreadable.
        """
        if not self.rollouts_path.exists():
            return []
        out: list[RolloutRecord] = []
        with open(self.rollouts_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(RolloutRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.warning(
                        f"{self.rollouts_path}:{lineno} is unreadable ({exc}); "
                        "skipping that rollout."
                    )
        return out

    def results(self) -> list[RolloutRecord]:
        """One scored result per plan index, in plan order.

        The JSONL is append-only, so a rollout's row can be *superseded* rather
        than edited — the GUI does this when the operator re-scores a rollout
        from the history list. The **last** row for an index is the verdict that
        stands, and it counts here only if it is a scored one, so an index whose
        latest row is aborted has no result: either it never really ran, or the
        operator has just discarded it.

        Filtering after taking the latest rather than before is what makes that
        discard possible; the other order would leave an earlier success
        standing behind the abort that was meant to remove it.

        Every rate in the report is computed from this, never from
        :meth:`records`, which would count a corrected rollout twice.
        """
        latest: dict[int, RolloutRecord] = {}
        for record in self.records():
            latest[record.index] = record
        return [
            latest[i] for i in sorted(latest) if latest[i].outcome.counts_toward_rate
        ]

    def completed_indices(self) -> set[int]:
        """Plan indices that already have a scored result.

        Aborted rollouts do *not* count: an aborted rollout is one that never
        really ran, so resume should offer it again.
        """
        return {r.index for r in self.results()}

    def pending(self) -> list[PlannedRollout]:
        done = self.completed_indices()
        return [p for p in self.load_plan() if p.index not in done]

    def next_pending(self) -> Optional[PlannedRollout]:
        return next(iter(self.pending()), None)

    def status(self) -> ExperimentStatus:
        scored = self.results()
        return ExperimentStatus(
            total=len(self.load_plan()),
            completed=len(scored),
            successes=sum(r.outcome is Outcome.SUCCESS for r in scored),
            failures=sum(r.outcome is Outcome.FAILURE for r in scored),
            aborted=sum(r.outcome is Outcome.ABORTED for r in self.records()),
        )

    def iter_records(self) -> Iterator[RolloutRecord]:
        yield from self.records()

    # -- episode naming ------------------------------------------------

    def episode_path(self, planned: PlannedRollout, attempt: int = 0) -> pathlib.Path:
        """Where rollout *planned* records its ``.vla``.

        The name carries the condition, so the episode files are browsable
        without cross-referencing the JSONL. ``attempt`` disambiguates a retry
        after an abort, which would otherwise overwrite the first try's file.
        """
        condition = "_".join(
            f"{name}-{level}" for name, level in sorted(planned.assignments.items())
        )
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        stem = f"rollout_{planned.index:04d}_{condition}{suffix}"
        return self.episodes_dir / f"{_safe(stem)}.vla"

    def attempt_count(self, index: int) -> int:
        """How many times rollout *index* has already been attempted."""
        return sum(1 for r in self.records() if r.index == index)

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _write_json(path: pathlib.Path, data) -> None:
        """Write via a temp file + rename so a crash cannot truncate the target."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
        tmp.replace(path)


def _as_list(matrix) -> list:
    """Numpy array to nested lists, for JSON."""
    import numpy as np

    return np.asarray(matrix).tolist()


def _safe(text: str) -> str:
    """Reduce *text* to characters that are safe in a filename."""
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in text)


def experiment_root(
    run: str, name: str, *, base: str | pathlib.Path = "data"
) -> pathlib.Path:
    """Canonical directory for experiment *name* under evaluation run *run*.

    Sits beside the batch evaluator's own output (``data/<run>/<suite>/…``)
    under an ``experiments/`` segment, so an operator-driven collection and a
    scripted sweep can share a run directory without colliding.
    """
    return pathlib.Path(base) / run / "experiments" / _safe(name)


__all__ = [
    "ExperimentExistsError",
    "ExperimentStatus",
    "ExperimentStore",
    "experiment_root",
]

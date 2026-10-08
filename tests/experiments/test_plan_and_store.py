"""The plan, the on-disk store, and the resume semantics built on them."""

from __future__ import annotations

import json

import pytest

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.factors import (
    AppearanceFactor,
    GridPositionFactor,
    OrientationFactor,
)
from policy_abstraction.experiments.grid import TableGrid
from policy_abstraction.experiments.outcomes import (
    FailureMode,
    Outcome,
    RolloutRecord,
)
from policy_abstraction.experiments.plan import build_plan
from policy_abstraction.experiments.store import (
    ExperimentExistsError,
    ExperimentStore,
)


@pytest.fixture
def grid():
    return TableGrid(
        origin=(0.3, -0.2, 0.02),
        u_axis=(0, 1, 0),
        v_axis=(1, 0, 0),
        rows=2,
        cols=3,
        name="cable",
    )


@pytest.fixture
def config(grid):
    return ExperimentConfig(
        name="test_experiment",
        task="Put the cable in the box.",
        factors=[GridPositionFactor("cable_position", grid)],
        repeats=2,
    )


@pytest.fixture
def store(tmp_path, config):
    return ExperimentStore.open(tmp_path / "exp", config)


def _record(index, assignments, outcome, **kwargs):
    return RolloutRecord(
        index=index, assignments=dict(assignments), outcome=outcome, **kwargs
    )


class TestPlan:
    def test_covers_the_cross_product_times_repeats(self, grid):
        plan = build_plan(
            [
                GridPositionFactor("pos", grid),
                AppearanceFactor("look", ["a", "b"]),
            ],
            repeats=3,
        )
        assert len(plan) == grid.num_cells * 2 * 3

    def test_repeats_of_a_cell_run_back_to_back_by_default(self, grid):
        """The object is staged once per cell, then every repeat runs there."""
        factor = GridPositionFactor("pos", grid)
        plan = build_plan([factor], repeats=3)
        cells = [p.assignments["pos"] for p in plan]
        for start in range(0, len(plan), 3):
            block = plan[start : start + 3]
            assert len({p.assignments["pos"] for p in block}) == 1
            assert [p.repeat for p in block] == [0, 1, 2]
        # ... and the cells themselves come in grid order, each exactly once.
        assert cells[::3] == [lv.id for lv in factor.levels]

    def test_sweep_grouping_is_repeat_major_so_a_truncated_run_still_covers(self, grid):
        """The reason for the alternative: stop early, keep balance."""
        factor = GridPositionFactor("pos", grid)
        plan = build_plan([factor], repeats=3, grouping="sweep")
        first_pass = plan[: grid.num_cells]
        assert len({p.assignments["pos"] for p in first_pass}) == grid.num_cells
        assert all(p.repeat == 0 for p in first_pass)

    def test_grouping_does_not_change_what_is_run(self, grid):
        factor = GridPositionFactor("pos", grid)
        by_condition = build_plan([factor], repeats=2)
        by_sweep = build_plan([factor], repeats=2, grouping="sweep")
        key = lambda p: (p.assignments["pos"], p.repeat)  # noqa: E731
        assert sorted(map(key, by_condition)) == sorted(map(key, by_sweep))

    def test_unknown_grouping_is_rejected(self, grid):
        with pytest.raises(ValueError, match="grouping"):
            build_plan([GridPositionFactor("pos", grid)], grouping="cell")  # type: ignore[arg-type]

    def test_indices_are_dense_and_ordered(self, grid):
        plan = build_plan([GridPositionFactor("pos", grid)], repeats=2)
        assert [p.index for p in plan] == list(range(len(plan)))

    def test_random_order_is_seeded_and_still_balanced(self, grid):
        factor = GridPositionFactor("pos", grid)
        a = build_plan([factor], repeats=2, order="random", seed=5)
        b = build_plan([factor], repeats=2, order="random", seed=5)
        c = build_plan([factor], repeats=2, order="random", seed=6)
        assert [p.assignments for p in a] == [p.assignments for p in b]
        assert [p.assignments for p in a] != [p.assignments for p in c]
        # Repeats still run back to back, over a shuffled cell order.
        assert a[0].assignments == a[1].assignments
        assert len({p.assignments["pos"] for p in a[::2]}) == grid.num_cells

    def test_random_sweeps_are_each_a_full_pass(self, grid):
        factor = GridPositionFactor("pos", grid)
        a = build_plan([factor], repeats=2, order="random", grouping="sweep", seed=5)
        for start in (0, grid.num_cells):
            block = a[start : start + grid.num_cells]
            assert len({p.assignments["pos"] for p in block}) == grid.num_cells

    def test_duplicate_factor_names_are_rejected(self, grid):
        """Names key the assignment dict, so a duplicate would drop one."""
        with pytest.raises(ValueError, match="unique"):
            build_plan(
                [
                    GridPositionFactor("cable", grid),
                    OrientationFactor("cable", [0, 90]),
                ]
            )

    def test_no_factors_is_rejected(self):
        with pytest.raises(ValueError):
            build_plan([])


class TestFactors:
    def test_grid_factor_can_be_restricted_to_reachable_cells(self, grid):
        factor = GridPositionFactor("pos", grid, cells=["A1", "B3"])
        assert [lv.id for lv in factor.levels] == ["A1", "B3"]

    def test_out_of_range_cell_is_rejected(self, grid):
        with pytest.raises(ValueError):
            GridPositionFactor("pos", grid, cells=["Z9"])

    def test_only_positional_factors_report_a_world_position(self, grid):
        pos = GridPositionFactor("pos", grid)
        assert pos.world_position("A1") is not None
        assert OrientationFactor("orient", [0]).world_position("000deg") is None
        assert AppearanceFactor("look", ["a"]).world_position("a") is None

    def test_orientation_level_ids_are_filename_safe(self):
        factor = OrientationFactor("orient", [0, 45, 90])
        assert [lv.id for lv in factor.levels] == ["000deg", "045deg", "090deg"]


class TestConfig:
    def test_yaml_round_trip_preserves_the_plan(self, config, tmp_path):
        path = config.save(tmp_path / "cfg.yaml")
        reloaded = ExperimentConfig.load(path)
        assert reloaded.to_dict() == config.to_dict()
        assert [p.to_dict() for p in reloaded.build_plan()] == [
            p.to_dict() for p in config.build_plan()
        ]

    def test_target_positions_are_recorded_for_positional_factors(self, config):
        targets = config.target_positions({"cable_position": "B2"})
        assert set(targets) == {"cable_position"}
        assert len(targets["cable_position"]) == 3

    def test_unknown_factor_kind_names_the_registry(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("name: x\nfactors:\n  - kind: teleport\n    name: t\n")
        with pytest.raises(ValueError, match="unknown factor kind"):
            ExperimentConfig.load(path)


class TestStoreResume:
    def test_status_starts_empty(self, store, config):
        status = store.status()
        assert status.total == len(config.build_plan())
        assert status.completed == 0

    def test_scored_rollouts_are_not_offered_again(self, store):
        planned = store.load_plan()[0]
        store.append(_record(planned.index, planned.assignments, Outcome.SUCCESS))
        assert store.next_pending().index == 1

    def test_aborted_rollouts_are_offered_again(self, store):
        """An abort means it never really ran, so the plan still needs it."""
        planned = store.load_plan()[0]
        store.append(_record(planned.index, planned.assignments, Outcome.ABORTED))
        assert store.next_pending().index == 0
        assert store.status().completed == 0
        assert store.status().aborted == 1

    def test_a_corrected_classification_supersedes_rather_than_doubles(self, store):
        """The GUI amends a verdict by appending; rates must not double-count."""
        planned = store.load_plan()[0]
        store.append(
            _record(
                planned.index,
                planned.assignments,
                Outcome.FAILURE,
                failure_mode=FailureMode.OTHER,
            )
        )
        store.append(
            _record(
                planned.index,
                planned.assignments,
                Outcome.FAILURE,
                failure_mode=FailureMode.GRASP_MISS,
            )
        )
        results = store.results()
        assert len(results) == 1
        assert results[0].failure_mode is FailureMode.GRASP_MISS
        assert store.status().completed == 1

    def test_a_row_written_with_one_mode_reads_back_as_a_list_of_one(self, store):
        """Records from before several modes could be selected still load."""
        from policy_abstraction.experiments.outcomes import RolloutRecord

        planned = store.load_plan()[0]
        row = _record(planned.index, planned.assignments, Outcome.FAILURE).to_dict()
        row["failure_mode"] = "dropped"
        del row["failure_modes"]
        old = RolloutRecord.from_dict(row)
        assert old.failure_mode is FailureMode.DROPPED
        assert old.failure_modes == [FailureMode.DROPPED]

    def test_several_modes_round_trip_and_keep_a_primary(self, store):
        from policy_abstraction.experiments.outcomes import RolloutRecord

        planned = store.load_plan()[0]
        record = _record(
            planned.index,
            planned.assignments,
            Outcome.FAILURE,
            failure_modes=[FailureMode.COLLISION, FailureMode.GRASP_MISS],
        )
        assert record.failure_mode is FailureMode.GRASP_MISS, "taxonomy order"
        assert record.failure_modes == [FailureMode.GRASP_MISS, FailureMode.COLLISION]
        store.append(record)
        back = store.results()[0]
        assert back.failure_modes == record.failure_modes
        assert back.to_dict()["failure_modes"] == ["grasp_miss", "collision"]
        assert RolloutRecord.from_dict(back.to_dict()) == back

    def test_none_is_not_a_mode_and_duplicates_collapse(self, store):
        from policy_abstraction.experiments.outcomes import normalize_failure_modes

        assert normalize_failure_modes(None) == ()
        assert normalize_failure_modes(FailureMode.NONE) == ()
        assert normalize_failure_modes(
            [FailureMode.DROPPED, FailureMode.NONE, FailureMode.DROPPED]
        ) == (FailureMode.DROPPED,)
        planned = store.load_plan()[0]
        success = _record(
            planned.index,
            planned.assignments,
            Outcome.SUCCESS,
            failure_mode=FailureMode.NONE,
        )
        assert success.failure_modes == []

    def test_discarding_a_scored_rollout_removes_its_result(self, store):
        """Re-scoring to aborted is how the GUI throws a mis-staged run away."""
        planned = store.load_plan()[0]
        store.append(_record(planned.index, planned.assignments, Outcome.SUCCESS))
        store.append(_record(planned.index, planned.assignments, Outcome.ABORTED))
        assert store.results() == []
        assert store.status().completed == 0
        assert store.next_pending().index == 0, "and its condition comes back"

    def test_an_abort_followed_by_a_real_attempt_still_counts(self, store):
        """The ordinary case, and the reason the *latest* row is the verdict."""
        planned = store.load_plan()[0]
        store.append(_record(planned.index, planned.assignments, Outcome.ABORTED))
        store.append(_record(planned.index, planned.assignments, Outcome.SUCCESS))
        assert [r.outcome for r in store.results()] == [Outcome.SUCCESS]

    def test_reopening_resumes_rather_than_restarting(self, tmp_path, config):
        store = ExperimentStore.open(tmp_path / "exp", config)
        planned = store.load_plan()[0]
        store.append(_record(planned.index, planned.assignments, Outcome.SUCCESS))
        reopened = ExperimentStore.open(tmp_path / "exp", config)
        assert reopened.status().completed == 1
        assert reopened.next_pending().index == 1

    def test_a_changed_plan_is_refused_not_silently_mixed(self, tmp_path, config):
        ExperimentStore.open(tmp_path / "exp", config)
        changed = ExperimentConfig.from_dict({**config.to_dict(), "repeats": 5})
        with pytest.raises(ExperimentExistsError, match="does not match"):
            ExperimentStore.open(tmp_path / "exp", changed)

    def test_overwrite_archives_rather_than_deletes(self, tmp_path, config):
        store = ExperimentStore.open(tmp_path / "exp", config)
        planned = store.load_plan()[0]
        store.append(_record(planned.index, planned.assignments, Outcome.SUCCESS))
        fresh = ExperimentStore.open(tmp_path / "exp", config, overwrite=True)
        assert fresh.status().completed == 0
        archived = list(tmp_path.glob("exp.superseded_*"))
        assert len(archived) == 1
        assert (archived[0] / "rollouts.jsonl").exists()

    def test_overwrite_can_delete_outright_when_asked(self, tmp_path, config):
        """What the GUI does after the operator has confirmed twice."""
        root = tmp_path / "exp"
        store = ExperimentStore.open(root, config)
        store.append(_record(0, {"cable_position": "A1"}, Outcome.SUCCESS))
        ExperimentStore.open(root, config, overwrite=True, archive=False)
        assert not list(tmp_path.glob("*.superseded_*"))
        assert ExperimentStore(root).record_count() == 0

    def test_record_count_sees_aborted_rows_too(self, store):
        assert store.record_count() == 0
        store.append(_record(0, {"cable_position": "A1"}, Outcome.ABORTED))
        store.append(_record(1, {"cable_position": "A2"}, Outcome.SUCCESS))
        assert store.record_count() == 2
        assert len(store.results()) == 1

    def test_a_truncated_final_line_does_not_break_the_log(self, store):
        planned = store.load_plan()[0]
        store.append(_record(planned.index, planned.assignments, Outcome.SUCCESS))
        with open(store.rollouts_path, "a", encoding="utf-8") as fh:
            fh.write('{"index": 1, "assignments": {"cable_posi')
        assert len(store.records()) == 1
        assert store.status().completed == 1

    def test_episode_paths_are_unique_per_attempt(self, store):
        planned = store.load_plan()[0]
        first = store.episode_path(planned, attempt=0)
        second = store.episode_path(planned, attempt=1)
        assert first != second
        assert "A1" in first.name

    def test_records_survive_a_schema_addition(self, store):
        """A row written by a newer version must still load in an older one."""
        planned = store.load_plan()[0]
        row = _record(planned.index, planned.assignments, Outcome.SUCCESS).to_dict()
        row["a_field_from_the_future"] = 42
        with open(store.rollouts_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        assert store.status().completed == 1


class TestActedOnObjects:
    """``actual_target`` and ``actual_targets`` are one fact in two columns."""

    def _record(self, **kwargs):
        from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord

        return RolloutRecord(index=0, assignments={}, outcome=Outcome.FAILURE, **kwargs)

    def test_the_single_column_derives_the_list(self):
        record = self._record(actual_target="black_cable")
        assert record.actual_targets == ["black_cable"]

    def test_the_list_derives_the_single_column(self):
        record = self._record(actual_targets=["black_cable", "red_mug"])
        assert record.actual_target == "black_cable"

    def test_assigning_either_updates_the_other(self):
        record = self._record()
        record.actual_targets = ["red_mug", "black_cable"]
        assert record.actual_target == "red_mug"
        record.actual_target = "black_cable"
        assert record.actual_targets == ["black_cable"]
        record.set_actual_targets(None)
        assert record.actual_target == "" and record.actual_targets == []

    def test_order_is_kept_and_duplicates_dropped(self):
        from policy_abstraction.experiments.outcomes import normalize_actual_targets

        assert normalize_actual_targets(["red_mug", "black_cable", "red_mug"]) == (
            "red_mug",
            "black_cable",
        )
        assert normalize_actual_targets("") == ()
        assert normalize_actual_targets(None) == ()

    def test_no_object_gives_way_to_a_named_one(self):
        from policy_abstraction.experiments.outcomes import (
            NO_OBJECT,
            normalize_actual_targets,
        )

        assert normalize_actual_targets([NO_OBJECT, "red_mug"]) == ("red_mug",)
        assert normalize_actual_targets([NO_OBJECT]) == (NO_OBJECT,)

    def test_round_trip_keeps_every_object(self):
        from policy_abstraction.experiments.outcomes import RolloutRecord

        record = self._record(actual_targets=["black_cable", "red_mug"])
        data = record.to_dict()
        assert data["actual_target"] == "black_cable"
        assert data["actual_targets"] == ["black_cable", "red_mug"]
        assert RolloutRecord.from_dict(data).actual_targets == [
            "black_cable",
            "red_mug",
        ]

    def test_a_row_written_before_the_list_existed_still_loads(self):
        from policy_abstraction.experiments.outcomes import RolloutRecord

        data = self._record(actual_target="black_cable").to_dict()
        del data["actual_targets"]
        loaded = RolloutRecord.from_dict(data)
        assert loaded.actual_target == "black_cable"
        assert loaded.actual_targets == ["black_cable"]

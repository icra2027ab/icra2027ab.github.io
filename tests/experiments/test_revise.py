"""Editing an experiment that already has results.

The rules this file pins down are the ones an operator's afternoon depends on:
an edit that cannot invalidate anything is applied silently, an edit that can
never is, and the only path that deletes rollouts is the one that was asked for
explicitly.
"""

from __future__ import annotations

import pytest

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.factors import GridPositionFactor
from policy_abstraction.experiments.grid import TableGrid
from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord
from policy_abstraction.experiments.revise import (
    ExperimentConflict,
    ExperimentReviser,
    RevisionAction,
    plan_signature,
)
from policy_abstraction.experiments.store import ExperimentStore, experiment_root


def make_config(*, rows: int = 2, cols: int = 3, repeats: int = 1, **kwargs):
    return ExperimentConfig(
        name="cable_in_box",
        task="Put the cable in the box.",
        factors=[
            GridPositionFactor(
                "cable_position",
                TableGrid(
                    origin=(0.3, -0.2, 0.0),
                    u_axis=(0, 1, 0),
                    v_axis=(1, 0, 0),
                    rows=rows,
                    cols=cols,
                    name="cable",
                ),
            )
        ],
        repeats=repeats,
        **kwargs,
    )


@pytest.fixture
def reviser(tmp_path):
    config = make_config()
    root = experiment_root("run", "cable_baseline", base=tmp_path)
    store = ExperimentStore.open(root, config)
    return ExperimentReviser(
        run="run",
        name="cable_baseline",
        config=config,
        store=store,
        base=tmp_path,
        source_config=tmp_path / "configs" / "cable.yaml",
    )


def record_one(store: ExperimentStore, index: int = 0) -> None:
    store.append(
        RolloutRecord(
            index=index,
            assignments={"cable_position": "A1"},
            outcome=Outcome.SUCCESS,
            duration_s=42.0,
        )
    )


class TestInspect:
    def test_an_identical_config_is_a_no_op(self, reviser):
        revision = reviser.inspect(make_config(), "cable_baseline")
        assert revision.action is RevisionAction.UNCHANGED
        assert not revision.needs_choice

    def test_an_edit_that_leaves_the_plan_alone_is_applied_in_place(self, reviser):
        record_one(reviser.store)
        revision = reviser.inspect(make_config(timeout_s=45.0), "cable_baseline")
        assert revision.action is RevisionAction.IN_PLACE
        assert not revision.plan_changed
        assert not revision.needs_choice

    def test_a_plan_change_with_nothing_collected_needs_no_permission(self, reviser):
        revision = reviser.inspect(make_config(rows=4), "cable_baseline")
        assert revision.action is RevisionAction.IN_PLACE
        assert revision.plan_changed
        assert not revision.needs_choice

    def test_a_plan_change_over_recorded_rollouts_asks(self, reviser):
        record_one(reviser.store)
        revision = reviser.inspect(make_config(rows=4), "cable_baseline")
        assert revision.action is RevisionAction.DESTRUCTIVE
        assert revision.needs_choice
        assert revision.records == 1
        assert revision.scored == 1
        assert (revision.old_total, revision.new_total) == (6, 12)

    def test_a_new_name_is_a_new_experiment(self, reviser):
        record_one(reviser.store)
        revision = reviser.inspect(make_config(rows=4), "cable_baseline_v2")
        assert revision.action is RevisionAction.CREATE
        assert not revision.needs_choice

    def test_the_same_plan_under_an_existing_name_resumes_it(self, reviser, tmp_path):
        other = experiment_root("run", "cable_guided", base=tmp_path)
        ExperimentStore.open(other, make_config())
        revision = reviser.inspect(make_config(), "cable_guided")
        assert revision.action is RevisionAction.RESUME
        assert not revision.needs_choice

    def test_an_aborted_row_still_counts_as_something_to_lose(self, reviser):
        reviser.store.append(
            RolloutRecord(index=0, assignments={}, outcome=Outcome.ABORTED)
        )
        revision = reviser.inspect(make_config(cols=5), "cable_baseline")
        assert revision.needs_choice
        assert (revision.records, revision.scored) == (1, 0)

    def test_inspect_writes_nothing(self, reviser):
        before = sorted(p.name for p in reviser.store.root.iterdir())
        reviser.inspect(make_config(rows=6), "somewhere_else")
        assert sorted(p.name for p in reviser.store.root.iterdir()) == before
        assert not experiment_root("run", "somewhere_else", base=reviser.base).exists()


class TestApply:
    def test_a_harmless_edit_keeps_the_results(self, reviser):
        record_one(reviser.store)
        applied = reviser.apply(make_config(timeout_s=45.0), "cable_baseline")
        assert applied.store.root == reviser.store.root
        assert applied.store.record_count() == 1
        assert applied.config.timeout_s == 45.0
        # The config travelling with the results is the edited one.
        assert applied.store.load_config().timeout_s == 45.0

    def test_a_plan_change_over_results_refuses_without_permission(self, reviser):
        record_one(reviser.store)
        with pytest.raises(ExperimentConflict) as excinfo:
            reviser.apply(make_config(rows=4), "cable_baseline")
        assert excinfo.value.revision.needs_choice
        # And nothing was touched on the way out.
        assert reviser.store.record_count() == 1
        assert len(reviser.store.load_plan()) == 6

    def test_starting_over_under_a_new_name_leaves_the_old_alone(self, reviser):
        record_one(reviser.store)
        old_root = reviser.store.root
        applied = reviser.apply(make_config(rows=4), "cable_baseline_v2")
        assert applied.store.root != old_root
        assert len(applied.store.load_plan()) == 12
        assert applied.store.record_count() == 0
        assert ExperimentStore(old_root).record_count() == 1

    def test_destroy_deletes_rather_than_archiving(self, reviser, tmp_path):
        record_one(reviser.store)
        root = reviser.store.root
        applied = reviser.apply(make_config(rows=4), "cable_baseline", destroy=True)
        assert applied.destroyed
        assert applied.store.root == root
        assert applied.store.record_count() == 0
        assert len(applied.store.load_plan()) == 12
        # Not renamed aside: the operator asked for it to be gone.
        assert not list(root.parent.glob("*.superseded_*"))

    def test_a_plan_rewrite_with_nothing_collected_needs_no_destroy_flag(self, reviser):
        applied = reviser.apply(make_config(cols=4), "cable_baseline")
        assert not applied.destroyed
        assert len(applied.store.load_plan()) == 8

    def test_the_reviser_tracks_what_it_applied(self, reviser):
        applied = reviser.apply(make_config(repeats=2), "cable_later")
        assert reviser.name == "cable_later"
        assert reviser.store.root == applied.store.root
        assert reviser.config.repeats == 2

    def test_a_rename_leaves_a_forwarding_address_on_the_old_one(self, reviser):
        """The launcher's default name still opens the old directory."""
        record_one(reviser.store)
        old_root = reviser.store.root
        applied = reviser.apply(make_config(rows=4), "cable_baseline_v2")
        meta = ExperimentStore(old_root).load_metadata()
        assert meta["superseded_by"] == "cable_baseline_v2"
        assert meta["superseded_at"] > 0
        # And the successor is not told it supersedes itself.
        assert "superseded_by" not in applied.store.load_metadata()

    def test_an_in_place_edit_leaves_no_forwarding_address(self, reviser):
        applied = reviser.apply(make_config(timeout_s=45.0), "cable_baseline")
        assert "superseded_by" not in applied.store.load_metadata()

    def test_the_settings_travel_with_the_results(self, reviser):
        """What makes the edit survive relaunching: the store's own copy."""
        record_one(reviser.store)
        applied = reviser.apply(make_config(rows=4, repeats=3), "cable_baseline_v2")
        reloaded = ExperimentStore(applied.store.root).load_config()
        assert reloaded.to_dict() == applied.config.to_dict()
        assert reloaded.factors[0].grid.rows == 4
        assert reloaded.repeats == 3

    def test_the_launch_snapshot_is_carried_onto_the_new_experiment(self, tmp_path):
        """It is how the next launch tells a GUI edit from a YAML edit."""
        config = make_config()
        root = experiment_root("run", "cable_baseline", base=tmp_path)
        reviser = ExperimentReviser(
            run="run",
            name="cable_baseline",
            config=config,
            store=ExperimentStore.open(root, config),
            base=tmp_path,
            metadata={"source_config_snapshot": config.to_dict()},
        )
        # No source_config, so the config dir would fall back to the repo's own
        # configs/experiments -- and this test is about the metadata, not the
        # YAML. Everything about that file is pinned below, under a tmp_path.
        applied = reviser.apply(
            make_config(rows=4), "cable_baseline_v2", write_config_file=False
        )
        assert applied.store.load_metadata()["source_config_snapshot"] == (
            config.to_dict()
        )

    def test_the_edited_settings_are_written_next_to_the_source(self, reviser):
        applied = reviser.apply(make_config(rows=4), "cable_baseline")
        assert applied.config_path is not None
        assert applied.config_path.name == "cable_baseline.yaml"
        assert ExperimentConfig.load(applied.config_path).to_dict() == (
            applied.config.to_dict()
        )

    def test_a_no_op_leaves_no_yaml_behind(self, reviser):
        """Opening the dialog, looking, and pressing OK writes nothing."""
        applied = reviser.apply(make_config(), "cable_baseline")
        assert applied.revision.action is RevisionAction.UNCHANGED
        assert applied.config_path is None
        assert not reviser.config_dir().exists()

    def test_the_operators_own_yaml_is_never_overwritten(self, tmp_path):
        source = tmp_path / "configs" / "cable_baseline.yaml"
        config = make_config()
        config.save(source)
        root = experiment_root("run", "cable_baseline", base=tmp_path)
        reviser = ExperimentReviser(
            run="run",
            name="cable_baseline",
            config=config,
            store=ExperimentStore.open(root, config),
            base=tmp_path,
            source_config=source,
        )
        applied = reviser.apply(make_config(rows=5), "cable_baseline")
        assert applied.config_path != source
        assert ExperimentConfig.load(source).factors[0].grid.rows == 2

    def test_the_hook_sees_the_new_store(self, reviser):
        seen = []
        reviser.on_applied = seen.append
        applied = reviser.apply(make_config(rows=4), "cable_next")
        assert [a.store.root for a in seen] == [applied.store.root]

    def test_a_raising_hook_does_not_lose_the_revision(self, reviser):
        def boom(_applied):
            raise RuntimeError("no")

        reviser.on_applied = boom
        assert reviser.apply(make_config(rows=4), "cable_next").store.exists


class TestSuggestName:
    def test_suggests_a_free_name(self, reviser):
        assert reviser.suggest_name("cable_baseline") == "cable_baseline_v2"

    def test_counts_up_past_names_already_taken(self, reviser, tmp_path):
        ExperimentStore.open(
            experiment_root("run", "cable_baseline_v2", base=tmp_path), make_config()
        )
        assert reviser.suggest_name("cable_baseline") == "cable_baseline_v3"

    def test_does_not_stack_suffixes(self, reviser):
        assert reviser.suggest_name("cable_baseline_v2") == "cable_baseline_v3"


class TestPlanSignature:
    def test_configs_that_stage_the_same_conditions_match(self):
        assert plan_signature(make_config()) == plan_signature(
            make_config(timeout_s=90.0)
        )

    def test_a_reshaped_grid_does_not(self):
        assert plan_signature(make_config()) != plan_signature(make_config(cols=4))

    def test_more_repeats_does_not(self):
        assert plan_signature(make_config()) != plan_signature(make_config(repeats=2))

    def test_regrouping_the_repeats_does_not(self):
        assert plan_signature(make_config(repeats=2)) != plan_signature(
            make_config(repeats=2, grouping="sweep")
        )

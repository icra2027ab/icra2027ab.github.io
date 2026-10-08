"""The disambiguation experiment: several objects staged, one of them named.

Three things carry this experiment and nothing else in the suite covers them:
the layouts are *spread* and *reproducible*, the two factors are wired to each
other correctly, and the prompt really does change per rollout — which is the
independent variable, so a template that silently failed to fill would produce
a whole afternoon of identically-prompted rollouts that look like data.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.factors import (
    GridPositionFactor,
    ObjectLayoutFactor,
    TargetObjectFactor,
    factor_from_dict,
)
from policy_abstraction.experiments.grid import TableGrid
from policy_abstraction.experiments.layout import (
    SceneObject,
    parse_objects,
    sample_layouts,
    sample_positions_uv,
)

OBJECTS = ["black cable", "white cable", "red cable"]


def make_region(width: float = 0.40, height: float = 0.30) -> TableGrid:
    return TableGrid(
        origin=(0.25, -0.20, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(1.0, 0.0, 0.0),
        cell_size_u=width,
        cell_size_v=height,
        rows=1,
        cols=1,
        marker_height=0.015,
        name="objects",
    )


def make_config(**overrides) -> ExperimentConfig:
    layout = {
        "kind": "object_layout",
        "name": "object_layout",
        "region": make_region().to_dict(),
        "objects": OBJECTS,
        "layouts": 4,
        "min_separation": 0.10,
        "margin": 0.05,
        "seed": 0,
        **overrides,
    }
    return ExperimentConfig(
        name="disambiguation",
        task="Put the {target_object} in the bag.",
        factors=[
            factor_from_dict(layout),
            factor_from_dict(
                {
                    "kind": "target_object",
                    "name": "target_object",
                    "layout": "object_layout",
                }
            ),
        ],
    )


def layout_of(config: ExperimentConfig) -> ObjectLayoutFactor:
    """The experiment's layout factor, narrowed from the base ``Factor`` type."""
    factor = config.factor("object_layout")
    assert isinstance(factor, ObjectLayoutFactor)
    return factor


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def layout_factor(config):
    return layout_of(config)


class TestSampling:
    def test_every_pair_clears_the_minimum_separation(self):
        points = sample_positions_uv(
            0.4, 0.3, 3, min_separation=0.1, rng=np.random.default_rng(7)
        )
        for a, b in itertools.combinations(points, 2):
            assert np.linalg.norm(a - b) >= 0.1

    def test_points_stay_inside_the_box(self):
        points = sample_positions_uv(
            0.4, 0.3, 4, min_separation=0.05, rng=np.random.default_rng(3)
        )
        assert (points >= 0).all()
        assert (points[:, 0] <= 0.4).all() and (points[:, 1] <= 0.3).all()

    def test_an_impossible_request_says_what_to_relax(self):
        with pytest.raises(ValueError, match="Enlarge the region"):
            sample_positions_uv(0.1, 0.1, 6, min_separation=0.2)

    def test_the_margin_is_kept_clear_of_the_region_edge(self):
        region = make_region()
        margin = 0.06
        layouts = sample_layouts(
            region, parse_objects(OBJECTS), count=3, margin=margin, seed=0
        )
        origin = np.asarray(region.origin)
        for layout in layouts:
            for position in layout.values():
                rel = position - origin
                u = float(np.dot(rel, np.asarray(region.u_axis)))
                v = float(np.dot(rel, np.asarray(region.v_axis)))
                assert margin <= u <= region.extent_u - margin
                assert margin <= v <= region.extent_v - margin

    def test_layouts_are_reproducible_across_constructions(self):
        """Both arms of a comparison must stage the identical scenes."""
        a = layout_of(make_config())
        b = layout_of(make_config())
        for level in a.levels:
            for obj in a.object_ids:
                assert a.position_of(level.id, obj) == pytest.approx(
                    b.position_of(level.id, obj)
                )

    def test_adding_a_layout_leaves_the_earlier_ones_alone(self):
        """So an experiment can be extended without invalidating its results."""
        four = layout_of(make_config(layouts=4))
        six = layout_of(make_config(layouts=6))
        for level in four.levels:
            for obj in four.object_ids:
                assert four.position_of(level.id, obj) == pytest.approx(
                    six.position_of(level.id, obj)
                )

    def test_layouts_actually_differ_from_each_other(self):
        """The whole point: position must not be a proxy for identity."""
        factor = layout_of(make_config(layouts=4))
        first = [factor.position_of(lv.id, "black_cable") for lv in factor.levels]
        spread = max(
            float(np.linalg.norm(a - b)) for a, b in itertools.combinations(first, 2)
        )
        assert spread > 0.05

    def test_objects_are_lifted_to_the_marker_height(self):
        factor = layout_of(make_config())
        for obj in factor.object_ids:
            assert factor.position_of("L1", obj)[2] == pytest.approx(0.015)


class TestSceneObjects:
    def test_a_bare_name_becomes_a_slug_and_a_label(self):
        obj = SceneObject.from_any("black USB cable")
        assert (obj.id, obj.label, obj.prompt_name) == (
            "black_usb_cable",
            "black USB cable",
            "black USB cable",
        )

    def test_the_prompt_name_can_differ_from_what_the_operator_reads(self):
        obj = SceneObject.from_any(
            {"id": "black_cable", "label": "black cable (coiled)", "prompt": "cable"}
        )
        assert obj.label == "black cable (coiled)"
        assert obj.prompt_name == "cable"

    def test_duplicate_ids_are_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            parse_objects(["red cable", "red cable"])


class TestFactorWiring:
    def test_the_target_factor_takes_its_levels_from_the_layout(self, config):
        target = config.factor("target_object")
        assert [lv.id for lv in target.levels] == [
            "black_cable",
            "white_cable",
            "red_cable",
        ]

    def test_a_dangling_layout_reference_fails_at_load(self):
        with pytest.raises(ValueError, match="no factor by that name"):
            ExperimentConfig(
                name="x",
                factors=[
                    factor_from_dict(
                        {
                            "kind": "target_object",
                            "name": "target_object",
                            "layout": "nope",
                        }
                    )
                ],
            )

    def test_pointing_at_the_wrong_kind_of_factor_fails_at_load(self):
        with pytest.raises(ValueError, match="must name an 'object_layout'"):
            ExperimentConfig(
                name="x",
                factors=[
                    GridPositionFactor("somewhere", make_region()),
                    TargetObjectFactor("target_object", layout="somewhere"),
                ],
            )

    def test_an_unlinked_target_factor_refuses_rather_than_guessing(self):
        unlinked = TargetObjectFactor("target_object", layout="object_layout")
        with pytest.raises(RuntimeError, match="not been linked"):
            unlinked.level("black_cable")

    def test_the_plan_is_every_layout_crossed_with_every_object(self, config):
        plan = config.build_plan()
        assert len(plan) == 4 * 3
        assert {
            (r.assignments["object_layout"], r.assignments["target_object"])
            for r in plan
        } == {
            (f"L{i + 1}", obj)
            for i in range(4)
            for obj in ("black_cable", "white_cable", "red_cable")
        }

    def test_the_scene_is_identical_across_a_layouts_targets(self, config):
        """Only the prompt may change between the rollouts of one layout."""
        scenes = {
            tuple(
                tuple(np.round(p, 6))
                for p in layout_of(config).positions("L2").values()
            )
        }
        assert len(scenes) == 1

    def test_the_layout_is_drawn_before_the_target_that_rings_it(self, config):
        priorities = [f.draw_priority for f in config.factors]
        assert priorities == [0, 1]

    def test_the_factors_round_trip_through_yaml(self, config, tmp_path):
        path = config.save(tmp_path / "exp.yaml")
        reloaded = ExperimentConfig.load(path)
        assert [r.assignments for r in reloaded.build_plan()] == [
            r.assignments for r in config.build_plan()
        ]
        assert layout_of(reloaded).position_of("L3", "red_cable") == pytest.approx(
            layout_of(config).position_of("L3", "red_cable")
        )


class TestPrompt:
    def test_the_template_is_filled_with_the_named_object(self, config):
        assert (
            config.task_for({"object_layout": "L1", "target_object": "white_cable"})
            == "Put the white cable in the bag."
        )

    def test_every_object_gets_its_own_prompt(self, config):
        prompts = {config.task_for(r.assignments) for r in config.build_plan()}
        assert prompts == {
            "Put the black cable in the bag.",
            "Put the white cable in the bag.",
            "Put the red cable in the bag.",
        }

    def test_a_task_without_placeholders_is_sent_verbatim(self, config):
        config.task = "Put all cables in the bag."
        assert (
            config.task_for({"object_layout": "L1", "target_object": "red_cable"})
            == "Put all cables in the bag."
        )

    def test_an_unfillable_template_survives_rather_than_raising(self, config):
        """Mid-session, a literal prompt is recoverable and a crash is not."""
        config.task = "Put the {no_such_factor} in the bag."
        assert (
            config.task_for({"object_layout": "L1", "target_object": "red_cable"})
            == "Put the {no_such_factor} in the bag."
        )


class TestRecordedGeometry:
    def test_every_object_position_is_recorded_not_just_the_target(self, config):
        targets = config.target_positions(
            {"object_layout": "L2", "target_object": "white_cable"}
        )
        assert set(targets) == {
            "object_layout.black_cable",
            "object_layout.white_cable",
            "object_layout.red_cable",
            "target_object",
        }

    def test_the_targets_position_is_where_the_layout_put_it(self, config):
        assignments = {"object_layout": "L3", "target_object": "red_cable"}
        targets = config.target_positions(assignments)
        assert targets["target_object"] == targets["object_layout.red_cable"]

    def test_without_a_layout_assignment_the_target_has_no_position(self, config):
        assert config.factor("target_object").world_position("red_cable") is None

    def test_staging_lines_name_every_object_and_its_place(self, config):
        lines = layout_of(config).staging_lines("L1")
        assert len(lines) == 1 + len(OBJECTS)
        for name in OBJECTS:
            assert any(name in line for line in lines)


class TestObjectOverride:
    def test_a_different_count_re_samples_the_layouts(self, config):
        wider = config.with_objects(["a cable", "b cable", "c cable", "d cable"])
        assert len(wider.factor("target_object").levels) == 4
        assert len(wider.build_plan()) == 4 * 4

    def test_the_override_reaches_the_prompt(self, config):
        renamed = config.with_objects(["blue mug"])
        plan = renamed.build_plan()
        assert renamed.task_for(plan[0].assignments) == "Put the blue mug in the bag."

    def test_replacing_objects_on_an_experiment_without_a_layout_is_refused(self):
        config = ExperimentConfig(
            name="x", factors=[GridPositionFactor("cable_position", make_region())]
        )
        with pytest.raises(ValueError, match="no 'object_layout' factor"):
            config.with_objects(["a"])


class TestOverlay:
    """The overlay must show the whole scene, and must not invent a target."""

    def _canvas(self, projectable: bool = True):
        from policy_abstraction.experiments.overlay import OverlayCanvas

        return OverlayCanvas(
            np.zeros((64, 64, 3), dtype=np.uint8),
            intrinsic=(
                np.array([[60.0, 0, 32.0], [0, 60.0, 32.0], [0, 0, 1.0]])
                if projectable
                else None
            ),
            extrinsic=np.eye(4) if projectable else None,
            preprocessed=False,
        )

    def test_marking_an_object_registers_where_it_is(self, layout_factor):
        canvas = self._canvas()
        layout_factor.annotate(canvas, "L1", {})
        for obj in layout_factor.object_ids:
            assert canvas.object_anchor(obj) is not None

    def test_highlighting_an_unmarked_object_draws_nothing(self):
        canvas = self._canvas()
        before = canvas.image.copy()
        canvas.highlight_object("never_marked", label="nope")
        assert np.array_equal(canvas.image, before)

    def test_an_uncalibrated_canvas_still_registers_anchors(self, layout_factor):
        """No projection, no marks — but the geometry is not lost."""
        canvas = self._canvas(projectable=False)
        layout_factor.annotate(canvas, "L1", {})
        assert canvas.object_anchor("black_cable") is not None
        assert not canvas.image.any()

    def test_without_calibration_the_badges_carry_the_coordinates(self, config):
        """The badge text is the whole instruction when nothing can be drawn."""
        from policy_abstraction.experiments.overlay import render_staging_view

        assignments = {"object_layout": "L1", "target_object": "black_cable"}
        annotated = render_staging_view(
            np.zeros((64, 480, 3), dtype=np.uint8),
            config.factors,
            assignments,
            scale=1.0,
        )
        # Badges are the only thing drawn without a projection, and there is
        # one per staging line rather than one per factor.
        assert annotated.any()


class TestGeometryPrintout:
    def test_calibration_lists_every_layout_and_object(self, layout_factor):
        lines = layout_factor.geometry_lines()
        assert "40 x 30 cm" in lines[0]
        for level in layout_factor.levels:
            assert any(f"layout {level.id}" in line for line in lines)
        assert sum(line.count("black cable") for line in lines) == len(
            layout_factor.levels
        )


class TestSession:
    """The prompt has to reach the policy, per rollout, or there is no experiment."""

    @pytest.fixture
    def session(self, tmp_path, config):
        from policy_abstraction.experiments.session import ExperimentSession
        from policy_abstraction.experiments.store import ExperimentStore
        from tests.experiments.conftest import FakeBackends

        backends = FakeBackends(runner_kwargs={"steps": 2, "tick": 0.001})
        store = ExperimentStore.open(tmp_path / "exp", config)
        session = ExperimentSession(backends, config, store, policy_setting="baseline")
        session.prepare_env()
        yield session
        session.shutdown(timeout=5)

    def _run_one(self, session):
        import time

        assert session.start_rollout()
        deadline = time.time() + 10
        while session.is_running and time.time() < deadline:
            time.sleep(0.01)
        assert not session.is_running

    def test_the_runner_is_built_with_the_filled_prompt(self, session):
        planned = session.current
        self._run_one(session)
        expected = session.config.task_for(planned.assignments)
        assert session.backends.built[0]["task_description"] == expected
        assert expected.startswith("Put the ") and "{" not in expected

    def test_the_prompt_is_written_into_the_record(self, session):
        planned = session.current
        self._run_one(session)
        record = session.last_record
        assert record.task == session.config.task_for(planned.assignments)

    def test_consecutive_rollouts_get_different_prompts(self, session):
        prompts = []
        for _ in range(3):
            prompts.append(session.config.task_for(session.current.assignments))
            self._run_one(session)
        assert len(set(prompts)) == 3

    def test_the_record_keeps_every_object_position(self, session):
        self._run_one(session)
        positions = session.last_record.target_positions
        assert len(positions) == len(OBJECTS) + 1
        assert all(len(p) == 3 for p in positions.values())

    def test_the_instructions_cover_every_object_plus_the_target(self, session):
        lines = session.instructions()
        assert len(lines) == 1 + len(OBJECTS) + 1
        assert "TARGET" in lines[-1]

    def test_a_skipped_condition_still_records_its_prompt(self, session):
        planned = session.current
        session.skip_current()
        assert session.last_record.task == session.config.task_for(planned.assignments)


class TestReport:
    def test_the_report_breaks_results_down_by_object(self, tmp_path, config):
        from policy_abstraction.experiments.outcomes import (
            FailureMode,
            Outcome,
            RolloutRecord,
        )
        from policy_abstraction.experiments.report import build_report
        from policy_abstraction.experiments.store import ExperimentStore

        store = ExperimentStore.open(tmp_path / "exp", config)
        for planned in config.build_plan():
            target = planned.assignments["target_object"]
            # The black cable is always found, the others never — the shape a
            # real disambiguation failure would have.
            succeeded = target == "black_cable"
            store.append(
                RolloutRecord(
                    index=planned.index,
                    assignments=dict(planned.assignments),
                    outcome=Outcome.SUCCESS if succeeded else Outcome.FAILURE,
                    failure_mode=(
                        FailureMode.NONE if succeeded else FailureMode.WRONG_TARGET
                    ),
                    task=config.task_for(planned.assignments),
                    target_positions=config.target_positions(planned.assignments),
                )
            )

        summary = build_report(store, config)
        by_object = summary["per_factor"]["target_object"]["levels"]
        assert by_object["black_cable"]["success_rate"] == pytest.approx(1.0)
        assert by_object["white_cable"]["success_rate"] == pytest.approx(0.0)
        assert by_object["white_cable"]["failure_modes"] == {"wrong_target": 4}
        # The nuisance axis should show nothing, which is the control.
        assert summary["per_factor"]["object_layout"]["rate_spread"] == pytest.approx(
            0.0
        )
        assert (store.report_dir / "report.md").exists()
        assert (store.report_dir / "target_object_rates.png").exists()
        assert (store.report_dir / "object_layout_rates.png").exists()


class TestProjectedPlot:
    def test_the_layout_map_is_drawn_when_there_is_a_frame_to_draw_on(
        self, tmp_path, config
    ):
        from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord
        from policy_abstraction.experiments.store import ExperimentStore
        from policy_abstraction.experiments.viz import plot_layouts_projected

        store = ExperimentStore.open(tmp_path / "exp", config)
        store.save_reference_view(
            np.full((64, 64, 3), 40, dtype=np.uint8),
            intrinsic=np.array([[60.0, 0, 32.0], [0, 60.0, 32.0], [0, 0, 1.0]]),
            extrinsic=np.eye(4),
            camera_name="agentview",
            preprocessed=False,
        )
        records = [
            RolloutRecord(
                index=r.index,
                assignments=dict(r.assignments),
                outcome=Outcome.SUCCESS,
            )
            for r in config.build_plan()
        ]
        out = plot_layouts_projected(
            records,
            layout_of(config),
            config.factor("target_object"),
            store,
            tmp_path / "layouts.png",
        )
        assert out is not None and out.exists()

    def test_without_a_reference_frame_it_declines_rather_than_raising(
        self, tmp_path, config
    ):
        from policy_abstraction.experiments.store import ExperimentStore
        from policy_abstraction.experiments.viz import plot_layouts_projected

        store = ExperimentStore.open(tmp_path / "exp", config)
        assert (
            plot_layouts_projected(
                [],
                layout_of(config),
                config.factor("target_object"),
                store,
                tmp_path / "layouts.png",
            )
            is None
        )


class TestConfusionMatrix:
    """Which object it went for against the one the prompt named.

    The disambiguation experiment is a classification task, so this is the
    result: the diagonal is the reference resolved and every other cell names
    the specific mistake. A success rate alone cannot tell "fails on the white
    cable" from "fails on the white cable by always going to the black one".
    """

    def _records(self, config, went_for):
        """One rollout per plan entry; *went_for* maps named -> acted-on id."""
        from policy_abstraction.experiments.outcomes import (
            FailureMode,
            Outcome,
            RolloutRecord,
        )

        out = []
        for planned in config.build_plan():
            named = planned.assignments["target_object"]
            acted = went_for(named)
            correct = acted == named
            out.append(
                RolloutRecord(
                    index=planned.index,
                    assignments=dict(planned.assignments),
                    outcome=Outcome.SUCCESS if correct else Outcome.FAILURE,
                    failure_mode=(
                        FailureMode.NONE if correct else FailureMode.WRONG_TARGET
                    ),
                    actual_target=acted,
                )
            )
        return out

    def test_a_perfect_policy_is_all_diagonal(self, config):
        from policy_abstraction.experiments.report import confusion_matrix

        matrix = confusion_matrix(self._records(config, lambda n: n), config)
        for named, row in matrix["rows"].items():
            assert row[named] == 4
            assert sum(row.values()) == 4, "nothing lands off the diagonal"
        assert matrix["accuracy"] == pytest.approx(1.0)

    def test_a_systematic_confusion_shows_up_as_one_column(self, config):
        """The finding a success rate cannot express."""
        from policy_abstraction.experiments.report import confusion_matrix

        matrix = confusion_matrix(
            self._records(config, lambda n: "black_cable"), config
        )
        assert matrix["rows"]["white_cable"]["black_cable"] == 4
        assert matrix["rows"]["red_cable"]["black_cable"] == 4
        assert matrix["rows"]["black_cable"]["black_cable"] == 4
        assert matrix["accuracy"] == pytest.approx(4 / 12)

    def test_never_committing_gets_its_own_column(self, config):
        from policy_abstraction.experiments.outcomes import NO_OBJECT
        from policy_abstraction.experiments.report import NOT_RESOLVED, confusion_matrix

        matrix = confusion_matrix(self._records(config, lambda n: NO_OBJECT), config)
        assert all(row[NOT_RESOLVED] == 4 for row in matrix["rows"].values())
        assert matrix["accuracy"] == pytest.approx(0.0)
        assert matrix["resolved"] == 12, "it was watched; that is a real observation"

    def test_going_for_several_objects_counts_under_each(self, config):
        """Like a failure with several modes: each confusion is a finding."""
        from policy_abstraction.experiments.outcomes import (
            FailureMode,
            Outcome,
            RolloutRecord,
        )
        from policy_abstraction.experiments.report import UNRECORDED, confusion_matrix

        planned = config.build_plan()[0]
        named = planned.assignments["target_object"]
        others = [o for o in ("black_cable", "white_cable", "red_cable") if o != named]
        record = RolloutRecord(
            index=planned.index,
            assignments=dict(planned.assignments),
            outcome=Outcome.FAILURE,
            failure_mode=FailureMode.WRONG_TARGET,
            actual_targets=others,
        )
        matrix = confusion_matrix([record], config)
        row = matrix["rows"][named]
        assert all(row[o] == 1 for o in others)
        assert row[named] == 0 and row[UNRECORDED] == 0
        assert matrix["scored"] == 1, "one rollout, however many objects"
        assert matrix["resolved"] == 1
        assert matrix["correct"] == 0
        assert matrix["multiple"] == 1
        assert matrix["accuracy"] == 0.0

    def test_reaching_the_named_object_among_others_is_still_resolved(self, config):
        from policy_abstraction.experiments.outcomes import (
            FailureMode,
            Outcome,
            RolloutRecord,
        )
        from policy_abstraction.experiments.report import confusion_matrix

        planned = config.build_plan()[0]
        named = planned.assignments["target_object"]
        other = next(o for o in ("black_cable", "white_cable") if o != named)
        record = RolloutRecord(
            index=planned.index,
            assignments=dict(planned.assignments),
            outcome=Outcome.FAILURE,
            failure_mode=FailureMode.DROPPED,
            actual_targets=[other, named],
        )
        matrix = confusion_matrix([record], config)
        assert matrix["rows"][named][named] == 1
        assert matrix["rows"][named][other] == 1
        assert matrix["correct"] == 1, "counted once, not once per cell"

    def test_rollouts_nobody_scored_are_held_out_not_assumed(self, config):
        """The honest treatment of data collected before this was asked for."""
        from policy_abstraction.experiments.report import UNRECORDED, confusion_matrix

        records = self._records(config, lambda n: "")
        matrix = confusion_matrix(records, config)
        assert all(row[UNRECORDED] == 4 for row in matrix["rows"].values())
        assert matrix["scored"] == 12
        assert matrix["resolved"] == 0, "none of them say what was reached"
        assert matrix["accuracy"] == 0.0

    def test_a_success_counts_as_the_named_object_however_it_was_logged(self, config):
        """That is what a success means, so it needs no separate answer."""
        from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord
        from policy_abstraction.experiments.report import confusion_matrix

        planned = config.build_plan()[0]
        named = planned.assignments["target_object"]
        record = RolloutRecord(
            index=planned.index,
            assignments=dict(planned.assignments),
            outcome=Outcome.SUCCESS,
        )
        matrix = confusion_matrix([record], config)
        assert matrix["rows"][named][named] == 1
        assert matrix["accuracy"] == pytest.approx(1.0)

    def test_an_experiment_with_no_named_target_has_no_matrix(self):
        """A positional experiment stages one object; nothing to confuse."""
        from policy_abstraction.experiments.config import ExperimentConfig
        from policy_abstraction.experiments.factors import GridPositionFactor
        from policy_abstraction.experiments.grid import TableGrid
        from policy_abstraction.experiments.report import confusion_matrix

        positional = ExperimentConfig(
            name="p",
            task="Put the cable in the box.",
            factors=[
                GridPositionFactor(
                    "cable_position",
                    TableGrid(
                        origin=(0, 0, 0),
                        u_axis=(0, 1, 0),
                        v_axis=(1, 0, 0),
                        rows=1,
                        cols=2,
                        name="cable",
                    ),
                )
            ],
        )
        assert confusion_matrix([], positional) is None

    def test_the_report_prints_it(self, config, tmp_path):
        from policy_abstraction.experiments.report import build_report
        from policy_abstraction.experiments.store import ExperimentStore

        store = ExperimentStore.open(tmp_path / "exp", config)
        for record in self._records(config, lambda n: "black_cable"):
            store.append(record)
        summary = build_report(store, config)

        assert summary["confusion_matrix"]["rows"]["red_cable"]["black_cable"] == 4
        text = (store.report_dir / "report.md").read_text()
        assert "## Confusion matrix" in text
        assert "went to the object named" in text
        assert "task success" in text
        assert (store.report_dir / "confusion_matrix.png").exists()

    def test_each_row_carries_its_task_success(self, config):
        """Reaching the right object and finishing the task are different."""
        from policy_abstraction.experiments.outcomes import FailureMode, Outcome
        from policy_abstraction.experiments.report import confusion_matrix

        records = self._records(config, lambda n: n)  # all reach the named one
        # ... but the first black-cable rollout dropped it on the way.
        dropped = next(
            r for r in records if r.assignments["target_object"] == "black_cable"
        )
        dropped.outcome = Outcome.FAILURE
        dropped.failure_mode = FailureMode.DROPPED
        matrix = confusion_matrix(records, config)
        black = matrix["success"]["black_cable"]
        assert black["n"] == 4 and black["successes"] == 3
        assert black["rate"] == pytest.approx(0.75)
        # The diagonal still says it went for the black cable every time.
        assert matrix["rows"]["black_cable"]["black_cable"] == 4
        assert matrix["objects"] == ["black_cable", "white_cable", "red_cable"]

    def test_the_plot_is_objects_by_objects(self, config, tmp_path):
        from policy_abstraction.experiments.outcomes import NO_OBJECT
        from policy_abstraction.experiments.report import confusion_matrix
        from policy_abstraction.experiments.viz import plot_confusion_matrix

        matrix = confusion_matrix(self._records(config, lambda n: NO_OBJECT), config)
        out = plot_confusion_matrix(matrix, tmp_path / "cm.png")
        assert out is not None and out.exists()
        assert (
            plot_confusion_matrix({"rows": {}, "columns": []}, tmp_path / "e.png")
            is None
        )


class TestOutcomesByCondition:
    """The cross-tab that needs no new data, so it works on every experiment."""

    def test_it_counts_each_level_by_how_it_ended(self, config):
        from policy_abstraction.experiments.factors import TargetObjectFactor
        from policy_abstraction.experiments.outcomes import (
            FailureMode,
            Outcome,
            RolloutRecord,
        )
        from policy_abstraction.experiments.report import outcomes_by_condition

        factor = next(f for f in config.factors if isinstance(f, TargetObjectFactor))
        records = []
        for planned in config.build_plan():
            named = planned.assignments["target_object"]
            good = named == "black_cable"
            records.append(
                RolloutRecord(
                    index=planned.index,
                    assignments=dict(planned.assignments),
                    outcome=Outcome.SUCCESS if good else Outcome.FAILURE,
                    failure_mode=FailureMode.NONE if good else FailureMode.DROPPED,
                )
            )
        matrix = outcomes_by_condition(records, factor)
        assert matrix["columns"] == ["success", "dropped"]
        assert matrix["rows"]["black_cable"] == {"success": 4, "dropped": 0}
        assert matrix["rows"]["white_cable"] == {"success": 0, "dropped": 4}
        assert matrix["labels"]["white_cable"] == "white cable"

    def test_only_the_modes_that_happened_get_a_column(self, config):
        """Seven empty columns would bury the two that carry the result."""
        from policy_abstraction.experiments.factors import TargetObjectFactor
        from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord
        from policy_abstraction.experiments.report import outcomes_by_condition

        factor = next(f for f in config.factors if isinstance(f, TargetObjectFactor))
        records = [
            RolloutRecord(
                index=p.index,
                assignments=dict(p.assignments),
                outcome=Outcome.SUCCESS,
            )
            for p in config.build_plan()
        ]
        assert outcomes_by_condition(records, factor)["columns"] == ["success"]

"""Overlay projection conventions, and the report's aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.envs.spec import CameraSpec
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
from policy_abstraction.experiments.overlay import (
    OverlayCanvas,
    preprocessed_for,
    render_staging_view,
    to_drawable,
)
from policy_abstraction.experiments.report import (
    _wilson_interval,
    duration_stats,
    render_markdown,
    summarize,
)
from policy_abstraction.experiments.store import ExperimentStore


@pytest.fixture
def grid():
    return TableGrid(
        origin=(-0.3, -0.25, 0.9),
        u_axis=(0, 1, 0),
        v_axis=(1, 0, 0),
        cell_size_u=0.125,
        cell_size_v=0.15,
        rows=3,
        cols=4,
        marker_height=0.03,
        name="cable",
    )


@pytest.fixture
def camera():
    """A downward-looking camera 1 m above the grid, world == base frame."""
    intrinsic = np.array([[300.0, 0, 128.0], [0, 300.0, 128.0], [0, 0, 1.0]])
    # camera-to-world: looking down -z, x right, y down in image.
    extrinsic = np.array(
        [
            [1.0, 0, 0, 0.0],
            [0, -1.0, 0, 0.0],
            [0, 0, -1.0, 1.9],
            [0, 0, 0, 1.0],
        ]
    )
    return intrinsic, extrinsic


@pytest.fixture
def image():
    return np.full((256, 256, 3), 40, dtype=np.uint8)


class TestPreprocessedFor:
    def test_matches_the_flip_u_rule(self):
        """Must agree with backends.resolve_flip_u: same question, same answer."""
        flipped = CameraSpec(
            name="c", role="world", image_key="c", image_transform="flip_vh"
        )
        upright = CameraSpec(
            name="c", role="world", image_key="c", image_transform="none"
        )
        assert preprocessed_for(flipped) is True
        assert preprocessed_for(upright) is False

    def test_defaults_to_the_simulator_convention(self):
        assert preprocessed_for(None) is True


class TestOverlayCanvas:
    def test_uncalibrated_canvas_degrades_instead_of_raising(self, image, grid):
        canvas = OverlayCanvas(image)
        assert not canvas.projectable
        # Every 3-D call is a no-op rather than an exception.
        canvas.draw_grid(grid)
        canvas.highlight_cell(grid, (0, 0))
        canvas.draw_orientation_arrow(grid, (0, 0), 45.0)
        canvas.shade_cell(grid, (0, 0), (255, 0, 0))
        assert canvas.cell_center_px(grid, (0, 0)) is None
        assert np.array_equal(canvas.image, image)

    def test_projection_is_independent_of_display_scale(self, image, grid, camera):
        """Scaling must not move a point relative to the picture."""
        k, e = camera
        one = OverlayCanvas(image, intrinsic=k, extrinsic=e, scale=1.0)
        three = OverlayCanvas(image, intrinsic=k, extrinsic=e, scale=3.0)
        centre = grid.cell_center((1, 2))[None, :]
        px1, ok1 = one.project(centre)
        px3, ok3 = three.project(centre)
        assert ok1[0] and ok3[0]
        assert np.allclose(px3[0], px1[0] * 3, atol=3)

    def test_drawing_mutates_a_copy_not_the_source(self, image, grid, camera):
        k, e = camera
        canvas = OverlayCanvas(image, intrinsic=k, extrinsic=e)
        canvas.draw_grid(grid)
        assert not np.array_equal(canvas.image, image)
        assert np.array_equal(image, np.full((256, 256, 3), 40, dtype=np.uint8))

    def test_highlight_registers_an_anchor_for_other_factors(self, image, grid, camera):
        k, e = camera
        canvas = OverlayCanvas(image, intrinsic=k, extrinsic=e)
        assert canvas.resolve_anchor(None, {}) is None
        canvas.highlight_cell(grid, (1, 1))
        assert canvas.resolve_anchor("cable", {}) == (grid, (1, 1))
        assert canvas.resolve_anchor(None, {}) == (grid, (1, 1))

    def test_wrong_shape_image_rejected(self):
        with pytest.raises(ValueError, match="RGB"):
            OverlayCanvas(np.zeros((8, 8), dtype=np.uint8))


class TestProjectedSuccessMap:
    def test_cell_tint_is_red_only_for_a_clean_failure(self):
        from policy_abstraction.experiments.viz import (
            CELL_FAILED_RGB,
            CELL_GREEN_HIGH_RGB,
            _cell_rgb,
        )

        assert _cell_rgb(0, 3) == CELL_FAILED_RGB
        assert _cell_rgb(3, 3) == CELL_GREEN_HIGH_RGB
        one, two = _cell_rgb(1, 3), _cell_rgb(2, 3)
        assert one != CELL_FAILED_RGB and one[1] > one[0]  # green, not red
        assert two[0] < one[0], "darker green as the rate climbs"


class TestDrawableText:
    @pytest.mark.parametrize(
        "raw,expected",
        [("45°", "45deg"), ("a — b", "a - b"), ("x…", "x...")],
    )
    def test_non_ascii_is_folded_not_dropped(self, raw, expected):
        """Hershey fonts are ASCII; unfolded text renders as '?'."""
        assert to_drawable(raw) == expected


class TestRenderStagingView:
    def test_draws_every_factor_without_a_camera(self, image, grid):
        factors = [
            GridPositionFactor("cable_position", grid),
            OrientationFactor("cable_orientation", [0, 90], anchor="cable"),
            AppearanceFactor("box_appearance", ["black", "white"]),
        ]
        out = render_staging_view(
            image,
            factors,
            {
                "cable_position": "B3",
                "cable_orientation": "090deg",
                "box_appearance": "black",
            },
            scale=2.0,
        )
        assert out.shape == (512, 512, 3)

    def test_missing_assignment_is_skipped_not_fatal(self, image, grid, camera):
        k, e = camera
        out = render_staging_view(
            image,
            [
                GridPositionFactor("cable_position", grid),
                AppearanceFactor("box", ["black"]),
            ],
            {"cable_position": "A1"},  # box unassigned
            intrinsic=k,
            extrinsic=e,
        )
        assert out.shape[:2] == (256, 256)

    def test_orientation_anchors_to_the_positional_factor(self, image, grid, camera):
        """The arrow must land in the cell the position factor selected."""
        k, e = camera
        pos = GridPositionFactor("cable_position", grid)
        orient = OrientationFactor("cable_orientation", [0], anchor="cable")
        a1 = render_staging_view(
            image,
            [pos, orient],
            {"cable_position": "A1", "cable_orientation": "000deg"},
            intrinsic=k,
            extrinsic=e,
        )
        c4 = render_staging_view(
            image,
            [pos, orient],
            {"cable_position": "C4", "cable_orientation": "000deg"},
            intrinsic=k,
            extrinsic=e,
        )
        assert not np.array_equal(a1, c4)


def _records(config, pattern):
    """One record per (cell, outcome) in *pattern* = {cell: [bool, ...]}."""
    out, index = [], 0
    for cell, outcomes in pattern.items():
        for ok in outcomes:
            out.append(
                RolloutRecord(
                    index=index,
                    assignments={"cable_position": cell},
                    outcome=Outcome.SUCCESS if ok else Outcome.FAILURE,
                    failure_mode=(FailureMode.NONE if ok else FailureMode.GRASP_MISS),
                    steps=100,
                    duration_s=30.0,
                )
            )
            index += 1
    return out


class TestSummarize:
    @pytest.fixture
    def config(self, grid):
        return ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)]
        )

    def test_rate_is_per_level_not_just_overall(self, config):
        summary = summarize(
            _records(config, {"A1": [True] * 3, "C4": [False] * 3}), config
        )
        levels = summary["per_factor"]["cable_position"]["levels"]
        assert levels["A1"]["success_rate"] == 1.0
        assert levels["C4"]["success_rate"] == 0.0
        assert summary["totals"]["success_rate"] == 0.5

    def test_rate_spread_is_the_distribution_shift_headline(self, config):
        flat = summarize(
            _records(config, {"A1": [True, False], "C4": [True, False]}), config
        )
        sloped = summarize(
            _records(config, {"A1": [True, True], "C4": [False, False]}), config
        )
        assert flat["per_factor"]["cable_position"]["rate_spread"] == 0.0
        assert sloped["per_factor"]["cable_position"]["rate_spread"] == 1.0
        assert sloped["per_factor"]["cable_position"]["best_level"] == "A1"
        assert sloped["per_factor"]["cable_position"]["worst_level"] == "C4"

    def test_aborted_rollouts_do_not_move_the_rate(self, config):
        records = _records(config, {"A1": [True, True]})
        records.append(
            RolloutRecord(
                index=99,
                assignments={"cable_position": "A1"},
                outcome=Outcome.ABORTED,
            )
        )
        summary = summarize(records, config)
        assert summary["totals"]["success_rate"] == 1.0
        assert summary["totals"]["collected"] == 2
        assert summary["totals"]["aborted"] == 1

    def test_uncollected_levels_are_named_not_scored_as_zero(self, config):
        summary = summarize(_records(config, {"A1": [True]}), config)
        factor = summary["per_factor"]["cable_position"]
        assert "C4" in factor["levels_without_data"]
        assert factor["levels"]["C4"]["n"] == 0
        # A cell with no data must not drag the spread to 100%.
        assert factor["rate_spread"] == 0.0

    def test_failure_modes_are_counted(self, config):
        summary = summarize(_records(config, {"A1": [False, False]}), config)
        assert summary["failure_modes"] == {"grasp_miss": 2}

    def test_a_failure_with_several_modes_counts_under_each(self, config):
        """Two failures, one of them ticked twice: three mode counts, but the
        shares are of failures, not of the mode total."""
        records = _records(config, {"A1": [False], "C4": [False]})
        records[0].set_failure_modes([FailureMode.WRONG_TARGET, FailureMode.DROPPED])
        records[1].set_failure_modes(FailureMode.DROPPED)
        summary = summarize(records, config)
        assert summary["failure_modes"] == {"dropped": 2, "wrong_target": 1}
        assert summary["totals"]["failures"] == 2
        table = summary["outcomes_by_condition"][0]
        assert set(table["columns"]) == {"success", "dropped", "wrong_target"}
        assert table["rows"]["A1"] == {"success": 0, "dropped": 1, "wrong_target": 1}
        assert table["rows"]["C4"] == {"success": 0, "dropped": 1, "wrong_target": 0}
        text = render_markdown(summary, config, [])
        assert "**Dropped** — 2 (100% of failures)" in text
        assert "**Wrong Target** — 1 (50% of failures)" in text
        assert "more than one mode" in text

    def test_empty_experiment_summarizes_without_dividing_by_zero(self, config):
        summary = summarize([], config)
        assert summary["totals"]["success_rate"] == 0.0
        assert summary["totals"]["collected"] == 0


class TestCompletionTime:
    """Every rollout's completion time is recorded, and the report says so."""

    @pytest.fixture
    def config(self, grid):
        return ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)]
        )

    def _timed(self, times):
        """One record per (seconds, success) pair."""
        return [
            RolloutRecord(
                index=i,
                assignments={"cable_position": "A1"},
                outcome=Outcome.SUCCESS if ok else Outcome.FAILURE,
                failure_mode=FailureMode.NONE if ok else FailureMode.GRASP_MISS,
                steps=100,
                duration_s=float(seconds),
            )
            for i, (seconds, ok) in enumerate(times)
        ]

    def test_stats_describe_the_spread_not_only_the_mean(self):
        stats = duration_stats([10.0, 20.0, 30.0, 40.0])
        assert stats["n"] == 4
        assert stats["mean_s"] == 25.0
        assert stats["median_s"] == 25.0
        assert (stats["min_s"], stats["max_s"]) == (10.0, 40.0)
        assert stats["total_s"] == 100.0
        assert stats["std_s"] == pytest.approx(11.18, abs=0.01)

    def test_unrecorded_times_are_dropped_not_counted_as_zero(self):
        assert duration_stats([0.0, 0.0, 50.0])["n"] == 1
        assert duration_stats([0.0, 0.0, 50.0])["mean_s"] == 50.0

    def test_no_times_at_all_is_zeroes_not_a_crash(self):
        assert duration_stats([])["n"] == 0
        assert duration_stats([])["median_s"] == 0.0

    def test_a_single_sample_is_its_own_quantiles(self):
        stats = duration_stats([77.0])
        assert stats["median_s"] == stats["p95_s"] == 77.0

    def test_successes_and_failures_are_timed_separately(self, config):
        summary = summarize(
            self._timed([(20, True), (40, True), (120, False), (120, False)]), config
        )
        timing = summary["completion_time"]
        assert timing["all"]["n"] == 4
        assert timing["successes"]["median_s"] == 30.0
        assert timing["failures"]["median_s"] == 120.0

    def test_timeouts_are_called_out(self, config):
        records = self._timed([(30, True), (120, False)])
        records[1].failure_mode = FailureMode.TIMEOUT
        timing = summarize(records, config)["completion_time"]
        assert timing["timeouts"]["n"] == 1
        assert timing["timeouts"]["median_s"] == 120.0

    def test_aborted_rollouts_are_timed_but_kept_out_of_the_rest(self, config):
        records = self._timed([(30, True)])
        records.append(
            RolloutRecord(
                index=9,
                assignments={"cable_position": "A1"},
                outcome=Outcome.ABORTED,
                duration_s=5.0,
            )
        )
        timing = summarize(records, config)["completion_time"]
        assert timing["all"]["n"] == 1
        assert timing["aborted"]["n"] == 1

    def test_each_level_carries_its_own_times(self, config):
        records = self._timed([(20, True), (60, True)])
        records[1].assignments = {"cable_position": "C4"}
        levels = summarize(records, config)["per_factor"]["cable_position"]["levels"]
        assert levels["A1"]["duration"]["median_s"] == 20.0
        assert levels["C4"]["duration"]["median_s"] == 60.0
        assert levels["A1"]["duration_successes"]["n"] == 1

    def test_the_warm_up_rollout_is_reported_but_not_averaged_in(self, config):
        """Its first inference compiles the checkpoint; that is not task time."""
        records = self._timed([(300, True), (40, True), (44, True)])
        records[0].warmup = True
        timing = summarize(records, config)["completion_time"]
        assert timing["all"]["n"] == 2
        assert timing["all"]["max_s"] == 44.0
        assert timing["warmup"]["n"] == 1
        assert timing["warmup"]["median_s"] == 300.0
        # Still available for anyone who wants the raw wall clock.
        assert timing["with_warmup"]["n"] == 3

    def test_the_headline_mean_skips_the_warm_up_too(self, config):
        records = self._timed([(300, True), (40, True), (44, True)])
        records[0].warmup = True
        totals = summarize(records, config)["totals"]
        assert totals["mean_duration_s"] == 42.0
        assert totals["warmup_rollouts"] == 1
        # The total robot time is the honest one: the warm-up did happen.
        assert totals["total_wall_clock_s"] == 384.0

    def test_a_warm_up_does_not_skew_its_cell(self, config):
        records = self._timed([(300, True), (40, True)])
        records[0].warmup = True
        levels = summarize(records, config)["per_factor"]["cable_position"]["levels"]
        assert levels["A1"]["duration"]["n"] == 1
        assert levels["A1"]["duration"]["median_s"] == 40.0
        # The rate still counts it: it ran, and it succeeded.
        assert levels["A1"]["n"] == 2

    def test_the_markdown_says_the_warm_up_was_left_out(self, config):
        records = self._timed([(300, True), (40, True), (44, False)])
        records[0].warmup = True
        text = render_markdown(summarize(records, config), config, [])
        assert "Warm-up, excluded above" in text
        assert "freshly loaded checkpoint" in text

    def test_the_markdown_reports_the_times(self, config):
        summary = summarize(self._timed([(20, True), (40, True), (120, False)]), config)
        text = render_markdown(summary, config, [])
        assert "## Completion time" in text
        assert "Successes" in text
        assert "faster" in text

    def test_an_experiment_with_no_times_skips_the_section(self, config):
        summary = summarize([], config)
        assert "## Completion time" not in render_markdown(summary, config, [])


class TestWilsonInterval:
    def test_no_samples_is_maximally_uncertain(self):
        assert _wilson_interval(0, 0) == (0.0, 1.0)

    def test_stays_inside_the_unit_interval(self):
        for successes, n in [(0, 3), (3, 3), (1, 2), (7, 10)]:
            low, high = _wilson_interval(successes, n)
            assert 0.0 <= low <= high <= 1.0

    def test_a_clean_sweep_keeps_honest_width(self):
        """The normal approximation would collapse this to zero width."""
        low, high = _wilson_interval(3, 3)
        assert low < 0.6
        assert high == pytest.approx(1.0)

    def test_more_samples_narrow_the_interval(self):
        few = _wilson_interval(5, 10)
        many = _wilson_interval(50, 100)
        assert (many[1] - many[0]) < (few[1] - few[0])


class TestReferenceView:
    def test_absent_reference_view_is_reported_not_raised(self, tmp_path):
        image, camera = ExperimentStore(tmp_path).load_reference_view()
        assert image is None and camera == {}

    def test_round_trips_the_camera(self, tmp_path, image, camera):
        k, e = camera
        store = ExperimentStore(tmp_path)
        store.save_reference_view(
            image, intrinsic=k, extrinsic=e, camera_name="agentview"
        )
        loaded, meta = store.load_reference_view()
        assert loaded is not None
        assert np.allclose(np.asarray(meta["intrinsic"]), k)
        assert meta["camera_name"] == "agentview"

    def test_first_frame_wins(self, tmp_path, image, camera):
        """A later frame would show a disturbed scene, or a moved camera."""
        k, e = camera
        store = ExperimentStore(tmp_path)
        store.save_reference_view(image, intrinsic=k, extrinsic=e)
        store.save_reference_view(np.full_like(image, 200), intrinsic=k, extrinsic=e)
        loaded, _ = store.load_reference_view()
        assert int(np.asarray(loaded)[..., :3].mean()) == 40


class TestReportRenders:
    """build_report must survive every stage of an experiment.

    An operator presses R at any time, including before the first
    rollout and on a half-collected grid. A plot that raises
    there loses the whole report, so the empty and partial cases
    are covered explicitly — an unbalanced zip in the dot
    layout used to break exactly the "no rollouts yet" case.
    """

    @pytest.fixture
    def store(self, tmp_path, grid):
        from policy_abstraction.experiments.store import ExperimentStore

        cfg = ExperimentConfig(
            name="r",
            factors=[
                GridPositionFactor("cable_position", grid),
                AppearanceFactor("box", ["black", "white"]),
            ],
            repeats=2,
        )
        return ExperimentStore.open(tmp_path / "r", cfg), cfg

    def _plots(self, store):
        return sorted(p.name for p in store.report_dir.glob("*.png"))

    def test_an_empty_experiment_reports(self, store):
        from policy_abstraction.experiments.report import build_report

        st, cfg = store
        summary = build_report(st, cfg)
        assert summary["totals"]["collected"] == 0
        assert (st.report_dir / "report.md").exists()
        # The grid maps still render, all cells "no data".
        assert "cable_position_grid_success.png" in self._plots(st)

    def test_a_partly_collected_experiment_reports(self, store):
        from policy_abstraction.experiments.report import build_report

        st, cfg = store
        plan = st.load_plan()
        for planned in plan[:3]:
            st.append(
                RolloutRecord(
                    index=planned.index,
                    assignments=dict(planned.assignments),
                    outcome=Outcome.SUCCESS,
                    steps=10,
                    duration_s=5.0,
                )
            )
        summary = build_report(st, cfg)
        assert summary["totals"]["collected"] == 3
        names = self._plots(st)
        assert "cable_position_grid_rollouts.png" in names
        assert "box_rates.png" in names

    def test_a_cell_with_several_repeats_renders(self, store):
        """The dot ring: n=0, n=1 and n>1 all have to work."""
        from policy_abstraction.experiments.report import build_report

        st, cfg = store
        plan = st.load_plan()
        same_cell = [p for p in plan if p.assignments["cable_position"] == "A1"]
        assert len(same_cell) > 1
        for i, planned in enumerate(same_cell):
            st.append(
                RolloutRecord(
                    index=planned.index,
                    assignments=dict(planned.assignments),
                    outcome=Outcome.SUCCESS if i % 2 else Outcome.FAILURE,
                    failure_mode=(FailureMode.NONE if i % 2 else FailureMode.DROPPED),
                    steps=10,
                    duration_s=5.0,
                )
            )
        build_report(st, cfg)
        assert "failure_modes.png" in self._plots(st)

    def test_the_projected_map_is_skipped_without_a_reference(self, store):
        """No saved frame: the abstract maps still render."""
        from policy_abstraction.experiments.report import build_report

        st, cfg = store
        build_report(st, cfg)
        assert "cable_position_on_camera.png" not in self._plots(st)

    def test_the_projected_map_renders_with_a_reference(self, store, image, camera):
        from policy_abstraction.experiments.report import build_report

        st, cfg = store
        k, e = camera
        st.save_reference_view(image, intrinsic=k, extrinsic=e)
        for planned in st.load_plan()[:2]:
            st.append(
                RolloutRecord(
                    index=planned.index,
                    assignments=dict(planned.assignments),
                    outcome=Outcome.SUCCESS,
                )
            )
        build_report(st, cfg)
        assert "cable_position_on_camera.png" in self._plots(st)

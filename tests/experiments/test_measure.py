"""Measuring staged object positions back from recordings.

The perception client is stood in for by a mask drawn by hand, so these pin
down everything around it: the back-projection, the choice among several
instances, the sidecar, and that the report draws the measured point.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.factors import GridPositionFactor
from policy_abstraction.experiments.grid import TableGrid
from policy_abstraction.experiments.measure import (
    FirstFrame,
    Measurement,
    measure_experiment,
    positions_from_masks,
)
from policy_abstraction.experiments.outcomes import Outcome, RolloutRecord
from policy_abstraction.experiments.report import build_report, summarize
from policy_abstraction.experiments.store import ExperimentStore


@pytest.fixture
def grid():
    return TableGrid(
        origin=(-0.3, -0.25, 0.0),
        u_axis=(0, 1, 0),
        v_axis=(1, 0, 0),
        cell_size_u=0.125,
        cell_size_v=0.15,
        rows=2,
        cols=2,
        marker_height=0.015,
        name="cable",
    )


@pytest.fixture
def camera():
    """Looking straight down from 1 m: pixel (u, v) -> world (x, y) is linear."""
    K = np.array([[300.0, 0, 128.0], [0, 300.0, 128.0], [0, 0, 1.0]])
    E = np.array(
        [
            [1.0, 0, 0, 0.0],
            [0, -1.0, 0, 0.0],
            [0, 0, -1.0, 1.0],
            [0, 0, 0, 1.0],
        ]
    )
    return K, E


def _frame(camera, depth_value=1.0):
    K, E = camera
    rgb = np.full((256, 256, 3), 90, dtype=np.uint8)
    depth = np.full((256, 256), depth_value, dtype=np.float32)
    return FirstFrame(
        rgb=rgb,
        depth_m=depth,
        intrinsic=K,
        extrinsic=E,
        timestep=0,
        pixels_mirrored=False,
    )


def _mask(cx, cy, half=6):
    m = np.zeros((256, 256), dtype=bool)
    m[cy - half : cy + half + 1, cx - half : cx + half + 1] = True
    return m


def _world_of(camera, cx, cy, depth=1.0):
    """Where a pixel at *depth* on this downward camera lands in the world."""
    K, E = camera
    x_cam = (cx - K[0, 2]) / K[0, 0] * depth
    y_cam = (cy - K[1, 2]) / K[1, 1] * depth
    return (E @ np.array([x_cam, y_cam, depth, 1.0]))[:3]


class TestPositionsFromMasks:
    def test_the_centroid_is_the_masked_points_back_projected(self, camera):
        frame = _frame(camera)
        masks = {1: ("cable", _mask(100, 140))}
        found = positions_from_masks(frame, masks, {"cable_position": "cable"})
        expected = _world_of(camera, 100, 140)
        assert np.allclose(found["cable_position"].position, expected, atol=1e-3)
        assert found["cable_position"].points == 13 * 13
        assert found["cable_position"].instances == 1

    def test_the_instance_nearest_the_nominal_position_wins(self, camera):
        frame = _frame(camera)
        near, far = _mask(100, 140), _mask(200, 60, half=10)  # far is bigger
        masks = {1: ("cable", far), 2: ("cable", near)}
        nominal = {"cable_position": _world_of(camera, 100, 140)}
        found = positions_from_masks(
            frame, masks, {"cable_position": "cable"}, nominal=nominal
        )
        assert np.allclose(
            found["cable_position"].position, nominal["cable_position"], atol=1e-3
        )
        assert found["cable_position"].instances == 2
        assert found["cable_position"].offset_m == pytest.approx(0.0, abs=1e-3)

    def test_without_a_nominal_position_the_biggest_instance_wins(self, camera):
        frame = _frame(camera)
        masks = {1: ("cable", _mask(100, 140)), 2: ("cable", _mask(200, 60, half=10))}
        found = positions_from_masks(frame, masks, {"cable_position": "cable"})
        assert np.allclose(
            found["cable_position"].position, _world_of(camera, 200, 60), atol=1e-3
        )

    def test_a_mask_on_a_depth_hole_is_not_a_measurement(self, camera):
        frame = _frame(camera)
        frame.depth_m[130:151, 90:111] = np.nan
        masks = {1: ("cable", _mask(100, 140))}
        assert positions_from_masks(frame, masks, {"cable_position": "cable"}) == {}

    def test_query_names_match_loosely(self, camera):
        frame = _frame(camera)
        masks = {1: ("Orange_Cable", _mask(100, 140))}
        found = positions_from_masks(frame, masks, {"k": "orange cable"})
        assert "k" in found

    def test_an_unrelated_detection_does_not_answer_the_query(self, camera):
        frame = _frame(camera)
        masks = {1: ("mug", _mask(100, 140))}
        assert positions_from_masks(frame, masks, {"cable_position": "cable"}) == {}


class TestFactorQueries:
    def test_the_object_defaults_to_the_noun_the_factor_is_named_after(self, grid):
        assert GridPositionFactor("cable_position", grid).object == "cable"
        assert GridPositionFactor("red_mug_position", grid).object == "red mug"
        assert GridPositionFactor("box", grid).object == "box"

    def test_an_explicit_object_round_trips_through_the_yaml(self, grid):
        factor = GridPositionFactor("cable_position", grid, object="orange usb cable")
        data = factor.to_dict()
        assert data["object"] == "orange usb cable"
        assert GridPositionFactor.from_dict(data).object == "orange usb cable"

    def test_the_config_keys_queries_like_the_recorded_positions(self, grid):
        config = ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)]
        )
        assignments = {"cable_position": "A1"}
        assert set(config.measurement_queries(assignments)) == set(
            config.target_positions(assignments)
        )


class TestMeasureExperiment:
    @pytest.fixture
    def config(self, grid):
        return ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)], repeats=1
        )

    @pytest.fixture
    def store(self, tmp_path, config, camera, monkeypatch):
        """An experiment with two scored rollouts and a fake recording each."""
        import policy_abstraction.experiments.measure as measure

        store = ExperimentStore.open(tmp_path / "exp", config)
        store.episodes_dir.mkdir(exist_ok=True)
        for index, cell, ok in ((0, "A1", True), (1, "B2", False)):
            (store.episodes_dir / f"rollout_{index:04d}_{cell}.vla").write_bytes(b"")
            store.append(
                RolloutRecord(
                    index=index,
                    assignments={"cable_position": cell},
                    outcome=Outcome.SUCCESS if ok else Outcome.FAILURE,
                    episode_path=f"episodes/rollout_{index:04d}_{cell}.vla",
                    target_positions=config.target_positions({"cable_position": cell}),
                    steps=10,
                    duration_s=5.0,
                )
            )
        # The recording reader is replaced: the .vla stubs hold nothing.
        monkeypatch.setattr(measure, "first_frame", lambda path, **kw: _frame(camera))
        K, E = camera
        store.save_reference_view(
            np.full((256, 256, 3), 90, dtype=np.uint8),
            intrinsic=K,
            extrinsic=E,
            camera_name="exterior",
            preprocessed=False,
        )
        return store

    def test_it_writes_the_sidecar_and_the_report_draws_from_it(
        self, store, config, camera
    ):
        calls = []

        def perceive(rgb, depth, K, E, names):
            calls.append(list(names))
            return {1: ("cable", _mask(100, 140))}

        data = measure_experiment(store, perceive, config=config)
        assert calls == [["cable"], ["cable"]]
        assert set(data["rollouts"]) == {"0", "1"}
        entry = data["rollouts"]["0"]["cable_position"]
        assert np.allclose(entry["position"], _world_of(camera, 100, 140), atol=1e-3)
        assert entry["offset_m"] > 0
        on_disk = json.loads(store.observed_positions_path.read_text())
        assert on_disk["rollouts"] == data["rollouts"]

        records = store.attach_observed_positions(store.results())
        assert all(r.observed_positions["cable_position"] for r in records)

        summary = build_report(store, config)
        stats = summary["measured_positions"]
        assert stats["measured"] == 2 and stats["total"] == 2
        assert stats["max_offset_m"] >= stats["median_offset_m"] > 0
        assert (store.report_dir / "cable_position_measured.png").exists()
        assert (store.report_dir / "cable_position_on_camera.png").exists()
        assert (
            "Measured object positions" in (store.report_dir / "report.md").read_text()
        )

    def test_measured_rollouts_are_skipped_unless_forced(self, store, config):
        seen = []

        def perceive(rgb, depth, K, E, names):
            seen.append(1)
            return {1: ("cable", _mask(100, 140))}

        measure_experiment(store, perceive, config=config)
        measure_experiment(store, perceive, config=config)
        assert len(seen) == 2
        measure_experiment(store, perceive, config=config, force=True, indices=[1])
        assert len(seen) == 3

    def test_a_miss_is_recorded_as_missing_not_as_a_position(self, store, config):
        data = measure_experiment(store, lambda *a: {}, config=config)
        assert data["rollouts"]["0"] == {"missing": ["cable_position"]}
        records = store.attach_observed_positions(store.results())
        assert records[0].observed_positions == {}
        # The report still renders, on the nominal positions.
        summary = build_report(store, config)
        assert summary["measured_positions"]["measured"] == 0

    def test_a_recording_that_cannot_be_read_does_not_stop_the_run(
        self, store, config, monkeypatch
    ):
        import policy_abstraction.experiments.measure as measure

        def broken(path, **kw):
            raise ValueError("no depth")

        monkeypatch.setattr(measure, "first_frame", broken)
        data = measure_experiment(store, lambda *a: {}, config=config)
        assert "error" in data["rollouts"]["0"]
        assert "error" in data["rollouts"]["1"]

    def test_a_miss_is_asked_again_once(self, store, config):
        answers = iter([{}, {1: ("cable", _mask(100, 140))}, {}, {}])
        calls = []

        def perceive(rgb, depth, K, E, names):
            calls.append(1)
            return next(answers)

        data = measure_experiment(store, perceive, config=config, indices=[0, 1])
        assert len(calls) == 4, "one retry each: rollout 0 hit on it, 1 did not"
        assert "cable_position" in data["rollouts"]["0"]
        assert data["rollouts"]["1"] == {"missing": ["cable_position"]}

    def test_query_overrides_reach_the_detector(self, store, config):
        asked = []

        def perceive(rgb, depth, K, E, names):
            asked.extend(names)
            return {1: ("orange cable", _mask(100, 140))}

        data = measure_experiment(
            store, perceive, config=config, queries={"cable_position": "orange cable"}
        )
        assert set(asked) == {"orange cable"}
        assert "cable_position" in data["rollouts"]["0"]


class TestReferenceView:
    def test_a_rollouts_first_frame_replaces_the_staging_view(
        self, tmp_path, grid, camera, monkeypatch
    ):
        import policy_abstraction.experiments.measure as measure
        from policy_abstraction.experiments.measure import set_reference_view

        config = ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)]
        )
        store = ExperimentStore.open(tmp_path / "exp", config)
        K, E = camera
        store.save_reference_view(
            np.zeros((8, 8, 3), dtype=np.uint8), intrinsic=K, extrinsic=E
        )
        store.episodes_dir.mkdir(exist_ok=True)
        (store.episodes_dir / "r.vla").write_bytes(b"")
        record = RolloutRecord(
            index=0,
            assignments={"cable_position": "B3"},
            outcome=Outcome.SUCCESS,
            episode_path="episodes/r.vla",
        )
        frame = _frame(camera)
        frame.camera = "exterior"
        asked = {}

        def fake_first_frame(path, **kw):
            asked.update(kw)
            return frame

        monkeypatch.setattr(measure, "first_frame", fake_first_frame)

        set_reference_view(store, record, timestep=40)
        assert asked == {"start": 40, "require_depth": False}
        image, cam = store.load_reference_view()
        assert image.shape == (256, 256, 3) and int(image[0, 0, 0]) == 90
        assert cam["camera_name"] == "exterior" and cam["preprocessed"] is False
        assert (store.root / "reference_frame.original.png").exists()
        assert (store.root / "reference_camera.original.json").exists()

    def test_without_replace_the_first_frame_stays(self, tmp_path, grid, camera):
        config = ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)]
        )
        store = ExperimentStore.open(tmp_path / "exp", config)
        store.save_reference_view(np.zeros((8, 8, 3), dtype=np.uint8))
        store.save_reference_view(np.full((8, 8, 3), 200, dtype=np.uint8))
        image, _ = store.load_reference_view()
        assert int(image[0, 0, 0]) == 0


class TestSummaryWithoutMeasurements:
    def test_the_offset_is_measured_in_the_grid_plane(self, grid):
        """The marker height is not a staging error."""
        from policy_abstraction.experiments.report import measured_position_stats

        config = ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)]
        )
        nominal = config.target_positions({"cable_position": "A1"})["cable_position"]
        on_table = [nominal[0] + 0.02, nominal[1], nominal[2] - grid.marker_height]
        record = RolloutRecord(
            index=0,
            assignments={"cable_position": "A1"},
            outcome=Outcome.SUCCESS,
            target_positions={"cable_position": nominal},
            observed_positions={"cable_position": on_table},
        )
        stats = measured_position_stats([record], config)
        assert stats["max_offset_m"] == pytest.approx(0.02, abs=1e-6)
        assert measured_position_stats([record])["max_offset_m"] > 0.02

    def test_records_without_measurements_report_zero_measured(self, grid):
        config = ExperimentConfig(
            name="t", factors=[GridPositionFactor("cable_position", grid)]
        )
        records = [
            RolloutRecord(
                index=0, assignments={"cable_position": "A1"}, outcome=Outcome.SUCCESS
            )
        ]
        assert summarize(records, config)["measured_positions"] == {
            "measured": 0,
            "total": 1,
        }

    def test_measurement_serialises(self):
        m = Measurement(position=[0.1, 0.2, 0.3], query="cable", points=5, instances=1)
        assert Measurement(**m.to_dict()) == m

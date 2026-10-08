"""Segmentation must run before the id -> name map is read.

A simulator's name map is a file attribute: it answers whenever asked. A real
recording has no names at all until its segmenter has produced them, so reading
the map first yields ``{}`` -- and an empty map is not inert. Every visible
segment id is then "not in the map", which
:func:`~policy_abstraction.utils.segmentation.get_robot_segment_ids` reads as
"a robot part the env deliberately left unnamed". The whole scene is classified
as robot, the object list handed to the VLM is empty, and the pairs that come
back can only relate the gripper to itself.

That failure is silent: analysis succeeds, guidance runs, and the rollout is
steered by nothing task-relevant.
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.utils.segmentation import get_robot_segment_ids

H = W = 8


class _RecordingEpisode:
    """A real recording: names exist only after ``segmentation()`` is called."""

    source_kind = "vla"
    task = "put all cables in the bag"
    frames_preprocessed = False
    demo_key = "episode_000"

    def __init__(self) -> None:
        self._names: dict[int, str] = {}
        self.calls: list[str] = []

    def __len__(self) -> int:
        return 4

    def rgb(self, t: int) -> np.ndarray:
        return np.zeros((H, W, 3), dtype=np.uint8)

    def segmentation(self, t: int) -> np.ndarray:
        self.calls.append("segmentation")
        labels = np.zeros((H, W), dtype=np.int32)
        labels[1:4, 1:4] = 1
        labels[4:7, 4:7] = 2
        self._names = {1: "cable", 2: "bag"}
        return labels

    def seg_name_map(self) -> dict[int, str]:
        self.calls.append("seg_name_map")
        return dict(self._names)

    def depth_m(self, t):
        return np.full((H, W), 1.0, dtype=np.float32)

    def intrinsic(self, t):
        return np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])

    def extrinsic(self, t):
        return np.eye(4)

    def ee_positions(self):
        return np.zeros((len(self), 3), dtype=np.float32)

    def ee_orientations(self):
        return np.zeros((len(self), 3), dtype=np.float32)

    def gripper_states(self):
        return np.zeros((len(self), 1), dtype=np.float32)

    def gripper_geometry(self):
        return "robotiq_2f140"

    def actions(self):
        return np.zeros((len(self) - 1, 7), dtype=np.float32)

    def frames_with_depth(self):
        return list(range(len(self)))


def test_empty_name_map_would_classify_every_object_as_robot():
    """The mechanism the ordering protects against, pinned directly."""
    visible = {1, 2}
    assert get_robot_segment_ids({}, all_seg_ids=visible) == visible
    assert get_robot_segment_ids({1: "cable", 2: "bag"}, all_seg_ids=visible) == set()


def test_analyze_segments_before_reading_the_name_map(monkeypatch):
    from policy_abstraction.analysis.demo_analyzer import DemoAnalyzer

    episode = _RecordingEpisode()

    captured: dict = {}

    def _fake_select(objects, *args, **kwargs):
        captured["objects"] = dict(objects)
        raise RuntimeError("stop after the object list is built")

    # The VLM call is irrelevant here; we only need the object list that would
    # be sent to it.
    monkeypatch.setattr(
        "policy_abstraction.analysis.demo_analyzer.select_distance_pairs",
        _fake_select,
        raising=False,
    )

    analyzer = DemoAnalyzer(cache_dir=None)
    try:
        analyzer.analyze(episode, episode.task)
    except Exception:
        pass

    assert episode.calls, "analyze() read neither segmentation nor the name map"
    assert episode.calls[0] == "segmentation", (
        "the name map was read before segmentation had produced any names; "
        f"call order was {episode.calls}"
    )
    if "objects" in captured:
        assert captured["objects"] == {1: "cable", 2: "bag"}


# ---------------------------------------------------------------------------
# Cached analysis must match the segmentation it is replayed against
# ---------------------------------------------------------------------------


class _Pair:
    def __init__(self, from_entity, to_entity, label="p"):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.label = label


def test_pairs_naming_absent_segment_ids_are_detected():
    """The check that decides whether a cached analysis can still be trusted.

    A real recording is re-segmented every run and numbers its instances
    afresh, so ids from an earlier analysis need not exist in this one.
    """
    from policy_abstraction.analysis.demo_analyzer import (
        _pair_seg_ids,
        _pairs_match_segmentation,
    )

    stale = [_Pair("ee", 0, "ee_to_cable"), _Pair(0, 1, "cable_to_bag")]
    assert _pair_seg_ids(stale) == {0, 1}
    # This run segmented the same scene as ids 1 and 2.
    assert not _pairs_match_segmentation(stale, {1, 2})
    assert _pairs_match_segmentation(stale, {0, 1, 2})


def test_robot_entities_never_invalidate_a_cache():
    """`ee` and the fingers come from proprioception, not from a mask, so they
    stay valid whatever the segmentation produced."""
    from policy_abstraction.analysis.demo_analyzer import (
        _pair_seg_ids,
        _pairs_match_segmentation,
    )

    robot_only = [
        _Pair("ee", "ee", "ee_down_alignment"),
        _Pair("gripper_l", "gripper_r", "aperture"),
    ]
    assert _pair_seg_ids(robot_only) == set()
    assert _pairs_match_segmentation(robot_only, set())

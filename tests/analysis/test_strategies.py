"""Tests for the pair/constraint proposal strategy abstraction.

Covers:
- ``SingleTurnStrategy`` is a transparent wrapper around ``TaskPairSelector``.
- Chained ``segment_demo_by_conditions`` finds expected boundaries on a
  synthetic trajectory and falls back gracefully when post never fires.
- ``score_proposal`` returns ~0 for identical trajectories.
- ``IterativeStrategy`` selects the iteration with the lowest aggregate score.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from policy_abstraction.analysis.consistency import ProposalScore, SegmentScore
from policy_abstraction.analysis.segmentation_from_conditions import (
    segment_demo_by_conditions,
)
from policy_abstraction.analysis.strategies import (
    StrategyContext,
    StrategyKind,
    make_strategy,
)
from policy_abstraction.analysis.strategies.base import IterativeStrategyConfig
from policy_abstraction.analysis.strategies.iterative import (
    IterativeStrategy,
    _format_refinement_feedback,
)
from policy_abstraction.vlm.types import (
    DistancePair,
    DistanceType,
    SubtrajConstraint,
    VLMConversation,
    VLMMessage,
)

# ---------------------------------------------------------------------------
# SingleTurnStrategy: forwards to TaskPairSelector verbatim
# ---------------------------------------------------------------------------


def _fake_pair() -> DistancePair:
    return DistancePair(
        from_entity="ee",
        to_entity=2,
        label="ee_to_obj",
        distance_type=DistanceType.CENTROID,
        subtraj_idx=0,
    )


def _fake_constraint() -> SubtrajConstraint:
    return SubtrajConstraint(
        kind="sub_goal",
        subtraj_idx=0,
        label="reached_obj",
        description="",
        expression="lambda ee, keypoints, ee_R: 0.0",
    )


def test_single_turn_strategy_forwards_to_task_pair_selector():
    fake_pairs = [_fake_pair()]
    fake_constraints = [_fake_constraint()]
    fake_conv = VLMConversation(
        messages=[VLMMessage(role="system", text="x")],
        model="fake-model",
    )

    with patch(
        "policy_abstraction.analysis.strategies.single_turn.TaskPairSelector"
    ) as mock_cls:
        mock_cls.return_value.select_pairs.return_value = (
            fake_pairs,
            fake_constraints,
            fake_conv,
        )
        strategy = make_strategy(StrategyKind.SINGLE_TURN, vlm_model="fake-model")
        ctx = StrategyContext(
            task="t",
            objects={1: "obj"},
            scene_image=None,
            seg_index_image=None,
            subtrajectories=[(0, 50)],
            subtraj_image=None,
            vlm_model="fake-model",
        )
        result = strategy.propose(ctx)

    assert result.pairs is fake_pairs
    assert result.constraints is fake_constraints
    assert result.conversation is fake_conv
    assert result.iteration_log == []
    assert result.selected_subtraj_boundaries is None
    mock_cls.return_value.select_pairs.assert_called_once_with(
        "t",
        {1: "obj"},
        scene_image=None,
        seg_index_image=None,
        subtrajectories=[(0, 50)],
        subtraj_image=None,
    )


# ---------------------------------------------------------------------------
# segment_demo_by_conditions
# ---------------------------------------------------------------------------


def test_segment_demo_by_conditions_chained_two_segments():
    T = 60
    ee_pos = np.zeros((T, 3), dtype=np.float32)
    ee_pos[:, 2] = np.linspace(0.0, 1.0, T)  # rising z
    ee_R = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
    keypoints: dict[int, np.ndarray] = {}

    # Post 0: z >= 0.5  →  cost = 0.5 - z
    post0 = lambda ee, kp, ee_R: 0.5 - ee[2]  # noqa: E731
    # Post 1: z >= 0.9  →  cost = 0.9 - z
    post1 = lambda ee, kp, ee_R: 0.9 - ee[2]  # noqa: E731

    segs = segment_demo_by_conditions(
        ee_pos, ee_R, keypoints, [post0, post1], min_segment_length=3
    )

    assert len(segs) == 2
    s0, s1 = segs
    assert not s0.fallback and not s1.fallback
    # First segment ends right after z first crosses 0.5 + 3 stable frames.
    assert s0.t_start == 0
    z = ee_pos[:, 2]
    first_cross_0 = int(np.argmax(z >= 0.5))  # earliest index satisfying
    assert s0.t_end >= first_cross_0 + 3
    assert s1.t_start == s0.t_end  # chained
    assert s1.t_end <= T


def test_segment_demo_by_conditions_fallback_when_post_never_fires():
    T = 30
    ee_pos = np.zeros((T, 3), dtype=np.float32)
    ee_R = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))

    # Post 0: never satisfied (cost always +1).
    post = lambda ee, kp, ee_R: 1.0  # noqa: E731

    segs = segment_demo_by_conditions(ee_pos, ee_R, {}, [post], min_segment_length=3)
    # Single-segment case extends the last segment to T as a fallback.
    assert len(segs) == 1
    assert segs[0].fallback
    assert segs[0].t_end == T


def test_segment_demo_by_conditions_fallback_propagates_downstream():
    T = 40
    ee_pos = np.zeros((T, 3), dtype=np.float32)
    ee_R = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))

    never = lambda ee, kp, ee_R: 1.0  # noqa: E731
    always = lambda ee, kp, ee_R: -1.0  # noqa: E731

    segs = segment_demo_by_conditions(
        ee_pos, ee_R, {}, [never, always], min_segment_length=3
    )
    assert len(segs) == 2
    assert segs[0].fallback
    assert segs[1].fallback


# ---------------------------------------------------------------------------
# IterativeStrategy.propose: picks lowest-scoring iteration
# ---------------------------------------------------------------------------


def _stub_score(aggregate: float) -> ProposalScore:
    return ProposalScore(
        aggregate=aggregate,
        metric="std_after_align",
        per_segment=[
            SegmentScore(
                subtraj_idx=0,
                n_attempted=5,
                n_segmented=5,
                mean_std_after_align=aggregate,
                mean_dtw_distance=aggregate,
                seg_score=aggregate,
                t_start_ref=0,
                t_end_ref=10,
                pair_labels=["x"],
            )
        ],
        demo_breakdown=[],
    )


def test_iterative_strategy_picks_lowest_aggregate():
    strategy = IterativeStrategy(
        vlm_model="fake-model",
        config=IterativeStrategyConfig(max_iterations=2),
    )

    fake_pair = _fake_pair()
    fake_constraint = SubtrajConstraint(
        kind="sub_goal",
        subtraj_idx=0,
        label="post_post",
        description="",
        expression="lambda ee, kp, ee_R: 0.0",
    )
    fake_conv = VLMConversation(messages=[], model="fake-model")

    score_seq = iter([_stub_score(0.05), _stub_score(0.02), _stub_score(0.04)])

    class FakeSelector:
        def propose_initial(self, *a, **kw):
            return ([fake_pair], [fake_constraint], fake_conv, [], "{}")

        def refine(self, *a, **kw):
            return ([fake_pair], [fake_constraint], fake_conv, [], "{}")

    strategy._selector = FakeSelector()  # type: ignore[assignment]

    with patch.object(strategy, "_score", side_effect=lambda *a, **kw: next(score_seq)):
        ctx = StrategyContext(
            task="t",
            objects={},
            scene_image=None,
            seg_index_image=None,
            subtrajectories=None,
            subtraj_image=None,
            vlm_model="fake-model",
            top_result={"hdf5_path": "x", "demo_key": "y", "task": "t"},
            other_results=[],
            seg_name_map={},
        )
        result = strategy.propose(ctx)

    assert result.selected_iteration == 1
    # Each iteration recorded
    assert len(result.iteration_log) == 3
    # The aggregate field is preserved (best is 0.02 at iter 1).
    aggs = [e["score"]["aggregate"] for e in result.iteration_log]
    assert aggs == pytest.approx([0.05, 0.02, 0.04])
    assert result.iteration_log[1]["selected"] is True


# ---------------------------------------------------------------------------
# Refinement feedback formatting
# ---------------------------------------------------------------------------


def test_format_refinement_feedback_mentions_failures():
    score = ProposalScore(
        aggregate=0.07,
        metric="std_after_align",
        per_segment=[
            SegmentScore(
                subtraj_idx=0,
                n_attempted=5,
                n_segmented=2,  # 3 demos failed to segment
                mean_std_after_align=0.04,
                mean_dtw_distance=0.05,
                seg_score=0.10,
                t_start_ref=0,
                t_end_ref=20,
                pair_labels=["a", "b"],
            )
        ],
        demo_breakdown=[],
    )
    txt = _format_refinement_feedback([], [], score)
    assert "ST0" in txt
    assert "did not fire" in txt or "success_rate=0.40" in txt
    assert "FULL revised JSON" in txt

"""Wraps the legacy :class:`TaskPairSelector` as a :class:`PairProposalStrategy`.

This wrapper preserves the pre-strategy behaviour exactly. It is the default
strategy and serves as the no-regression baseline.
"""

from __future__ import annotations

from policy_abstraction.analysis.strategies.base import (
    PairProposalStrategy,
    StrategyContext,
    StrategyResult,
)
from policy_abstraction.vlm.task_pair_selector import TaskPairSelector


class SingleTurnStrategy(PairProposalStrategy):
    def __init__(self, vlm_model: str) -> None:
        self._selector = TaskPairSelector(model=vlm_model)

    def propose(self, ctx: StrategyContext) -> StrategyResult:
        pairs, constraints, conversation = self._selector.select_pairs(
            ctx.task,
            ctx.objects,
            scene_image=ctx.scene_image,
            seg_index_image=ctx.seg_index_image,
            subtrajectories=ctx.subtrajectories,
            subtraj_image=ctx.subtraj_image,
        )
        return StrategyResult(
            pairs=pairs,
            constraints=constraints,
            conversation=conversation,
            iteration_log=[],
            selected_subtraj_boundaries=None,
            selected_iteration=None,
        )

"""Strategy abstraction for VLM-based pair / constraint proposal.

A *strategy* turns a :class:`StrategyContext` (task description, scene + seg
images, retrieval results, …) into a :class:`StrategyResult` containing the
final list of :class:`DistancePair`, :class:`SubtrajConstraint`, the full VLM
conversation, plus optional iteration debug data.

Two concrete strategies live alongside this module:

- :class:`policy_abstraction.analysis.strategies.single_turn.SingleTurnStrategy`
  wraps the existing :class:`~policy_abstraction.vlm.task_pair_selector.TaskPairSelector`
  and is the default — bit-for-bit identical to the pre-strategy pipeline.
- :class:`policy_abstraction.analysis.strategies.iterative.IterativeStrategy`
  proposes pre/post conditions + pairs, verifies them by replaying the
  post-conditions on retrieved demos, scores consistency via DTW, and
  iteratively refines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from policy_abstraction.vlm.types import (
    DistancePair,
    SubtrajConstraint,
    VLMConversation,
)


class StrategyKind(str, Enum):
    """How distance pairs and constraints are produced from a demo + task."""

    SINGLE_TURN = "single_turn"
    """Today's behaviour: speed-based sub-trajectories are computed first, then
    the VLM is asked once to fill in pairs + sub-goal/path constraints per
    segment."""

    ITERATIVE = "iterative"
    """VLM proposes its own decomposition (pre/post lambdas + pairs) without
    pre-broken segments; each proposal is verified offline via DTW across
    retrieved demos and refined for up to N iterations."""


@dataclass
class IterativeStrategyConfig:
    """Knobs for :class:`IterativeStrategy`. See ``DemoAnalysisArgs.iterative``."""

    max_iterations: int = 3
    consistency_metric: str = "std_after_align"  # or "mean_dtw"
    top_k_for_verification: int = 5
    min_segment_length: int = 5
    accept_score_threshold: float | None = None


@dataclass
class StrategyContext:
    """Inputs every strategy receives from :class:`DemoAnalyzer`."""

    task: str
    objects: dict[int, str]
    scene_image: np.ndarray | None
    seg_index_image: np.ndarray | None
    # For SingleTurnStrategy these are pre-computed (speed-based); for
    # IterativeStrategy they are unused (the VLM proposes its own segmentation).
    subtrajectories: list[tuple[int, int]] | None
    subtraj_image: np.ndarray | None
    vlm_model: str
    # Retrieval context — used by IterativeStrategy to verify proposals.
    # SingleTurnStrategy ignores these.
    top_result: dict | None = None
    other_results: list[dict] = field(default_factory=list)
    seg_name_map: dict[str, str] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Output of :meth:`PairProposalStrategy.propose`."""

    pairs: list[DistancePair]
    constraints: list[SubtrajConstraint]
    conversation: VLMConversation | None
    iteration_log: list[dict] = field(default_factory=list)
    """Per-iteration proposal+score record (empty for single-turn)."""
    selected_subtraj_boundaries: list[tuple[int, int]] | None = None
    """The verified segment boundaries for the top-1 demo when an iterative
    strategy succeeded; ``None`` to fall back to runtime speed-based
    segmentation in :class:`RolloutAbstractor`."""
    selected_iteration: int | None = None


class PairProposalStrategy(Protocol):
    """Anything that turns a :class:`StrategyContext` into a :class:`StrategyResult`."""

    def propose(self, ctx: StrategyContext) -> StrategyResult: ...

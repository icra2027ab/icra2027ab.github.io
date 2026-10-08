"""Strategy abstraction for VLM-based pair / constraint proposal.

See :mod:`policy_abstraction.analysis.strategies.base` for the protocol and
data classes.
"""

from __future__ import annotations

from policy_abstraction.analysis.strategies.base import (
    IterativeStrategyConfig,
    PairProposalStrategy,
    StrategyContext,
    StrategyKind,
    StrategyResult,
)


def make_strategy(
    kind: StrategyKind,
    *,
    vlm_model: str,
    iterative_config: IterativeStrategyConfig | None = None,
) -> PairProposalStrategy:
    """Factory: construct the chosen strategy with its dependencies."""
    if kind == StrategyKind.SINGLE_TURN:
        from policy_abstraction.analysis.strategies.single_turn import (
            SingleTurnStrategy,
        )

        return SingleTurnStrategy(vlm_model=vlm_model)
    if kind == StrategyKind.ITERATIVE:
        from policy_abstraction.analysis.strategies.iterative import IterativeStrategy

        cfg = iterative_config or IterativeStrategyConfig()
        return IterativeStrategy(vlm_model=vlm_model, config=cfg)
    raise ValueError(f"Unknown strategy kind: {kind!r}")


__all__ = [
    "IterativeStrategyConfig",
    "PairProposalStrategy",
    "StrategyContext",
    "StrategyKind",
    "StrategyResult",
    "make_strategy",
]

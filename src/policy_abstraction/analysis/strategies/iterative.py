"""Iterative pre/post-condition strategy.

Loop:
    1. Initial VLM proposal: K sub-trajectories, each with pre/post + pairs.
    2. Verify offline by replaying post-conditions on top-1 + retrieved peers
       and DTW-aligning each segment's distance trajectories. Score = post-DTW
       std (or mean DTW path distance) divided by per-segment success rate.
    3. Build a structured per-segment feedback string and ask the VLM to
       refine. Repeat for up to ``max_iterations``.
    4. Return the iteration with the best (lowest) aggregate score.

The selected proposal's segment boundaries on the top-1 demo are returned
alongside the pairs/constraints so :class:`RolloutAbstractor` can use the
verified segmentation at runtime instead of recomputing speed-based segments.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from loguru import logger

from policy_abstraction.analysis.consistency import (
    ProposalScore,
    ref_segment_boundaries,
    score_proposal,
)
from policy_abstraction.analysis.strategies.base import (
    IterativeStrategyConfig,
    PairProposalStrategy,
    StrategyContext,
    StrategyResult,
)
from policy_abstraction.vlm.iterative_pair_selector import IterativePairSelector
from policy_abstraction.vlm.types import (
    DistancePair,
    SubtrajConstraint,
    VLMConversation,
)


@dataclasses.dataclass
class _Iteration:
    pairs: list[DistancePair]
    constraints: list[SubtrajConstraint]
    conversation: VLMConversation
    openai_messages: list[dict]
    assistant_text: str
    score: ProposalScore


class IterativeStrategy(PairProposalStrategy):
    def __init__(self, vlm_model: str, config: IterativeStrategyConfig) -> None:
        self._vlm_model = vlm_model
        self._config = config
        self._selector: IterativePairSelector | None = None  # lazy

    def _get_selector(self) -> IterativePairSelector:
        if self._selector is None:
            self._selector = IterativePairSelector(model=self._vlm_model)
        return self._selector

    def propose(self, ctx: StrategyContext) -> StrategyResult:
        if ctx.top_result is None:
            logger.warning(
                "IterativeStrategy: no top retrieval result supplied; "
                "falling back to a single uncached VLM turn (no verification)."
            )
        selector = self._get_selector()
        peers = ctx.other_results[: max(self._config.top_k_for_verification - 1, 0)]

        history: list[_Iteration] = []
        try:
            (
                pairs,
                constraints,
                conv,
                msgs,
                assistant_text,
            ) = selector.propose_initial(
                ctx.task,
                ctx.objects,
                scene_image=ctx.scene_image,
                seg_index_image=ctx.seg_index_image,
            )
        except Exception as exc:
            logger.error(f"IterativeStrategy: initial proposal failed: {exc}")
            raise

        score = self._score(ctx, peers, pairs, constraints)
        history.append(
            _Iteration(pairs, constraints, conv, msgs, assistant_text, score)
        )
        logger.info(
            f"[IterativeStrategy] iter 0 aggregate={score.aggregate:.4f} "
            f"(metric={score.metric})"
        )
        if (
            self._config.accept_score_threshold is not None
            and np.isfinite(score.aggregate)
            and score.aggregate <= self._config.accept_score_threshold
        ):
            return self._build_result(history, selected=0)

        for it in range(1, self._config.max_iterations + 1):
            feedback = _format_refinement_feedback(
                pairs, constraints, history[-1].score
            )
            try:
                (
                    pairs,
                    constraints,
                    conv,
                    msgs,
                    assistant_text,
                ) = selector.refine(history[-1].openai_messages, conv, feedback)
            except Exception as exc:
                logger.warning(
                    f"IterativeStrategy: refinement turn {it} failed: {exc}; "
                    "stopping early."
                )
                break
            score = self._score(ctx, peers, pairs, constraints)
            history.append(
                _Iteration(pairs, constraints, conv, msgs, assistant_text, score)
            )
            logger.info(
                f"[IterativeStrategy] iter {it} aggregate={score.aggregate:.4f}"
            )
            if (
                self._config.accept_score_threshold is not None
                and np.isfinite(score.aggregate)
                and score.aggregate <= self._config.accept_score_threshold
            ):
                break

        # Pick best (lowest) finite aggregate; if none finite, pick the last.
        finite = [
            (i, h.score.aggregate)
            for i, h in enumerate(history)
            if np.isfinite(h.score.aggregate)
        ]
        if finite:
            best_idx = min(finite, key=lambda x: x[1])[0]
        else:
            best_idx = len(history) - 1
            logger.warning(
                "[IterativeStrategy] no iteration produced a finite aggregate "
                "score; selecting the last proposal as a best-effort fallback."
            )
        return self._build_result(history, selected=best_idx)

    def _score(
        self,
        ctx: StrategyContext,
        peers: list[dict],
        pairs: list[DistancePair],
        constraints: list[SubtrajConstraint],
    ) -> ProposalScore:
        if ctx.top_result is None or not pairs:
            return ProposalScore(
                aggregate=float("inf"),
                metric=self._config.consistency_metric,
                per_segment=[],
                demo_breakdown=[],
            )
        try:
            return score_proposal(
                top_result=ctx.top_result,
                other_results=peers,
                top_seg_map=ctx.seg_name_map,
                pairs=pairs,
                constraints=constraints,
                metric=self._config.consistency_metric,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                min_segment_length=self._config.min_segment_length,
            )
        except Exception as exc:
            logger.warning(f"[IterativeStrategy] scoring failed: {exc}")
            return ProposalScore(
                aggregate=float("inf"),
                metric=self._config.consistency_metric,
                per_segment=[],
                demo_breakdown=[],
            )

    def _build_result(self, history: list[_Iteration], selected: int) -> StrategyResult:
        chosen = history[selected]
        boundaries = ref_segment_boundaries(chosen.score)
        log: list[dict] = []
        for i, h in enumerate(history):
            log.append(_iteration_to_log_entry(i, h))
        log[selected]["selected"] = True
        return StrategyResult(
            pairs=chosen.pairs,
            constraints=chosen.constraints,
            conversation=chosen.conversation,
            iteration_log=log,
            selected_subtraj_boundaries=boundaries,
            selected_iteration=selected,
        )


# ---------------------------------------------------------------------------
# Iteration log + feedback helpers
# ---------------------------------------------------------------------------


def _pair_to_dict(p: DistancePair) -> dict:
    return {
        "from_entity": p.from_entity,
        "to_entity": p.to_entity,
        "label": p.label,
        "distance_type": (
            p.distance_type.value
            if hasattr(p.distance_type, "value")
            else p.distance_type
        ),
        "axis": p.axis,
        "plane": p.plane,
        "ee_vector": p.ee_vector,
        "subtraj_idx": p.subtraj_idx,
    }


def _constraint_to_dict(c: SubtrajConstraint) -> dict:
    return {
        "kind": c.kind,
        "subtraj_idx": c.subtraj_idx,
        "label": c.label,
        "description": c.description,
        "expression": c.expression,
        "pairs_affected": c.pairs_affected,
    }


def _iteration_to_log_entry(idx: int, h: _Iteration) -> dict:
    seg_scores = [
        {
            "subtraj_idx": s.subtraj_idx,
            "n_attempted": s.n_attempted,
            "n_segmented": s.n_segmented,
            "mean_std_after_align": _safe_float(s.mean_std_after_align),
            "mean_dtw_distance": _safe_float(s.mean_dtw_distance),
            "seg_score": _safe_float(s.seg_score),
            "t_start_ref": s.t_start_ref,
            "t_end_ref": s.t_end_ref,
            "pair_labels": s.pair_labels,
        }
        for s in h.score.per_segment
    ]
    demo_breakdown = [
        {
            "demo_key": r.demo_key,
            "hdf5_path": r.hdf5_path,
            "segments": r.segments,
        }
        for r in h.score.demo_breakdown
    ]
    return {
        "iteration": idx,
        "proposal": {
            "pairs": [_pair_to_dict(p) for p in h.pairs],
            "constraints": [_constraint_to_dict(c) for c in h.constraints],
        },
        "score": {
            "aggregate": _safe_float(h.score.aggregate),
            "metric": h.score.metric,
            "per_segment": seg_scores,
        },
        "demo_breakdown": demo_breakdown,
        "assistant_text": h.assistant_text,
        "selected": False,
    }


def _safe_float(x: float) -> Optional[float]:
    if x is None:
        return None
    fx = float(x)
    if not np.isfinite(fx):
        return None
    return fx


def _format_refinement_feedback(
    pairs: list[DistancePair],
    constraints: list[SubtrajConstraint],
    score: ProposalScore,
) -> str:
    """Build the user message sent in the next refinement turn.

    Includes per-segment success rate, post-DTW std (the metric the user
    asked us to optimise), and a short narrative of what to fix.
    """
    if not score.per_segment:
        return (
            "The previous proposal could not be scored (no segments were "
            "produced or no demos could be loaded). Please return a fresh "
            "proposal with simpler, clearly-grounded post conditions."
        )

    lines = [
        "Your previous proposal was scored against retrieved demos.",
        f"Aggregate consistency score (lower is better, metric={score.metric}): "
        f"{_fmt(score.aggregate)}",
        "",
        "Per-segment breakdown:",
    ]
    for s in score.per_segment:
        success = s.n_segmented / max(s.n_attempted, 1)
        lines.append(
            f"- ST{s.subtraj_idx}: segmented {s.n_segmented}/{s.n_attempted} "
            f"demos (success_rate={success:.2f}) | "
            f"mean_std_after_align={_fmt(s.mean_std_after_align)} | "
            f"mean_dtw_distance={_fmt(s.mean_dtw_distance)} | "
            f"seg_score={_fmt(s.seg_score)} | "
            f"ref_window=[{s.t_start_ref}, {s.t_end_ref})"
        )
        if s.pair_labels:
            lines.append(f"    pairs: {s.pair_labels}")
        if s.n_segmented < s.n_attempted:
            lines.append(
                "    NOTE: post_condition did not fire for some demos — "
                "consider whether your threshold is too tight or whether "
                "the geometry you reference (entity IDs) is reliable."
            )

    lines += [
        "",
        "Suggestions:",
        "- For segments with low success rate, relax the post threshold or "
        "switch to an entity that is robustly visible across all demos.",
        "- For segments with high seg_score (high std after DTW), the chosen "
        "pairs vary inconsistently across demos — consider replacing them "
        "with quantities that are more directly task-relevant (e.g. surface "
        "vs centroid, or a single axis projection rather than full 3-D).",
        "- Keep the chaining rule: ST(i+1).pre should follow from ST(i).post.",
        "",
        "Return the FULL revised JSON object using the same schema. Keep "
        "good segments unchanged.",
    ]
    return "\n".join(lines)


def _fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.4f}"

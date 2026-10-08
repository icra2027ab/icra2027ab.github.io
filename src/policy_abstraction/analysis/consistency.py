"""Score how consistent a pair/condition proposal is across retrieved demos.

Pipeline:
    1. For each demo, derive segment boundaries from the proposed
       post-conditions (:func:`segment_demo_by_conditions`).
    2. For each segment ``i``, compute the per-frame distance trajectory
       restricted to the pairs scoped to that segment.
    3. DTW-align each peer demo's segment trajectory to the top-1 reference's
       segment trajectory (via the existing
       :func:`policy_abstraction.evaluation.retrieval._dtw_warp_to_ref`).
    4. Average post-DTW std (or mean DTW path distance) over time and pairs;
       penalise by inverse success rate to discount proposals where many
       demos failed to segment.

The result feeds the iterative-strategy refinement loop (the VLM is shown a
breakdown of which segments scored worst and why) and is also dumped to the
recorder metadata so a human can inspect how the iterations evolved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import h5py
import numpy as np
from loguru import logger

from policy_abstraction.analysis.segmentation_from_conditions import (
    load_demo_pose_and_keypoints,
    segment_demo_by_conditions,
)
from policy_abstraction.vlm.types import DistancePair, SubtrajConstraint


@dataclass
class SegmentScore:
    subtraj_idx: int
    n_attempted: int
    n_segmented: int
    mean_std_after_align: float
    mean_dtw_distance: float
    seg_score: float
    """Composite score (lower is better) — driven by ``metric`` and divided by
    ``max(success_rate, 1e-3)`` so failed segmentations are penalised."""
    t_start_ref: int
    t_end_ref: int
    pair_labels: list[str] = field(default_factory=list)


@dataclass
class DemoSegmentReport:
    demo_key: str
    hdf5_path: str
    segments: list[dict]


@dataclass
class ProposalScore:
    aggregate: float
    metric: str
    per_segment: list[SegmentScore]
    demo_breakdown: list[DemoSegmentReport]


def _post_conditions_by_subtraj(
    constraints: list[SubtrajConstraint],
) -> dict[int, SubtrajConstraint]:
    """Pick the post constraint for each subtraj_idx by label suffix ``_post``."""
    out: dict[int, SubtrajConstraint] = {}
    for c in constraints:
        if c.kind == "sub_goal" and c.label.endswith("_post"):
            out[c.subtraj_idx] = c
    return out


def _compile_post_lambdas(
    constraints: list[SubtrajConstraint],
) -> tuple[list, list[SubtrajConstraint]]:
    """Compile the chained list of post lambdas in subtraj_idx order.

    Returns ``(callables, post_constraints)`` where both lists have length K.
    """
    by_idx = _post_conditions_by_subtraj(constraints)
    if not by_idx:
        return [], []
    K = max(by_idx.keys()) + 1
    fns = []
    posts: list[SubtrajConstraint] = []
    for i in range(K):
        c = by_idx.get(i)
        if c is None:
            # Missing — synthesise a never-satisfied stub so the segmenter
            # produces a fallback rather than crashing.
            fns.append(lambda ee, kp, ee_R: 1.0)
            posts.append(
                SubtrajConstraint(
                    kind="sub_goal",
                    subtraj_idx=i,
                    label=f"st{i}_post_missing",
                    description="missing post condition",
                    expression="lambda ee, keypoints, ee_R: 1.0",
                )
            )
        else:
            try:
                fns.append(c.compile())
            except Exception as exc:
                logger.warning(
                    f"[consistency] post for subtraj {i} failed to compile: {exc}"
                )
                fns.append(lambda ee, kp, ee_R: 1.0)
            posts.append(c)
    return fns, posts


def _entity_ids_from_pairs(pairs: list[DistancePair]) -> set[int]:
    out: set[int] = set()
    for p in pairs:
        for e in (p.from_entity, p.to_entity):
            if isinstance(e, int):
                out.add(e)
    return out


def _entity_ids_from_constraints(constraints: list[SubtrajConstraint]) -> set[int]:
    """Heuristically extract integer keypoint IDs referenced in lambdas.

    The constraint sandbox accesses keypoints via ``keypoints[<id>]``. We don't
    parse the AST — we just pre-fill any int seen in :class:`DistancePair`. The
    :class:`_CarryForwardKeypoints` sandbox covers anything we miss.
    """
    return set()


def _load_seg_name_map(hdf5_path: str) -> dict[str, str]:
    try:
        with h5py.File(hdf5_path, "r") as f:
            raw = f["data"].attrs.get("segmentation_name_mapping", None)
            return json.loads(raw) if raw else {}
    except Exception as exc:
        logger.debug(f"[consistency] could not load seg map: {exc}")
        return {}


def score_proposal(
    top_result: dict,
    other_results: list[dict],
    top_seg_map: dict[str, str],
    pairs: list[DistancePair],
    constraints: list[SubtrajConstraint],
    metric: Literal["std_after_align", "mean_dtw"] = "std_after_align",
    min_segment_length: int = 5,
) -> ProposalScore:
    """Score a proposal by replaying its post-conditions across demos.

    The top-1 demo's segmentation is the reference time axis. Each peer demo's
    pairs are remapped to its segmentation IDs, distances are computed
    per-frame, sliced per segment, and DTW-aligned to the reference slice.

    Returns
    -------
    :class:`ProposalScore` with per-segment + aggregate metrics, plus a
    JSON-serialisable demo breakdown for the recorder sidecar.
    """
    from policy_abstraction.analysis.demo_episode import build_demo_episode
    from policy_abstraction.evaluation.retrieval import (
        _compute_raw_traj,
        _dtw_warp_to_ref,
        _remap_pairs_for_demo,
    )

    post_fns, post_constraints = _compile_post_lambdas(constraints)
    if not post_fns:
        return ProposalScore(
            aggregate=float("inf"),
            metric=metric,
            per_segment=[],
            demo_breakdown=[],
        )

    K = len(post_fns)

    # ---- Reference (top-1) -------------------------------------------------
    entity_ids = _entity_ids_from_pairs(pairs) | _entity_ids_from_constraints(
        constraints
    )
    ref_loaded = load_demo_pose_and_keypoints(
        top_result["source_path"], top_result["demo_key"], entity_ids
    )
    if ref_loaded is None:
        return ProposalScore(
            aggregate=float("inf"),
            metric=metric,
            per_segment=[],
            demo_breakdown=[],
        )
    ref_ee, ref_R, ref_kp = ref_loaded
    ref_segments = segment_demo_by_conditions(
        ref_ee, ref_R, ref_kp, post_fns, min_segment_length=min_segment_length
    )
    pair_labels = [p.label for p in pairs]
    if not pair_labels:
        return ProposalScore(
            aggregate=float("inf"),
            metric=metric,
            per_segment=[],
            demo_breakdown=[],
        )
    ref_traj = _compute_raw_traj(build_demo_episode(top_result), pairs, pair_labels)
    if ref_traj is None:
        return ProposalScore(
            aggregate=float("inf"),
            metric=metric,
            per_segment=[],
            demo_breakdown=[],
        )

    # ---- Pair-to-segment column map (which columns belong to subtraj i) ----
    cols_for_subtraj: dict[int, list[int]] = {}
    for col, p in enumerate(pairs):
        if p.subtraj_idx is None:
            # Cross-segment pair — applies to every segment.
            for i in range(K):
                cols_for_subtraj.setdefault(i, []).append(col)
        else:
            cols_for_subtraj.setdefault(p.subtraj_idx, []).append(col)

    # ---- Each peer demo: load, segment, compute traj, slice per segment ----
    demo_breakdown: list[DemoSegmentReport] = [
        DemoSegmentReport(
            demo_key=top_result.get("demo_key", "ref"),
            hdf5_path=top_result.get("source_path", ""),
            segments=[
                {
                    "subtraj_idx": s.subtraj_idx,
                    "t_start": s.t_start,
                    "t_end": s.t_end,
                    "fallback": s.fallback,
                    "reason": s.reason,
                }
                for s in ref_segments
            ],
        )
    ]

    per_segment_slices: dict[int, list[np.ndarray]] = {i: [] for i in range(K)}
    n_attempted: dict[int, int] = {i: 1 for i in range(K)}  # ref counts as 1
    n_segmented: dict[int, int] = {
        i: (1 if i < len(ref_segments) and not ref_segments[i].fallback else 0)
        for i in range(K)
    }

    # Reference slices
    for i, seg in enumerate(ref_segments):
        if seg.fallback or i not in cols_for_subtraj:
            continue
        cols = cols_for_subtraj[i]
        per_segment_slices[i].append(ref_traj[seg.t_start : seg.t_end, cols])

    for other in other_results:
        other_seg_map = _load_seg_name_map(other["source_path"])
        if other_seg_map and other_seg_map != top_seg_map:
            mapped_pairs = _remap_pairs_for_demo(pairs, top_seg_map, other_seg_map)
        else:
            mapped_pairs = pairs
        if not mapped_pairs:
            continue

        mapped_entity_ids = _entity_ids_from_pairs(mapped_pairs)
        loaded = load_demo_pose_and_keypoints(
            other["source_path"], other["demo_key"], mapped_entity_ids
        )
        if loaded is None:
            continue
        ee_p, ee_R_o, kp_o = loaded
        peer_segments = segment_demo_by_conditions(
            ee_p, ee_R_o, kp_o, post_fns, min_segment_length=min_segment_length
        )
        peer_traj = _compute_raw_traj(
            build_demo_episode(other), mapped_pairs, [p.label for p in mapped_pairs]
        )
        # Build a map back to the reference column order
        label_to_peer_col = {p.label: col for col, p in enumerate(mapped_pairs)}

        demo_breakdown.append(
            DemoSegmentReport(
                demo_key=other.get("demo_key", "?"),
                hdf5_path=other.get("source_path", ""),
                segments=[
                    {
                        "subtraj_idx": s.subtraj_idx,
                        "t_start": s.t_start,
                        "t_end": s.t_end,
                        "fallback": s.fallback,
                        "reason": s.reason,
                    }
                    for s in peer_segments
                ],
            )
        )

        for i, seg in enumerate(peer_segments):
            n_attempted[i] = n_attempted.get(i, 0) + 1
            if seg.fallback or peer_traj is None or i not in cols_for_subtraj:
                continue
            ref_cols = cols_for_subtraj[i]
            # Translate ref column indices back to peer column indices via labels.
            peer_cols: list[int] = []
            keep_mask: list[bool] = []
            for ref_col in ref_cols:
                lbl = pair_labels[ref_col]
                if lbl in label_to_peer_col:
                    peer_cols.append(label_to_peer_col[lbl])
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            if not peer_cols:
                continue
            peer_slice = peer_traj[seg.t_start : seg.t_end, peer_cols]
            n_segmented[i] = n_segmented.get(i, 0) + 1
            # Pad missing columns with NaN so the stack is rectangular.
            full = np.full(
                (peer_slice.shape[0], len(ref_cols)), np.nan, dtype=np.float32
            )
            j = 0
            for k, kept in enumerate(keep_mask):
                if kept:
                    full[:, k] = peer_slice[:, j]
                    j += 1
            per_segment_slices[i].append(full)

    # ---- Aggregate per-segment scores --------------------------------------
    per_segment: list[SegmentScore] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for i in range(K):
        ref_seg = ref_segments[i] if i < len(ref_segments) else None
        if ref_seg is None or ref_seg.fallback:
            seg_score = float("inf")
            t_start_ref, t_end_ref = (
                (ref_seg.t_start, ref_seg.t_end) if ref_seg is not None else (0, 0)
            )
            per_segment.append(
                SegmentScore(
                    subtraj_idx=i,
                    n_attempted=n_attempted.get(i, 0),
                    n_segmented=n_segmented.get(i, 0),
                    mean_std_after_align=float("nan"),
                    mean_dtw_distance=float("nan"),
                    seg_score=seg_score,
                    t_start_ref=t_start_ref,
                    t_end_ref=t_end_ref,
                    pair_labels=[pair_labels[c] for c in cols_for_subtraj.get(i, [])],
                )
            )
            continue

        slices = per_segment_slices[i]
        ref_slice = slices[0]
        peer_slices = slices[1:]
        if ref_slice.shape[0] < 2:
            mean_std = float("nan")
            mean_dtw = float("nan")
        elif not peer_slices:
            mean_std = 0.0
            mean_dtw = 0.0
        else:
            warped = [ref_slice]
            dtw_dists: list[float] = []
            for ps in peer_slices:
                w = _dtw_warp_to_ref(ref_slice, ps)
                warped.append(w)
                # Frobenius distance between ref and warped peer.
                diff = ref_slice - w
                d = float(np.sqrt(np.nanmean(diff * diff)))
                dtw_dists.append(d)
            stack = np.stack(warped, axis=0)
            mean_std = float(np.nanmean(np.nanstd(stack, axis=0)))
            mean_dtw = float(np.nanmean(dtw_dists)) if dtw_dists else 0.0

        attempted = max(n_attempted.get(i, 1), 1)
        success_rate = n_segmented.get(i, 0) / attempted
        denom = max(success_rate, 1e-3)
        chosen = mean_std if metric == "std_after_align" else mean_dtw
        seg_score = (chosen if np.isfinite(chosen) else float("inf")) / denom

        weight = max(ref_seg.t_end - ref_seg.t_start, 1)
        if np.isfinite(seg_score):
            weighted_sum += seg_score * weight
            total_weight += weight

        per_segment.append(
            SegmentScore(
                subtraj_idx=i,
                n_attempted=attempted,
                n_segmented=n_segmented.get(i, 0),
                mean_std_after_align=mean_std,
                mean_dtw_distance=mean_dtw,
                seg_score=seg_score,
                t_start_ref=ref_seg.t_start,
                t_end_ref=ref_seg.t_end,
                pair_labels=[pair_labels[c] for c in cols_for_subtraj.get(i, [])],
            )
        )

    aggregate = weighted_sum / total_weight if total_weight > 0 else float("inf")

    return ProposalScore(
        aggregate=aggregate,
        metric=metric,
        per_segment=per_segment,
        demo_breakdown=demo_breakdown,
    )


def ref_segment_boundaries(
    proposal_score: ProposalScore,
) -> list[tuple[int, int]] | None:
    """Extract ``(t_start, t_end)`` for the top-1 demo's segments.

    Returns ``None`` if any segment is a fallback — in that case the runtime
    speed-based segmentation should be used instead.
    """
    if not proposal_score.demo_breakdown:
        return None
    ref_breakdown = proposal_score.demo_breakdown[0]
    if any(s.get("fallback") for s in ref_breakdown.segments):
        return None
    return [(s["t_start"], s["t_end"]) for s in ref_breakdown.segments]

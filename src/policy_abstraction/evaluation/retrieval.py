"""Retrieval and demo-analysis helpers for the evaluation pipeline.

Separates the retrieval/analysis orchestration from the main evaluation script
so that ``evaluate.py`` stays focused on the episode loop.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import io
import pathlib

import numpy as np
from loguru import logger
from PIL import Image

# ---------------------------------------------------------------------------
# Multi-demo trajectory helpers
# ---------------------------------------------------------------------------


def _compute_raw_traj(
    episode,
    pairs: list,
    pair_labels: list[str],
    erode_px: int = 3,
) -> np.ndarray | None:
    """Per-frame distance trajectory for one demo, whatever its format.

    Replicates the core logic of ``RolloutAbstractor._precompute_ref_traj`` as
    a standalone function so it can be applied to several retrieved demos
    without constructing a full :class:`RolloutAbstractor`.

    Reads through the :class:`~policy_abstraction.analysis.demo_episode.DemoEpisode`
    contract, so a simulator demo and a real ``.vla`` recording take the same
    path. Only the timesteps the episode reports under ``frames_with_depth()``
    can be reconstructed — every frame for a simulator, the recorded stride for
    a real rollout — and the gaps between them are carried forward, which is
    what the function already did for frames whose reconstruction failed.

    Parameters
    ----------
    episode:
        The demo to measure.
    pairs:
        List of :class:`~policy_abstraction.vlm.types.DistancePair` objects
        (potentially re-mapped to match this demo's segmentation IDs).
    pair_labels:
        Ordered list of label strings matching *pairs*.
    erode_px:
        Erosion kernel radius applied to the segmentation mask.

    Returns
    -------
    np.ndarray or None
        ``[T, n_pairs]`` float32 array, or ``None`` on failure.
    """
    from policy_abstraction.evaluation.abstraction import _gripper_aperture_scalar
    from policy_abstraction.features.base import compute_distances
    from policy_abstraction.features.centroid import CentroidDistance
    from policy_abstraction.utils.gripper import compute_finger_tip_positions
    from policy_abstraction.utils.segmentation import (
        build_segmented_pointcloud,
        erode_segmentation,
    )

    feature = CentroidDistance()
    n_pairs = len(pair_labels)

    try:
        T = len(episode)
        ee_pos = episode.ee_positions()
        depth_frames = episode.frames_with_depth()
    except Exception as exc:
        logger.warning(
            f"[multi-demo] Could not open {getattr(episode, 'demo_key', '?')}: {exc}"
        )
        return None

    if T == 0 or not depth_frames:
        logger.warning(
            f"[multi-demo] {getattr(episode, 'demo_key', '?')} has no depth frames; "
            "cannot build a reference trajectory."
        )
        return None

    # Orientation, aperture and finger-tip pairs are measured from the same
    # per-frame quantities the guidance reference uses. Passing none of them
    # (which this did) does not skip those pairs -- their features silently
    # return 0.0, so every gripper-relative column came out a flat zero line.
    ee_ori = episode.ee_orientations()
    gripper_scalar = _gripper_aperture_scalar(episode.gripper_states())
    finger_l = finger_r = None
    needs_fingers = any(
        e in ("gripper_l", "gripper_r")
        for p in pairs
        for e in (p.from_entity, p.to_entity)
    )
    if needs_fingers and ee_ori is not None and gripper_scalar is not None:
        try:
            finger_l, finger_r = compute_finger_tip_positions(
                np.asarray(ee_pos, dtype=np.float64),
                np.asarray(ee_ori, dtype=np.float64),
                np.asarray(episode.gripper_states(), dtype=np.float64),
                geometry=episode.gripper_geometry(),
            )
        except Exception as exc:
            logger.debug(f"[multi-demo] finger-tip reconstruction failed: {exc}")

    # A format that stores frames rotated needs the u mirror undone on
    # back-projection; one that publishes them upright does not.
    preprocessed = bool(getattr(episode, "frames_preprocessed", True))

    traj = np.full((T, n_pairs), np.nan, dtype=np.float32)
    for t in depth_frames:
        if t >= T:
            continue
        try:
            seg = episode.segmentation(t)
            depth = episode.depth_m(t)
            K = episode.intrinsic(t)
            E = episode.extrinsic(t)
            if seg is None or depth is None or K is None or E is None:
                continue
            if erode_px > 0:
                seg = erode_segmentation(seg, erode_px)
            K3 = np.asarray(K)
            K3 = K3[:3, :3] if K3.shape == (4, 4) else K3
            pcd = build_segmented_pointcloud(
                episode.rgb(t),
                depth,
                seg,
                K3,
                np.asarray(E),
                preprocessed=preprocessed,
            )
            if pcd.shape[0] == 0:
                continue
            finger_pos_t = (
                {
                    "gripper_l": finger_l[t].astype(np.float32),
                    "gripper_r": finger_r[t].astype(np.float32),
                }
                if finger_l is not None and finger_r is not None
                else None
            )
            dists = compute_distances(
                pairs,
                feature,
                pcd,
                np.asarray(ee_pos[t], dtype=np.float32),
                finger_pos_t,
                ee_ori=np.asarray(ee_ori[t]) if ee_ori is not None else None,
                gripper_state=(
                    float(gripper_scalar[t, 0]) if gripper_scalar is not None else None
                ),
            )
            for i, lbl in enumerate(pair_labels):
                if lbl in dists:
                    traj[t, i] = dists[lbl]
        except Exception as exc:
            logger.debug(f"[multi-demo] traj step {t} failed: {exc}")

    return _fill_traj_gaps(traj, pair_labels)


def _fill_traj_gaps(traj: np.ndarray, pair_labels: list[str]) -> np.ndarray:
    """Make a sparsely-measured ``[T, n_pairs]`` trajectory dense, in place.

    Frames that could not be reconstructed hold their column's previous
    value. Ordering matters: the carry-forward copies row ``t-1`` into row
    ``t``, so row 0 has to be seeded *first* — a NaN left there propagates
    down the whole column and only row 0 ever gets repaired.

    The seed is the column's first real measurement rather than 0.0. With
    depth recorded on a stride the first reconstructable frame need not be
    ``t=0``, and back-filling the first true value beats inventing a zero
    distance that the aligner would then try to match.
    """
    T = traj.shape[0]
    if T == 0:
        return traj

    valid = ~np.isnan(traj)
    measured = np.asarray(valid.any(axis=0))
    first_valid = np.asarray(np.argmax(valid, axis=0))
    for i in np.flatnonzero(measured):
        traj[0, i] = traj[first_valid[i], i]

    unmeasured = np.flatnonzero(~measured)
    if unmeasured.size:
        missing = [pair_labels[int(i)] for i in unmeasured]
        logger.warning(
            f"[multi-demo] No frame yielded a distance for {missing} — those "
            "entities were never both reconstructed. The pairs are zeroed and "
            "carry no guidance signal."
        )
        traj[0, unmeasured] = 0.0

    for t in range(1, T):
        nan_mask = np.isnan(traj[t])
        if nan_mask.any():
            traj[t, nan_mask] = traj[t - 1, nan_mask]

    return traj


def _remap_pairs_for_demo(
    pairs: list,
    source_seg_map: dict[str, str],
    target_seg_map: dict[str, str],
) -> list:
    """Remap :class:`~policy_abstraction.vlm.types.DistancePair` entity IDs.

    Builds a name-based correspondence between *source_seg_map* (from the
    top-1 demo) and *target_seg_map* (from the candidate demo), then returns
    a copy of *pairs* with entity IDs substituted accordingly.  Pairs whose
    entities cannot be matched are dropped.

    Parameters
    ----------
    pairs:
        Original ``DistancePair`` list (entities reference top-1 seg IDs).
    source_seg_map:
        ``{str(id): name}`` from the top-1 demo.
    target_seg_map:
        ``{str(id): name}`` from the target demo.

    Returns
    -------
    list
        Re-mapped ``DistancePair`` list (may be shorter than *pairs* on
        failure to match some entities).
    """
    # Build lookup tables
    src_id_to_name: dict[int, str] = {int(k): v for k, v in source_seg_map.items()}
    tgt_name_to_id: dict[str, int] = {}
    for id_str, name in target_seg_map.items():
        tgt_name_to_id[name] = int(id_str)
    # Case-insensitive fallback map
    tgt_name_lower: dict[str, int] = {k.lower(): v for k, v in tgt_name_to_id.items()}

    remapped = []
    for pair in pairs:
        p = copy.copy(pair)
        ok = True
        for attr in ("from_entity", "to_entity"):
            entity = getattr(p, attr)
            if entity == "ee":
                continue
            if not isinstance(entity, int):
                continue
            name = src_id_to_name.get(entity)
            if name is None:
                ok = False
                break
            new_id = tgt_name_to_id.get(name) or tgt_name_lower.get(name.lower())
            if new_id is None:
                ok = False
                break
            setattr(p, attr, new_id)
        if ok:
            remapped.append(p)

    return remapped


def _dtw_warp_to_ref(
    ref_traj: np.ndarray,
    query_traj: np.ndarray,
) -> np.ndarray:
    """DTW-align *query_traj* to the time axis of *ref_traj*.

    Uses :func:`dtaidistance.dtw_ndim.warping_path` (multi-dimensional DTW)
    to find the optimal alignment.  The warping path maps each reference
    timestep ``i`` to one or more query timesteps ``j``; the returned array
    averages the query values for all ``j`` mapped to each ``i``.

    NaN values in *ref_traj* / *query_traj* are replaced with ``0`` before
    DTW (to avoid library errors) but the output is always float32 and may
    contain NaN for timesteps where no valid query frame was mapped.

    Parameters
    ----------
    ref_traj:
        ``[T_ref, n_pairs]`` reference trajectory (from top-1 demo).
    query_traj:
        ``[T_query, n_pairs]`` trajectory to align.

    Returns
    -------
    np.ndarray
        ``[T_ref, n_pairs]`` float32 warped trajectory.
    """
    from dtaidistance import dtw_ndim

    T_ref, n_pairs = ref_traj.shape

    # Replace NaN with 0 for DTW distance computation
    ref_filled = np.where(np.isnan(ref_traj), 0.0, ref_traj).astype(np.double)
    qry_filled = np.where(np.isnan(query_traj), 0.0, query_traj).astype(np.double)

    try:
        path = dtw_ndim.warping_path(ref_filled, qry_filled)
    except Exception as exc:
        logger.warning(
            f"[DTW] warping_path failed: {exc}; falling back to linear interpolation."
        )
        # Linear resampling fallback
        T_q = query_traj.shape[0]
        new_t = np.linspace(0, T_q - 1, T_ref)
        warped = np.zeros((T_ref, n_pairs), dtype=np.float32)
        for c in range(n_pairs):
            col = query_traj[:, c].astype(np.float32)
            warped[:, c] = np.interp(new_t, np.arange(T_q), col)
        return warped

    # Aggregate: for each ref index i, average query frames mapped to it
    sums = np.zeros((T_ref, n_pairs), dtype=np.float64)
    counts = np.zeros(T_ref, dtype=np.int32)
    for i_ref, j_qry in path:
        sums[i_ref] += query_traj[j_qry]
        counts[i_ref] += 1

    warped = np.full((T_ref, n_pairs), np.nan, dtype=np.float32)
    valid = counts > 0
    warped[valid] = (sums[valid] / counts[valid, np.newaxis]).astype(np.float32)
    return warped


def compute_multi_demo_stats(
    ref_traj: np.ndarray,
    other_trajs: list[np.ndarray],
    pair_labels: list[str],
) -> dict:
    """Compute per-timestep statistics across DTW-aligned demo trajectories.

    Parameters
    ----------
    ref_traj:
        ``[T_ref, n_pairs]`` reference trajectory (from the top-1 demo).
    other_trajs:
        List of ``[T_i, n_pairs]`` trajectories for the other retrieved demos.
        Each is DTW-aligned to *ref_traj*'s time axis before statistics are
        computed.
    pair_labels:
        Ordered list of pair label strings (length ``n_pairs``).

    Returns
    -------
    dict with keys:
        ``ref_traj``    — ``[T, n_pairs]`` top-1 reference
        ``mean``        — ``[T, n_pairs]`` mean over all aligned demos
        ``std``         — ``[T, n_pairs]`` std dev
        ``min``         — ``[T, n_pairs]`` per-element minimum
        ``max``         — ``[T, n_pairs]`` per-element maximum
        ``pair_labels`` — list of label strings
        ``n_demos``     — total number of demos included (including top-1)
    """
    T_ref, n_pairs = ref_traj.shape
    aligned: list[np.ndarray] = [ref_traj.astype(np.float32)]

    for traj in other_trajs:
        if traj is None or traj.shape[0] == 0:
            continue
        if traj.shape[1] != n_pairs:
            logger.warning(
                f"[multi-demo stats] Skipping demo with {traj.shape[1]} pairs "
                f"(expected {n_pairs})."
            )
            continue
        warped = _dtw_warp_to_ref(ref_traj, traj)
        aligned.append(warped)

    stacked = np.stack(aligned, axis=0)  # [N, T, n_pairs]
    mean = np.nanmean(stacked, axis=0).astype(np.float32)
    std = np.nanstd(stacked, axis=0).astype(np.float32)
    mn = np.nanmin(stacked, axis=0).astype(np.float32)
    mx = np.nanmax(stacked, axis=0).astype(np.float32)

    return {
        "ref_traj": ref_traj.astype(np.float32),
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "pair_labels": pair_labels,
        "n_demos": len(aligned),
    }


# ---------------------------------------------------------------------------
# Main component builders
# ---------------------------------------------------------------------------


def build_retrieval_components(args):
    """Instantiate retriever and (optionally) demo analyzer from ``args``.

    Returns
    -------
    tuple[EmbeddingRetriever | None, DemoAnalyzer | None]
        Both are ``None`` when retrieval is disabled or the index is empty.
    """
    if not args.retrieval.enable:
        return None, None

    from policy_abstraction.retrieval import (
        DatasetIndex,
        DemoEncoder,
        EmbeddingRetriever,
    )

    project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    dataset_dirs: list[pathlib.Path | str] = (
        list(args.retrieval.dataset_dirs)
        if args.retrieval.dataset_dirs
        else [
            str(project_root / "datasets" / "libero_spatial_new"),
            str(project_root / "datasets" / "libero_object_new"),
            str(project_root / "datasets" / "libero_goal_new"),
            str(project_root / "datasets" / "libero_10_new"),
        ]
    )

    # One encoder for both jobs: auto-encoding a missing cache (if it comes to
    # that) and embedding the live query below. Built on first use so a run
    # against an already-encoded dataset loads it exactly once, at the point
    # the log says so.
    _encoder: list = []

    def _get_encoder():
        if not _encoder:
            logger.info(
                f"Loading retrieval encoder "
                f"(DINOv2={args.retrieval.dino_model}, "
                f"CLIP={args.retrieval.clip_model})…"
            )
            _encoder.append(
                DemoEncoder(
                    dino_model=args.retrieval.dino_model,
                    clip_model=args.retrieval.clip_model,
                )
            )
        return _encoder[0]

    logger.info(f"Building retrieval index from: {dataset_dirs}")
    index = DatasetIndex(
        dataset_dirs,
        auto_encode=args.retrieval.auto_encode,
        encoder_factory=_get_encoder,
    )
    if index.size == 0:
        raise RuntimeError(
            f"Retrieval index is empty — no *_embeddings.npz files found in "
            f"{dataset_dirs}. Run scripts/encode_datasets.py first, or pass "
            "--retrieval.no-enable to disable retrieval."
        )

    encoder = _get_encoder()
    retriever = EmbeddingRetriever(index=index, encoder=encoder)
    logger.info(f"Retrieval index ready ({index.size} demos).")

    analyzer = None
    if args.demo_analysis.enable:
        from policy_abstraction.analysis import DemoAnalyzer
        from policy_abstraction.analysis.strategies import (
            IterativeStrategyConfig,
            StrategyKind,
        )

        cache_dir = (
            args.demo_analysis.vlm_cache_dir
            if getattr(args.demo_analysis, "cache_pairs", True)
            else None
        )
        strategy = getattr(args.demo_analysis, "strategy", StrategyKind.SINGLE_TURN)
        iterative_cfg: IterativeStrategyConfig | None = None
        if strategy == StrategyKind.ITERATIVE:
            it = args.demo_analysis.iterative
            iterative_cfg = IterativeStrategyConfig(
                max_iterations=it.max_iterations,
                consistency_metric=it.consistency_metric,
                top_k_for_verification=it.top_k_for_verification,
                min_segment_length=it.min_segment_length,
                accept_score_threshold=it.accept_score_threshold,
            )
        analyzer = DemoAnalyzer(
            vlm_model=args.demo_analysis.vlm_model,
            cache_dir=cache_dir,
            gripper_speed_threshold=args.subtraj.gripper_speed_threshold,
            subtraj_min_length=args.subtraj.min_length,
            strategy=strategy,
            iterative_config=iterative_cfg,
        )
        logger.info(
            f"Demo analyzer ready (VLM model: {args.demo_analysis.vlm_model}, "
            f"strategy: {strategy.value}"
            + (f", cache_dir: {cache_dir}" if cache_dir else "")
            + ")."
        )

    return retriever, analyzer


def _analysis_cache_key(source_path: str, demo_key: str) -> str:
    """Cache key for one demo's VLM analysis.

    The obvious key -- the source path's stem -- is not unique for a ``.vla``
    source, because every recorded dataset ends in a directory literally named
    ``episodes``. Two different demo folders therefore shared one cache entry,
    and the second run silently reused the first's distance pairs, whose
    segmentation ids referred to a different scene.

    So the key carries a digest of the *resolved* source path as well. The
    readable part is kept in front so the cache directory stays greppable.
    """
    import hashlib

    resolved = pathlib.Path(source_path).resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    # For ``.../<task>/episodes`` the parent is the informative half.
    label = resolved.stem
    if label in ("episodes", "") and resolved.parent.name:
        label = f"{resolved.parent.name}_{label}".strip("_")
    return f"{label}_{digest}/{demo_key}"


def _check_extrinsic(episode, measure, tolerance_m: float = 0.15) -> None:
    """Warn when a demo's calibration disagrees with its own depth.

    Guidance measures distances in the robot frame, so a wrong extrinsic gives
    confidently wrong numbers rather than an error. Worth one loud line before
    an experiment is run on top of it.
    """
    try:
        error = measure(episode)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never break the run
        logger.debug(f"[calibration] check failed: {exc}")
        return
    if error is None:
        return
    if error == float("inf"):
        logger.warning(
            f"[calibration] {getattr(episode, 'demo_key', '?')}: the recorded "
            "extrinsic never places the end-effector inside the camera image, "
            "so depth reconstructed with it lands in the wrong place and "
            "guidance distances would be wrong. The usual cause is a base-frame "
            "mismatch rather than a bad calibration: UR's `base` and `base_link` "
            "sit 180 degrees apart about z, and mixing them mirrors the scene in "
            "x and y while leaving the table horizontal. Check "
            "--env.calibration-base-frame before re-running the hand-eye "
            "calibration (scripts/setup/ur5_calibration_from_kalib.py)."
        )
    elif error > tolerance_m:
        logger.warning(
            f"[calibration] {getattr(episode, 'demo_key', '?')}: extrinsic "
            f"disagrees with the episode's own depth by {error:.2f} m at the "
            "end-effector. Guidance distances will carry that error."
        )
    else:
        logger.info(
            f"[calibration] extrinsic agrees with depth to {error * 100:.1f} cm "
            "at the end-effector."
        )


def run_retrieval_and_analysis(
    recorder,
    retriever,
    analyzer,
    task_description: str,
    first_frame: np.ndarray,
    k: int = 5,
    segmenter=None,
) -> tuple:
    """Retrieve similar demos, save a grid figure, and optionally analyze top-1.

    *segmenter* is forwarded to every episode that needs one: a simulator
    demo carries ground-truth masks, while a real recording has none and is
    segmented from RGB-D at analysis time.

    Retrieves *k* demos, runs VLM analysis on the top-1 result to extract
    distance pairs, then computes per-frame distance trajectories for all
    retrieved demos and DTW-aligns demos 2…k to the top-1 reference.
    Multi-demo statistics (mean / std / min / max over time) are stored in
    *viz_data* under the key ``"multi_demo_stats"`` for downstream
    visualization.

    Results are written into *recorder*:
    - ``retrieval/query_grid``      — matplotlib figure as uint8 RGB (in .vla)
    - ``analysis/seg_annotated_image`` — seg-ID overlay (in .vla, if analysis)
    - ``retrieval_results``         — ranked list of matches (in .json sidecar)
    - ``vlm_analysis``              — VLM prompt/response and distance pairs (.json)

    Failures are caught and logged as warnings so the episode continues.

    Returns
    -------
    tuple[dict | None, DemoAnalysisResult | None, dict[str, str], dict, DemoEpisode | None]
        ``(top_result, analysis, seg_name_map, viz_data, top_episode)`` where *top_result* is the
        highest-ranked retrieval match dict, *analysis* is the
        :class:`~policy_abstraction.analysis.demo_analyzer.DemoAnalysisResult`
        for the top-1 demo, *seg_name_map* is the raw
        ``{str(seg_id): name}`` mapping from the HDF5 attribute, and
        *viz_data* is a dict of kwargs for
        :meth:`~policy_abstraction.visualization.rerun.RerunVisualizer.log_retrieval_info`,
        and *top_episode* is the opened
        :class:`~policy_abstraction.analysis.demo_episode.DemoEpisode` for the
        top-1 demo -- handed on so the abstractor reuses this one rather than
        re-opening (and, for a real recording, re-segmenting) the same episode.
        Any component may be ``None`` / empty on failure.
    """
    from policy_abstraction.analysis.demo_episode import (
        build_demo_episode,
        extrinsic_disagreement,
    )
    from policy_abstraction.retrieval.visualization import render_retrieval_figure

    results: list[dict] = []
    viz_data: dict = {}

    # ---- Retrieval ----------------------------------------------------------
    try:
        results = retriever.retrieve(
            {"frames": [first_frame], "task": task_description}, k=k
        )
        query_grid = render_retrieval_figure(task_description, first_frame, results)
        retrieval_results_meta = [
            {
                "rank": i + 1,
                "task": r["task"],
                "source_kind": r["source_kind"],
                "source_path": r["source_path"],
                "demo_key": r["demo_key"],
                "score": float(r["score"]),
                "visual_score": float(r.get("visual_score", 0)),
                "text_score": float(r.get("text_score", 0)),
            }
            for i, r in enumerate(results)
        ]
        if recorder is not None:
            recorder.record(retrieval_figure=query_grid)
            recorder.update_metadata("retrieval_results", retrieval_results_meta)
        viz_data["query_grid"] = query_grid
        viz_data["retrieval_results"] = retrieval_results_meta
        logger.info(
            f"Retrieved {len(results)} demos. "
            f"Top: {results[0]['task']!r} (score={results[0]['score']:.3f})"
        )
    except Exception as exc:
        logger.error(f"Retrieval failed: {exc}")
        raise

    # ---- Demo analysis (VLM) on top-1 result --------------------------------
    if analyzer is None or not results:
        return results[0] if results else None, None, {}, viz_data, None

    top = results[0]
    seg_name_map: dict[str, str] = {}
    analysis = None
    try:
        top_episode = build_demo_episode(top, segmenter=segmenter)
        _check_extrinsic(top_episode, extrinsic_disagreement)
        cache_key = _analysis_cache_key(top["source_path"], top["demo_key"])
        analysis = analyzer.analyze(
            top_episode,
            top["task"],
            cache_key=cache_key,
            top_result=top,
            other_results=results[1:],
        )
        seg_name_map = {str(k): v for k, v in top_episode.seg_name_map().items()}

        if analysis.seg_index_image is not None:
            if recorder is not None:
                recorder.record(analysis_seg_image=analysis.seg_index_image)
            viz_data["seg_annotated_image"] = analysis.seg_index_image

        if analysis.scene_image is not None:
            if recorder is not None:
                recorder.record(vlm_scene_image=analysis.scene_image)
            viz_data["vlm_scene_image"] = analysis.scene_image

        if analysis.subtraj_image is not None:
            if recorder is not None:
                recorder.record(vlm_subtraj_image=analysis.subtraj_image)
            viz_data["vlm_subtraj_image"] = analysis.subtraj_image

        if analysis.pairs_selection_conversation is not None:
            conv = analysis.pairs_selection_conversation
            msgs = conv.messages
            user_images_b64: list[str] = []
            if len(msgs) > 1:
                for img_arr in msgs[1].images:
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    user_images_b64.append(
                        base64.b64encode(buf.getvalue()).decode("utf-8")
                    )
            vlm_conv_meta = {
                "model": conv.model,
                "system_prompt": msgs[0].text if msgs else "",
                "user_text": msgs[1].text if len(msgs) > 1 else "",
                "user_images_b64": user_images_b64,
                "assistant_response": msgs[2].text if len(msgs) > 2 else "",
                "cache_file": analysis.vlm_cache_file,
            }
            if recorder is not None:
                recorder.update_metadata("vlm_conversation", vlm_conv_meta)
            viz_data["vlm_conversation"] = vlm_conv_meta

        vlm_analysis_meta = {
            "retrieved_task": top["task"],
            "retrieved_source_path": top["source_path"],
            "retrieved_demo_key": top["demo_key"],
            "vlm_prompt_text": analysis.vlm_prompt_text,
            "vlm_response_json": analysis.vlm_response_json,
            "interacting_ids": sorted(analysis.interacting_ids),
            "objects": {str(k): v for k, v in analysis.objects.items()},
            "distance_pairs": [dataclasses.asdict(p) for p in analysis.distance_pairs],
        }
        if recorder is not None:
            recorder.update_metadata("vlm_analysis", vlm_analysis_meta)
        viz_data["vlm_analysis"] = vlm_analysis_meta
        logger.info(
            f"VLM analysis done. Distance pairs: "
            f"{[(p.from_entity, p.to_entity, p.label) for p in analysis.distance_pairs]}"
        )

        if analysis.iteration_log:
            iterative_meta = {
                "iterations": analysis.iteration_log,
                "selected_iteration": analysis.selected_iteration,
                "selected_subtraj_boundaries": (
                    [list(b) for b in analysis.selected_subtraj_boundaries]
                    if analysis.selected_subtraj_boundaries is not None
                    else None
                ),
            }
            if recorder is not None:
                recorder.update_metadata("iterative_analysis", iterative_meta)
            viz_data["iterative_analysis"] = iterative_meta
            logger.info(
                f"Iterative analysis: ran {len(analysis.iteration_log)} iteration(s); "
                f"selected iter {analysis.selected_iteration}."
            )
    except Exception as exc:
        logger.error(f"Demo analysis failed: {exc}")
        raise

    # ---- Multi-demo trajectory statistics -----------------------------------
    if analysis is not None and analysis.distance_pairs:
        pair_labels = [p.label for p in analysis.distance_pairs]
        other_trajs: list[np.ndarray] = []

        # Compute top-1 reference trajectory
        ref_traj = _compute_raw_traj(top_episode, analysis.distance_pairs, pair_labels)

        if ref_traj is not None:
            # Compute trajectories for demos 2..k
            for other_result in results[1:]:
                try:
                    other_episode = build_demo_episode(
                        other_result, segmenter=segmenter
                    )
                    # The other demo's seg_name_map, for entity remapping.
                    other_seg_map: dict[str, str] = {}
                    try:
                        other_seg_map = {
                            str(k): v for k, v in other_episode.seg_name_map().items()
                        }
                    except Exception as exc:
                        logger.debug(f"[multi-demo] Could not load seg_name_map: {exc}")

                    # Remap pairs if seg maps differ
                    if other_seg_map and other_seg_map != seg_name_map:
                        remapped_pairs = _remap_pairs_for_demo(
                            analysis.distance_pairs, seg_name_map, other_seg_map
                        )
                    else:
                        remapped_pairs = analysis.distance_pairs

                    if not remapped_pairs:
                        logger.debug(
                            f"[multi-demo] No remapped pairs for {other_result['demo_key']}; skipping."
                        )
                        continue

                    # Use only labels that survived remapping
                    remap_labels = [p.label for p in remapped_pairs]
                    traj = _compute_raw_traj(
                        other_episode, remapped_pairs, remap_labels
                    )

                    if traj is not None:
                        # If labels differ (some pairs dropped), pad/align columns
                        if remap_labels == pair_labels:
                            other_trajs.append(traj)
                        else:
                            # Fill missing columns with NaN
                            full_traj = np.full(
                                (traj.shape[0], len(pair_labels)),
                                np.nan,
                                dtype=np.float32,
                            )
                            for col, lbl in enumerate(remap_labels):
                                if lbl in pair_labels:
                                    full_traj[:, pair_labels.index(lbl)] = traj[:, col]
                            other_trajs.append(full_traj)

                except Exception as exc:
                    logger.warning(
                        f"[multi-demo] Failed to compute trajectory for "
                        f"{other_result.get('demo_key')}: {exc}"
                    )

            logger.info(
                f"[multi-demo] Computed trajectories for {len(other_trajs) + 1}/{len(results)} demos."
            )
            stats = compute_multi_demo_stats(ref_traj, other_trajs, pair_labels)
            viz_data["multi_demo_stats"] = stats
        else:
            logger.warning(
                "[multi-demo] Failed to compute top-1 reference trajectory; skipping stats."
            )

    return top, analysis, seg_name_map, viz_data, top_episode

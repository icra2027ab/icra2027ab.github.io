"""Monotonic Trajectory Alignment for matching live episode state to a reference rollout."""

from __future__ import annotations

import enum

import numpy as np
from loguru import logger


class AlignmentMode(str, enum.Enum):
    """How per-feature temporal alignment is performed."""

    UNIFIED = "unified"
    PER_FEATURE = "per_feature"
    MIXED = "mixed"


class MonotonicTrajectoryAligner:
    """Online greedy monotonic alignment of a live distance sequence to a reference trajectory.

    The aligner maintains a cursor ``aligned_t`` that only advances along the
    reference trajectory (monotonicity constraint).  At each call to
    :meth:`align` it searches a window of ``window`` frames ahead of the
    current cursor **within the active sub-trajectory** and moves to whichever
    reference frame minimises the MSE over the pair dimensions that are valid
    (non-NaN) for *both* the reference frame and the live observation.

    NaN values in the reference trajectory are preserved as-is; they are never
    used in distance computations.  If the live vector contains NaN for any
    pair, a warning is emitted and those dimensions are excluded from that
    step's alignment.

    When ``aligned_t`` approaches the end of the current sub-trajectory
    (within ``subtraj_end_threshold`` frames), the aligner also evaluates the
    start of the next sub-trajectory.  If the best score there is lower than
    the best in-segment score by a relative margin of ``subtraj_switch_margin``,
    the aligner switches to the next sub-trajectory.

    Parameters
    ----------
    ref_traj:
        ``[T_ref, n_pairs]`` float32 array of pre-computed reference distances.
        NaN values are expected and treated as "not observed at this timestep".
    pair_labels:
        List of ``n_pairs`` string labels in the same column order as
        *ref_traj* (used for logging / result packaging).
    window:
        Maximum number of frames to look ahead from the current cursor.
        ``None`` searches to the end of the current sub-trajectory.
        Default is ``10``.
    subtrajectories:
        Optional list of ``(start, end)`` index pairs (end exclusive) that
        partition the reference trajectory into sub-segments.  When provided,
        alignment is restricted to the active segment, and pair relevance is
        determined per-segment.
    subtraj_end_threshold:
        Number of frames before the end of the current sub-trajectory at
        which the aligner starts evaluating the next sub-trajectory as a
        potential switch target.  Default is ``10``.
    subtraj_switch_margin:
        Relative improvement (0–1) required for the aligner to switch to the
        next sub-trajectory.  The next segment must achieve a MSE that is
        lower than the current segment's best MSE by at least this fraction.
        E.g. ``0.1`` means "10 % lower".  Default is ``0.1``.
    subtraj_pair_map:
        Optional explicit mapping ``{subtraj_idx: [col_idx, ...]}`` that
        declares which distance-pair columns are relevant for each
        sub-trajectory.  When provided this takes priority over the
        automatic std-dev heuristic in :meth:`_build_col_masks`.
        Column indices not listed for a given sub-trajectory are excluded
        from that segment's alignment search.  Pairs absent from **all**
        sub-trajectories (e.g. cross-segment pairs) should be omitted from
        the map so they fall back to an all-True mask.
    normalize_distances:
        When ``True`` (default), distances are normalized per sub-trajectory
        before computing MSE.  For each distance-pair column *i*, the scale
        is defined as ``ref[start, i] - ref[end-1, i]`` (the signed change
        over the sub-trajectory) and the offset is ``ref[end-1, i]``.  This
        makes a normalized value of ``1`` at the start and ``0`` at the end
        of every sub-trajectory, so alignment is driven by *relative progress*
        rather than absolute distance.  Pairs whose scale is near-zero (constant
        distance) or whose start/end values are NaN fall back to no
        normalization (``scale=1, offset=0``).  When no sub-trajectories are
        configured, the full reference trajectory is used as the single segment.
    subtraj_time_weight:
        Weight for the temporal progress dimension added to the MSE.  For
        each candidate reference frame *t* in a segment ``[start, end)``, the
        time residual is ``(t - start) / duration - live_step / duration``
        where *duration* ``= end - start`` and *live_step* is the number of
        ``align()`` calls made since entering the current sub-trajectory.
        The weighted squared residual ``subtraj_time_weight * residual²`` is
        added to the numerator and ``subtraj_time_weight`` to the denominator
        of the per-frame MSE, treating execution pace as one extra alignment
        dimension.  Set to ``0.0`` to disable.  Default is ``1.0``.
    monotonic:
        When ``True`` the cursor can only advance along the reference
        trajectory — each call to :meth:`align` searches only frames at or
        after the current ``aligned_t``.  When ``False`` (default) the cursor
        is free to move in either direction within the current sub-trajectory
        (the ``window`` parameter is ignored), and sub-trajectory switching
        is bidirectional: the aligner may switch back to the previous
        sub-trajectory when it is a significantly better MSE fit and the
        current sub-trajectory's sub-goal has not yet been satisfied.
    max_advance:
        Maximum number of reference frames that ``aligned_t`` may advance
        per call to :meth:`align`.  When ``None`` (default) there is no
        cap beyond the window / sub-trajectory boundary.  Useful for
        preventing large jumps when the action chunk size is small relative
        to the reference trajectory length.
    alignment_mode:
        How per-feature temporal alignment is performed.  See
        :class:`AlignmentMode` for details.  Default is ``UNIFIED``.
    mixed_variation:
        Half-width of the search window around the unified cursor for
        ``MIXED`` mode.  Default is ``5``.
    temporal_factor_temperature:
        Temperature for the softmax-over-lag weighting used to compute
        per-pair temporal guidance factors.  Default is ``1.0``.
    temporal_factor_floor:
        Minimum weight floor so no relevant pair
        gets zero guidance.  Default is ``0.05``.
    sg_switch_persistence:
        Number of consecutive timesteps a sub-goal condition must remain
        satisfied (for forward switching) or unsatisfied (for backward
        switching) before the aligner acts on it.  This prevents transient
        sub-goal signals from triggering premature switches.  Default is ``5``.
    """

    def __init__(
        self,
        ref_traj: np.ndarray,
        pair_labels: list[str],
        window: int | None = 10,
        subtrajectories: list[tuple[int, int]] | None = None,
        subtraj_end_threshold: int = 15,
        subtraj_switch_margin: float = 0.1,
        subtraj_pair_map: dict[int, list[int]] | None = None,
        normalize_distances: bool = True,
        subtraj_time_weight: float = 1.0,
        monotonic: bool = False,
        max_advance: int | float | None = None,
        alignment_mode: AlignmentMode = AlignmentMode.UNIFIED,
        mixed_variation: int = 5,
        temporal_factor_temperature: float = 100.0,
        temporal_factor_floor: float = 0.05,
        sg_switch_persistence: int = 5,
    ) -> None:
        self._ref = ref_traj.astype(np.float32)  # raw, NaN preserved
        self._labels = pair_labels
        self._window = window
        self._T_ref = ref_traj.shape[0]
        self._aligned_t: int = 0
        self._normalize_distances = normalize_distances
        self._subtraj_time_weight = subtraj_time_weight
        self._monotonic = monotonic
        self._max_advance = int(max_advance) if max_advance is not None else None

        # Alignment mode parameters
        self._alignment_mode = alignment_mode
        self._mixed_variation = mixed_variation
        self._temporal_factor_temperature = temporal_factor_temperature
        self._temporal_factor_floor = temporal_factor_floor

        n_pairs = ref_traj.shape[1] if ref_traj.ndim > 1 else 1
        self._aligned_ts: np.ndarray = np.zeros(n_pairs, dtype=np.int32)
        self._per_feature_mse: np.ndarray | None = None
        self._guidance_weights: np.ndarray = np.zeros(n_pairs, dtype=np.float32)

        self._subtrajectories: list[tuple[int, int]] = (
            list(subtrajectories) if subtrajectories else []
        )
        self._current_subtraj_idx: int = 0
        self._subtraj_live_steps: int = 0  # steps taken in the current sub-trajectory
        self._current_mse: float = float("inf")
        self._last_search_start: int = 0
        self._last_search_end: int = 0
        self._subtraj_end_threshold = subtraj_end_threshold
        self._subtraj_switch_margin = subtraj_switch_margin
        self._subtraj_pair_map: dict[int, list[int]] | None = subtraj_pair_map
        self._sg_switch_persistence = sg_switch_persistence
        self._cur_sg_true_count: int = 0
        self._prev_sg_false_count: int = 0

        # Pre-compute per-subtrajectory relevant column masks (non-constant columns).
        self._subtraj_col_masks: list[np.ndarray] = self._build_col_masks()
        # Pre-compute per-subtrajectory normalization parameters.
        self._norm_scales: list[np.ndarray]
        self._norm_offsets: list[np.ndarray]
        self._norm_scales, self._norm_offsets = self._build_norm_params()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def aligned_t(self) -> int:
        """Current aligned reference timestep index."""
        return self._aligned_t

    @property
    def T_ref(self) -> int:
        """Length of the reference trajectory."""
        return self._T_ref

    @property
    def current_mse(self) -> float:
        """MSE at the last aligned reference frame (updated by :meth:`align`)."""
        return self._current_mse

    @property
    def current_subtraj_idx(self) -> int | None:
        """Index of the current sub-trajectory, or ``None`` if none configured."""
        if not self._subtrajectories:
            return None
        return self._current_subtraj_idx

    @property
    def last_search_window(self) -> tuple[int, int]:
        """The ``(start, end)`` search window used by the most recent :meth:`align` call."""
        return (self._last_search_start, self._last_search_end)

    @property
    def aligned_ts(self) -> np.ndarray:
        """Per-pair aligned reference timestep indices ``[n_pairs]`` int32."""
        return self._aligned_ts

    @property
    def guidance_weights(self) -> np.ndarray:
        """Per-pair temporal guidance weight array ``[n_pairs]`` float32."""
        return self._guidance_weights

    @property
    def per_feature_mse(self) -> np.ndarray | None:
        """Per-pair MSE at each per-feature aligned_t ``[n_pairs]`` float32, or ``None``."""
        return self._per_feature_mse

    def align_all(
        self,
        live_vec: np.ndarray,
        subtraj_sg_satisfied: dict[int, bool] | None = None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Advance the unified cursor and compute per-pair alignment indices.

        Delegates to :meth:`align` for the unified cursor (which handles
        sub-trajectory switching), then computes per-pair ``aligned_ts``
        based on the configured :class:`AlignmentMode`:

        * **UNIFIED** — all entries equal the unified cursor.
        * **PER_FEATURE** — each pair is independently aligned within the
          same search window used by the unified pass.
        * **MIXED** — each pair is independently aligned within a window
          of ``±mixed_variation`` frames around the unified cursor.

        Returns
        -------
        (aligned_t, aligned_ts, guidance_weights)
            ``aligned_t``:  int — unified cursor (for sub-traj management,
            pose guidance, visualisation).
            ``aligned_ts``: ``[n_pairs]`` int32 — per-pair cursors.
            ``guidance_weights``: ``[n_pairs]`` float32 — per-pair guidance
            weights (uniform ``1.0`` in UNIFIED mode).
        """
        aligned_t = self.align(live_vec, subtraj_sg_satisfied=subtraj_sg_satisfied)
        n_pairs = self._ref.shape[1] if self._ref.ndim > 1 else 1

        if self._alignment_mode == AlignmentMode.UNIFIED:
            self._aligned_ts = np.full(n_pairs, aligned_t, dtype=np.int32)
            self._per_feature_mse = None
            # Start with uniform weights, then zero out irrelevant features.
            weights = np.ones(n_pairs, dtype=np.float32)
            if self._subtrajectories and self._subtraj_col_masks:
                col_mask = self._subtraj_col_masks[self._current_subtraj_idx]
                weights = np.where(col_mask, weights, 0.0)
            n_relevant = np.count_nonzero(weights)
            if n_relevant > 0:
                weights[weights > 0] = 1.0 / n_relevant
            self._guidance_weights = weights
        else:
            if self._alignment_mode == AlignmentMode.PER_FEATURE:
                pf_start, pf_end = self.last_search_window
            else:  # MIXED
                pf_start = max(0, aligned_t - self._mixed_variation)
                pf_end = min(self._T_ref, aligned_t + self._mixed_variation + 1)

            pf_aligned_t, pf_mse = self.align_per_feature(live_vec, pf_start, pf_end)
            if (
                self._alignment_mode == AlignmentMode.MIXED
                and self._max_advance is not None
            ):
                pf_aligned_t = np.minimum(
                    pf_aligned_t, self._aligned_ts + self._max_advance
                ).astype(np.int32)
            self._aligned_ts = pf_aligned_t
            self._per_feature_mse = pf_mse
            self._guidance_weights = self._compute_guidance_weights(pf_aligned_t)

        return aligned_t, self._aligned_ts, self._guidance_weights

    def align(
        self,
        live_vec: np.ndarray,
        subtraj_sg_satisfied: dict[int, bool] | None = None,
    ) -> int:
        """Advance the cursor to best match *live_vec*.

        Only dimensions that are non-NaN in **both** *live_vec* and the
        candidate reference frame contribute to the MSE.  A warning is logged
        whenever *live_vec* contains NaN values.

        Parameters
        ----------
        live_vec:
            ``[n_pairs]`` float32 array of distances from the current live
            observation.  May contain NaN for pairs whose entities were not
            visible in the current point cloud.
        subtraj_sg_satisfied:
            Mapping from sub-trajectory index to whether its sub-goal
            constraints are currently satisfied.  Only used when
            ``monotonic=False`` and sub-trajectories are configured.
            Sub-trajectories absent from the dict are treated as ``None``
            (unknown).

            For the **current** sub-trajectory:

            * ``True``  — sub-goal satisfied; forward switching is preferred
              (current subtraj's threshold is relaxed) and backward switching
              is suppressed.
            * ``False`` — sub-goal not yet satisfied; backward switching is
              allowed when the previous sub-trajectory is a better MSE fit.

            For the **previous** sub-trajectory:

            * ``True``  — sub-goal still satisfied; backward switching is
              blocked (we shouldn't undo completed work).
            * ``False`` — sub-goal no longer satisfied; backward switching is
              allowed (e.g. the robot dropped the object).

        Returns
        -------
        int
            Updated ``aligned_t`` in ``[0, T_ref - 1]``.
        """
        if self._T_ref == 0:
            return 0

        live_nan_mask = np.isnan(live_vec)
        if live_nan_mask.any():
            nan_labels = [self._labels[i] for i in np.where(live_nan_mask)[0]]
            logger.warning(
                f"[MTA] Live distance vector has NaN for pairs {nan_labels}; "
                "those dimensions will be excluded from alignment."
            )

        # ---- No sub-trajectory info: windowed greedy search -----------------
        if not self._subtrajectories:
            if self._monotonic:
                search_start = self._aligned_t
                search_end = (
                    self._T_ref
                    if self._window is None
                    else min(self._aligned_t + self._window, self._T_ref)
                )
            else:
                search_start = 0
                search_end = self._T_ref
            self._last_search_start = search_start
            self._last_search_end = search_end
            all_cols = np.ones(self._ref.shape[1], dtype=bool)
            ns = self._norm_scales[0] if self._normalize_distances else None
            no = self._norm_offsets[0] if self._normalize_distances else None
            best_t, best_mse = self._best_in_range(
                search_start,
                search_end,
                live_vec,
                all_cols,
                ns,
                no,
                live_subtraj_step=self._subtraj_live_steps,
                subtraj_start=0,
                subtraj_duration=self._T_ref,
            )
            if self._max_advance is not None:
                best_t = min(best_t, self._aligned_t + self._max_advance)
            self._aligned_t = min(best_t, self._T_ref - 1)
            self._current_mse = best_mse
            self._subtraj_live_steps += 1
            return self._aligned_t

        # ---- Sub-trajectory-aware alignment ---------------------------------
        cur_idx = self._current_subtraj_idx
        cur_start, cur_end = self._subtrajectories[cur_idx]
        next_idx = cur_idx + 1

        # Resolve per-subtraj satisfaction from the dict (None = unknown).
        _sg = subtraj_sg_satisfied or {}
        cur_sg = _sg.get(cur_idx)  # current subtraj's sub-goal status
        prev_sg = _sg.get(cur_idx - 1) if cur_idx > 0 else None

        # Update persistence counters for sub-goal switching hysteresis.
        if cur_sg is True:
            self._cur_sg_true_count += 1
        else:
            self._cur_sg_true_count = 0

        if prev_sg is False or (prev_sg is None and cur_idx > 0):
            self._prev_sg_false_count += 1
        else:
            self._prev_sg_false_count = 0

        # Mandatory switch: cursor has reached the end of the current segment.
        # Only enforced in monotonic mode (non-monotonic cursor can move freely
        # within the segment and may not advance to the boundary).
        if (
            self._monotonic
            and self._aligned_t >= cur_end - 1
            and next_idx < len(self._subtrajectories)
        ):
            logger.debug(
                f"[MTA] Mandatory switch sub-trajectory {cur_idx} → {next_idx} "
                f"(cursor at boundary {self._aligned_t}/{cur_end - 1})"
            )
            self._current_subtraj_idx = next_idx
            self._subtraj_live_steps = 0
            self._cur_sg_true_count = 0
            self._prev_sg_false_count = 0
            cur_idx = next_idx
            _, cur_end = self._subtrajectories[cur_idx]
            next_idx = cur_idx + 1

        # In monotonic mode the search starts at the current cursor and is
        # capped by the window.  In non-monotonic mode the full sub-trajectory
        # is always searched (window ignored).
        cur_start_idx, _ = self._subtrajectories[cur_idx]
        if self._monotonic:
            search_start = self._aligned_t
            search_end = (
                cur_end
                if self._window is None
                else min(self._aligned_t + self._window, cur_end)
            )
        else:
            search_start = cur_start_idx
            search_end = cur_end

        self._last_search_start = search_start
        self._last_search_end = search_end
        cur_duration = cur_end - cur_start_idx
        cur_mask = self._subtraj_col_masks[cur_idx]
        cur_ns = self._norm_scales[cur_idx] if self._normalize_distances else None
        cur_no = self._norm_offsets[cur_idx] if self._normalize_distances else None
        best_t, best_mse = self._best_in_range(
            search_start,
            search_end,
            live_vec,
            cur_mask,
            cur_ns,
            cur_no,
            live_subtraj_step=self._subtraj_live_steps,
            subtraj_start=cur_start_idx,
            subtraj_duration=cur_duration,
        )

        # ---- Backward switch (non-monotonic only) ---------------------------
        # Allowed only when:
        #   - current sub-goal is NOT satisfied (cur_sg is not True)
        #   - previous sub-goal is NOT satisfied (prev_sg is not True),
        #     i.e. the condition that was achieved has been undone (e.g. object dropped)
        did_backward_switch = False
        if (
            not self._monotonic
            and cur_idx > 0
            and cur_sg is not True
            and prev_sg is not True
        ):
            prev_idx = cur_idx - 1
            prev_start, prev_end = self._subtrajectories[prev_idx]
            prev_mask = self._subtraj_col_masks[prev_idx]
            prev_ns = self._norm_scales[prev_idx] if self._normalize_distances else None
            prev_no = (
                self._norm_offsets[prev_idx] if self._normalize_distances else None
            )
            prev_duration = prev_end - prev_start

            prev_best_t, prev_mse = self._best_in_range(
                prev_start,
                prev_end,
                live_vec,
                prev_mask,
                prev_ns,
                prev_no,
                live_subtraj_step=self._subtraj_live_steps,
                subtraj_start=prev_start,
                subtraj_duration=prev_duration,
            )

            if self._prev_sg_false_count >= self._sg_switch_persistence:
                logger.debug(
                    f"[MTA] Backward switch sub-trajectory {cur_idx} → {prev_idx} "
                    f"(mse {best_mse:.4f} → {prev_mse:.4f}, "
                    f"sg_satisfied={subtraj_sg_satisfied})"
                )
                self._current_subtraj_idx = prev_idx
                self._subtraj_live_steps = 0
                self._cur_sg_true_count = 0
                self._prev_sg_false_count = 0
                best_t = prev_best_t
                best_mse = prev_mse
                # Update loop variables so the forward-switch check below uses
                # the new current segment.
                cur_idx = prev_idx
                cur_start_idx = prev_start
                cur_end = prev_end
                cur_duration = prev_duration
                cur_mask = prev_mask
                cur_ns = prev_ns
                cur_no = prev_no
                next_idx = cur_idx + 1
                search_start = prev_start
                search_end = prev_end
                did_backward_switch = True

        # ---- Check for forward sub-trajectory transition --------------------
        # Use the union of both masks so the MSE comparison is over the same
        # feature dimensions (comparing cur_mask MSE vs nxt_mask MSE would be
        # meaningless since the two masks may cover completely different pairs).
        #
        # In non-monotonic mode with a satisfied sub-goal, skip the distance
        # threshold so the forward switch can happen anywhere in the segment.
        frames_to_end = cur_end - self._aligned_t
        sg_forces_forward = (not self._monotonic) and (
            self._cur_sg_true_count >= self._sg_switch_persistence
        )
        within_threshold = frames_to_end <= self._subtraj_end_threshold

        if (
            not did_backward_switch
            and (within_threshold or sg_forces_forward)
            and next_idx < len(self._subtrajectories)
        ):
            nxt_start, nxt_end = self._subtrajectories[next_idx]
            nxt_search_end = (
                nxt_end
                if (self._window is None or not self._monotonic)
                else min(nxt_start + self._window, nxt_end)
            )
            nxt_mask = self._subtraj_col_masks[next_idx]
            nxt_ns = self._norm_scales[next_idx] if self._normalize_distances else None
            nxt_no = self._norm_offsets[next_idx] if self._normalize_distances else None
            combined_mask = cur_mask | nxt_mask

            _, cur_mse_combined = self._best_in_range(
                search_start,
                search_end,
                live_vec,
                combined_mask,
                cur_ns,
                cur_no,
                live_subtraj_step=self._subtraj_live_steps,
                subtraj_start=cur_start_idx,
                subtraj_duration=cur_duration,
            )
            nxt_best_t, nxt_mse_combined = self._best_in_range(
                nxt_start,
                nxt_search_end,
                live_vec,
                combined_mask,
                nxt_ns,
                nxt_no,
                live_subtraj_step=0,  # would be step 0 in the new sub-trajectory
                subtraj_start=nxt_start,
                subtraj_duration=nxt_end - nxt_start,
            )

            # When the sub-goal is confirmed satisfied, require only a smaller
            # relative improvement to trigger the forward switch.
            if sg_forces_forward:
                logger.debug(
                    f"[MTA] {'SG-driven ' if sg_forces_forward else ''}forward switch "
                    f"sub-trajectory {cur_idx} → {next_idx} "
                    f"(combined mse {cur_mse_combined:.4f} → {nxt_mse_combined:.4f})"
                )
                self._current_subtraj_idx = next_idx
                self._subtraj_live_steps = 0
                self._cur_sg_true_count = 0
                self._prev_sg_false_count = 0
                best_t = nxt_best_t

        # ---- Apply max_advance cap ----------------------------------------
        if self._max_advance is not None:
            best_t = min(best_t, self._aligned_t + self._max_advance)

        self._aligned_t = min(best_t, self._T_ref - 1)
        self._current_mse = best_mse
        self._subtraj_live_steps += 1
        return self._aligned_t

    def ref_distances_at(self, t: int | None = None) -> dict[str, float]:
        """Return the reference distances at timestep *t* (default: current cursor).

        Parameters
        ----------
        t:
            Reference timestep index.  Defaults to ``self.aligned_t``.

        Returns
        -------
        dict mapping pair label → reference distance value (may be NaN).
        """
        idx = self._aligned_t if t is None else t
        idx = max(0, min(idx, self._T_ref - 1))
        row = self._ref[idx]
        return {
            label: float(val) for label, val in zip(self._labels, row, strict=False)
        }

    def align_per_feature(
        self,
        live_vec: np.ndarray,
        search_start: int,
        search_end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the best-matching reference frame for each distance pair independently.

        Unlike :meth:`align`, this does **not** update the cursor or any internal
        state.  It is a stateless per-column argmin-MSE query over
        ``[search_start, search_end)``.

        Parameters
        ----------
        live_vec:
            ``[n_pairs]`` float32 array of current live distances.
        search_start:
            First reference frame to evaluate (inclusive).
        search_end:
            Last reference frame (exclusive).

        Returns
        -------
        (per_feature_t, per_feature_mse)
            ``per_feature_t``:   int array ``[n_pairs]`` — best frame per column.
            ``per_feature_mse``: float32 array ``[n_pairs]`` — MSE at best frame.
        """
        n_pairs = self._ref.shape[1]
        per_feature_t = np.full(n_pairs, self._aligned_t, dtype=np.int32)
        per_feature_mse = np.full(n_pairs, np.inf, dtype=np.float32)

        if search_start >= search_end:
            return per_feature_t, per_feature_mse

        # Determine normalization params for the active sub-trajectory
        subtraj_idx = self._current_subtraj_idx if self._subtrajectories else 0
        norm_scale = (
            self._norm_scales[subtraj_idx] if self._normalize_distances else None
        )
        norm_offset = (
            self._norm_offsets[subtraj_idx] if self._normalize_distances else None
        )

        ref_seg = self._ref[search_start:search_end].copy()  # [W, n_pairs]
        live = live_vec.copy()

        ref_nan = np.isnan(ref_seg)
        live_nan = np.isnan(live)

        # Fill NaN before arithmetic
        ref_filled = np.where(ref_nan, 0.0, ref_seg)
        live_filled = np.where(live_nan, 0.0, live)

        # Apply normalization
        if norm_scale is not None and norm_offset is not None:
            ref_filled = (ref_filled - norm_offset[None, :]) / norm_scale[None, :]
            live_filled = (live_filled - norm_offset) / norm_scale

        # Per-column squared differences  [W, n_pairs]
        diffs = ref_filled - live_filled[None, :]
        sq = diffs * diffs

        # Mask out invalid entries (NaN in ref or live)
        valid = ~ref_nan & ~live_nan[None, :]  # [W, n_pairs]
        sq = np.where(valid, sq, np.inf)

        # Per-column argmin
        best_local = np.argmin(sq, axis=0)  # [n_pairs]
        for i in range(n_pairs):
            if valid[:, i].any():
                per_feature_t[i] = search_start + int(best_local[i])
                per_feature_mse[i] = float(sq[best_local[i], i])

        return per_feature_t, per_feature_mse

    def _compute_guidance_weights(
        self,
        per_feature_t: np.ndarray,
        base_factor: float = 1.0,
    ) -> np.ndarray:
        """Compute per-pair guidance weights from temporal alignment spread.

        Features that are more "behind" (lower ``aligned_t``) get higher weight
        so guidance pulls harder on the lagging dimensions.

        Uses softmax over ``(max_t - t_i) / temperature``, scaled so the sum
        of weights is 1.

        Only features relevant to the current sub-trajectory (per
        ``_subtraj_col_masks``) participate in the softmax; irrelevant features
        receive 0.
        """
        n = len(per_feature_t)
        valid = np.isfinite(per_feature_t.astype(float))

        # Restrict to features relevant to the current sub-trajectory.
        if self._subtrajectories and self._subtraj_col_masks:
            col_mask = self._subtraj_col_masks[self._current_subtraj_idx]
            relevant = valid & col_mask
        else:
            relevant = np.zeros_like(valid, dtype=bool)

        if not relevant.any() or n == 0:
            return np.full(n, 0, dtype=np.float32)

        max_t = float(np.max(per_feature_t[relevant]))
        lag = max_t - per_feature_t.astype(np.float64) + 10  # [n], >= 0

        temp = max(self._temporal_factor_temperature, 1e-6)
        # Softmax in the shifted form: subtracting the largest logit changes
        # nothing mathematically but keeps exp() from overflowing to inf when
        # the lag is large relative to the temperature, which turned the
        # weights into inf/inf = NaN and, downstream, the whole guided action
        # chunk into NaN.
        logits = lag / temp
        logits = logits - np.max(logits[relevant])
        exp_lag_masked = np.where(relevant, np.exp(logits), 0.0)
        total = exp_lag_masked.sum()
        if not np.isfinite(total) or total <= 0.0:
            return np.full(n, 0, dtype=np.float32)
        weights = exp_lag_masked / total  # softmax over relevant only

        # Assign zero weight to irrelevant features.
        weights = np.where(relevant, weights, 0)

        return weights.astype(np.float32)

    def reset(self) -> None:
        """Reset the cursor and sub-trajectory index to the beginning."""
        self._aligned_t = 0
        self._current_subtraj_idx = 0
        self._subtraj_live_steps = 0
        self._current_mse = float("inf")
        self._cur_sg_true_count = 0
        self._prev_sg_false_count = 0
        n_pairs = self._ref.shape[1] if self._ref.ndim > 1 else 1
        self._aligned_ts = np.zeros(n_pairs, dtype=np.int32)
        self._per_feature_mse = None
        self._guidance_weights = np.zeros(n_pairs, dtype=np.float32)

    def progress(self) -> float:
        """Fractional completion in ``[0.0, 1.0]`` (``aligned_t / (T_ref - 1)``)."""
        return self._aligned_t / max(self._T_ref - 1, 1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _best_in_range(
        self,
        start: int,
        end: int,
        live_vec: np.ndarray,
        col_mask: np.ndarray,
        norm_scale: np.ndarray | None = None,
        norm_offset: np.ndarray | None = None,
        live_subtraj_step: int | None = None,
        subtraj_start: int | None = None,
        subtraj_duration: int | None = None,
    ) -> tuple[int, float]:
        """Return ``(best_t, best_mse)`` over reference frames ``[start, end)``.

        For each candidate frame the distance is computed only over dimensions
        where *both* the reference value and the live value are non-NaN **and**
        the column is selected by *col_mask*.  MSE is normalised by the number
        of valid dimensions so scores are comparable across frames and segments.

        If a candidate frame has no valid dimensions in common with *live_vec*,
        it is assigned ``inf`` MSE and will not be selected.

        When *norm_scale* and *norm_offset* are provided, both reference and
        live values are normalized as ``(x - norm_offset) / norm_scale`` before
        computing differences.  Invalid (NaN) dimensions are zeroed out by the
        *valid* mask regardless, so normalization of those positions is harmless.

        When *live_subtraj_step*, *subtraj_start*, and *subtraj_duration* are
        all provided (and ``self._subtraj_time_weight > 0``), execution pace is
        included as one extra weighted dimension.  For candidate frame *t*::

            time_residual = (t - subtraj_start) / subtraj_duration
                          - live_subtraj_step / subtraj_duration

        The weighted squared residual ``w * time_residual²`` is added to the
        numerator and *w* to the denominator of the per-frame MSE so that time
        is treated as one more alignment feature with equal (or custom) weight.
        """
        if start >= end:
            return self._aligned_t, float("inf")

        ref_seg = self._ref[start:end]  # [W, n_pairs]
        live_nan = np.isnan(live_vec)  # [n_pairs]
        ref_nan = np.isnan(ref_seg)  # [W, n_pairs]

        # Valid: col_mask AND ref non-NaN AND live non-NaN
        valid = col_mask[None, :] & ~ref_nan & ~live_nan[None, :]  # [W, n_pairs]

        # Fill NaN before subtraction so invalid entries are numerically zero;
        # the valid mask will then zero out their squared contribution.
        ref_filled = np.where(ref_nan, 0.0, ref_seg)
        live_filled = np.where(live_nan, 0.0, live_vec)

        # Apply per-segment normalization: map [ref_start, ref_end] → [1, 0].
        if norm_scale is not None and norm_offset is not None:
            ref_filled = (ref_filled - norm_offset[None, :]) / norm_scale[None, :]
            live_filled = (live_filled - norm_offset) / norm_scale

        diffs = (ref_filled - live_filled[None, :]) * valid  # zero out invalid dims
        sum_sq = (diffs * diffs).sum(axis=1)  # [W]
        n_valid = valid.sum(axis=1).astype(np.float32)  # [W]

        # ---- Optional time-progress dimension --------------------------------
        tw = self._subtraj_time_weight
        if (
            tw > 0
            and live_subtraj_step is not None
            and subtraj_start is not None
            and subtraj_duration is not None
            and subtraj_duration > 0
        ):
            candidate_t = np.arange(start, end, dtype=np.float32)  # [W]
            ref_time_norm = (candidate_t - subtraj_start) / subtraj_duration
            live_time_norm = live_subtraj_step / subtraj_duration
            time_residual = ref_time_norm - live_time_norm  # [W]
            sum_sq = sum_sq + tw * (time_residual**2)
            n_valid = n_valid + tw

        # Frames with no valid dimensions get inf so they're never chosen.
        with np.errstate(invalid="ignore", divide="ignore"):
            mse_vals = np.where(n_valid > 0, sum_sq / n_valid, np.inf)

        best_local = int(np.argmin(mse_vals))
        return start + best_local, float(mse_vals[best_local])

    def _build_norm_params(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute per-segment normalization scales and offsets.

        For each sub-trajectory (or the full trajectory when none are configured)
        with frame range ``[start, end)``:

        * ``offset[i] = ref[end-1, i]``  (target value at the end of the segment)
        * ``scale[i]  = ref[start, i] - ref[end-1, i]``  (total signed change)

        Normalized value = ``(x - offset) / scale``, giving ``1`` at the start
        and ``0`` at the end of each segment.

        Pairs where either boundary is NaN, or whose scale is near-zero, fall
        back to ``scale=1, offset=0`` (raw distances, no normalization).
        """
        segs = self._subtrajectories if self._subtrajectories else [(0, self._T_ref)]

        scales: list[np.ndarray] = []
        offsets: list[np.ndarray] = []
        for start, end in segs:
            ref_s = self._ref[start].copy()
            ref_e = self._ref[max(start, end - 1)].copy()
            scale = ref_s - ref_e
            # Pairs with NaN boundary or near-zero scale: no normalization
            invalid = np.isnan(ref_s) | np.isnan(ref_e) | (np.abs(scale) < 0.02)
            scale = np.where(invalid, 1.0, scale).astype(np.float32)
            offset = np.where(invalid, 0.0, ref_e).astype(np.float32)
            scales.append(scale)
            offsets.append(offset)
        return scales, offsets

    def _build_col_masks(self) -> list[np.ndarray]:
        """Build a boolean column mask for each sub-trajectory.

        If *subtraj_pair_map* was provided at construction, each sub-trajectory
        uses exactly the columns listed there.  Columns not listed for any
        sub-trajectory are assumed to be global (included in every mask).

        Without *subtraj_pair_map*, falls back to a std-dev heuristic: a
        column is considered *relevant* for a segment if its reference values
        have non-zero standard deviation (ignoring NaN) within that segment.

        Falls back to an all-True mask when no sub-trajectories are configured
        or a segment is too short.
        """
        n_pairs = self._ref.shape[1] if self._ref.ndim > 1 else 1

        if not self._subtrajectories:
            return [np.ones(n_pairs, dtype=bool)]

        # ------------------------------------------------------------------
        # Explicit mapping path
        # ------------------------------------------------------------------
        if self._subtraj_pair_map is not None:
            # Columns that appear in *any* sub-trajectory entry.
            all_mapped_cols: set[int] = set()
            for cols in self._subtraj_pair_map.values():
                all_mapped_cols.update(cols)

            masks = []
            for s_idx in range(len(self._subtrajectories)):
                mask = np.zeros(n_pairs, dtype=bool)
                # Columns explicitly listed for this sub-trajectory.
                for col in self._subtraj_pair_map.get(s_idx, []):
                    if 0 <= col < n_pairs:
                        mask[col] = True
                # Columns not listed in *any* sub-trajectory are global —
                # include them in every segment's mask.
                for col in range(n_pairs):
                    if col not in all_mapped_cols:
                        mask[col] = True
                if not mask.any():
                    mask = np.ones(n_pairs, dtype=bool)  # safety fallback
                masks.append(mask)
            return masks

        # ------------------------------------------------------------------
        # Heuristic path (std-dev fallback)
        # ------------------------------------------------------------------
        masks = []
        for start, end in self._subtrajectories:
            seg = self._ref[start:end]  # [L, n_pairs]
            if seg.shape[0] < 2:
                masks.append(np.ones(n_pairs, dtype=bool))
                continue
            std = np.nanstd(seg, axis=0)  # ignore NaN when computing std
            mask = std > 0
            if not mask.any():
                mask = np.ones(n_pairs, dtype=bool)  # fallback: all columns
            masks.append(mask)
        return masks

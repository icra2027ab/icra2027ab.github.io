"""Spatial consistency scoring for point correspondences.

Given N matched point pairs (reference -> observed), scores each correspondence
by how well it agrees with the geometric transform implied by the best matches.
Uses the top-k highest-confidence matches as anchors, fits a single affine
transform, and scores remaining points by their residual from that transform.
"""

from __future__ import annotations

import numpy as np


def _fit_affine(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an affine transform dst = A @ src + b via least-squares.

    Args:
        src: ``(M, D)`` source points.
        dst: ``(M, D)`` destination points.

    Returns:
        A: ``(D, D)`` linear part.
        b: ``(D,)`` translation.
    """
    # Build system: [src, 1] @ W = dst  where W = [[A^T], [b^T]]
    M, D = src.shape
    ones = np.ones((M, 1), dtype=src.dtype)
    X = np.concatenate([src, ones], axis=1)  # (M, D+1)
    W, _, _, _ = np.linalg.lstsq(X, dst, rcond=None)  # (D+1, D)
    A = W[:D, :].T  # (D, D)
    b = W[D, :]  # (D,)
    return A, b


def score_spatial_consistency(
    ref_pts: np.ndarray,
    obs_pts: np.ndarray,
    *,
    n_anchors: int | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """Score each correspondence by spatial consistency with top matches.

    Points must be sorted by feature-similarity score in **descending** order
    (best match first). The first ``n_anchors`` points are treated as trusted
    anchors that define the affine transform; all points are then scored by
    their residual from that transform.

    Args:
        ref_pts:   ``(N, D)`` reference coordinates (D = 2 or 3).
        obs_pts:   ``(N, D)`` observed coordinates, same ordering as *ref_pts*.
        n_anchors: Number of top matches to use as anchors.
                   Defaults to ``D + 1`` (3 for 2-D, 4 for 3-D).
        sigma:     Gaussian decay parameter for converting residuals to scores.
                   If *None*, estimated as ``max(median(anchor_residuals), 1e-6) * 3``.

    Returns:
        ``(N,)`` float64 array of scores in ``[0, 1]``.
        Anchor points always receive a score of 1.0.
    """
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    obs_pts = np.asarray(obs_pts, dtype=np.float64)

    N, D = ref_pts.shape
    if obs_pts.shape != (N, D):
        raise ValueError(
            f"Shape mismatch: ref_pts {ref_pts.shape} vs obs_pts {obs_pts.shape}"
        )

    min_sample = D + 1
    if n_anchors is None:
        n_anchors = min_sample
    if n_anchors < min_sample:
        raise ValueError(
            f"Need at least {min_sample} anchors for {D}-D affine, got {n_anchors}"
        )
    if N < n_anchors:
        raise ValueError(f"Not enough points ({N}) for {n_anchors} anchors")

    # Fit affine from anchors
    A, b = _fit_affine(ref_pts[:n_anchors], obs_pts[:n_anchors])

    # Compute residuals for all points
    predicted = ref_pts @ A.T + b  # (N, D)
    residuals = np.linalg.norm(obs_pts - predicted, axis=1)  # (N,)

    # Auto-estimate sigma from anchor fit quality + overall scale
    if sigma is None:
        anchor_res = residuals[:n_anchors]
        # Use a scale-aware default: combine anchor residual with point spread
        ref_spread = np.linalg.norm(ref_pts.max(axis=0) - ref_pts.min(axis=0))
        sigma = float(max(np.median(anchor_res), ref_spread * 0.01, 1e-6) * 3.0)

    scores = np.exp(-0.5 * (residuals / sigma) ** 2)
    # Anchors are trusted — force score to 1.0
    scores[:n_anchors] = 1.0

    return scores

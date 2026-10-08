"""PnP with outliers, and the checks that judge it."""

import numpy as np
import pytest

from policy_abstraction.calibration.geometry import invert, pose_to_matrix, project
from policy_abstraction.calibration.solve import (
    depth_agreement,
    depth_at,
    judge,
    pose_difference,
    solve_extrinsic,
)

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def _scene(n=60, seed=0):
    """*n* TCP positions that all land inside a 640x480 image."""
    rng = np.random.default_rng(seed)
    T_cam_base = invert(pose_to_matrix([0.3, -0.8, 0.6, -2.0, 0.3, 0.2]))
    m = 20 * n
    P = np.column_stack(
        [
            rng.uniform(-0.6, -0.1, m),
            rng.uniform(-0.4, 0.3, m),
            rng.uniform(0.0, 0.5, m),
        ]
    )
    uv, z = project(K, T_cam_base, P)
    inside = (
        np.all(np.isfinite(uv), axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < 640)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < 480)
    )
    keep = np.flatnonzero(inside)[:n]
    assert len(keep) == n, "scene too small for the requested count"
    return T_cam_base, P[keep], uv[keep], z[keep]


def test_recovers_the_pose_from_clean_data():
    T, P, uv, _ = _scene()
    sol = solve_extrinsic(P, uv, K)
    d = pose_difference(sol.T_cam_parent, T)
    assert d.translation_m < 1e-4 and d.rotation_deg < 1e-3
    np.testing.assert_allclose(
        sol.T_parent_cam @ sol.T_cam_parent, np.eye(4), atol=1e-9
    )
    assert sol.n_used == 60


def test_rejects_outliers_and_reports_them():
    T, P, uv, _ = _scene()
    uv = uv + np.random.default_rng(1).normal(0, 0.5, uv.shape)
    bad = [3, 17, 40]
    uv[bad] += 80.0  # the tracker jumped to the other finger
    sol = solve_extrinsic(P, uv, K, ransac_px=4.0)
    assert not sol.used[bad].any()
    assert sol.n_used >= 55
    d = pose_difference(sol.T_cam_parent, T)
    assert d.translation_m < 0.01 and d.rotation_deg < 0.5
    assert np.all(sol.reproj_err_px[bad] > 50)


def test_candidates_mask_excludes_moving_frames():
    T, P, uv, _ = _scene()
    cand = np.ones(len(P), bool)
    cand[::3] = False
    uv[::3] += 200.0  # garbage where the arm was moving
    sol = solve_extrinsic(P, uv, K, candidates=cand)
    assert not sol.used[::3].any()
    assert sol.n_candidates == int(cand.sum())
    d = pose_difference(sol.T_cam_parent, T)
    assert d.translation_m < 1e-4


def test_refuses_too_few_points():
    _, P, uv, _ = _scene(n=8)
    with pytest.raises(ValueError, match="usable frames"):
        solve_extrinsic(P, uv, K)


def test_nan_pixels_are_not_candidates():
    _, P, uv, _ = _scene()
    uv[:5] = np.nan
    sol = solve_extrinsic(P, uv, K)
    assert not sol.candidates[:5].any()
    assert np.all(np.isnan(sol.reproj_err_px[:5]))


def test_depth_at_takes_the_window_median_and_ignores_zeros():
    depth = np.zeros((10, 10), np.float32)
    depth[4:7, 4:7] = 1.0
    depth[5, 5] = 0.0  # a hole right on the pixel
    assert depth_at(depth, (5, 5)) == pytest.approx(1.0)
    assert np.isnan(depth_at(depth, (50, 5)))
    assert np.isnan(depth_at(None, (5, 5)))


def test_depth_agreement_matches_the_solved_distance():
    T, P, uv, z = _scene(n=10)
    maps = []
    for i in range(10):
        d = np.full((480, 640), z[i] + 0.01, np.float32)  # sensor sees 1 cm further
        maps.append(d)
    agr = depth_agreement(maps, uv, T, P)
    assert agr.n_valid == 10
    assert agr.median_m == pytest.approx(0.01, abs=1e-6)
    assert agr.median_abs_m == pytest.approx(0.01, abs=1e-6)


def test_judge_levels():
    T, P, uv, z = _scene()
    sol = solve_extrinsic(P, uv, K)
    good = depth_agreement([np.full((480, 640), zi, np.float32) for zi in z], uv, T, P)
    assert judge(sol, good).level == "good"
    off = depth_agreement(
        [np.full((480, 640), zi + 0.2, np.float32) for zi in z], uv, T, P
    )
    v = judge(sol, off)
    assert v.level == "bad" and not v.ok
    assert any("depth" in r for r in v.reasons)
    # No depth at all is a "check", never silently good.
    assert judge(sol, depth_agreement([None] * 60, uv, T, P)).level == "check"

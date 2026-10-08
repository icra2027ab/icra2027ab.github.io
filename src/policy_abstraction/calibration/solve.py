"""Camera pose from tracked pixels and known 3-D positions, with checks.

Kalib's own PnP feeds every frame to ``cv2.solvePnP`` with no outlier
rejection, so one frame where the tracker slipped onto the wrong finger pulls
the whole solution. Here the settled frames go through ``solvePnPRansac``
first, the inliers are refined with Levenberg-Marquardt, and the per-frame
residuals are kept so the operator can see *which* frames disagree.

Two independent checks follow the solve:

* **Reprojection** — the residual PnP minimised. Small residuals with a wrong
  answer are possible when the poses barely vary, so this alone is not proof.
* **Depth agreement** — the RealSense's own depth at the tracked pixel versus
  the solved distance to the point. Depth never entered the solve, so this is
  the check that catches a mirrored or otherwise self-consistent-but-wrong
  extrinsic. ``retrieval._check_extrinsic`` does the same at rollout start.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import cv2
import numpy as np

from policy_abstraction.calibration.geometry import (
    invert,
    is_rigid,
    project,
    rotation_angle_deg,
    transform_points,
)


@dataclasses.dataclass
class ExtrinsicSolution:
    """A solved camera pose plus the evidence behind it."""

    T_cam_parent: np.ndarray
    """OpenCV's answer: ``X_cam = T_cam_parent @ X_parent``."""
    T_parent_cam: np.ndarray
    """The camera's pose in the parent frame — what ``extrinsic`` means in the
    env's calibration file."""
    used: np.ndarray
    """``(N,)`` bool: frames that entered the solve as inliers."""
    candidates: np.ndarray
    """``(N,)`` bool: frames that were eligible (settled and visible)."""
    reproj_err_px: np.ndarray
    """``(N,)`` per-frame reprojection error, nan where the point was not
    tracked or fell behind the camera."""
    mean_px: float
    median_px: float
    max_inlier_px: float

    @property
    def n_used(self) -> int:
        return int(self.used.sum())

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.sum())


def solve_extrinsic(
    points_parent: np.ndarray,
    points_uv: np.ndarray,
    K: np.ndarray,
    *,
    candidates: Optional[np.ndarray] = None,
    ransac_px: float = 4.0,
    min_points: int = 12,
    seed: int = 0,
) -> ExtrinsicSolution:
    """RANSAC PnP + LM refinement over the candidate frames.

    Args:
        points_parent: ``(N, 3)`` position of the tracked point per frame, in
            the parent frame.
        points_uv: ``(N, 2)`` tracked pixel per frame (nan when lost).
        K: 3x3 intrinsics of the stream the pixels were tracked on.
        candidates: ``(N,)`` bool mask of frames allowed into the solve
            (settled + visible). Defaults to every frame with finite pixels.
        ransac_px: inlier threshold, pixels.
        min_points: refuse to solve with fewer eligible frames than this; a
            PnP on a handful of nearly coplanar points is well-defined and
            wrong.
    """
    P = np.asarray(points_parent, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    n = P.shape[0]
    if uv.shape[0] != n:
        raise ValueError(f"{n} 3-D points but {uv.shape[0]} pixels")
    finite = np.all(np.isfinite(uv), axis=1) & np.all(np.isfinite(P), axis=1)
    cand = finite if candidates is None else (np.asarray(candidates, bool) & finite)
    if int(cand.sum()) < min_points:
        raise ValueError(
            f"only {int(cand.sum())} usable frames (settled and tracked); need at "
            f"least {min_points}. Capture more poses, or check that the tracked "
            "point stayed in view."
        )

    idx = np.flatnonzero(cand)
    obj = np.ascontiguousarray(P[idx]).reshape(-1, 1, 3)
    img = np.ascontiguousarray(uv[idx]).reshape(-1, 1, 2)
    dist = np.zeros((4, 1))
    cv2.setRNGSeed(int(seed))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        K,
        dist,
        iterationsCount=2000,
        reprojectionError=float(ransac_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError(
            "PnP found no consistent camera pose. The clicked point probably did "
            "not stay on the same physical spot, or the poses were too similar."
        )
    inl = np.asarray(inliers, dtype=np.int64).reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)
    # A second pass at the same threshold with the refined pose: RANSAC's
    # inlier set came from its best *hypothesis*, not the refined answer.
    R, _ = cv2.Rodrigues(rvec)
    T_cam_parent = np.eye(4)
    T_cam_parent[:3, :3] = R
    T_cam_parent[:3, 3] = tvec.reshape(3)
    proj, _ = project(K, T_cam_parent, P)
    err = np.linalg.norm(proj - uv, axis=1)
    used = cand & np.isfinite(err) & (err < float(ransac_px))
    if int(used.sum()) >= 6:
        idx2 = np.flatnonzero(used)
        rvec, tvec = cv2.solvePnPRefineLM(
            np.ascontiguousarray(P[idx2]).reshape(-1, 1, 3),
            np.ascontiguousarray(uv[idx2]).reshape(-1, 1, 2),
            K,
            dist,
            rvec,
            tvec,
        )
        R, _ = cv2.Rodrigues(rvec)
        T_cam_parent[:3, :3] = R
        T_cam_parent[:3, 3] = tvec.reshape(3)
        proj, _ = project(K, T_cam_parent, P)
        err = np.linalg.norm(proj - uv, axis=1)
        used = cand & np.isfinite(err) & (err < float(ransac_px))

    if not is_rigid(T_cam_parent):
        raise ValueError("PnP returned a non-rigid transform; this is a bug.")
    used_err = err[used]
    return ExtrinsicSolution(
        T_cam_parent=T_cam_parent,
        T_parent_cam=invert(T_cam_parent),
        used=used,
        candidates=cand,
        reproj_err_px=err,
        mean_px=float(used_err.mean()) if used_err.size else float("nan"),
        median_px=float(np.median(used_err)) if used_err.size else float("nan"),
        max_inlier_px=float(used_err.max()) if used_err.size else float("nan"),
    )


def depth_at(depth_m: Optional[np.ndarray], uv, radius: int = 2) -> float:
    """Median valid depth in a small window around a pixel, or nan.

    A single pixel on a fingertip edge is as likely to hit the background as
    the finger; the median over a 5x5 window ignoring zeros is what the
    startup check in ``retrieval._check_extrinsic`` effectively measures too.
    """
    if depth_m is None:
        return float("nan")
    u, v = float(uv[0]), float(uv[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return float("nan")
    h, w = depth_m.shape[:2]
    cu, cv_ = int(round(u)), int(round(v))
    if not (0 <= cu < w and 0 <= cv_ < h):
        return float("nan")
    patch = depth_m[
        max(0, cv_ - radius) : cv_ + radius + 1, max(0, cu - radius) : cu + radius + 1
    ]
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.median(valid)) if valid.size else float("nan")


@dataclasses.dataclass
class DepthAgreement:
    """Depth-sensor distance minus solved distance, per frame."""

    residual_m: np.ndarray
    """``(N,)`` ``depth - z_solved`` in metres, nan where unavailable."""

    @property
    def n_valid(self) -> int:
        return int(np.isfinite(self.residual_m).sum())

    @property
    def median_abs_m(self) -> float:
        r = self.residual_m[np.isfinite(self.residual_m)]
        return float(np.median(np.abs(r))) if r.size else float("nan")

    @property
    def median_m(self) -> float:
        r = self.residual_m[np.isfinite(self.residual_m)]
        return float(np.median(r)) if r.size else float("nan")


def depth_agreement(
    depth_maps: list[Optional[np.ndarray]],
    points_uv: np.ndarray,
    T_cam_parent: np.ndarray,
    points_parent: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> DepthAgreement:
    """Compare the sensor's depth at each tracked pixel with the solved depth.

    On the fingertips the sensor sees the finger surface, which sits a few
    millimetres in front of the TCP point between the pads, so a residual of
    ``-0.5 .. -1`` cm is expected there; a residual of tens of centimetres, or
    a sign that flips between frames, means the extrinsic is wrong.
    """
    uv = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    z = transform_points(T_cam_parent, points_parent)[:, 2]
    n = uv.shape[0]
    res = np.full(n, np.nan)
    for i in range(n):
        if mask is not None and not mask[i]:
            continue
        d = depth_at(depth_maps[i], uv[i]) if i < len(depth_maps) else float("nan")
        if np.isfinite(d) and np.isfinite(z[i]) and z[i] > 0:
            res[i] = d - z[i]
    return DepthAgreement(residual_m=res)


@dataclasses.dataclass
class PoseDifference:
    translation_m: float
    rotation_deg: float


def pose_difference(T_a: np.ndarray, T_b: np.ndarray) -> PoseDifference:
    """How far two poses of the same frame are apart."""
    D = invert(np.asarray(T_a)) @ np.asarray(T_b)
    return PoseDifference(
        translation_m=float(np.linalg.norm(D[:3, 3])),
        rotation_deg=rotation_angle_deg(D[:3, :3]),
    )


@dataclasses.dataclass
class Verdict:
    ok: bool
    level: str
    """``good`` / ``check`` / ``bad``."""
    reasons: list[str]


def judge(
    solution: ExtrinsicSolution,
    depth: Optional[DepthAgreement],
    *,
    reproj_good_px: float = 2.5,
    reproj_bad_px: float = 6.0,
    depth_good_m: float = 0.02,
    depth_bad_m: float = 0.05,
    min_used: int = 15,
) -> Verdict:
    """Turn the numbers into one of three colours the operator can act on."""
    reasons: list[str] = []
    level = "good"

    def worse(new: str) -> None:
        nonlocal level
        order = {"good": 0, "check": 1, "bad": 2}
        if order[new] > order[level]:
            level = new

    if solution.n_used < min_used:
        reasons.append(
            f"only {solution.n_used} frames survived as inliers (want >= {min_used})"
        )
        worse("check")
    frac = solution.n_used / max(1, solution.n_candidates)
    if frac < 0.6:
        reasons.append(
            f"{solution.n_used}/{solution.n_candidates} settled frames are inliers; "
            "the tracker may have drifted"
        )
        worse("check" if frac >= 0.4 else "bad")
    if solution.mean_px > reproj_bad_px:
        reasons.append(f"mean reprojection error {solution.mean_px:.1f} px is high")
        worse("bad")
    elif solution.mean_px > reproj_good_px:
        reasons.append(f"mean reprojection error {solution.mean_px:.1f} px")
        worse("check")
    if depth is not None and depth.n_valid >= 5:
        m = depth.median_abs_m
        if m > depth_bad_m:
            reasons.append(
                f"depth disagrees with the solved pose by {m * 100:.1f} cm (median)"
            )
            worse("bad")
        elif m > depth_good_m:
            reasons.append(f"depth agrees only to {m * 100:.1f} cm (median)")
            worse("check")
        else:
            reasons.append(f"depth agrees to {m * 100:.1f} cm (median)")
    elif depth is not None:
        reasons.append("too few depth readings at the tracked point to cross-check")
        worse("check")
    if not reasons:
        reasons.append(
            f"{solution.n_used} inliers, mean reprojection {solution.mean_px:.2f} px"
        )
    return Verdict(ok=level != "bad", level=level, reasons=reasons)


__all__ = [
    "DepthAgreement",
    "ExtrinsicSolution",
    "PoseDifference",
    "Verdict",
    "depth_agreement",
    "depth_at",
    "judge",
    "pose_difference",
    "solve_extrinsic",
]

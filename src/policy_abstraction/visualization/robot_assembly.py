"""Bolt a gripper MJCF onto an arm MJCF, and pose its linkage from an aperture.

Why this exists
---------------
Menagerie's UR5e is arm-only, so a rollout on the real cell used to render a
bare tool flange where the Robotiq 2F-140 actually is. Menagerie has no 2F-140
either — only the 2F-85 — so the model is vendored from an open PR; see
``../assets/mujoco_menagerie/PROVENANCE.md`` for where it came from and what was
checked before trusting it.

Drawing it takes two steps that are easy to get subtly wrong:

**Assembly.** :func:`compose_arm_with_gripper` uses ``MjSpec.attach`` to graft
the gripper onto the arm's ``attachment_site``, which is how menagerie composes
its own scenes. Every gripper name gains a prefix, so the arm's joint indices —
and therefore every existing caller that writes ``qpos[:6]`` — are untouched.

**Posing.** This is the part that does not work by default. The Robotiq is a
*closed* chain: each finger is a four-bar whose loop MuJoCo holds shut with
``connect`` equality constraints. Those constraints are solved in the dynamics,
not in kinematics, and the visualiser only ever calls ``mj_kinematics``. Writing
a driver angle and calling it done leaves the coupler and follower at whatever
they were, and the finger renders as four disconnected links pointing off in
different directions. :class:`GripperLinkage` closes the loop up front instead:
it solves the free-air pose once per model, over a grid of apertures, and the
render path just interpolates.

The free-air pose
-----------------
Each side has four joints and one position-level closure, so two degrees of
freedom remain. The gripper is *adaptive*: the second one is what lets the
fingertip roll around an object after the pad touches it. In free air a spring
holds the coupler against a mechanical stop, which is what keeps the two pads
parallel as they close. Pinning the coupler at that stop and solving the closure
reproduces a full dynamics settle to 3e-4 rad and 0.02 mm of aperture — see
``tests/visualization/test_gripper_linkage.py``, which asserts exactly that.

Aperture, not joint angle, is the input
---------------------------------------
Callers pass metres between the pads, and :meth:`GripperLinkage.qpos_for_aperture`
inverts the linkage. That keeps the drawn pads on the same fingertips that
``policy_steering``'s FK computes distances from, which is the whole point of
rendering the gripper at all — the two would otherwise disagree by up to a
centimetre with nothing on screen to say which was right.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

# Models vendored because upstream menagerie does not ship them yet.
VENDORED_MENAGERIE = Path(__file__).resolve().parents[1] / "assets" / "mujoco_menagerie"

# Gripper joint names, once any attach prefix and the ``_joint`` suffix are off.
_JOINT_RE = re.compile(
    r"^(?:.*/)?(left|right)_(driver|coupler|spring_link|follower)_joint$"
)

# How far past the model's authored "open" limit the linkage may be posed. The
# 2F-140's driver range stops at 0, which is only a 128.6 mm aperture, while the
# hardware's stroke — and so the aperture guidance asks us to draw — is 140 mm.
# The closure still solves cleanly out here (residuals stay at 1e-14 m); the
# authored limit is a control-side bound, and this module does no control.
_OPEN_MARGIN_RAD = 0.20

# Grid resolution for the aperture table. 96 samples over ~140 mm leaves the
# linear interpolation between them wrong by well under 0.05 mm.
_TABLE_SAMPLES = 96


def _model_xml_in(directory: Path) -> Path | None:
    """The one model MJCF inside a menagerie model directory, if present.

    Menagerie does not name the file after the directory — ``robotiq_2f140/``
    holds ``2f140.xml`` — so the file is found by elimination rather than by
    construction. ``scene.xml`` is the demo scene (ground plane, lights, a box
    to grasp), not the model.
    """
    if not directory.is_dir():
        return None
    candidates = [p for p in sorted(directory.glob("*.xml")) if p.name != "scene.xml"]
    return candidates[0] if len(candidates) == 1 else None


def menagerie_checkout() -> Path | None:
    """The ``mujoco_menagerie`` clone ``robot_descriptions`` keeps, if it exists.

    Found from the documented cache location rather than by importing one of the
    description modules: importing ``robot_descriptions.ur5e_mj_description``
    *clones the repository* as an import side effect, and this function only
    answers "is it already on disk?".
    """
    root = (
        Path(
            os.path.expanduser(
                os.environ.get(
                    "ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions"
                )
            )
        )
        / "mujoco_menagerie"
    )
    return root if root.is_dir() else None


def gripper_mjcf_path(name: str) -> Path:
    """Locate a gripper MJCF by menagerie directory name (e.g. ``robotiq_2f140``).

    Prefers a real menagerie checkout when it has the model, so that once PR
    #161 lands the vendored copy stops being used without anyone editing code.
    """
    checkout = menagerie_checkout()
    if checkout is not None:
        upstream = _model_xml_in(checkout / name)
        if upstream is not None:
            return upstream

    vendored = _model_xml_in(VENDORED_MENAGERIE / name)
    if vendored is None:
        available = sorted(p.name for p in VENDORED_MENAGERIE.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"No gripper MJCF {name!r}: not in the menagerie cache and not "
            f"vendored under {VENDORED_MENAGERIE} (have: {available})"
        )
    return vendored


# ---------------------------------------------------------------------------
# Linkage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperLinkage:
    """Free-air joint angles of a closed-chain gripper, indexed by aperture.

    ``apertures`` is ascending and ``table[i]`` holds the full gripper qpos
    vector that opens the pads to ``apertures[i]`` metres.
    """

    qpos_adr: np.ndarray  # [n_joints] qpos addresses in the composed model
    apertures: np.ndarray  # [N] metres, ascending
    table: np.ndarray  # [N, n_joints] radians
    joint_names: tuple[str, ...]

    @property
    def aperture_range(self) -> tuple[float, float]:
        return float(self.apertures[0]), float(self.apertures[-1])

    def qpos_for_aperture(self, aperture_m: float) -> np.ndarray:
        """Joint angles that open the pads to *aperture_m* metres.

        Clamped to what the linkage can reach: asking for more than the
        mechanism's span silently returns the fully-open pose, since a
        visualiser refusing to draw is worse than one drawing the nearest
        reachable thing.
        """
        a = float(np.clip(aperture_m, self.apertures[0], self.apertures[-1]))
        return np.array(
            [
                np.interp(a, self.apertures, self.table[:, j])
                for j in range(self.table.shape[1])
            ]
        )

    def write_into(self, qpos: np.ndarray, aperture_m: float) -> None:
        """Write the pose for *aperture_m* into *qpos* in place."""
        qpos[self.qpos_adr] = self.qpos_for_aperture(aperture_m)


def _gripper_joints(model, mj) -> dict[tuple[str, str], int]:
    """Map ``(side, role) -> joint id`` for every Robotiq-style linkage joint."""
    found: dict[tuple[str, str], int] = {}
    for j in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or ""
        m = _JOINT_RE.match(name)
        if m:
            found[(m.group(1), m.group(2))] = j
    return found


def _pad_geoms(model, mj) -> tuple[int, int]:
    """Ids of the left/right gripping pads (``*_pad1`` collision boxes)."""
    out: dict[str, int] = {}
    for g in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or ""
        for side in ("left", "right"):
            if name.endswith(f"{side}_pad1"):
                out[side] = g
    if len(out) != 2:
        raise ValueError(f"expected left/right *_pad1 pad geoms, found {sorted(out)}")
    return out["left"], out["right"]


def _pad_gap(model, data, gl: int, gr: int) -> float:
    """Signed gap between the two pads' facing surfaces, in metres.

    Measured from the box corners rather than centre-to-centre minus a
    thickness, so it stays correct as the pads tilt through the stroke and
    needs no assumption about which local axis is the pad's normal.
    """
    centres = data.geom_xpos[[gl, gr]]
    axis = centres[1] - centres[0]
    axis /= np.linalg.norm(axis)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    proj = []
    for g in (gl, gr):
        corners = (data.geom_xmat[g].reshape(3, 3) @ (signs * model.geom_size[g]).T).T
        proj.append((corners + data.geom_xpos[g]) @ axis)
    return float(proj[1].min() - proj[0].max())


def _closure_residual(model, data, mj, qpos: np.ndarray) -> np.ndarray:
    """World-space error of every ``connect`` equality, given a candidate qpos."""
    data.qpos[:] = qpos
    mj.mj_kinematics(model, data)
    res = []
    for i in range(model.neq):
        if model.eq_type[i] != mj.mjtEq.mjEQ_CONNECT:
            continue
        b1, b2 = model.eq_obj1id[i], model.eq_obj2id[i]
        p1 = data.xpos[b1] + data.xmat[b1].reshape(3, 3) @ model.eq_data[i][:3]
        p2 = data.xpos[b2] + data.xmat[b2].reshape(3, 3) @ model.eq_data[i][3:6]
        res.append(p1 - p2)
    return np.concatenate(res) if res else np.zeros(0)


def build_linkage(
    model, mj, *, base_qpos: np.ndarray | None = None
) -> GripperLinkage | None:
    """Solve a closed-chain gripper's free-air pose across its whole stroke.

    Returns ``None`` when *model* has no Robotiq-style linkage, which is the
    normal case for LIBERO/RoboCasa's prismatic-fingered Panda.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415

    joints = _gripper_joints(model, mj)
    roles = {"driver", "coupler", "spring_link", "follower"}
    if {r for _, r in joints} != roles or {s for s, _ in joints} != {"left", "right"}:
        return None

    adr = {k: model.jnt_qposadr[v] for k, v in joints.items()}
    order = [
        (s, r)
        for s in ("right", "left")
        for r in ("driver", "coupler", "spring_link", "follower")
    ]
    qpos_adr = np.array([adr[k] for k in order])
    names = tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joints[k]) for k in order)

    data = mj.MjData(model)
    gl, gr = _pad_geoms(model, mj)
    base = np.array(model.qpos0 if base_qpos is None else base_qpos, dtype=np.float64)

    def pose(driver: float, unknowns: np.ndarray) -> np.ndarray:
        q = base.copy()
        for i, side in enumerate(("right", "left")):
            q[adr[(side, "driver")]] = driver
            # The free-air stop: a spring holds the coupler here, which is what
            # keeps the pads parallel until something touches them.
            q[adr[(side, "coupler")]] = 0.0
            q[adr[(side, "spring_link")]] = unknowns[2 * i]
            q[adr[(side, "follower")]] = unknowns[2 * i + 1]
        return q

    def solve(driver: float, guess: np.ndarray) -> tuple[np.ndarray, float]:
        sol = least_squares(
            lambda u: _closure_residual(model, data, mj, pose(driver, u)),
            guess,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        return sol.x, float(np.abs(sol.fun).max())

    # Which end of the authored range opens the gripper? The 2F-85 and the
    # 2F-140 disagree on the driver's sign, so ask the model rather than assume.
    jr = model.jnt_range[joints[("right", "driver")]]
    ends = []
    for end in jr:
        u, _ = solve(float(end), np.array([end, -end, end, -end]))
        data.qpos[:] = pose(float(end), u)
        mj.mj_kinematics(model, data)
        ends.append(_pad_gap(model, data, gl, gr))
    open_end, closed_end = (jr[0], jr[1]) if ends[0] > ends[1] else (jr[1], jr[0])
    open_end = float(open_end + np.sign(open_end - closed_end) * _OPEN_MARGIN_RAD)

    # Walk from open to closed, keeping the run over which the aperture falls
    # monotonically. Past full closure the four-bar crosses a singularity and
    # the gap starts growing again; that branch is not a pose the gripper has.
    apertures: list[float] = []
    rows: list[np.ndarray] = []
    guess = np.array([open_end, -open_end, open_end, -open_end])
    worst = 0.0
    for driver in np.linspace(open_end, float(closed_end), _TABLE_SAMPLES):
        guess, resid = solve(float(driver), guess)
        worst = max(worst, resid)
        data.qpos[:] = pose(float(driver), guess)
        mj.mj_kinematics(model, data)
        gap = _pad_gap(model, data, gl, gr)
        if apertures and gap >= apertures[-1]:
            break
        row = [driver, 0.0, guess[0], guess[1], driver, 0.0, guess[2], guess[3]]
        apertures.append(gap)
        rows.append(np.array(row))
        if gap <= 0.0:
            break

    if len(apertures) < 2:
        raise RuntimeError("gripper linkage solve produced no usable aperture range")
    if worst > 1e-6:
        raise RuntimeError(
            f"gripper linkage closure did not converge (residual {worst:.2e} m)"
        )

    apertures_arr = np.array(apertures)[::-1]
    table = np.stack(rows)[::-1]
    logger.info(
        f"[robot_assembly] Solved {len(apertures)}-point gripper linkage: "
        f"aperture {apertures_arr[0] * 1000:.1f}–{apertures_arr[-1] * 1000:.1f} mm, "
        f"closure residual <= {worst:.1e} m"
    )
    return GripperLinkage(qpos_adr, apertures_arr, table, names)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssembledRobot:
    model: object  # mujoco.MjModel
    linkage: GripperLinkage | None
    arm_nq: int  # qpos entries belonging to the arm, which come first


@lru_cache(maxsize=4)
def compose_arm_with_gripper(
    arm_description: str,
    gripper_name: str,
    *,
    site: str = "attachment_site",
    prefix: str | None = None,
    mount_yaw_deg: float = 0.0,
) -> AssembledRobot:
    """Attach *gripper_name* to *arm_description*'s tool site and compile.

    Args:
        arm_description: a ``robot_descriptions`` entry, e.g. ``ur5e_mj_description``.
        gripper_name: a menagerie directory name, e.g. ``robotiq_2f140``.
        site: the arm site to mount on. Menagerie arms all expose
            ``attachment_site`` at the tool flange.
        prefix: name prefix for the gripper's bodies/joints. Defaults to
            ``"<gripper_name>/"``.
        mount_yaw_deg: rotation of the gripper about the flange axis before it
            is bolted on. A gripper can be mounted at any of several yaws and
            menagerie's default is only one of them; rendered at the wrong one
            the fingers open across the real ones. Measure it against the
            cell, and keep it in step with the ``finger_axis`` of the gripper
            geometry YAML so the drawn pads and the fingertips guidance
            measures from stay the same points. On this UR5e the Robotiq
            2F-140 opens along tool x, which is menagerie's default (0°).
    """
    import mujoco  # noqa: PLC0415
    from robot_descriptions.loaders.mujoco import (
        load_robot_description,  # noqa: PLC0415
    )

    # Resolve through the loader first so the arm is downloaded if need be,
    # then re-open the same file as a spec (only specs can be attached).
    load_robot_description(arm_description)
    import importlib  # noqa: PLC0415

    arm_path = importlib.import_module(
        f"robot_descriptions.{arm_description}"
    ).MJCF_PATH

    arm = mujoco.MjSpec.from_file(str(arm_path))  # ty: ignore[unresolved-attribute]
    arm_nq = mujoco.MjModel.from_xml_path(str(arm_path)).nq  # ty: ignore[unresolved-attribute]

    if site not in {s.name for s in arm.sites}:
        raise ValueError(
            f"{arm_description} has no site {site!r} to mount a gripper on "
            f"(sites: {sorted(s.name for s in arm.sites)})"
        )

    gripper = mujoco.MjSpec.from_file(str(gripper_mjcf_path(gripper_name)))  # ty: ignore[unresolved-attribute]
    if mount_yaw_deg:
        _yaw_root_body(gripper, mount_yaw_deg)
    arm.attach(gripper, prefix=prefix or f"{gripper_name}/", site=arm.site(site))
    model = arm.compile()

    linkage = build_linkage(model, mujoco)
    logger.info(
        f"[robot_assembly] Composed {arm_description} + {gripper_name}: "
        f"nq={model.nq} (arm {arm_nq}), "
        f"{'closed-chain gripper posed by aperture' if linkage else 'no gripper linkage'}"
    )
    return AssembledRobot(model=model, linkage=linkage, arm_nq=arm_nq)


def _yaw_root_body(spec, yaw_deg: float) -> None:
    """Rotate *spec*'s root body about its own z by *yaw_deg*.

    Applied to the gripper before attachment, so the rotation is about the
    flange axis it is mounted on. Quaternions are MuJoCo's ``wxyz``.
    """
    import math  # noqa: PLC0415

    root = spec.worldbody.bodies[0]
    w2, x2, y2, z2 = (float(v) for v in root.quat)
    half = math.radians(yaw_deg) / 2.0
    w1, z1 = math.cos(half), math.sin(half)
    # (w1, 0, 0, z1) * (w2, x2, y2, z2)
    root.quat = [
        w1 * w2 - z1 * z2,
        w1 * x2 - z1 * y2,
        w1 * y2 + z1 * x2,
        w1 * z2 + z1 * w2,
    ]


__all__ = [
    "AssembledRobot",
    "GripperLinkage",
    "VENDORED_MENAGERIE",
    "build_linkage",
    "compose_arm_with_gripper",
    "gripper_mjcf_path",
]

"""Shared robot geometry config loader.

Loads YAML config files from ``policy_steering/configs/`` so that the same
constants are used by both this package and ``policy_abstraction.utils.gripper``.

Why the geometry is passed around rather than fixed
---------------------------------------------------
This repo runs *cross-embodiment*: the retrieved demonstration comes from one
robot (LIBERO's Franka Panda) while the live rollout may be on another (a UR5e
with a Robotiq 2F-140). The two sides of a comparison therefore need
*different* finger geometry at the same instant, so a module-level constant —
or a process-wide "active gripper" — cannot be correct. Call sites resolve a
:class:`GripperGeometry` for the embodiment they are describing and pass it in.

Three coordinate frames show up around the gripper; keeping them distinct is
the whole point of this file:

``action``
    What the policy emits. Panda: ``[-1, 1]``. UR5e: ``[0, 1]`` openness.
``qpos``
    What the env *observes*, after ``GripperSpec.aperture_from_qpos``. Panda:
    ``q1 - q2`` already in metres. UR5e: normalised openness.
``metric``
    Aperture in metres. Every distance/feature computation lives here.

``action`` and ``qpos`` coincide for the Robotiq (both normalised) and differ
for the Panda (whose qpos is metric but whose actions are not), which is
exactly the case that a single shared mapping would silently get wrong.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from functools import lru_cache

import yaml

_CONFIG_DIR = pathlib.Path(__file__).parent / "configs"

DEFAULT_GRIPPER = "franka_panda"

# Positive floor for the four-bar radicand (m^2); see `finger_z_offset`.
_SPAN_FLOOR = 1e-12

# Grip-site axes a pair of jaws can open along. z is the tool axis itself.
_FINGER_AXES = {"x": 0, "y": 1}


@lru_cache(maxsize=None)
def load_robot_config(name: str = DEFAULT_GRIPPER) -> dict:
    """Load a robot config YAML from ``configs/<name>.yaml``."""
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"No robot config {name!r} in {_CONFIG_DIR}; available: {available}"
        )
    with path.open("r") as f:
        return yaml.safe_load(f)


def _lerp(value, src_lo: float, src_hi: float, dst_lo: float, dst_hi: float):
    """Affine remap. Deliberately unclipped so gradients flow through it."""
    return dst_lo + (value - src_lo) * ((dst_hi - dst_lo) / (src_hi - src_lo))


@dataclass(frozen=True)
class GripperGeometry:
    """Finger geometry plus the two aperture mappings for one gripper.

    Resolve with :func:`load_gripper_geometry`, keyed by the same identifier
    ``policy_abstraction.envs.spec.GripperSpec.geometry_id`` carries, so the
    env spec picks the geometry and no call site hardcodes a robot.
    """

    name: str
    finger_x_rest: float
    finger_z_from_grip: float
    action_open: float
    action_close: float
    metric_open: float
    metric_close: float
    qpos_open: float
    qpos_close: float
    # Radius of the circle the pad face rides when the fingers are not
    # prismatic (metres). None => prismatic, and finger_z_from_grip is a
    # constant. Fitted to the linkage, not read off a drawing -- see the
    # provenance note in configs/robotiq_2f140.yaml.
    arc_radius: float | None = None
    # Which axis of the grip-site frame the jaws open along: "x" or "y". The
    # left tip sits on the negative side, the right tip on the positive side.
    # This is a property of how the gripper is bolted to the flange, not of
    # the gripper itself: the 2F-140 on the UR5e opens along tool x, the Panda
    # along its EEF y. Getting it wrong rotates every finger-anchored distance
    # a quarter turn about the tool axis.
    finger_axis: str = "y"

    def __post_init__(self) -> None:
        if self.finger_axis not in _FINGER_AXES:
            raise ValueError(
                f"finger_axis must be one of {sorted(_FINGER_AXES)}, "
                f"got {self.finger_axis!r}"
            )

    @property
    def finger_axis_index(self) -> int:
        """Column of the grip-site frame the jaws open along (0 = x, 1 = y)."""
        return _FINGER_AXES[self.finger_axis]

    @property
    def finger_open_direction(self) -> tuple[float, float, float]:
        """Unit vector in the grip-site frame from the left tip towards the right."""
        d = [0.0, 0.0, 0.0]
        d[self.finger_axis_index] = 1.0
        return (d[0], d[1], d[2])

    @property
    def is_four_bar(self) -> bool:
        """Whether this gripper's pads swing on an arc rather than sliding."""
        return self.arc_radius is not None

    def finger_z_offset(self, half_aperture):
        """Pad z-offset from the grip site at a given half-aperture (m).

        A prismatic gripper holds its tips at a fixed ``finger_z_from_grip``.
        A four-bar swings them on a circle of radius ``arc_radius`` centred on
        the grip site, so the half-aperture and the pad's z are two coordinates
        of the same point and z falls monotonically as the gripper opens.
        Measuring from the closed pose keeps ``finger_z_from_grip`` meaning
        what it always did -- the offset when the gripper is shut.

        Works on floats, numpy arrays and torch/jax tensors alike: it only uses
        arithmetic and the ``**0.5`` power, so gradients flow.
        """
        if not self.is_four_bar:
            return self.finger_z_from_grip
        r2 = self.arc_radius**2
        # The pad rides a circle of radius `arc_radius` whose coordinate is the
        # half-aperture itself, so z falls monotonically as the gripper opens.
        # Clamped so a half-aperture beyond the linkage's reach saturates at the
        # pivot height instead of producing NaN through the square root.
        #
        # The clamp has to keep the radicand strictly positive, not merely
        # non-negative: d/dx sqrt(x) is infinite at 0, and under autograd the
        # masked-out branch still multiplies that infinity by the zero mask,
        # which is NaN. Guidance denoises from pure noise, so out-of-reach
        # apertures are routine there, and one NaN in this gradient poisons the
        # whole action chunk. The floor is far below the linkage's precision.
        span = r2 - half_aperture**2
        span = span * (span > 0) + _SPAN_FLOOR * (span <= 0)
        return self.finger_z_from_grip + span**0.5 - self.arc_radius

    def action_to_metric(self, action):
        """Policy gripper-action channel → metric aperture (m).

        Unclipped by design: downstream FK tolerates slightly out-of-range
        values and clipping would zero the gradient at the extremes.
        """
        return _lerp(
            action,
            self.action_close,
            self.action_open,
            self.metric_close,
            self.metric_open,
        )

    def qpos_to_metric(self, aperture):
        """Observed aperture (post ``aperture_from_qpos``) → metric aperture (m)."""
        return _lerp(
            aperture,
            self.qpos_close,
            self.qpos_open,
            self.metric_close,
            self.metric_open,
        )


@lru_cache(maxsize=None)
def load_gripper_geometry(name: str = DEFAULT_GRIPPER) -> GripperGeometry:
    """Resolve a :class:`GripperGeometry` by config name (e.g. ``robotiq_2f140``)."""
    cfg = load_robot_config(name)["gripper"]
    missing = [
        k
        for k in (
            "finger_x_rest",
            "finger_z_from_grip",
            "action_open",
            "action_close",
            "metric_open",
            "metric_close",
            "qpos_open",
            "qpos_close",
        )
        if k not in cfg
    ]
    if missing:
        raise KeyError(f"robot config {name!r} is missing gripper keys: {missing}")
    arc = cfg.get("four_bar") or {}
    return GripperGeometry(
        name=name,
        finger_x_rest=float(cfg["finger_x_rest"]),
        finger_z_from_grip=float(cfg["finger_z_from_grip"]),
        action_open=float(cfg["action_open"]),
        action_close=float(cfg["action_close"]),
        metric_open=float(cfg["metric_open"]),
        metric_close=float(cfg["metric_close"]),
        qpos_open=float(cfg["qpos_open"]),
        qpos_close=float(cfg["qpos_close"]),
        arc_radius=float(arc["radius"]) if "radius" in arc else None,
        finger_axis=str(cfg.get("finger_axis", "y")),
    )


@lru_cache(maxsize=None)
def gripper_geometry(name: str = DEFAULT_GRIPPER) -> tuple[float, float]:
    """Return ``(finger_x_rest, finger_z_from_grip)`` in metres."""
    g = load_gripper_geometry(name)
    return g.finger_x_rest, g.finger_z_from_grip


@lru_cache(maxsize=None)
def gripper_action_bounds(
    name: str = DEFAULT_GRIPPER,
) -> tuple[float, float, float, float]:
    """Return ``(action_open, action_close, metric_open, metric_close)``.

    Defines the linear mapping from the policy's gripper action channel
    (e.g. pi0 outputs roughly ``[-1, 1]``) to the metric joint-space
    aperture ``q1 - q2`` in metres.  Both the JAX-side guidance FK and
    the numpy dynamics-integrated reference chunk use this so that every
    distance computation lives in the same metric frame.
    """
    g = load_gripper_geometry(name)
    return (g.action_open, g.action_close, g.metric_open, g.metric_close)


__all__ = [
    "DEFAULT_GRIPPER",
    "GripperGeometry",
    "gripper_action_bounds",
    "gripper_geometry",
    "load_gripper_geometry",
    "load_robot_config",
]

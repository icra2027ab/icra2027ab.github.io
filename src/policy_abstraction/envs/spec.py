"""Environment specification schema.

`EnvSpec` describes the camera layout, robot layout, and policy I/O contract
of an environment in a serialisable form. Each environment adapter (LIBERO,
RoboCasa, real robot, etc.) emits one of these once at startup; downstream
modules consume it instead of hardcoding camera names, obs keys, or robot-
specific math.

Wire format: every dataclass below has `to_dict()` / `from_dict()` so the spec
can travel through msgpack-numpy with `WebsocketEnvClient` like any other RPC
payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

import numpy as np


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Inline copy of :func:`policy_abstraction.utils.transforms.quat2axisangle`.

    Duplicated here so :mod:`policy_abstraction.envs.spec` stays leaf-level
    and importable from the simulator-side venv (which only has LIBERO and
    not the ML/3D stack pulled in by ``policy_abstraction.utils.__init__``).
    """
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

CAMERA_ROLES = ("world", "wrist", "auxiliary")
IMAGE_TRANSFORMS = ("none", "flip_vh")


@dataclass(frozen=True)
class CameraSpec:
    """Describes one camera available in the environment's observation."""

    name: str
    role: str  # one of CAMERA_ROLES
    image_key: str
    depth_key: Optional[str] = None
    segmentation_key: Optional[str] = None
    intrinsic_key: Optional[str] = None
    extrinsic_key: Optional[str] = None
    depth_near: float = 0.0
    depth_far: float = 0.0
    # True when the camera publishes depth already in metres. Simulators emit a
    # normalised z-buffer that consumers convert with depth_near/depth_far;
    # real depth sensors emit metres directly. Consumers must branch on this
    # rather than sniffing the value range — a close-up scene is legitimately
    # all-under-1 m and would otherwise be "converted" a second time.
    depth_is_metric: bool = False
    image_transform: str = "none"  # one of IMAGE_TRANSFORMS
    is_primary: bool = False

    def __post_init__(self):
        if self.role not in CAMERA_ROLES:
            raise ValueError(f"role must be one of {CAMERA_ROLES}, got {self.role!r}")
        if self.image_transform not in IMAGE_TRANSFORMS:
            raise ValueError(
                f"image_transform must be one of {IMAGE_TRANSFORMS}, "
                f"got {self.image_transform!r}"
            )

    def apply_image_transform(self, img: np.ndarray) -> np.ndarray:
        if self.image_transform == "flip_vh":
            return img[::-1, ::-1]
        return img

    @property
    def pixels_mirrored(self) -> bool:
        """Whether a projected ``u`` must be mirrored to land on the displayed frame.

        ``apply_image_transform`` rotates a ``flip_vh`` frame by 180°, and the
        axis-corrected intrinsic already accounts for the v flip — so u, and
        only u, is left needing ``(width - 1) - u``. A camera that publishes
        upright frames (``none``, which is every real camera here) needs no
        correction at all.

        The single definition of that rule. Anything projecting world points
        onto a displayed frame must consult it rather than assuming: assuming
        ``True`` is invisible on a simulator's square, pre-flipped frames and
        puts every overlay on the wrong side of a real camera's.
        """
        return self.image_transform != "none"

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "CameraSpec":
        return cls(**data)


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------

# String-keyed operations so the spec stays serialisable. Each op maps qpos
# (shape [N] or [..., N]) to a scalar aperture value. Add new ops here as
# new gripper types are supported.
APERTURE_OPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Parallel-jaw with two opposing fingers (Panda convention): q1 - q2.
    "diff": lambda q: np.asarray(q)[..., 0] - np.asarray(q)[..., 1],
    # Single-DOF gripper: pass the only qpos through.
    "identity": lambda q: np.asarray(q)[..., 0],
    # Parallel-jaw where both fingers carry the same magnitude: q1 + q2.
    "sum": lambda q: np.asarray(q)[..., 0] + np.asarray(q)[..., 1],
}


@dataclass(frozen=True)
class GripperSpec:
    """Gripper geometry and qpos→aperture conversion."""

    geometry_id: str  # registry key for utils/gripper.py finger geometry
    finger_count: int = 2
    aperture_op: str = "diff"  # one of APERTURE_OPS
    speed_threshold: float = 0.002  # default sub-trajectory segmentation threshold

    def __post_init__(self):
        if self.aperture_op not in APERTURE_OPS:
            raise ValueError(
                f"aperture_op must be one of {sorted(APERTURE_OPS)}, "
                f"got {self.aperture_op!r}"
            )

    def aperture_from_qpos(self, qpos) -> np.ndarray:
        return APERTURE_OPS[self.aperture_op](np.asarray(qpos))

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "GripperSpec":
        return cls(**data)


# ---------------------------------------------------------------------------
# State layout
# ---------------------------------------------------------------------------

# A small set of named ops compose the policy state vector from raw obs.
# Every op consumes `obs[obs_key]` and emits a 1D float32 segment.
STATE_OPS = ("pos", "quat_to_axisangle", "gripper_aperture", "scalar", "vector")


@dataclass(frozen=True)
class StateField:
    """One contiguous segment of the policy state vector."""

    op: str  # one of STATE_OPS
    obs_key: str

    def __post_init__(self):
        if self.op not in STATE_OPS:
            raise ValueError(f"op must be one of {STATE_OPS}, got {self.op!r}")

    def to_dict(self) -> dict:
        return {"op": self.op, "obs_key": self.obs_key}

    @classmethod
    def from_dict(cls, data: dict) -> "StateField":
        return cls(**data)


@dataclass(frozen=True)
class StateLayout:
    """Composes the policy state vector from raw obs + a GripperSpec."""

    fields: tuple[StateField, ...]

    def build(
        self,
        obs: dict,
        gripper_spec: GripperSpec,
        *,
        dtype: Any = None,
    ) -> np.ndarray:
        """Concatenate the configured fields into a 1-D state vector.

        When *dtype* is None, segments are read at their native dtype and
        numpy's promotion rules choose the output dtype. Pass a concrete
        dtype (e.g. ``np.float32``) when the consumer requires a specific
        precision — the cast happens once on the concatenated result.
        """
        parts: list[np.ndarray] = []
        for f in self.fields:
            if f.op == "pos":
                parts.append(np.asarray(obs[f.obs_key]).reshape(-1))
            elif f.op == "quat_to_axisangle":
                parts.append(_quat2axisangle(obs[f.obs_key]).reshape(-1))
            elif f.op == "gripper_aperture":
                aperture = gripper_spec.aperture_from_qpos(obs[f.obs_key])
                parts.append(np.atleast_1d(aperture))
            elif f.op == "scalar":
                parts.append(np.atleast_1d(np.asarray(obs[f.obs_key])))
            elif f.op == "vector":
                parts.append(np.asarray(obs[f.obs_key]).reshape(-1))
            else:  # pragma: no cover — guarded by __post_init__
                raise ValueError(f"unknown state op: {f.op!r}")
        out = np.concatenate(parts)
        return out.astype(dtype) if dtype is not None else out

    def to_dict(self) -> dict:
        return {"fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, data: dict) -> "StateLayout":
        return cls(fields=tuple(StateField.from_dict(f) for f in data["fields"]))


# ---------------------------------------------------------------------------
# Action layout
# ---------------------------------------------------------------------------


# Control space of the action vector. Sim envs (LIBERO / RoboCasa / CALVIN)
# emit ``eef_delta`` — normalised Cartesian deltas consumed by an OSC-style
# controller. Real arms driven by a joint-servo interface (UR5e + servoJ)
# emit ``joint_absolute`` — absolute joint targets in radians.
ACTION_SPACES = ("eef_delta", "eef_absolute", "joint_absolute")


@dataclass(frozen=True)
class ActionLayout:
    """Describes the action vector produced by the policy.

    Slices are half-open ``[start, stop)`` ranges in the action dim. Negative
    indices are *not* supported here — emit absolute indices so consumers
    don't need to know the action dim.

    ``eef_pos_slice`` / ``eef_rot_slice`` describe the *end-effector* channels
    and are ``None`` for joint-space action vectors; ``joint_pos_slice``
    describes the joint channels and is ``None`` for Cartesian ones. Consumers
    that need to know which is which should branch on :attr:`space` rather
    than on slice nullability.
    """

    action_dim: int
    eef_pos_slice: Optional[tuple[int, int]] = None  # e.g. (0, 3) — EEF translation
    eef_rot_slice: Optional[tuple[int, int]] = None  # e.g. (3, 6) — axis-angle
    gripper_dim: Optional[int] = None  # absolute index of the gripper channel
    integrate_eef: bool = True  # if True, EEF dims are deltas; gripper is absolute
    joint_pos_slice: Optional[tuple[int, int]] = None  # e.g. (0, 6) — joint targets
    space: str = "eef_delta"  # one of ACTION_SPACES

    def __post_init__(self):
        if self.space not in ACTION_SPACES:
            raise ValueError(
                f"space must be one of {ACTION_SPACES}, got {self.space!r}"
            )
        for name in ("eef_pos_slice", "eef_rot_slice", "joint_pos_slice"):
            sl = getattr(self, name)
            if sl is None:
                continue
            lo, hi = sl
            if not (0 <= lo < hi <= self.action_dim):
                raise ValueError(
                    f"{name}={sl} out of bounds for action_dim={self.action_dim}"
                )
        if self.gripper_dim is not None and not (
            0 <= self.gripper_dim < self.action_dim
        ):
            raise ValueError(
                f"gripper_dim={self.gripper_dim} out of bounds for action_dim={self.action_dim}"
            )
        if self.space == "joint_absolute" and self.joint_pos_slice is None:
            raise ValueError("space='joint_absolute' requires joint_pos_slice")
        if self.space != "joint_absolute" and self.eef_pos_slice is None:
            raise ValueError(f"space={self.space!r} requires eef_pos_slice")

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "ActionLayout":
        # tuples survive msgpack as lists; coerce slices back. Unknown keys
        # from a newer peer are dropped so an older client can still parse a
        # spec published by a newer server.
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in data.items() if k in known}
        for key in ("eef_pos_slice", "eef_rot_slice", "joint_pos_slice"):
            if d.get(key) is not None:
                d[key] = tuple(d[key])
        return cls(**d)


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotSpec:
    """Robot-side obs keys, gripper convention, and action/state layout."""

    eef_pos_key: str
    eef_quat_key: str
    gripper_qpos_key: str
    joint_pos_key: str
    state_layout: StateLayout
    gripper: GripperSpec
    action_layout: ActionLayout
    # For visualisation / FK. Either a mujoco_menagerie path or a
    # robot_descriptions package name; the visualiser already accepts both.
    urdf_path: Optional[str] = None
    robot_description: Optional[str] = None
    # Robot-side entity names used by the VLM correspondence module. Must
    # cover at least the end-effector; gripper-finger entries are optional
    # for non-parallel-jaw grippers.
    entity_names: tuple[str, ...] = ("ee", "gripper_l", "gripper_r")
    # Some sims (e.g. LIBERO) require negating the last visualizer joint to
    # render the gripper correctly. Capture that here rather than hardcoding.
    visualizer_negate_gripper_joint: bool = False

    def to_dict(self) -> dict:
        return {
            "eef_pos_key": self.eef_pos_key,
            "eef_quat_key": self.eef_quat_key,
            "gripper_qpos_key": self.gripper_qpos_key,
            "joint_pos_key": self.joint_pos_key,
            "state_layout": self.state_layout.to_dict(),
            "gripper": self.gripper.to_dict(),
            "action_layout": self.action_layout.to_dict(),
            "urdf_path": self.urdf_path,
            "robot_description": self.robot_description,
            "entity_names": list(self.entity_names),
            "visualizer_negate_gripper_joint": self.visualizer_negate_gripper_joint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RobotSpec":
        return cls(
            eef_pos_key=data["eef_pos_key"],
            eef_quat_key=data["eef_quat_key"],
            gripper_qpos_key=data["gripper_qpos_key"],
            joint_pos_key=data["joint_pos_key"],
            state_layout=StateLayout.from_dict(data["state_layout"]),
            gripper=GripperSpec.from_dict(data["gripper"]),
            action_layout=ActionLayout.from_dict(data["action_layout"]),
            urdf_path=data.get("urdf_path"),
            robot_description=data.get("robot_description"),
            entity_names=tuple(
                data.get("entity_names", ("ee", "gripper_l", "gripper_r"))
            ),
            visualizer_negate_gripper_joint=data.get(
                "visualizer_negate_gripper_joint", False
            ),
        )


# ---------------------------------------------------------------------------
# Policy I/O
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyIOKeys:
    """Maps internal logical roles to the keys the policy expects.

    `images` is a list of (camera_name, policy_input_key) pairs. Camera names
    must match an entry in :class:`EnvSpec.cameras`.

    `state_layout` describes how to build the policy's expected state vector
    from the raw observation. This may differ from
    :attr:`RobotSpec.state_layout` (the internal robot state used for
    dynamics / visualisation): e.g. LIBERO's pi0 policy expects the raw
    2-DOF gripper qpos, while the internal state uses the 1-D aperture.
    When unset, the consumer falls back to :attr:`RobotSpec.state_layout`.
    """

    state_key: str = "observation/state"
    prompt_key: str = "prompt"
    images: tuple[tuple[str, str], ...] = ()
    state_layout: Optional[StateLayout] = None

    def to_dict(self) -> dict:
        return {
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
            "images": [list(pair) for pair in self.images],
            "state_layout": self.state_layout.to_dict()
            if self.state_layout is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyIOKeys":
        sl = data.get("state_layout")
        return cls(
            state_key=data.get("state_key", "observation/state"),
            prompt_key=data.get("prompt_key", "prompt"),
            images=tuple(tuple(p) for p in data.get("images", ())),
            state_layout=StateLayout.from_dict(sl) if sl is not None else None,
        )


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSpec:
    """Full environment description — published by adapters, consumed by core."""

    cameras: tuple[CameraSpec, ...]
    robot: RobotSpec
    policy_input_keys: PolicyIOKeys
    # Free-form metadata that doesn't fit elsewhere (env name, version, etc.).
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.cameras:
            raise ValueError("EnvSpec requires at least one camera")
        primaries = [c for c in self.cameras if c.is_primary]
        if len(primaries) > 1:
            raise ValueError(
                f"at most one camera may have is_primary=True (got {len(primaries)})"
            )
        names = [c.name for c in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError(f"camera names must be unique, got {names}")
        for cam_name, _ in self.policy_input_keys.images:
            if cam_name not in names:
                raise ValueError(
                    f"policy_input_keys references unknown camera {cam_name!r}; "
                    f"available: {names}"
                )

    @property
    def primary_camera(self) -> CameraSpec:
        """The primary camera. The first ``is_primary=True`` if any, else
        ``cameras[0]`` — so envs that don't tag a primary still have a
        defined primary view."""
        return next(
            (c for c in self.cameras if c.is_primary),
            self.cameras[0],
        )

    @property
    def primary_camera_idx(self) -> int:
        return next(
            (i for i, c in enumerate(self.cameras) if c.is_primary),
            0,
        )

    @property
    def wrist_camera(self) -> Optional[CameraSpec]:
        """First camera with ``role == "wrist"``, or ``None`` if absent."""
        return next((c for c in self.cameras if c.role == "wrist"), None)

    def camera_by_name(self, name: str) -> CameraSpec:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(
            f"no camera named {name!r}; available: {[c.name for c in self.cameras]}"
        )

    def to_dict(self) -> dict:
        return {
            "cameras": [c.to_dict() for c in self.cameras],
            "robot": self.robot.to_dict(),
            "policy_input_keys": self.policy_input_keys.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EnvSpec":
        return cls(
            cameras=tuple(CameraSpec.from_dict(c) for c in data["cameras"]),
            robot=RobotSpec.from_dict(data["robot"]),
            policy_input_keys=PolicyIOKeys.from_dict(data["policy_input_keys"]),
            metadata=dict(data.get("metadata", {})),
        )


__all__ = [
    "CAMERA_ROLES",
    "IMAGE_TRANSFORMS",
    "STATE_OPS",
    "ACTION_SPACES",
    "APERTURE_OPS",
    "CameraSpec",
    "GripperSpec",
    "StateField",
    "StateLayout",
    "ActionLayout",
    "RobotSpec",
    "PolicyIOKeys",
    "EnvSpec",
]

"""
Shared VLM types.

DistancePair
  Represents a directed feature pair: ``from_entity`` → ``to_entity``.
  Entities are either a robot-entity string (``"ee"``, ``"gripper_l"``,
  ``"gripper_r"``) or an integer segmentation ID.  ``label`` is a
  human-readable name for the feature (e.g. "ee_to_handle").

DistanceType
  How the scalar distance between two entities is computed:
  - ``centroid``: Euclidean distance between the 3-D centroids.
  - ``surface``: Shortest distance between the mesh/point-cloud surfaces.

  Either type can be combined with an optional ``DistancePair.axis`` or
  ``DistancePair.plane`` to restrict the measurement to a specific component:

  - ``axis [x,y,z]``: signed scalar projection onto the given direction (1-D).
    Use for measuring distance along a single world-space direction, e.g.
    ``[0,0,1]`` for vertical clearance.
  - ``plane [nx,ny,nz]``: magnitude of the in-plane component after removing
    the contribution along the given normal (2-D).  Use for lateral alignment,
    e.g. normal ``[0,0,1]`` gives the XY-plane distance.

  At most one of ``axis``/``plane`` should be set; if neither is set the full
  3-D distance is used.

  Additional distance types:

  - ``orientation``: Projection of a configurable body-frame vector of the
    end-effector (default: z-axis ``[0,0,1]``, the "approach direction") onto
    a world-space reference direction.  Set ``ee_vector`` to choose which
    body-frame axis to track (e.g. ``[1,0,0]`` for the gripper x-axis).
    With ``axis`` set, returns the signed cosine of the angle between the
    body vector and the axis (in ``[-1, 1]``); use ``axis=[0,0,-1]`` to
    measure "approach from above" (+1 = pointing straight down).  With
    ``plane`` set, returns the magnitude of the body vector projected into
    the plane perpendicular to the normal (in ``[0, 1]``).  Both entities
    should be ``"ee"`` (self-referencing).
  - ``gripper_aperture``: Scalar gripper opening value.  Both entities should
    be gripper entities (``"gripper_l"``, ``"gripper_r"``).

SubtrajConstraint
  A constraint associated with a specific sub-trajectory.  Two kinds are
  supported:

  - ``sub_goal``: Must be satisfied *at the end* of the sub-trajectory.
    Used to decide when to transition to the next sub-trajectory.
  - ``path``: Must be satisfied *throughout* the sub-trajectory.  When
    violated, the importance of the referenced distance pairs in the
    guidance process is increased.

  Each constraint is expressed as a Python lambda string with signature
  ``lambda ee, keypoints, ee_R: <expr>`` where ``ee`` is the end-effector
  position (3-D array), ``keypoints`` is a ``dict[int | str, np.ndarray]``
  mapping entity IDs to their 3-D centroids, and ``ee_R`` is the world-from-EE
  rotation matrix (shape ``(3, 3)``).  ``ee_R[:, 2]`` is the gripper's body
  z-axis ("approach direction") expressed in world coordinates.  The
  constraint is satisfied when the returned scalar cost is ≤ 0.

VLMMessage
  A single turn in a VLM conversation, with an optional list of images.

VLMConversation
  Full conversation with the VLM, including all messages and images, so it
  can be inspected and visualized later (e.g. in Rerun).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal

import numpy as np


class DistanceType(str, Enum):
    """How the scalar distance between two entities is computed."""

    CENTROID = "centroid"
    SURFACE = "surface"
    ORIENTATION = "orientation"
    """Projection of a configurable body-frame vector of the end-effector
    (default: z-axis ``[0,0,1]``, the "approach direction") onto a world-space
    reference direction.  Set ``DistancePair.ee_vector`` to choose which
    body-frame axis to track (e.g. ``[1,0,0]`` for the gripper x-axis).
    With ``axis`` set, returns the signed cosine of the angle between the
    body vector and the axis (in ``[-1, 1]``); use ``axis=[0,0,-1]`` to
    measure "approach from above" (+1 = pointing straight down).  With
    ``plane`` set, returns the magnitude of the body vector projected into
    the plane perpendicular to the normal (in ``[0, 1]``); use
    ``plane=[0,0,1]`` to measure tilt away from vertical (0 = perfectly
    vertical, 1 = horizontal).  Both ``from_entity`` and ``to_entity``
    should be ``"ee"`` (self-referencing)."""
    GRIPPER_APERTURE = "gripper_aperture"
    """Direct gripper opening value as a scalar "distance".  Both
    ``from_entity`` and ``to_entity`` should be gripper entities
    (``"gripper_l"``, ``"gripper_r"``)."""


ROBOT_ENTITIES: frozenset[str] = frozenset({"ee", "gripper_l", "gripper_r"})
"""Entity strings that refer to robot-attached parts (not scene objects)."""


@dataclass
class DistancePair:
    """A labeled pair of entities whose pairwise distance is task-relevant.

    Each entity is either a robot-entity string (``"ee"``, ``"gripper_l"``,
    ``"gripper_r"``) or an int (segmentation instance ID).

    Attributes:
        from_entity:   Source entity (``"ee"``, ``"gripper_l"``,
                       ``"gripper_r"``, or segmentation ID).
        to_entity:     Target entity (``"ee"``, ``"gripper_l"``,
                       ``"gripper_r"``, or segmentation ID).
        label:         Human-readable name (e.g. ``"ee_to_handle"``).
        distance_type: How the scalar distance is computed.
        axis:          Optional unit axis ``[x, y, z]``.  When set, the signed
                       scalar projection of the distance vector onto this
                       direction is returned (1-D measurement).
        plane:         Optional plane normal ``[nx, ny, nz]``.  When set, the
                       magnitude of the distance vector's in-plane component
                       (perpendicular to the normal) is returned (2-D
                       measurement).  Use for lateral / alignment distances.
        reasoning:     VLM-generated justification for the pair selection.

    At most one of ``axis`` and ``plane`` should be set.
    """

    from_entity: int | str
    to_entity: int | str
    label: str
    distance_type: DistanceType = DistanceType.CENTROID
    axis: list[float] | None = field(default=None)
    plane: list[float] | None = field(default=None)
    ee_vector: list[float] | None = field(default=None)
    """Body-frame vector ``[x, y, z]`` of the EE whose world-frame projection
    is measured.  Only used when ``distance_type`` is ``ORIENTATION``.
    Defaults to ``[0, 0, 1]`` (gripper z-axis / approach direction)."""
    subtraj_idx: int | None = field(default=None)
    """Index of the sub-trajectory this pair is most informative for (0-based).
    ``None`` means the pair is relevant across multiple sub-trajectories."""
    reasoning: str | None = field(default=None)
    """Short justification from the VLM for choosing this pair."""


@dataclass
class SubtrajConstraint:
    """A constraint associated with a specific sub-trajectory.

    Attributes:
        kind:             ``"sub_goal"`` (checked at end of sub-trajectory to
                          trigger transition) or ``"path"`` (checked throughout;
                          violation boosts the weight of *pairs_affected*).
        subtraj_idx:      0-based index of the sub-trajectory this constraint
                          belongs to.
        label:            Short, unique identifier (e.g. ``"ee_near_mug_sg"``).
        description:      Human-readable description of what must hold.
        expression:       Python lambda string with signature
                          ``lambda ee, keypoints, ee_R: <scalar>`` where
                          ``ee`` is the end-effector position
                          (``np.ndarray`` shape ``(3,)``), ``keypoints`` is a
                          ``dict[int | str, np.ndarray]`` mapping entity IDs
                          to 3-D centroids, and ``ee_R`` is the world-from-EE
                          rotation matrix (``np.ndarray`` shape ``(3, 3)``).
                          ``ee_R[:, 2]`` is the gripper's body z-axis
                          ("approach direction") in world coordinates.  The
                          constraint is satisfied when the returned cost ≤ 0.
        pairs_affected:   Labels of :class:`DistancePair` instances whose
                          guidance weight increases when this constraint is
                          violated (only meaningful for ``"path"`` constraints).
        reasoning:        VLM-generated justification.
    """

    kind: Literal["sub_goal", "path"]
    subtraj_idx: int
    label: str
    description: str
    expression: str
    pairs_affected: list[str] = field(default_factory=list)
    reasoning: str | None = field(default=None)

    def compile(self) -> Callable[[np.ndarray, dict, np.ndarray], float]:
        """Compile *expression* into a callable constraint function.

        Returns:
            A callable ``f(ee, keypoints, ee_R) -> float`` where cost ≤ 0
            means the constraint is satisfied.  ``ee_R`` is the world-from-EE
            rotation matrix (shape ``(3, 3)``).

        Raises:
            ValueError: If the expression cannot be compiled or does not
                        produce a callable.
        """
        import numpy as np  # noqa: F401 – available inside lambda scope

        try:
            fn = eval(self.expression, {"np": np, "__builtins__": {}})  # noqa: S307
        except Exception as exc:
            raise ValueError(
                f"SubtrajConstraint '{self.label}': failed to compile expression: {exc}"
            ) from exc
        if not callable(fn):
            raise ValueError(
                f"SubtrajConstraint '{self.label}': expression did not "
                "evaluate to a callable."
            )
        return fn


@dataclass
class VLMMessage:
    """A single message in a VLM conversation.

    Attributes:
        role:   ``"system"``, ``"user"``, or ``"assistant"``.
        text:   Text content of the message.
        images: Optional list of ``[H, W, 3]`` uint8 RGB images attached to
                this message (only meaningful for ``"user"`` messages).
    """

    role: Literal["system", "user", "assistant"]
    text: str
    images: list[np.ndarray] = field(default_factory=list)


@dataclass
class VLMConversation:
    """Full conversation with the VLM, preserved for inspection and visualization.

    Attributes:
        messages: Ordered list of messages (system → user → assistant).
        model:    Model identifier used for the request.
    """

    messages: list[VLMMessage]
    model: str

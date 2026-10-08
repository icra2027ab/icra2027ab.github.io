"""Visualizer components for the EpisodeRunner action loop.

A Visualizer consumes timestep data (observations, actions, robot state,
action chunks, …) and renders it for human inspection.

The default ``RerunVisualizer`` streams data to a `Rerun <https://rerun.io>`_
viewer in real time.  It optionally loads a robot URDF/MJCF and renders the
robot pose alongside the camera feeds, using either a
`mujoco_menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_
asset path/name or a
`robot_descriptions <https://github.com/robot-descriptions/robot_descriptions.py>`_
package name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseVisualizer(ABC):
    """Abstract base class for episode visualisers."""

    @abstractmethod
    def setup(
        self, task_description: str = "", *, policy_label: str | None = None
    ) -> None:
        """
        Initialise the visualiser before an episode starts.

        Args:
            task_description: Human-readable task string shown in the UI.
            policy_label: Optional short string describing the policy being
                run (e.g. ``"groot"``, ``"openpi"``, ``"replay"``). Surfaced
                next to the task panel in viewers that support it.
        """

    def set_env_spec(self, env_spec: Any) -> None:  # noqa: B027
        """Tell the visualiser which env spec is in use (no-op by default)."""
        pass

    @abstractmethod
    def update(
        self,
        *,
        timestep: int,
        obs: dict[str, Any] | None = None,
        images: dict[str, np.ndarray] | None = None,
        encoded_obs: dict[str, Any] | None = None,
        robot_state: np.ndarray | None = None,
        robot_joints: np.ndarray | None = None,
        action: np.ndarray | None = None,
        action_chunks: np.ndarray | None = None,
        predicted_states: np.ndarray | None = None,
        action_chunk_states: np.ndarray | None = None,
        reward: float | None = None,
        done: bool = False,
        camera_params: dict[str, dict] | None = None,
        abstraction_result=None,
        frame_bundle=None,
        guidance_info: dict | None = None,
        replan_steps: int | None = None,
    ) -> None:
        """
        Push one timestep of data to the visualiser.

        Args:
            timestep:         Current environment step index.
            obs:              Raw observation dict from the environment.
            encoded_obs:      Encoded observation from the ``ObsEncoder``.
            robot_state:      Proprioceptive state vector.
            action:           The single action executed at this timestep.
            action_chunks:    Full action-chunk prediction ``[N, T, D]`` or
                              ``[T, D]``.
            predicted_states: States predicted by the ``DynamicsModel``
                              ``[T, state_dim]``.
            action_chunk_states: Predicted states for each action chunk
                                 ``[N, T, state_dim]``.
            reward:           Scalar reward received after executing *action*.
            done:             Whether the episode terminated.
            camera_params:    Dict mapping camera name to ``{"intrinsic": K,
                              "extrinsic": E}`` (3×3 and 4×4 arrays).
        """

    @abstractmethod
    def close(self) -> None:
        """Tear down the visualiser after the episode ends."""

    def set_robot_base(  # noqa: B027
        self,
        base_pos: np.ndarray,
        base_ori: np.ndarray,
    ) -> None:
        """Set the robot base pose for correct world-frame rendering.

        Args:
            base_pos: Robot base position ``[3]`` in world frame
                      (from ``env.robots[0].base_pos``).
            base_ori: Robot base orientation ``[3, 3]`` rotation matrix in
                      world frame (from ``env.robots[0].base_ori``).

        The default implementation is a no-op; override in subclasses that
        render a robot model.
        """
        pass

    def log_retrieval_info(self, **kwargs: Any) -> None:  # noqa: B027
        """Log retrieval results (no-op by default; override to implement)."""
        pass

    def log_abstraction_setup(self, **kwargs: Any) -> None:  # noqa: B027
        """Log abstraction setup data (no-op by default; override to implement)."""
        pass

    def send_blueprint(self) -> None:  # noqa: B027
        """Send the viewer blueprint (no-op by default; override to implement)."""
        pass

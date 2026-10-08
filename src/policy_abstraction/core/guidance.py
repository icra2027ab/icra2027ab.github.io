"""
Differentiable guidance for flow-matching and diffusion policies.

Provides:
  1. SO3Guidance           – a differentiable function mapping the current robot
                             state + scene point cloud → scalar loss derived from
                             distance features, whose gradient steers denoising.
  2. ActionSpaceGuidance   – wraps SO3Guidance with the robot dynamics model so
                             that gradients are expressed in the policy's action
                             space (e.g. delta-EEF or joint-velocity).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class SO3Guidance:
    """Differentiable guidance: distance features → scalar loss → policy gradients."""

    def compute_loss(
        self, robot_state, pcd, reference_features: dict
    ) -> "torch.Tensor":
        raise NotImplementedError


class ActionSpaceGuidance:
    """Wraps SO3Guidance with robot dynamics to give gradients in action space."""

    def __init__(self, so3_guidance: SO3Guidance, dynamics_model):
        self.so3_guidance = so3_guidance
        self.dynamics_model = dynamics_model

    def grad(self, actions, robot_state, pcd, reference_features: dict):
        raise NotImplementedError

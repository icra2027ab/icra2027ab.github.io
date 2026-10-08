"""Abstract environment client interface."""

from abc import ABC, abstractmethod


class BaseEnvClient(ABC):
    """Minimal interface that any simulator backend must implement."""

    @abstractmethod
    def make_env(self, *args, **kwargs) -> dict:
        """Instantiate a new environment episode. Returns env metadata."""

    @abstractmethod
    def reset(self) -> dict:
        """Reset environment and return initial observation dict."""

    @abstractmethod
    def step(self, action) -> tuple:
        """Step environment. Returns (obs, reward, done, info)."""

    @abstractmethod
    def seed(self, n: int) -> None:
        """Set the random seed."""

    @abstractmethod
    def close(self) -> None:
        """Clean up the environment."""

    def get_camera_params(self, camera_names: list) -> dict:
        """Return intrinsic + extrinsic matrices per camera."""
        raise NotImplementedError

    def get_robot_base(self) -> dict:
        """Return robot base position and orientation."""
        raise NotImplementedError

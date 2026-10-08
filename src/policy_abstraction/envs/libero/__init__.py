"""LIBERO and LIBERO-PRO environment sub-package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from policy_abstraction.envs.libero.perturbation import PerturbationManager
    from policy_abstraction.envs.libero.server import LiberoEnvServer

from policy_abstraction.envs.libero.config import (
    DEFAULT_RESIZE_SIZE,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    TASK_MAX_STEPS,
)
from policy_abstraction.envs.libero.utils import (
    get_libero_env,
    get_max_steps_for_suite,
    start_libero_env_server,
)

# PerturbationManager and LiberoEnvServer depend on LIBERO-PRO / libero / robosuite
# which are only available in the dedicated simulator virtualenv.  Lazy-load them
# so that lightweight consumers (e.g. the evaluation loop) can import this package
# without those heavy dependencies installed.

__all__ = [
    "LIBERO_DUMMY_ACTION",
    "LIBERO_ENV_RESOLUTION",
    "TASK_MAX_STEPS",
    "DEFAULT_RESIZE_SIZE",
    "get_max_steps_for_suite",
    "start_libero_env_server",
    "get_libero_env",
    "PerturbationManager",
    "LiberoEnvServer",
]


def __getattr__(name: str):
    if name == "PerturbationManager":
        from policy_abstraction.envs.libero.perturbation import PerturbationManager

        return PerturbationManager
    if name == "LiberoEnvServer":
        from policy_abstraction.envs.libero.server import LiberoEnvServer

        return LiberoEnvServer
    raise AttributeError(
        f"module 'policy_abstraction.envs.libero' has no attribute {name!r}"
    )

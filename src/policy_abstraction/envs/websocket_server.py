"""Backward-compatibility shim.

The implementation moved to :mod:`policy_abstraction.envs.libero.server`
(generic plumbing in :mod:`policy_abstraction.envs.base_server`,
robosuite/MuJoCo helpers in
:mod:`policy_abstraction.envs.robosuite_geom_utils`). New code should import
from those modules directly.
"""

from policy_abstraction.envs.libero.server import LiberoEnvServer
from policy_abstraction.envs.robosuite_geom_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

__all__ = [
    "LiberoEnvServer",
    "get_camera_intrinsic_matrix",
    "get_camera_extrinsic_matrix",
]

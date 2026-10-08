"""Policy steering: guidance functions for action-space trajectory correction.

Public API
----------
- :data:`GP_KEYS` / :data:`GP_SENTINEL_KEY` — canonical parameter key names
- :func:`extract_guidance_params` — extract & convert guidance params from obs dict
- :func:`build_guidance_fns` — JAX loss+grad from guidance params (requires JAX)
- :func:`compute_guidance_velocity` — JAX velocity correction (requires JAX)

Torch users should import directly from ``policy_steering.guidance_fns_torch``
to avoid the JAX dependency.
"""

from policy_steering.params import (
    GP_KEYS,
    GP_SENTINEL_KEY,
    extract_guidance_params,
)

__all__ = [
    "GP_KEYS",
    "GP_SENTINEL_KEY",
    "extract_guidance_params",
]

# JAX-backed symbols are optional: the Gr00t venv imports this package
# without JAX installed. Re-export them at the top level when available
# so existing JAX callers (openpi, dummy policy) keep working unchanged.
try:
    from policy_steering.guidance_fns import (
        build_guidance_fns,
        compute_guidance_velocity,
        compute_guidance_velocity_multi_alpha,
    )
except ImportError:
    pass
else:
    __all__ += [
        "build_guidance_fns",
        "compute_guidance_velocity",
        "compute_guidance_velocity_multi_alpha",
    ]

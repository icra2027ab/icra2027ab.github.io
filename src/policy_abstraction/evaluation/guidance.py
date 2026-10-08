"""Client-side helpers for packing differentiable guidance into obs dicts.

Two modes are supported:

**Callable mode** (legacy, bypasses JIT)::

    from policy_abstraction.evaluation.guidance import pack_guidance
    import jax.numpy as jnp

    def my_guidance(x_t):
        # x_t: [total_horizon, action_dim] — unbatched, JAX-traceable
        # Must return a JAX scalar; jax.vmap(jax.grad(...)) is applied server-side.
        target = jnp.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0])
        return -jnp.sum((x_t[0, :3] - target[:3]) ** 2)

    obs = pack_guidance(obs, my_guidance)

**Params mode** (preferred, JIT-compatible)::

    from policy_abstraction.evaluation.guidance import pack_guidance_params

    obs = pack_guidance_params(obs, {
        "gradient_directions": np.array(..., dtype=np.float32),  # (N, 3)
        "expected_errors":     np.array(..., dtype=np.float32),  # (N,)
        "reference_points":    np.array(..., dtype=np.float32),  # (N, 3)
    })

    action = policy_client.infer(obs)["actions"]

In params mode the guidance function is reconstructed server-side from plain JAX
arrays, which allows ``module_jit`` to stay active.

Callable guidance function contract:
    - Signature: ``fn(x_t: jnp.ndarray[total_horizon, action_dim]) -> scalar``
    - Input is unbatched; ``jax.vmap`` is applied server-side over the batch dimension.
    - Must return a JAX scalar (e.g. ``jnp.sum(...)``), NOT a Python float.
    - Must use only ``jnp`` operations — must be JAX-traceable.
"""

from __future__ import annotations

import cloudpickle
import numpy as np

from policy_abstraction.evaluation.guidance_builder import GP_KEYS, GP_SENTINEL_KEY

# Sentinel key used by both client and server (callable path)
GUIDANCE_KEY = "__guidance_fn__"


def pack_guidance(obs: dict, guidance_fn) -> dict:
    """Serialize *guidance_fn* with cloudpickle and insert it into *obs*.

    Args:
        obs: The observation dict to be sent to the policy server.
        guidance_fn: A JAX-differentiable callable. See module docstring for contract.

    Returns:
        A shallow copy of *obs* with ``GUIDANCE_KEY`` set to a ``np.uint8`` array
        containing the cloudpickle bytes. The original dict is not mutated.
    """
    obs = dict(obs)
    obs[GUIDANCE_KEY] = np.frombuffer(cloudpickle.dumps(guidance_fn), dtype=np.uint8)
    return obs


def unpack_guidance(obs: dict) -> tuple:
    """Extract and deserialize the guidance function from *obs*, if present.

    Args:
        obs: Observation dict received by the server (may or may not contain guidance).

    Returns:
        ``(guidance_fn_or_None, obs_without_key)`` where *obs_without_key* has the
        sentinel key removed so downstream input transforms don't see it.
    """
    if GUIDANCE_KEY not in obs:
        return None, obs
    arr = obs[GUIDANCE_KEY]
    obs = {k: v for k, v in obs.items() if k != GUIDANCE_KEY}
    fn = cloudpickle.loads(arr.tobytes())
    return fn, obs


def pack_guidance_params(obs: dict, guidance_params: dict) -> dict:
    """Insert explicit guidance parameter arrays into *obs* (JIT-compatible path).

    Unlike :func:`pack_guidance`, no Python callables are involved — all values
    are plain numpy arrays that survive JAX tracing and ``module_jit``.

    Args:
        obs: The observation dict to be sent to the policy server.
        guidance_params: Dict produced by
            :func:`~policy_abstraction.evaluation.guidance_builder.build_distance_guidance_params`.
            Keys are the ``__gp_*__`` sentinel strings defined in that module.

    Returns:
        Shallow copy of *obs* with all ``__gp_*__`` keys inserted.
    """
    obs = dict(obs)
    for key in GP_KEYS:
        if key in guidance_params:
            obs[key] = guidance_params[key]
    return obs


def unpack_guidance_params(obs: dict) -> tuple[dict | None, dict]:
    """Extract guidance parameter arrays from *obs*, if present.

    Returns:
        ``(guidance_params_or_None, obs_without_keys)`` — all ``__gp_*__`` keys
        are removed from the returned obs so downstream transforms don't see them.
    """
    if GP_SENTINEL_KEY not in obs:
        return None, obs
    params = {k: obs[k] for k in GP_KEYS if k in obs}
    obs = {k: v for k, v in obs.items() if k not in GP_KEYS}
    return params, obs


def guidance_request_flags(
    *,
    debug: bool = False,
    normalize: bool = True,
    line_search: bool = True,
    max_alpha: float = 1.0,
    multi_alpha: bool = False,
    multi_alpha_combine: str = "sequential",
) -> dict:
    """The ``__gp_*__`` control flags a guided request carries beside its
    geometry, as :class:`~policy_abstraction.evaluation.episode_runner.EpisodeRunner`
    sends them.

    Shared with the guided policy warm-up, which has to send a request shaped
    exactly like a rollout's so the compiler sees the same graph.
    """
    import numpy as np

    flags: dict = {
        "__gp_normalize__": bool(normalize),
        "__gp_line_search__": bool(line_search),
        "__gp_max_alpha__": np.float32(max_alpha),
        "__gp_multi_alpha__": bool(multi_alpha),
        "__gp_multi_alpha_combine__": np.int32(
            {"sequential": 0, "sum": 1, "weighted": 2, "per_timestep": 3}.get(
                multi_alpha_combine, 0
            )
        ),
    }
    if debug:
        flags["__gp_debug_guidance__"] = True
    return flags

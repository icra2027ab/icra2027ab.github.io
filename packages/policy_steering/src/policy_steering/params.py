"""Canonical guidance-parameter keys and extraction helpers.

This module is the single source of truth for the ``__gp_*__`` key names
used to pass structured guidance arrays through observation/options dicts.

The module is backend-agnostic: extracted values are returned as numpy
arrays so the same wire format can be consumed by JAX (pi0grounded) or
PyTorch (gr00t grounded) callers, who convert to their own tensor type.
"""

from __future__ import annotations

import numpy as np

# Sentinel key — if present in obs/options, the server uses the params path.
GP_SENTINEL_KEY = "__gp_expected_dists__"

# All guidance-param keys (for strip-out on the server side).
GP_KEYS = (
    "__gp_ee_pos_0__",
    "__gp_dt__",
    "__gp_expected_dists__",
    "__gp_expected_stds__",
    "__gp_dist_types__",
    "__gp_waypoints__",
    "__gp_has_axis__",
    "__gp_axes__",
    "__gp_has_plane__",
    "__gp_plane_normals__",
    "__gp_surface_counts__",
    "__gp_surface_pts__",
    "__gp_guidance_weights__",
    "__gp_guidance_factor__",
    "__gp_debug_guidance__",
    # Source entity type (0=EE, 1=finger_l, 2=finger_r, 3=orientation, 4=gripper)
    "__gp_source_type__",
    # EE orientation and gripper state for FK-based guidance
    "__gp_ee_ori_0__",
    "__gp_gripper_0__",
    # Per-pair body-frame vector for orientation guidance
    "__gp_ee_vectors__",
    # Per-pair body-frame offset (N,3) of an EEF-attached object (source_type
    # 5): source_pos = ee_pos + R_eef @ offset, so an object-to-object pair
    # propagates gradients to the EEF pose. Zeros for non-attached pairs.
    "__gp_ee_offsets__",
    # Multi-chunk and search control
    "__gp_num_chunks__",
    "__gp_line_search__",
    "__gp_normalize__",
    "__gp_max_alpha__",
    # Per-pair multi-alpha guidance (torch/GR00T head only). When the bool is
    # False (default) the single-shared-alpha path runs unchanged. The combine
    # code selects the merge rule: 0=sequential, 1=sum, 2=weighted.
    "__gp_multi_alpha__",
    "__gp_multi_alpha_combine__",
)

# Boolean keys that must stay as Python bools (not JAX/torch arrays) to
# avoid TracerBoolConversionError inside jit / shape-only branching.
_BOOL_KEYS = {
    "__gp_debug_guidance__",
    "__gp_line_search__",
    "__gp_normalize__",
    "__gp_multi_alpha__",
}


def extract_guidance_params(obs: dict) -> tuple[dict | None, dict]:
    """Extract guidance parameter arrays from *obs*.

    If the sentinel key is absent, returns ``(None, obs)`` unchanged.
    Otherwise returns ``(guidance_params, cleaned_obs)`` where:

    - ``guidance_params`` has all ``__gp_*__`` values converted to
      ``np.asarray`` (except bool keys which stay as Python ``bool``).
      Backend-specific callers should convert these to ``jnp.asarray``
      or ``torch.as_tensor`` at the call site.
    - ``cleaned_obs`` has all ``__gp_*__`` keys removed.
    """
    if GP_SENTINEL_KEY not in obs:
        return None, obs

    guidance_params: dict = {}
    for k in GP_KEYS:
        if k in obs:
            v = obs[k]
            if k in _BOOL_KEYS:
                guidance_params[k] = bool(v)
            else:
                guidance_params[k] = np.asarray(v)

    cleaned_obs = {k: v for k, v in obs.items() if k not in GP_KEYS}
    return guidance_params, cleaned_obs

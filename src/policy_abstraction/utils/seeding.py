"""Centralized seeding for reproducibility across all RNGs.

Call :func:`seed_everything` once at the top of every entrypoint, before any
model loading or environment construction. Library code should *not* re-call
``seed_everything`` — pass derived state (e.g. a split JAX key) explicitly.

Note: ``torch.use_deterministic_algorithms(True)`` and ``cudnn.deterministic``
are intentionally NOT set here. Bitwise CUDA determinism is not a goal of this
repo and toggling those flags breaks several kernels used by the policy server.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
from loguru import logger

_JAX_KEY: Any = None


def seed_everything(seed: int) -> int:
    """Seed Python ``random``, NumPy, PyTorch, and JAX from a single integer.

    Returns the seed for downstream wiring (e.g. into env constructors).
    Torch and JAX are imported lazily so this stays usable on CPU-only setups
    that may not have either installed.

    ``PYTHONHASHSEED`` is exported so child processes (servers spawned by the
    main entrypoint) inherit a deterministic hash seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import jax

        global _JAX_KEY
        _JAX_KEY = jax.random.PRNGKey(seed)
    except ImportError:
        pass

    logger.info(f"Seeded all RNGs with seed={seed}")
    return seed


def get_jax_key() -> Any:
    """Return the root JAX PRNGKey set by the most recent :func:`seed_everything`."""
    if _JAX_KEY is None:
        raise RuntimeError("seed_everything() must be called before get_jax_key()")
    return _JAX_KEY

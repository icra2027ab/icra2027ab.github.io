"""Tests for `policy_abstraction.utils.seeding`."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np

from policy_abstraction.utils.seeding import get_jax_key, seed_everything


def test_seed_everything_returns_seed():
    assert seed_everything(123) == 123


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(42)
    a = np.random.rand(5)
    seed_everything(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_makes_python_random_reproducible():
    import random

    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_pythonhashseed_exported():
    seed_everything(99)
    assert os.environ.get("PYTHONHASHSEED") == "99"


def test_seed_everything_across_subprocesses():
    """Two subprocesses with the same seed must produce identical numpy output."""
    code = textwrap.dedent(
        """
        from policy_abstraction.utils.seeding import seed_everything
        import numpy as np
        seed_everything(7)
        print(np.random.rand(3).tolist())
        """
    )
    out1 = subprocess.check_output([sys.executable, "-c", code]).decode().strip()
    out2 = subprocess.check_output([sys.executable, "-c", code]).decode().strip()
    assert out1 == out2


def test_get_jax_key_raises_before_seed(monkeypatch):
    # Reset the module-level key so the test is independent of test order.
    import policy_abstraction.utils.seeding as seeding_mod

    monkeypatch.setattr(seeding_mod, "_JAX_KEY", None)
    try:
        get_jax_key()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when called before seed_everything")

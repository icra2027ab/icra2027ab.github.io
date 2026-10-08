"""CALVIN env adapter — server-side, runs in third_party/calvin/.venv.

Mirrors :mod:`policy_abstraction.envs.libero` and
:mod:`policy_abstraction.envs.robocasa`. Importing the top-level package does
NOT pull in pybullet/hydra; the actual server lives in
:mod:`policy_abstraction.envs.calvin.server` and is only imported by the
``serve_calvin_env.py`` entry-point.
"""

from __future__ import annotations

__all__: list[str] = []

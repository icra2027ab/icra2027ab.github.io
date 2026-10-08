"""UR5e real-robot env adapter — server-side, runs in the physical-ai-sdk venv.

Mirrors :mod:`policy_abstraction.envs.libero`,
:mod:`policy_abstraction.envs.robocasa` and
:mod:`policy_abstraction.envs.calvin`. Importing the top-level package does
NOT pull in ``rtde_control`` / ``lerobot``; the actual server lives in
:mod:`policy_abstraction.envs.ur5.server` and is only imported by the
``serve_ur5_env.py`` entry-point.
"""

from __future__ import annotations

__all__: list[str] = []

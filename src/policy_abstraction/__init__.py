"""
policy_abstraction
==================
Improves out-of-distribution robotics policy performance by retrieving relevant
in-distribution demonstrations, abstracting robot execution via relative-distance
features to object affordances, solving the demonstration-to-scene correspondence
problem through VLMs, and steering flow-matching / diffusion policies with the
resulting gradients.

Public re-exports of the most commonly used types are listed below once the
sub-packages are mature.
"""

from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

_load_dotenv(PROJECT_ROOT / ".env", override=False)

__all__ = ["PROJECT_ROOT"]

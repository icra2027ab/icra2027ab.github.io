"""VLM interfaces for correspondence solving."""

from __future__ import annotations

from typing import TYPE_CHECKING

from policy_abstraction.vlm.types import DistancePair

if TYPE_CHECKING:
    from policy_abstraction.vlm.task_pair_selector import TaskPairSelector

__all__ = ["DistancePair", "TaskPairSelector"]


def __getattr__(name: str):
    # Lazy-load TaskPairSelector to avoid importing openai at package load time.
    if name == "TaskPairSelector":
        from policy_abstraction.vlm.task_pair_selector import TaskPairSelector

        return TaskPairSelector
    raise AttributeError(f"module 'policy_abstraction.vlm' has no attribute {name!r}")

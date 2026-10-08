"""Observation encoder components for the EpisodeRunner action loop.

An ObsEncoder transforms a raw environment observation (and, optionally,
the current proprioceptive robot state) into a feature representation that
can be passed to downstream components such as a visualiser or a learned
policy.

The default ``IdentityObsEncoder`` returns the raw observation dict
unchanged so that the pipeline is a no-op by default.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseObsEncoder(ABC):
    """Abstract base class for observation encoders."""

    @abstractmethod
    def encode(
        self,
        obs: dict[str, Any],
        robot_state: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Encode a raw observation into a feature representation.

        Args:
            obs:         Raw observation dictionary from the environment.
            robot_state: Optional proprioceptive state vector of shape
                         ``[state_dim]``.  May be ``None`` if the encoder
                         does not require it.

        Returns:
            A (possibly transformed) observation dictionary.  The returned
            dict must be usable by downstream components; at minimum it
            should preserve any keys that later pipeline stages expect.
        """


# ---------------------------------------------------------------------------
# Default: identity – return the raw observation unchanged
# ---------------------------------------------------------------------------


class IdentityObsEncoder(BaseObsEncoder):
    """Pass-through encoder that returns the raw observation dict as-is.

    ``robot_state`` is ignored; it is accepted only to satisfy the interface.
    """

    def encode(
        self,
        obs: dict[str, Any],
        robot_state: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return *obs* unchanged.

        Args:
            obs:         Raw observation dictionary.
            robot_state: Ignored.

        Returns:
            The same ``obs`` dictionary, unmodified.
        """
        return obs

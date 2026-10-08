"""Retrieval of relevant in-distribution demonstrations."""

from abc import ABC, abstractmethod
from typing import Any


class DemoRetriever(ABC):
    """Abstract demo retriever interface."""

    @abstractmethod
    def retrieve(self, query: Any, k: int = 5) -> list:
        """Return the k most relevant demonstrations for the given query.

        Parameters
        ----------
        query:
            Query embedding or observation.
        k:
            Number of demonstrations to retrieve.

        Returns
        -------
        list
            List of demo metadata dicts (keys: ``demo_key``, ``score``, ...).
        """

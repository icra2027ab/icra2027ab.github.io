"""Lazy loader so optional models are only constructed when first needed."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from loguru import logger

T = TypeVar("T")


def resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    logger.warning("cuda requested but unavailable; falling back to cpu")
    return "cpu"


class LazyModel(Generic[T]):
    """Defer a model's construction to first ``get()``.

    Thread-safety is not attempted — callers are expected to initialize models
    from a single thread, as with the reference eval script.
    """

    def __init__(self, loader: Callable[[], T], name: str = "model") -> None:
        self._loader = loader
        self._instance: T | None = None
        self._name = name

    def is_loaded(self) -> bool:
        return self._instance is not None

    def get(self) -> T:
        if self._instance is None:
            logger.info(f"[lazy] loading {self._name}")
            self._instance = self._loader()
        return self._instance

    def unload(self) -> None:
        self._instance = None

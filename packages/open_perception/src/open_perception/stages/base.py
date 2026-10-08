"""Stage protocol and registry.

Stages declare their pipeline-data dependencies via two class-level (or
instance-level) attributes so `PerceptionConfig.validate()` can warn the
user about misordered or under-provisioned configs:

- ``requires``: tuple of context-field keys the stage reads. Validation
  warns if no earlier stage (or input) has produced them.
- ``provides``: tuple of context-field keys the stage writes / populates.
  Validation accumulates them as the pipeline order is walked.

Field-key vocabulary (keep these strings stable; validation matches by
exact string):

==============================  =============================================
Key                             Source
==============================  =============================================
``rgb``                         input (always present)
``depth``                       input — per-view, may be None
``camera.intrinsics``           input — per-view, may be None
``camera.extrinsics``           input — per-view, may be None
``instances``                   produced by ``sam3``
``mask_scores``                 produced by ``sam3``
``crops``                       produced by ``filter.size``
``bboxes``                      produced by ``filter.size``
``crop_features.<encoder>``     produced by ``disambiguate.clip`` / ``.dino``
``scores.<source>``             produced by ``disambiguate.*``
``detections``                  produced by ``sam3`` (raw) or ``assign``
``detections.pointcloud``       produced by ``reconstruct``
``detections.embeddings``       produced by ``encode``
==============================  =============================================
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from ..context import DetectionContext


@runtime_checkable
class Stage(Protocol):
    name: str

    def __call__(self, ctx: DetectionContext) -> DetectionContext: ...


# Stages may *optionally* declare these class attrs to opt into config-time
# dependency validation (`PerceptionConfig.validate`). They're read via
# ``getattr(stage, attr, default)`` so they're not part of the structural
# protocol contract — leaving them out is fine, the stage just won't
# contribute to dependency tracking.
#   requires: ClassVar[tuple[str, ...]]   # context-field keys read
#   provides: ClassVar[tuple[str, ...]]   # context-field keys written
#   runs_on_track: ClassVar[bool]         # also runs in tracking-only mode


@runtime_checkable
class MultiViewStage(Protocol):
    """Stage that operates on all views' contexts at once.

    Per-view stages (the existing ``Stage`` protocol) get one
    ``DetectionContext`` per call; multi-view stages receive the full
    list and may mutate any context — used for cross-view operations
    like instance-id alignment or projecting one view's masks into
    others.

    Identification: the runner detects multi-view stages via
    ``getattr(stage, "multiview", False)``. Implementations set the
    attribute to ``True``.
    """

    name: str
    multiview: bool

    def __call__(self, ctxs: list[DetectionContext]) -> list[DetectionContext]: ...


# Factories receive the OpenPerception backing services (model holders, config,
# reference bank getter, etc.) plus any stage-specific params, and return a
# Stage or MultiViewStage instance — the runner dispatches via the
# ``multiview`` marker attribute, not the protocol type.
StageFactory = Callable[..., "Stage | MultiViewStage"]


class Registry:
    def __init__(self) -> None:
        self._factories: dict[str, StageFactory] = {}

    def register(self, name: str) -> Callable[[StageFactory], StageFactory]:
        def decorator(factory: StageFactory) -> StageFactory:
            if name in self._factories:
                raise ValueError(f"stage {name!r} already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def add(self, name: str, factory: StageFactory) -> None:
        self._factories[name] = factory

    def get(self, name: str) -> StageFactory:
        if name not in self._factories:
            raise KeyError(f"stage {name!r} not registered")
        return self._factories[name]

    def names(self) -> set[str]:
        return set(self._factories.keys())


REGISTRY = Registry()

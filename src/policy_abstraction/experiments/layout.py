"""Scattering several named objects across a region of the table.

The disambiguation experiment asks a different question from the positional
one. There, one object moves and the prompt is fixed; here several objects sit
on the table at once, the prompt names one of them, and the thing being measured
is whether the policy goes to the object it was *told* to go to rather than to
the nearest one, the biggest one, or the one it saw most often in training.

That needs scene *layouts*: a set of positions, one per object, that

* lie inside a region the operator can reach and the camera can see,
* are far enough apart that "it went to the red mug" is unambiguous to the
  operator and to the policy alike — :data:`DEFAULT_MIN_SEPARATION`, 10 cm,
* and move between layouts, so that "which object" is decorrelated from "which
  place". Without that a policy that always grasps whatever is on the left
  would score 1/3 and look like it was reading the prompt.

Sampling is rejection sampling against the minimum separation, seeded, so the
same config yields the same layouts on every machine and both arms of a
baseline/guided comparison stage the identical scenes.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from policy_abstraction.experiments.grid import TableGrid

DEFAULT_MIN_SEPARATION = 0.10
"""Metres between object centres. Two objects closer than this are hard for an
operator to stage repeatably and hard to call a grasp between."""

DEFAULT_MARGIN = 0.05
"""Metres kept clear inside the region's edge, so an object's *body* stays in
the region even though only its centre is sampled."""

_ATTEMPTS_PER_POINT = 200
_RESTARTS = 200


@dataclasses.dataclass(frozen=True)
class SceneObject:
    """One physical object the operator puts on the table.

    Attributes:
        id: Filename- and record-safe identifier. Written into level ids and
            paths, so it must not change once collection has started.
        label: What the operator reads on the overlay and in the GUI.
        prompt_name: The noun phrase substituted into the task template — what
            the *policy* is told. Defaults to :attr:`label`. Kept separate
            because the two audiences want different things: an operator
            benefits from "red mug (tall, on its side)", a pi05 checkpoint
            wants exactly the phrasing it was trained on.
    """

    id: str
    label: str = ""
    prompt_name: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("a scene object needs a non-empty id")
        object.__setattr__(self, "label", self.label or self.id.replace("_", " "))
        object.__setattr__(self, "prompt_name", self.prompt_name or self.label)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "prompt_name": self.prompt_name}

    @classmethod
    def from_any(cls, data: "str | dict | SceneObject") -> "SceneObject":
        """Accept a bare name, a mapping, or an instance.

        A YAML author writing three objects should be able to write three
        strings; one who needs a different prompt phrasing writes a mapping.
        """
        if isinstance(data, SceneObject):
            return data
        if isinstance(data, str):
            return cls(id=_slug(data), label=data)
        if isinstance(data, dict):
            name = str(data.get("id") or data.get("name") or "")
            label = str(data.get("label") or data.get("description") or "")
            if not name and label:
                name = _slug(label)
            return cls(
                id=name,
                label=label,
                prompt_name=str(data.get("prompt_name") or data.get("prompt") or ""),
            )
        raise TypeError(f"cannot read a scene object from {type(data).__name__}")


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    if not slug:
        raise ValueError(f"cannot derive an id from {text!r}; give one explicitly")
    return slug


def parse_objects(
    items: Sequence["str | dict | SceneObject"],
) -> tuple[SceneObject, ...]:
    """Read the object list from a config, rejecting duplicate ids."""
    objects = tuple(SceneObject.from_any(item) for item in items)
    if not objects:
        raise ValueError("at least one scene object is required")
    ids = [o.id for o in objects]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(
            f"scene object ids must be unique; duplicated: {duplicates}. "
            "Two objects the prompt cannot tell apart are two objects the "
            "experiment cannot tell apart either."
        )
    return objects


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _capacity_hint(width: float, height: float, min_separation: float) -> int:
    """Roughly how many points of the given spacing fit in a ``width x height`` box.

    A hexagonal-packing estimate, used only to write a useful error message —
    the sampler's authority on feasibility is that it failed, not this number.
    """
    if min_separation <= 0:
        return 10**6
    rows = int(np.floor(height / (min_separation * np.sqrt(3) / 2))) + 1
    cols = int(np.floor(width / min_separation)) + 1
    return max(0, rows * cols)


def sample_positions_uv(
    width: float,
    height: float,
    count: int,
    *,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """``[count, 2]`` points in a ``width x height`` box, pairwise >= *min_separation*.

    Rejection sampling with restarts. Uniform-with-rejection rather than a
    jittered lattice because a lattice would put every object at one of a small
    set of places, and the whole point of re-sampling per layout is that the
    policy cannot learn the places.

    Raises:
        ValueError: if no arrangement was found. The message says what to
            relax — the region is too small, the separation too large, or there
            are too many objects — because those are the only three fixes.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if width <= 0 or height <= 0:
        raise ValueError(f"region must have positive extent, got {width} x {height}")
    rng = rng if rng is not None else np.random.default_rng(0)

    for _ in range(_RESTARTS):
        points: list[np.ndarray] = []
        for _ in range(count):
            for _ in range(_ATTEMPTS_PER_POINT):
                candidate = np.array(
                    [rng.uniform(0.0, width), rng.uniform(0.0, height)]
                )
                if all(
                    float(np.linalg.norm(candidate - p)) >= min_separation
                    for p in points
                ):
                    points.append(candidate)
                    break
            else:
                break
        if len(points) == count:
            return np.stack(points)

    raise ValueError(
        f"could not place {count} objects at least {min_separation * 100:.0f} cm "
        f"apart in a {width * 100:.0f} x {height * 100:.0f} cm area "
        f"(roughly {_capacity_hint(width, height, min_separation)} fit at best, "
        "and random placement needs slack on top of that). Enlarge the region, "
        "shrink `margin`, lower `min_separation`, or use fewer objects."
    )


def sample_layouts(
    region: TableGrid,
    objects: Sequence[SceneObject],
    *,
    count: int,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    margin: float = DEFAULT_MARGIN,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], ...]:
    """*count* layouts, each mapping every object id to a lifted world position.

    Each layout gets its own sub-seed, so adding a layout to a config leaves the
    earlier ones untouched — an experiment can be extended without invalidating
    the rollouts already collected against it.
    """
    if count < 1:
        raise ValueError(f"a layout factor needs at least one layout, got {count}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    width = region.extent_u - 2 * margin
    height = region.extent_v - 2 * margin
    if width <= 0 or height <= 0:
        raise ValueError(
            f"a {margin * 100:.0f} cm margin leaves nothing of a "
            f"{region.extent_u * 100:.0f} x {region.extent_v * 100:.0f} cm region. "
            "Shrink `margin` or enlarge the region."
        )

    layouts: list[dict[str, np.ndarray]] = []
    for index in range(count):
        uv = sample_positions_uv(
            width,
            height,
            len(objects),
            min_separation=min_separation,
            rng=np.random.default_rng([seed, index]),
        )
        layouts.append(
            {
                obj.id: region.point_at(
                    margin + float(u), margin + float(v), lifted=True
                )
                for obj, (u, v) in zip(objects, uv, strict=True)
            }
        )
    return tuple(layouts)


def layout_label(index: int) -> str:
    """Human-facing name for the *index*-th layout: ``L1``, ``L2``, …"""
    return f"L{index + 1}"


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_MIN_SEPARATION",
    "SceneObject",
    "layout_label",
    "parse_objects",
    "sample_layouts",
    "sample_positions_uv",
]

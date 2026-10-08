"""Segmentation for demos that carry none of their own.

A simulator writes a ground-truth mask and an id → name map next to every
frame. A real recording has neither: it is RGB, depth and proprioception, and
nothing in it labels its own pixels. Analysis needs those labels, so they have
to be produced.

:class:`PerceptionDemoSegmenter` produces them with the same open-vocabulary
stack the live rollout already uses — names from a VLM, masks from
``OpenPerceptionClient`` (detect on the first frame it sees, track after) — so
a retrieved demo and the live scene are segmented by the same machinery, and
their instance ids mean the same kind of thing on both sides of the
correspondence step.

It is written to the ``segmenter`` contract that
:class:`~policy_abstraction.analysis.demo_episode.VlaEpisode` expects::

    (rgb, depth_m, K, E, task) -> (labels, {instance_id: name})

so any other way of getting masks — a hand annotation, a different detector, a
cached mask file — can be dropped in without touching the analyzer.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

_SYSTEM_PROMPT = (
    "You name the physical objects a robot must reason about to perform a "
    "task, given a photograph of its workspace.\n\n"
    'Return JSON: {"names": [...], "reasoning": "..."}.\n\n'
    "Rules:\n"
    "- Name only objects visible in the image.\n"
    "- Use short, concrete, singular noun phrases a detector can ground "
    '("cable", "bag", "bowl"), not descriptions or roles.\n'
    "- Include the objects the task acts on and any object it references "
    "(containers, targets, landmarks).\n"
    "- Exclude the robot arm, its gripper, and the work surface itself.\n"
    "- Prefer 2-6 names. Lowercase."
)


def _encode(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class PerceptionDemoSegmenter:
    """Name a demo's objects once, then segment its frames from RGB-D."""

    def __init__(
        self,
        perception_client,
        vlm_model: str = "gpt-5.4",
        cache_dir: Path | str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        object_names: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        perception_client:
            An :class:`~policy_abstraction.perception.open_perception_client.OpenPerceptionClient`
            (or anything with its ``(rgbs, depths, cameras, names)`` call and a
            ``reset()``). **Not** the one driving the live scene: detection is
            stateful, and interleaving two scenes through one tracker would mix
            their instance ids. Construct a second client for the demo.
        object_names:
            Skip the VLM and use these names. Useful for tests and for re-runs
            where the vocabulary is already known.
        """
        self._perception = perception_client
        self._vlm_model = vlm_model
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._api_key = api_key
        self._base_url = base_url
        self._names: list[str] | None = list(object_names) if object_names else None
        self._started = False

    # ------------------------------------------------------------------
    # Object naming
    # ------------------------------------------------------------------

    def _cache_path(self, task: str) -> Path | None:
        if self._cache_dir is None:
            return None
        import hashlib

        digest = hashlib.sha1(task.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"demo_objects_{digest}.json"

    def object_names(self, rgb: np.ndarray, task: str) -> list[str]:
        """Names of the task-relevant objects in this demo, inferred once."""
        if self._names is not None:
            return self._names

        path = self._cache_path(task)
        if path is not None and path.exists():
            try:
                self._names = list(json.loads(path.read_text())["names"])
                return self._names
            except Exception as exc:
                logger.debug(f"[DemoSegmenter] Could not read name cache: {exc}")

        from policy_abstraction.vlm._client import make_vlm_client
        from policy_abstraction.vlm._retry import chat_completion_with_retry

        names: list[str] = []
        try:
            client = make_vlm_client(self._api_key, self._base_url)
            response = chat_completion_with_retry(
                client,
                log_prefix="DemoSegmenter",
                model=self._vlm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_encode(rgb)}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": f"Task: {task}"},
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(response.choices[0].message.content or "{}")
            seen: set[str] = set()
            for raw in data.get("names", []):
                if not isinstance(raw, str):
                    continue
                name = raw.strip().lower()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        except Exception as exc:
            logger.warning(f"[DemoSegmenter] VLM naming failed: {exc}")

        self._names = names
        if names and path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"task": task, "names": names}, indent=2))
            except Exception as exc:
                logger.debug(f"[DemoSegmenter] Could not write name cache: {exc}")
        return names

    # ------------------------------------------------------------------
    # segmenter contract
    # ------------------------------------------------------------------

    def __call__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        K: np.ndarray | None,
        E: np.ndarray | None,
        task: str,
    ) -> tuple[np.ndarray | None, dict[int, str]]:
        """Return ``(labels, {instance_id: name})`` for one frame.

        ``labels`` is ``(H, W)`` int32 with 0 for background, matching the
        convention of a simulator's segmentation buffer, so the same overlay
        and point-cloud code serves both.
        """
        if depth is None or K is None or E is None:
            logger.warning(
                "[DemoSegmenter] Frame has no depth/intrinsic/extrinsic; "
                "cannot segment. A recording needs --env.calibration-path and "
                "a non-zero --output.depth-stride to be analysable."
            )
            return None, {}

        names = self.object_names(rgb, task)
        if not names:
            logger.warning("[DemoSegmenter] No object names; cannot segment.")
            return None, {}

        from open_perception import CameraInfo  # lazy: sys.path-mounted

        if not self._started:
            self._perception.reset()
            self._started = True

        # open_perception reads depth in metres and ignores non-finite pixels;
        # NaN is how the episode marks "no return".
        try:
            per_view = self._perception(
                [np.asarray(rgb, dtype=np.uint8)],
                [np.asarray(depth, dtype=np.float32)],
                [CameraInfo(intrinsics=np.asarray(K), extrinsics=np.asarray(E))],
                names,
            )
        except Exception as exc:
            logger.warning(f"[DemoSegmenter] Perception call failed: {exc}")
            return None, {}

        masks = per_view[0] if per_view else {}
        if not masks:
            return None, {}

        labels = np.zeros(np.asarray(rgb).shape[:2], dtype=np.int32)
        id_to_name: dict[int, str] = {}
        # Paint smallest-first so a large container cannot bury the object
        # inside it; the same ordering the live overlay uses.
        for inst_id, (name, mask) in sorted(
            masks.items(), key=lambda kv: -int(np.count_nonzero(kv[1][1]))
        ):
            m = np.asarray(mask, dtype=bool)
            if m.shape != labels.shape or not m.any():
                continue
            labels[m] = int(inst_id)
            id_to_name[int(inst_id)] = str(name)

        return labels, id_to_name

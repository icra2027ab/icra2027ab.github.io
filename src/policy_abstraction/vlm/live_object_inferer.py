"""VLM-based inference of relevant object names for the current live scene.

Used by :class:`policy_abstraction.evaluation.abstraction.RolloutAbstractor`
to bootstrap perception in a real-life-parity setting where the live
environment does not provide a ground-truth segmentation name mapping.
The VLM consumes the task instruction, the retrieved demo's relevant
object list (with names + scene/seg-overlay images), and the live RGB,
and returns a list of live-object name strings. Those names are then
fed to an open-vocabulary segmenter (PREDICTED mode) or matched against
the GT scene name map (GT modes) to recover seg IDs.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PIL import Image

from policy_abstraction.vlm._client import make_vlm_client
from policy_abstraction.vlm._orientation import to_vlm_orientation
from policy_abstraction.vlm._retry import chat_completion_with_retry

if TYPE_CHECKING:
    from policy_abstraction.analysis.demo_analyzer import DemoAnalysisResult


class LiveObjectInferer:
    """Queries a VLM to infer relevant live-scene object names.

    Lifecycle: construct once per process (loads the system prompt and
    instantiates the OpenAI client). Call :meth:`infer` once per episode
    setup. Results are cached on disk by ``(task, demo_key)`` when a
    ``cache_dir`` is provided, so re-running the same episode does not
    incur a second VLM call.
    """

    _SYSTEM_PROMPT = (
        Path(__file__).parent / "prompts" / "live_object_inferer_system.txt"
    ).read_text()

    def __init__(
        self,
        model: str = "gpt-5.4",
        cache_dir: Path | str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = make_vlm_client(api_key, base_url)
        self._model = model
        self._cache_dir: Path | None = (
            Path(cache_dir) if cache_dir is not None else None
        )

    # ------------------------------------------------------------------
    # Cache helpers (mirror DemoAnalyzer's pattern)
    # ------------------------------------------------------------------

    def _cache_path(self, cache_key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        safe_key = cache_key.replace("/", "__").replace(" ", "_")
        # The ``_v2`` suffix retires entries written with the earlier prompt,
        # which echoed demo names ("blue cup") for live objects of another
        # colour; those cached names would keep sending the detector after
        # objects that are not on the table.
        return self._cache_dir / f"live_objects_v2__{safe_key}.json"

    def _load_cache(self, cache_key: str) -> list[str] | None:
        path = self._cache_path(cache_key)
        if path is None or not path.exists():
            return None
        try:
            with path.open() as f:
                data = json.load(f)
            names = data.get("names")
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                logger.info(f"[LiveObjectInferer] Loaded cached names from {path}")
                return names
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[LiveObjectInferer] Failed to load cache {path}: {exc}")
        return None

    def _save_cache(
        self,
        cache_key: str,
        names: list[str],
        reasoning: str,
        prompt_text: str,
        response_json: str,
    ) -> None:
        path = self._cache_path(cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(
                {
                    "names": names,
                    "reasoning": reasoning,
                    "prompt": prompt_text,
                    "response": response_json,
                },
                f,
                indent=2,
            )
        logger.info(f"[LiveObjectInferer] Cached names to {path}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def infer(
        self,
        task_description: str,
        demo_analysis: "DemoAnalysisResult",
        live_scene_image: np.ndarray,
        *,
        cache_key: str | None = None,
        live_object_vocabulary: list[str] | None = None,
        retrieved_task: str | None = None,
    ) -> list[str]:
        """Infer the names of live-scene objects relevant to the task.

        *task_description* is the live episode's instruction and drives which
        roles matter; *retrieved_task* is the demo's own instruction, given so
        the VLM can see how the two phrasings differ (colour names vs.
        positional references, say) instead of assuming the demo names
        describe the live objects. Names come back describing the live
        objects as they look, which is what the open-vocabulary detector
        needs.

        When *live_object_vocabulary* is supplied (GT modes, where the
        simulator's seg-name map is known), the VLM is constrained to
        return names verbatim from that list; off-list names are dropped
        with a warning. The vocabulary is also folded into the cache key
        so cached names from a different scene are not reused.

        Returns an empty list on VLM failure or empty response — callers
        should treat that as a fail-soft signal and disable abstraction
        for the episode.
        """
        effective_cache_key = cache_key
        if effective_cache_key is not None and live_object_vocabulary:
            vocab_fingerprint = hashlib.sha1(
                "|".join(sorted(live_object_vocabulary)).encode("utf-8")
            ).hexdigest()[:12]
            effective_cache_key = f"{cache_key}::vocab={vocab_fingerprint}"

        if effective_cache_key is not None:
            cached = self._load_cache(effective_cache_key)
            if cached is not None:
                return cached

        demo_objects_str = "\n".join(
            f"  {id_}: {name}" for id_, name in sorted(demo_analysis.objects.items())
        )
        demo_task_line = (
            f"Retrieved demo task: {retrieved_task}\n"
            if retrieved_task
            else "Retrieved demo task: (not available)\n"
        )
        prompt_text = (
            f"Live task: {task_description}\n\n"
            f"{demo_task_line}"
            f"Retrieved demo objects (id: name):\n{demo_objects_str}\n\n"
            "Name the live-scene objects the LIVE task needs, each described "
            "by how it looks in the live image (colour, material, print, "
            "shape + object type). The demo objects tell you which roles "
            "matter; do not reuse a demo name for a live object that looks "
            "different."
        )
        if live_object_vocabulary:
            vocab_lines = "\n".join(f"  - {n}" for n in live_object_vocabulary)
            prompt_text += (
                f"\n\nLive-scene candidate object names (closed vocabulary, "
                f"return names verbatim from this list only):\n{vocab_lines}"
            )

        user_content: list[dict] = []
        for img, _label in [
            (demo_analysis.scene_image, "Demo scene (RGB)"),
            (demo_analysis.seg_index_image, "Demo scene (segmentation overlay)"),
            (live_scene_image, "Live scene (RGB)"),
        ]:
            if img is not None:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._encode_image(img)}",
                            "detail": "low",
                        },
                    }
                )
        user_content.append({"type": "text", "text": prompt_text})

        response_json = "{}"
        names: list[str] = []
        reasoning = ""

        try:
            response = chat_completion_with_retry(
                self._client,
                log_prefix="LiveObjectInferer",
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},  # type: ignore[arg-type]
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            response_json = response.choices[0].message.content or "{}"
            data = json.loads(response_json)
            raw_names = data.get("names", [])
            reasoning = str(data.get("reasoning", ""))
            seen: set[str] = set()
            for n in raw_names:
                if not isinstance(n, str):
                    continue
                norm = n.strip().lower()
                if norm and norm not in seen:
                    seen.add(norm)
                    names.append(norm)
        except Exception as exc:
            logger.warning(f"[LiveObjectInferer] VLM query failed: {exc}")

        if live_object_vocabulary:
            allowed = {v.strip().lower() for v in live_object_vocabulary if v.strip()}
            kept: list[str] = []
            rejected: list[str] = []
            for n in names:
                (kept if n in allowed else rejected).append(n)
            if rejected:
                logger.warning(
                    f"[LiveObjectInferer] Dropping off-vocabulary names {rejected}; "
                    f"VLM must pick verbatim from the supplied live vocabulary "
                    f"({len(allowed)} candidates)."
                )
            names = kept

        if effective_cache_key is not None and names:
            self._save_cache(
                effective_cache_key, names, reasoning, prompt_text, response_json
            )

        return names

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        pil_img = Image.fromarray(to_vlm_orientation(img).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

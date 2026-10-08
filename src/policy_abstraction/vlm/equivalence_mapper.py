"""VLM-based semantic equivalence mapping between retrieved and live scene objects."""

from __future__ import annotations

import base64
import dataclasses
import io
import json

import numpy as np
from PIL import Image

from policy_abstraction.vlm._client import make_vlm_client
from policy_abstraction.vlm._orientation import to_vlm_orientation
from policy_abstraction.vlm._retry import chat_completion_with_retry
from policy_abstraction.vlm.types import (
    ROBOT_ENTITIES,
    DistancePair,
    VLMConversation,
    VLMMessage,
)


@dataclasses.dataclass
class ObjectMapping:
    """Result of VLM-based equivalence mapping between two scenes.

    Maps retrieved-demo object IDs to live-scene object IDs so that
    ``DistancePair`` instances computed on the retrieved demo can be
    re-expressed in terms of objects visible in the current episode.
    """

    retrieved_objects: dict[int, str]
    """``{seg_id: name}`` for the retrieved demo (source space)."""

    live_objects: dict[int, str]
    """``{seg_id: name}`` for the current live scene (target space)."""

    id_map: dict[int | str, int | str]
    """Mapping from retrieved ID (or ``"ee"``) → live ID (or ``"ee"``).

    ``"ee"`` is always mapped to ``"ee"``.  Retrieved IDs with no semantic
    counterpart in the live scene are absent from this dict.
    """

    unmapped_retrieved_ids: list[int]
    """Retrieved IDs for which no live counterpart was found."""

    vlm_prompt_text: str
    """Reconstructed text sent to the VLM."""

    vlm_response_json: str
    """Raw JSON string returned by the VLM."""

    conversation: VLMConversation | None = None
    """Full system + user (with images) + assistant trace, for visualization.

    Populated only on the live flow; serialised forms (e.g. recorder
    metadata) carry an equivalent base64-encoded dict on the side.
    """


class ObjectEquivalenceMapper:
    """Uses a VLM to establish semantic equivalence between two sets of objects.

    Given the object lists from a retrieved demo and the current live scene
    (plus optional context images for both), queries the VLM to identify which
    live object corresponds to each retrieved object.  The resulting
    :class:`ObjectMapping` is used to translate :class:`~policy_abstraction.vlm.types.DistancePair`
    instances from retrieved-ID space into live-scene-ID space.

    Parameters
    ----------
    model:
        OpenAI model name (e.g. ``"gpt-4o"`` or ``"gpt-5.4"``).
    api_key:
        Optional API key; falls back to ``OPENAI_API_KEY`` env var.
    """

    _SYSTEM_PROMPT = (
        "You are a robotics expert. You are given two lists of objects from two different "
        "robot manipulation scenes, together with the natural-language task description for each. "
        "The first list is from a *retrieved demonstration*; the second is from the *current live scene*. "
        "The two scenes are related but NOT identical: object names, categories, colours, object positions, and even the "
        "task itself may differ. The retrieved demo is being used as a reference for a (possibly different) "
        "live task, so you must reason about each object's *role* in its respective task rather than relying "
        "on lexical name overlap. Two objects with different names (e.g. 'mug' vs. 'cup', 'bowl' vs. 'plate', "
        "'red block' vs. 'cube') can still be semantically equivalent if they play the same role; conversely, "
        "two identically-named objects can be non-equivalent if the live task uses them differently. "
        "Carefully compare the retrieved-demo task and the live task: identify what is being manipulated, "
        "what acts as the target/container/landmark, and which objects are merely scene clutter, then map "
        "retrieved objects to the live object that fills the analogous role. "
        "If no equivalent exists for a retrieved object (e.g. its role is absent in the live task), omit it. "
        "\n\n"
        "DISAMBIGUATING MULTIPLE INSTANCES OF THE SAME OBJECT. When the live scene "
        "contains multiple objects with the same name or role (e.g. two bowls, three "
        "blocks), the live and/or retrieved task description almost always contains a "
        '*spatial referring expression* identifying the correct one ("the bowl next to '
        'the cookie box", "the bowl on the stove", "the bowl not between the plate '
        'and the ramekin", "top drawer"). You MUST parse this expression and use the '
        "provided 2D pixel centroids (``px=(u, v)``) and, when available, 3D world "
        "centroids (``world=(x, y, z)``) of every candidate plus the named referent(s) "
        "to pick the candidate whose centroid satisfies the relation:\n"
        '  * "next to X" / "beside X" -> smallest planar distance to X.\n'
        '  * "on X" / "on top of X" -> roughly centred over X in the image AND '
        "higher Z (world z) than X.\n"
        '  * "between A and B" -> roughly collinear with A and B and bracketed by them.\n'
        '  * "not between A and B" -> the candidate that is NOT bracketed by A and B.\n'
        'Negations ("not", "other than") invert the choice. Apply the same rule on '
        "the retrieved side: the demo's retrieved object also obeys its own referring "
        "expression, which lets you align the candidates symmetrically. "
        "If no spatial expression disambiguates a multi-candidate match, fall back to "
        "the segmentation overlay images and pick the candidate visually closest to the "
        "referent.\n\n"
        "Return JSON only, with the reasoning first and the mapping second: "
        '{"reasoning": "<concise step-by-step justification: identify the role of '
        "each retrieved object, the referring expression in each task, and how the "
        'provided centroids disambiguate multi-instance candidates>", '
        '"mapping": [{"retrieved_id": <int>, "live_id": <int>}, ...]}. '
        "The reasoning must come before the mapping in the JSON object. "
        "Output nothing outside the JSON."
    )

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = make_vlm_client(api_key, base_url)
        self._model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def map_objects(
        self,
        retrieved_objects: dict[int, str],
        live_objects: dict[int, str],
        retrieved_scene_image: np.ndarray | None = None,
        live_scene_image: np.ndarray | None = None,
        retrieved_seg_image: np.ndarray | None = None,
        live_seg_image: np.ndarray | None = None,
        retrieved_task: str | None = None,
        live_task: str | None = None,
        retrieved_centroids_2d: dict[int, tuple[int, int]] | None = None,
        live_centroids_2d: dict[int, tuple[int, int]] | None = None,
        live_centroids_3d: dict[int, np.ndarray] | None = None,
    ) -> ObjectMapping:
        """Query the VLM to match retrieved objects to live objects.

        Sends up to four images (retrieved scene, retrieved seg-overlay, live
        scene, live seg-overlay) plus a text summary of both object lists and,
        when available, the natural-language task description for each scene.

        Parameters
        ----------
        retrieved_objects:
            ``{seg_id: name}`` from the retrieved demo.
        live_objects:
            ``{seg_id: name}`` from the current live scene.
        retrieved_scene_image:
            Optional ``[H, W, 3]`` uint8 RGB of the retrieved demo first frame.
        live_scene_image:
            Optional ``[H, W, 3]`` uint8 RGB of the current live first frame.
        retrieved_seg_image:
            Optional ``[H, W, 3]`` uint8 segmentation overlay for the retrieved demo.
        live_seg_image:
            Optional ``[H, W, 3]`` uint8 segmentation overlay for the live scene.
        retrieved_task:
            Natural-language task description used in the retrieved demo. Helps
            the VLM reason about each retrieved object's role.
        live_task:
            Natural-language task description for the current live episode.
            The VLM is asked to attend to differences between the two tasks
            when establishing equivalence.

        Returns
        -------
        ObjectMapping
            Always returns a valid mapping; on VLM failure returns one with an
            empty ``id_map`` so callers can degrade gracefully.
        """

        def _format_object_line(
            sid: int,
            name: str,
            px: tuple[int, int] | None,
            world: np.ndarray | None,
        ) -> str:
            parts = [f"  {sid}: {name}"]
            if px is not None:
                parts.append(f"px=({int(px[0])}, {int(px[1])})")
            if world is not None:
                w = np.asarray(world, dtype=np.float64).ravel()
                if w.size >= 3:
                    parts.append(f"world=({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})")
            return "   ".join(parts)

        retrieved_str = "\n".join(
            _format_object_line(
                id_,
                name,
                (retrieved_centroids_2d or {}).get(id_),
                None,
            )
            for id_, name in sorted(retrieved_objects.items())
        )
        live_str = "\n".join(
            _format_object_line(
                id_,
                name,
                (live_centroids_2d or {}).get(id_),
                (live_centroids_3d or {}).get(id_),
            )
            for id_, name in sorted(live_objects.items())
        )

        labelled_images: list[tuple[str, np.ndarray]] = [
            (label, img)
            for label, img in [
                ("Retrieved scene (RGB)", retrieved_scene_image),
                ("Retrieved scene (segmentation overlay)", retrieved_seg_image),
                ("Live scene (RGB)", live_scene_image),
                ("Live scene (segmentation overlay)", live_seg_image),
            ]
            if img is not None
        ]

        image_legend = (
            "\n\nImages (in order):\n"
            + "\n".join(
                f"  {i + 1}. {label}" for i, (label, _) in enumerate(labelled_images)
            )
            if labelled_images
            else ""
        )
        retrieved_task_str = (retrieved_task or "").strip() or "(unspecified)"
        live_task_str = (live_task or "").strip() or "(unspecified)"
        task_block = (
            f'Retrieved demo task: "{retrieved_task_str}"\n\n'
            f'Live scene task:    "{live_task_str}"\n\n\n'
            "Note: object *names* in the two lists below come from independent "
            "labellers and may differ even when objects are semantically equivalent. "
            "The two tasks may also differ — attend to those differences when deciding "
            "which retrieved object plays the same role as a live object.\n\n"
        )
        has_retrieved_px = bool(retrieved_centroids_2d)
        has_live_px = bool(live_centroids_2d)
        has_live_world = bool(live_centroids_3d)
        retrieved_header = (
            "Retrieved demo objects (id: name"
            + ("   px=(u, v)" if has_retrieved_px else "")
            + "):"
        )
        live_header = (
            "Live scene objects (id: name"
            + ("   px=(u, v)" if has_live_px else "")
            + ("   world=(x, y, z)" if has_live_world else "")
            + "):"
        )
        prompt_text = (
            f"{task_block}"
            f"{retrieved_header}\n{retrieved_str}\n\n"
            f"{live_header}\n{live_str}"
            f"{image_legend}"
        )

        user_content: list[dict] = []
        for _label, img in labelled_images:
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

        vlm_response_json = "{}"
        raw_mapping: list[dict] = []

        try:
            response = chat_completion_with_retry(
                self._client,
                log_prefix="ObjectEquivalenceMapper",
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},  # type: ignore[arg-type]
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            vlm_response_json = response.choices[0].message.content or "{}"
            data = json.loads(vlm_response_json)
            raw_mapping = data.get("mapping", [])
        except Exception as exc:
            import warnings

            warnings.warn(
                f"ObjectEquivalenceMapper: VLM query failed: {exc}", stacklevel=2
            )

        id_map: dict[int | str, int | str] = {"ee": "ee"}
        for entry in raw_mapping:
            try:
                ret_id = int(entry["retrieved_id"])
                liv_id = int(entry["live_id"])
                if ret_id in retrieved_objects and liv_id in live_objects:
                    id_map[ret_id] = liv_id
            except (KeyError, ValueError, TypeError):
                continue

        unmapped = [rid for rid in retrieved_objects if rid not in id_map]

        conversation = VLMConversation(
            messages=[
                VLMMessage(role="system", text=self._SYSTEM_PROMPT),
                VLMMessage(
                    role="user",
                    text=prompt_text,
                    images=[img for _, img in labelled_images],
                ),
                VLMMessage(role="assistant", text=vlm_response_json),
            ],
            model=self._model,
        )

        return ObjectMapping(
            retrieved_objects=retrieved_objects,
            live_objects=live_objects,
            id_map=id_map,
            unmapped_retrieved_ids=unmapped,
            vlm_prompt_text=prompt_text,
            vlm_response_json=vlm_response_json,
            conversation=conversation,
        )

    def translate_pairs(
        self,
        pairs: list[DistancePair],
        mapping: ObjectMapping,
    ) -> list[DistancePair]:
        """Rewrite ``DistancePair`` instances to use live-scene IDs.

        Pairs whose ``from_entity`` or ``to_entity`` has no entry in
        ``mapping.id_map`` are dropped silently.  ``"ee"`` entities are always
        preserved (they map to themselves).

        Returns
        -------
        list[DistancePair]
            Translated pairs (may be shorter than *pairs*).
        """
        translated: list[DistancePair] = []
        for pair in pairs:
            from_e: int | str | None = (
                pair.from_entity
                if pair.from_entity in ROBOT_ENTITIES
                else mapping.id_map.get(int(pair.from_entity))  # type: ignore[arg-type]
            )
            to_e: int | str | None = (
                pair.to_entity
                if pair.to_entity in ROBOT_ENTITIES
                else mapping.id_map.get(int(pair.to_entity))  # type: ignore[arg-type]
            )
            if from_e is not None and to_e is not None:
                translated.append(
                    DistancePair(
                        from_entity=from_e,
                        to_entity=to_e,
                        label=pair.label,
                        distance_type=pair.distance_type,
                        axis=pair.axis,
                        plane=pair.plane,
                        ee_vector=pair.ee_vector,
                        subtraj_idx=pair.subtraj_idx,
                        reasoning=pair.reasoning,
                    )
                )
        return translated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        """Encode a uint8 numpy image ``[H, W, 3]`` to a base64 PNG string.

        Applies :func:`to_vlm_orientation` so the VLM receives a
        scene-consistent (image-left == scene-left) view, undoing the
        column mirror that PA's ``flip_vh`` reference baked in.
        """
        pil_img = Image.fromarray(to_vlm_orientation(img).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

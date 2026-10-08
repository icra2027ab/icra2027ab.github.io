"""SAM3 "segment everything" + VLM re-rank vs GT masks.

Pipeline:
  1. SAM3 with a generic text prompt (default ``"object"``) enumerates all
     instance masks in the scene.
  2. Instances are filtered (min area) and capped to ``max_instances``; the
     kept masks are numbered 1..N.
  3. An *annotated* image is built: the RGB scene with each mask tinted and
     its number drawn at the mask centroid (same convention as
     ``task_pair_selector`` uses for its ``seg_index_image``).
  4. The raw RGB + annotated image + list of requested object names are sent
     to an OpenAI VLM. The VLM returns ``{object_name: [instance_numbers]}``.
  5. The final prediction for each prompt is the union of the chosen SAM3
     masks; objects not mapped by the VLM get an empty mask.

Output schema matches the other perception-ablation scripts so numbers are
directly comparable.

Note: one VLM call per evaluation frame, so keep ``--max-demos-per-file``
low or you will burn tokens quickly.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import tyro
from _common import BaseArgs, _overlay_masks, normalize_name, run_pipeline
from loguru import logger
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from _common import FailureSample


@dataclasses.dataclass
class Args(BaseArgs):
    out_dir: str = "tmp/sam3_vlm_eval"
    sam3_model: str = "facebook/sam3"
    sam3_prompt: str = "object"
    """Generic text prompt used to make SAM3 enumerate all instances."""
    score_threshold: float = 0.05
    mask_threshold: float = 0.4
    min_mask_area: int = 50
    """Drop SAM3 instances with fewer than this many pixels (noise)."""
    max_instances: int = 40
    """Keep at most this many top-scoring SAM3 instances per frame."""
    vlm_model: str = "gpt-5.4"
    """OpenAI model to use for mask disambiguation."""
    image_detail: str = "high"
    """Image ``detail`` field sent to the VLM (``low`` or ``high``). The
    numbered markers are small, so ``high`` is usually worth the extra tokens."""
    annotation_alpha: float = 0.5
    """Mask-tint alpha for the annotated image."""
    annotation_font_scale: float = 0.4
    annotation_font_thickness: int = 1


# ---------------------------------------------------------------------------
# Annotation: overlay coloured masks and numbered labels on the RGB scene.
# ---------------------------------------------------------------------------


def _make_palette(n: int) -> np.ndarray:
    """Return ``n`` distinct RGB colours in [0, 255]."""
    cmap = plt.get_cmap("tab20")
    return (np.array([cmap(i % 20)[:3] for i in range(n)]) * 255.0).astype(np.uint8)


def annotate_instances(
    rgb: np.ndarray,
    instances: np.ndarray,
    *,
    alpha: float,
    font_scale: float,
    font_thickness: int,
) -> np.ndarray:
    """Return an RGB image with each mask tinted and numbered at its centroid.

    Instances are numbered 1..N so the VLM sees 1-indexed markers
    (easier to read than 0-indexed). Numbering matches the order of
    ``instances[i]`` (the caller controls this — e.g. score-sorted).
    """
    import cv2

    out = rgb.copy()
    n = instances.shape[0]
    if n == 0:
        return out

    palette = _make_palette(n)
    tinted = out.astype(np.float32)
    for i in range(n):
        m = instances[i]
        color = palette[i].astype(np.float32)
        tinted[m] = (1.0 - alpha) * tinted[m] + alpha * color
    out = np.clip(tinted, 0, 255).astype(np.uint8)

    # Draw numeric marker at the centroid of each mask with a filled background
    # rectangle for legibility. Colour matches the mask tint.
    for i in range(n):
        ys, xs = np.where(instances[i])
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(i + 1)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        pad = 3
        cv2.rectangle(
            out,
            (cx - tw // 2 - pad, cy - th - pad),
            (cx + tw // 2 + pad, cy + baseline + pad),
            (0, 0, 0),
            -1,
        )
        color = tuple(int(c) for c in palette[i].tolist())
        cv2.putText(
            out,
            label,
            (cx - tw // 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# VLM prompt
# ---------------------------------------------------------------------------


_VLM_SYSTEM_PROMPT = (
    "You are a robotics-vision expert. You are given two images of the same scene: "
    "(1) the raw RGB image, and (2) an annotated image where a segmentation model "
    "has produced candidate instance masks, each tinted with a distinct colour and "
    "labelled with a numeric marker (1, 2, 3, …) at its centroid. "
    "Along with the images you receive a list of object names that must be "
    "identified in the scene. "
    "For each requested object name, decide which numbered instance(s) from the "
    "annotated image correspond to that object. You may select multiple instances "
    "if the segmentation model split one object into several masks; prefer a "
    "single mask when one is clearly correct. If no candidate matches the object, "
    "return an empty list for that name. Only select from the numbers that "
    "actually appear in the annotated image.\n\n"
    'Return JSON only with the schema: {"assignments": [{"object": <name>, '
    '"instance_ids": [<int>, ...]}, ...]}. No explanation.'
)


def _encode_image(img: np.ndarray) -> str:
    pil_img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Sam3VlmPredictor:
    name = "sam3_vlm"

    def __init__(
        self,
        sam3_model_id: str,
        device: str,
        sam3_prompt: str,
        score_threshold: float,
        mask_threshold: float,
        min_mask_area: int,
        max_instances: int,
        vlm_model: str,
        image_detail: str,
        annotation_alpha: float,
        annotation_font_scale: float,
        annotation_font_thickness: int,
        api_key: str | None = None,
    ) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        from policy_abstraction.vlm._client import make_vlm_client

        self._torch = torch
        self.device = (
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if self.device != device:
            logger.warning(
                f"requested device={device} unavailable, falling back to {self.device}"
            )
        logger.info(f"loading {sam3_model_id} on {self.device}")
        self.sam3_processor = Sam3Processor.from_pretrained(sam3_model_id)
        self.sam3 = Sam3Model.from_pretrained(sam3_model_id).to(self.device).eval()

        self._client = make_vlm_client(api_key)
        self.vlm_model = vlm_model
        self.image_detail = image_detail

        self.sam3_prompt = sam3_prompt
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.min_mask_area = min_mask_area
        self.max_instances = max_instances
        self.annotation_alpha = annotation_alpha
        self.annotation_font_scale = annotation_font_scale
        self.annotation_font_thickness = annotation_font_thickness

    # ------------------------------------------------------------------

    def _run_sam3(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(instances[N,H,W] bool, scores[N])`` sorted by descending score."""
        torch = self._torch
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)
        img_inputs = self.sam3_processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            vision_embeds = self.sam3.get_vision_features(
                pixel_values=img_inputs.pixel_values
            )
        text_inputs = self.sam3_processor(
            text=self.sam3_prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.sam3(vision_embeds=vision_embeds, **text_inputs)
        target_sizes = img_inputs.get("original_sizes")
        target_sizes = target_sizes.tolist() if target_sizes is not None else [[H, W]]
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )[0]
        masks = results.get("masks")
        scores = results.get("scores")
        if masks is None or len(masks) == 0:
            return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
        if hasattr(masks, "cpu"):
            inst = masks.cpu().numpy().astype(bool)
        else:
            inst = np.stack([np.asarray(m).astype(bool) for m in masks], axis=0)
        if scores is not None and hasattr(scores, "cpu"):
            score_arr = scores.cpu().numpy().astype(np.float32)
        elif scores is not None:
            score_arr = np.asarray(scores, dtype=np.float32)
        else:
            score_arr = np.zeros((inst.shape[0],), dtype=np.float32)
        order = np.argsort(-score_arr)
        inst = inst[order]
        score_arr = score_arr[order]
        return inst, score_arr

    def _filter(
        self, instances: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop too-small instances and cap to ``max_instances``."""
        if instances.shape[0] == 0:
            return instances, scores
        keep: list[int] = []
        for i in range(instances.shape[0]):
            if int(instances[i].sum()) < self.min_mask_area:
                continue
            keep.append(i)
            if len(keep) >= self.max_instances:
                break
        if not keep:
            return (
                np.zeros((0, *instances.shape[1:]), dtype=bool),
                np.zeros((0,), dtype=np.float32),
            )
        return instances[keep], scores[keep]

    def _query_vlm(
        self, rgb: np.ndarray, annotated: np.ndarray, prompts: list[str], n_inst: int
    ) -> tuple[dict[str, list[int]], str, str]:
        """Ask the VLM to map each prompt to a list of instance numbers.

        Returns ``({prompt: [1-based ids], ...}, raw_response_json, user_text)``.
        Invalid IDs (outside ``1..n_inst``) are filtered out. ``user_text`` is
        the text block sent to the VLM (for failure-case introspection).
        """
        objects_str = "\n".join(f"  - {p}" for p in prompts)
        markers_str = ", ".join(str(i + 1) for i in range(n_inst)) or "(none)"
        text = (
            f"Candidate instance markers visible in the annotated image: {markers_str}.\n\n"
            f"Objects to identify (return a list entry for every one, even if empty):\n"
            f"{objects_str}"
        )
        user_content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image(rgb)}",
                    "detail": self.image_detail,
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_image(annotated)}",
                    "detail": self.image_detail,
                },
            },
            {"type": "text", "text": text},
        ]
        raw = "{}"
        try:
            response = self._client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},  # type: ignore[arg-type]
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = response.choices[0].message.content or "{}"
        except Exception as exc:
            logger.warning(f"VLM request failed: {exc}")
            return {p: [] for p in prompts}, raw, text

        assignments: dict[str, list[int]] = {p: [] for p in prompts}
        try:
            data = json.loads(raw)
            for entry in data.get("assignments", []):
                name = entry.get("object")
                ids = entry.get("instance_ids", [])
                if not isinstance(name, str) or name not in assignments:
                    continue
                clean: list[int] = []
                for v in ids:
                    try:
                        iv = int(v)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= iv <= n_inst:
                        clean.append(iv)
                # De-dup while preserving order.
                seen: set[int] = set()
                assignments[name] = [x for x in clean if not (x in seen or seen.add(x))]
        except json.JSONDecodeError as exc:
            logger.warning(f"VLM returned non-JSON response: {exc}")
        return assignments, raw, text

    # ------------------------------------------------------------------

    def predict(self, rgb: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        H, W = rgb.shape[:2]
        empty = np.zeros((H, W), dtype=bool)
        out: dict[str, np.ndarray] = {p: empty.copy() for p in prompts}

        # Per-frame debug state for failure-sample rendering. Reset every call.
        self.last_debug: dict[str, Any] = {
            "rgb": rgb,
            "instances": np.zeros((0, H, W), dtype=bool),
            "annotated": rgb.copy(),
            "assignments": {p: [] for p in prompts},
            "raw_response": "{}",
            "user_text": "",
            "system_prompt": _VLM_SYSTEM_PROMPT,
            "vlm_model": self.vlm_model,
        }

        instances, scores = self._run_sam3(rgb)
        instances, scores = self._filter(instances, scores)
        if instances.shape[0] == 0:
            logger.warning("SAM3 produced no instances for this frame")
            return out

        annotated = annotate_instances(
            rgb,
            instances,
            alpha=self.annotation_alpha,
            font_scale=self.annotation_font_scale,
            font_thickness=self.annotation_font_thickness,
        )

        assignments, raw_response, user_text = self._query_vlm(
            rgb, annotated, prompts, instances.shape[0]
        )

        for prompt, ids in assignments.items():
            if not ids:
                continue
            mask = np.zeros((H, W), dtype=bool)
            for iv in ids:
                mask |= instances[iv - 1]
            out[prompt] = mask

        self.last_debug["instances"] = instances
        self.last_debug["annotated"] = annotated
        self.last_debug["assignments"] = assignments
        self.last_debug["raw_response"] = raw_response
        self.last_debug["user_text"] = user_text
        return out


# ---------------------------------------------------------------------------
# Failure visualisation: show annotated image + chosen markers + GT.
# ---------------------------------------------------------------------------


def _failure_extra(predictor: Sam3VlmPredictor, prompt: str) -> dict[str, Any]:
    """Snapshot the predictor's last-frame state for this prompt.

    Must copy arrays — ``last_debug`` is overwritten on the next frame. The
    whole VLM query (system prompt + user text + images) and raw response
    are captured so a failure case can be fully reproduced offline.
    """
    d = predictor.last_debug
    return {
        "annotated": d["annotated"].copy(),
        "instances": d["instances"].copy(),
        "assignments": {k: list(v) for k, v in d["assignments"].items()},
        "chosen_ids": list(d["assignments"].get(prompt, [])),
        "raw_response": d["raw_response"],
        "user_text": d["user_text"],
        "system_prompt": d["system_prompt"],
        "vlm_model": d["vlm_model"],
    }


def _save_vlm_failure_figure(out_path: "Path", s: "FailureSample") -> None:
    """Save the failure figure plus a sidecar text file with the full VLM query.

    The PNG is a 2x2 grid of [raw RGB | annotated VLM input | final prediction
    | GT mask]. Alongside it, ``<out_path>.txt`` records the system prompt,
    user prompt text, VLM model, raw JSON response, and parsed per-prompt
    assignments — everything needed to reproduce or audit the call.
    """
    annotated = s.extra.get("annotated")
    chosen_ids = s.extra.get("chosen_ids", [])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].imshow(s.rgb)
    axes[0, 0].set_title("raw RGB (VLM input 1)", fontsize=9)
    axes[0, 0].axis("off")

    if annotated is not None:
        axes[0, 1].imshow(annotated)
        axes[0, 1].set_title("annotated candidates (VLM input 2)", fontsize=9)
    else:
        axes[0, 1].imshow(s.rgb)
        axes[0, 1].set_title("annotated (missing)", fontsize=9)
    axes[0, 1].axis("off")

    pred_overlay = _overlay_masks(
        s.rgb, s.pred_mask[None, ...], colors=np.array([[230, 60, 60]])
    )
    axes[1, 0].imshow(pred_overlay)
    chosen_str = ", ".join(str(i) for i in chosen_ids) if chosen_ids else "(none)"
    axes[1, 0].set_title(
        f"final pred '{s.prompt}'  IoU={s.iou:.3f}\n"
        f"area={int(s.pred_mask.sum())}px  chosen=[{chosen_str}]",
        fontsize=9,
    )
    axes[1, 0].axis("off")

    gt_overlay = _overlay_masks(
        s.rgb, s.gt_mask[None, ...], colors=np.array([[60, 200, 90]])
    )
    axes[1, 1].imshow(gt_overlay)
    axes[1, 1].set_title(f"GT '{s.prompt}'  area={int(s.gt_mask.sum())}px", fontsize=9)
    axes[1, 1].axis("off")

    fig.suptitle(f"{s.suite} :: {s.demo}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # Sidecar with the full VLM query + response. Name mirrors the PNG so
    # they stay paired when sorted.
    sidecar = out_path.with_suffix(out_path.suffix + ".txt")
    assignments = s.extra.get("assignments", {})
    lines = [
        f"frame:        {s.suite} :: {s.demo}",
        f"target name:  {s.prompt}",
        f"iou:          {s.iou:.4f}",
        f"pred area:    {int(s.pred_mask.sum())} px",
        f"gt area:      {int(s.gt_mask.sum())} px",
        f"vlm_model:    {s.extra.get('vlm_model', '?')}",
        f"chosen ids:   {chosen_ids}",
        "",
        "=== VLM system prompt ===",
        s.extra.get("system_prompt", ""),
        "",
        "=== VLM user text ===",
        s.extra.get("user_text", ""),
        "",
        "=== VLM raw response ===",
        s.extra.get("raw_response", ""),
        "",
        "=== parsed assignments (all prompts this frame) ===",
    ]
    for name, ids in assignments.items():
        lines.append(f"  {name}: {ids}")
    sidecar.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    predictor = Sam3VlmPredictor(
        sam3_model_id=args.sam3_model,
        device=args.device,
        sam3_prompt=args.sam3_prompt,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_mask_area=args.min_mask_area,
        max_instances=args.max_instances,
        vlm_model=args.vlm_model,
        image_detail=args.image_detail,
        annotation_alpha=args.annotation_alpha,
        annotation_font_scale=args.annotation_font_scale,
        annotation_font_thickness=args.annotation_font_thickness,
    )

    tag = f"vlm_s{args.score_threshold:.2f}_m{args.mask_threshold:.2f}"
    extra_meta: dict[str, Any] = {
        "sam3_model": args.sam3_model,
        "sam3_prompt": args.sam3_prompt,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "min_mask_area": args.min_mask_area,
        "max_instances": args.max_instances,
        "vlm_model": args.vlm_model,
        "image_detail": args.image_detail,
    }
    run_pipeline(
        args,
        predictor,
        tag=tag,
        extra_meta=extra_meta,
        failure_figure_fn=_save_vlm_failure_figure,
        failure_extra_fn=_failure_extra,
    )


if __name__ == "__main__":
    # ``normalize_name`` is imported for parity with the other ablations (the
    # driver handles normalisation); referenced here to quiet lint warnings.
    _ = normalize_name
    main(tyro.cli(Args))

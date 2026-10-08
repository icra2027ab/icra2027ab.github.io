"""Streaming video-tracker wrappers for HF mask-propagation models.

Both SAM3's `Sam3TrackerVideoModel` + `Sam3TrackerVideoProcessor` and SAM2's
`Sam2VideoModel` + `Sam2VideoProcessor` expose the same `InferenceSession`
streaming surface:

  1. `initialize(first_frame, {obj_id: mask_at_frame0})`
       Starts a session with ``video=None``, adds the first frame, registers
       one conditioning mask per object, runs the model at frame 0.
  2. `step(next_frame)`
       Preprocesses the frame and calls
       ``model.forward(session, frame=pixel_values)``. The model's streaming
       path appends the frame and propagates memory from conditioning frames.
  3. `reset()` clears the session.

`BaseVideoMaskTracker` implements that flow once; `Sam3VideoTracker` is a
thin subclass that loads the SAM3 classes. See `sam2_tracker.Sam2VideoTracker`
for the faster, mask-only SAM2 variant (SAM2 cannot segment from text but
propagates given masks more cheaply).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ..utils.iou import mask_iou


class BaseVideoMaskTracker:
    """Streaming-session wrapper shared by the SAM2/SAM3 video trackers."""

    # Subclasses override to load the concrete (model, processor) pair.
    def _load(self, model_id: str, device: str) -> tuple[Any, Any]:
        raise NotImplementedError

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        *,
        log_tag: str = "video_tracker",
        dtype: Any | None = None,
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.model_id = model_id
        self._log_tag = log_tag
        # Pick bf16 automatically on CUDA where supported — fp32 inference
        # here costs 2-3x on Ampere+ for no visible quality difference.
        if dtype is None:
            if (
                device == "cuda"
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            ):
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        self.dtype = dtype
        logger.info(f"[{log_tag}] loading {model_id} on {device} ({dtype})")
        self.model, self.processor = self._load(model_id, device)
        if dtype is not torch.float32:
            self.model = self.model.to(dtype=dtype)

        self._session: Any | None = None
        self._frame_idx: int = -1
        self._obj_ids: list[int] = []

    # ------------------------------------------------------------------

    def initialize(
        self,
        first_frame: np.ndarray,
        masks_by_obj_id: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Start a new session, seed each object with its frame-0 mask.

        Returns the refined masks for frame 0 as ``{obj_id: mask[H, W] bool}``.
        """
        torch = self._torch
        H, W = first_frame.shape[:2]

        # Per-frame mask-memory features and vision-feature caches default to
        # `inference_device` (CUDA), so output_dict_per_obj grows on the GPU
        # every step() with no eviction — long episodes OOM in the memory
        # encoder. Force per-frame state to CPU (the processor's intended
        # default); the model copies the last `num_maskmem` slices back to
        # GPU per forward via .to(device, non_blocking=True).
        self._session = self.processor.init_video_session(
            video=None,
            inference_device=self.device,
            inference_state_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        pixel_values = self._preprocess_frame(first_frame)
        # Register frame 0 via add_new_frame (explicit index) so the conditioning
        # call knows which frame to attach the mask to.
        self._session.add_new_frame(pixel_values=pixel_values, frame_idx=0)
        self._frame_idx = 0
        self._obj_ids = sorted(masks_by_obj_id.keys())

        # Add every object in a single call — looping per-object re-triggers
        # the default `clear_old_inputs=True`, which wipes prior objects'
        # conditioning masks and leaves the session in an inconsistent state
        # (failures surface as "maskmem_features ... cannot be empty").
        self.processor.add_inputs_to_inference_session(
            inference_session=self._session,
            frame_idx=0,
            obj_ids=self._obj_ids,
            input_masks=[
                masks_by_obj_id[oid].astype(np.uint8) for oid in self._obj_ids
            ],
            original_size=(H, W),
        )

        with torch.inference_mode():
            output = self.model(inference_session=self._session, frame_idx=0)
        return self._postprocess(output, (H, W))

    def step(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        """Propagate to the next frame and return ``{obj_id: mask[H, W] bool}``."""
        if self._session is None:
            raise RuntimeError(
                "tracker session not initialized; call initialize() first"
            )
        torch = self._torch
        H, W = frame.shape[:2]
        pixel_values = self._preprocess_frame(frame)
        # Streaming mode: pass `frame=` — the model runs add_new_frame internally.
        with torch.inference_mode():
            output = self.model(inference_session=self._session, frame=pixel_values)
        self._frame_idx += 1
        return self._postprocess(output, (H, W))

    def add_object_with_points(
        self,
        frame_rgb: np.ndarray,
        obj_id: int,
        points_uv: np.ndarray,
        point_labels: np.ndarray,
    ) -> np.ndarray:
        """Mid-stream: register a new tracked object via point prompts.

        ``points_uv`` is ``(P, 2)`` ``(x, y)`` pixel coords; ``point_labels``
        is ``(P,)`` ints (1 = foreground, 0 = background). Uses
        ``clear_old_inputs=False`` so prior conditioning frames + objects
        survive (the same failure mode flagged in ``initialize`` applies).
        Returns the refined mask for ``obj_id`` at the current frame.
        """
        return self._add_object(
            frame_rgb,
            obj_id,
            input_points=[[points_uv.astype(float).tolist()]],
            input_labels=[[point_labels.astype(int).tolist()]],
        )

    def add_object_with_mask(
        self, frame_rgb: np.ndarray, obj_id: int, mask: np.ndarray
    ) -> np.ndarray:
        """Mid-stream: register a new tracked object via a mask prompt.

        Same semantics as ``add_object_with_points`` but the prompt is
        an existing ``(H, W)`` boolean mask instead of point coords.
        """
        return self._add_object(
            frame_rgb,
            obj_id,
            input_masks=[mask.astype(np.uint8)],
        )

    def _add_object(
        self,
        frame_rgb: np.ndarray,
        obj_id: int,
        **prompt_kwargs,
    ) -> np.ndarray:
        if self._session is None:
            raise RuntimeError(
                "tracker session not initialized; call initialize() first"
            )
        torch = self._torch
        H, W = frame_rgb.shape[:2]
        frame_idx = max(self._frame_idx, 0)
        self.processor.add_inputs_to_inference_session(
            inference_session=self._session,
            frame_idx=frame_idx,
            obj_ids=[obj_id],
            original_size=(H, W),
            clear_old_inputs=False,
            **prompt_kwargs,
        )
        if obj_id not in self._obj_ids:
            self._obj_ids = sorted(self._obj_ids + [obj_id])
        with torch.inference_mode():
            output = self.model(inference_session=self._session, frame_idx=frame_idx)
        masks = self._postprocess(output, (H, W))
        return masks.get(obj_id, np.zeros((H, W), dtype=bool))

    def reset(self) -> None:
        if self._session is not None:
            try:
                self._session.reset_inference_session()
            except Exception as exc:
                logger.warning(
                    f"[{self._log_tag}] reset_inference_session failed: {exc}"
                )
        self._session = None
        self._frame_idx = -1
        self._obj_ids = []
        # Force Python GC then release freed CUDA tensors back to the device.
        # reset_inference_session() clears Python references to the SAM2/SAM3
        # memory bank tensors, but PyTorch keeps them in its allocator pool
        # until empty_cache() is called — causing unbounded growth across episodes.
        import gc

        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def is_initialized(self) -> bool:
        return self._session is not None

    # ------------------------------------------------------------------
    # Internals

    def _preprocess_frame(self, frame: np.ndarray):
        torch = self._torch
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            out = image_processor(images=frame, return_tensors="pt")
            tensor = out.get("pixel_values")
            if tensor is not None:
                return tensor.to(device=self.device, dtype=self.dtype)
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _postprocess(
        self, output: Any, original_size: tuple[int, int]
    ) -> dict[int, np.ndarray]:
        pred_masks = getattr(output, "pred_masks", None)
        if pred_masks is None and isinstance(output, dict):
            pred_masks = output.get("pred_masks")
        if pred_masks is None:
            return {
                obj_id: np.zeros(original_size, dtype=bool) for obj_id in self._obj_ids
            }
        post = self.processor.post_process_masks(
            [pred_masks], original_sizes=[list(original_size)], binarize=True
        )[0]
        if hasattr(post, "cpu"):
            post = post.cpu().numpy()
        arr = np.asarray(post).astype(bool)
        # Expected shape: (num_objs, 1, H, W) — squeeze the channel dim.
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim == 2:
            # Single-object edge case → (1, H, W).
            arr = arr[None]
        out: dict[int, np.ndarray] = {}
        n = arr.shape[0]
        for obj_idx in range(n):
            try:
                obj_id = int(self._session.obj_idx_to_id(obj_idx))
            except Exception:
                obj_id = (
                    self._obj_ids[obj_idx] if obj_idx < len(self._obj_ids) else obj_idx
                )
            out[obj_id] = arr[obj_idx]
        return out


class Sam3VideoTracker(BaseVideoMaskTracker):
    """Thin wrapper around `Sam3TrackerVideoModel` for streaming propagation."""

    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda") -> None:
        super().__init__(model_id, device, log_tag="sam3_video")

    def _load(self, model_id: str, device: str):
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
        model = Sam3TrackerVideoModel.from_pretrained(model_id).to(device).eval()
        return model, processor


def drift_ok(new_mask: np.ndarray, anchor_mask: np.ndarray, threshold: float) -> bool:
    """Return True iff the propagated mask's IoU with the anchor exceeds `threshold`."""
    if not new_mask.any():
        return False
    return mask_iou(new_mask, anchor_mask) >= threshold

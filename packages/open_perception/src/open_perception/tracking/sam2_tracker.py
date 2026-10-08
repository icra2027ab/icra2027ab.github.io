"""SAM2 video-tracker wrapper.

SAM2 cannot segment from text prompts — use it as a faster alternative to
`Sam3VideoTracker` once SAM3 has produced the initial masks at frame 0.
Both trackers share the same streaming `InferenceSession` API, so swapping
between them only changes which HF weights are loaded.
"""

from __future__ import annotations

from .sam3_tracker import BaseVideoMaskTracker


class Sam2VideoTracker(BaseVideoMaskTracker):
    """Thin wrapper around `Sam2VideoModel` for streaming mask propagation."""

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-tiny",
        device: str = "cuda",
    ) -> None:
        super().__init__(model_id, device, log_tag="sam2_video")

    def _load(self, model_id: str, device: str):
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        processor = Sam2VideoProcessor.from_pretrained(model_id)
        model = Sam2VideoModel.from_pretrained(model_id).to(device).eval()
        return model, processor

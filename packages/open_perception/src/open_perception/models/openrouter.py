"""OpenRouter (OpenAI-wire) open-vocabulary box/point detector.

Routes the same detection prompt as :class:`GeminiBoxDetector` through any
OpenAI-compatible vision model served by OpenRouter — e.g.
``google/gemini-3-flash-preview``, ``openai/gpt-4o``,
``qwen/qwen2.5-vl-72b-instruct``. The image is sent as a base64 ``data:`` URL in
the chat-completions ``image_url`` content block.

The OpenAI SDK is imported lazily by ``utils.openai_client.make_openai_client``,
which resolves ``OPENROUTER_API_KEY`` (preferred) / ``OPENAI_API_KEY`` and the
OpenRouter base URL. Model ids are passed verbatim (OpenRouter-namespaced).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ..utils.openai_client import OPENROUTER_DEFAULT_BASE_URL, make_openai_client
from .box_detect import VlmBoxDetectorBase


@dataclass
class OpenRouterBoxDetector(VlmBoxDetectorBase):
    """Box/point proposer backed by an OpenAI-wire model via OpenRouter."""

    model_id: str = "google/gemini-3-flash-preview"
    base_url: str | None = None
    max_tokens: int = 4096

    def _ensure_client(self):
        if self._client is None:
            # Default to OpenRouter so OPENROUTER_API_KEY is used (and an
            # OpenAI key isn't mis-routed to the OpenRouter endpoint).
            self._client = make_openai_client(
                base_url=self.base_url or OPENROUTER_DEFAULT_BASE_URL
            )
        return self._client

    def _generate(self, rgb: np.ndarray, prompt: str) -> str | None:
        from PIL import Image

        client = self._ensure_client()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        kwargs: dict = {"max_tokens": self.max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — surface as no-detections, not a crash
            logger.warning(f"[openrouter] request failed ({self.model_id}): {exc}")
            return None
        if not resp.choices:
            return None
        return resp.choices[0].message.content

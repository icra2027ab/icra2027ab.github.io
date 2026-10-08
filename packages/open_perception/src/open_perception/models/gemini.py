"""Gemini-native open-vocabulary box/point detector.

Talks Google's ``generateContent`` API directly. The prompts, JSON parsing, and
0-1000 → pixel conversion live on :class:`VlmBoxDetectorBase`; this subclass only
implements the request transport.

The ``google-genai`` SDK is an optional dependency (``open-perception[gemini]``)
and is imported lazily inside the methods, so ``import open_perception`` and
eager pipeline preloading work even when the extra / API key are absent. The
genai client itself is created on first ``detect()`` call and reads the API key
from the environment (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``).

The robotics-ER models are not served on OpenRouter; for OpenAI-wire models
(e.g. ``google/gemini-3-flash-preview``) use
:class:`open_perception.models.openrouter.OpenRouterBoxDetector` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .box_detect import VlmBoxDetectorBase


@dataclass
class GeminiBoxDetector(VlmBoxDetectorBase):
    """Box/point proposer backed by Google's native Gemini API."""

    model_id: str = "gemini-robotics-er-1.6-preview"
    thinking_budget: int = 0

    def _ensure_client(self) -> Any:
        if self._client is None:
            from google import genai  # type: ignore  # optional `gemini` extra

            # Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
            self._client = genai.Client()
        return self._client

    def _generate(self, rgb: np.ndarray, prompt: str) -> str | None:
        from google.genai import types as genai_types  # type: ignore  # optional
        from PIL import Image

        client = self._ensure_client()
        response = client.models.generate_content(
            model=self.model_id,
            contents=[Image.fromarray(rgb), prompt],
            config=genai_types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
            ),
        )
        return response.text

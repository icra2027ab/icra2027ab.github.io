"""Factory for the OpenAI-compatible client shared by every VLM client.

OpenRouter (https://openrouter.ai/api/v1) speaks the OpenAI wire protocol, so
the same ``openai`` SDK can target either provider. This factory recognizes a
dedicated ``OPENROUTER_API_KEY`` so VLM calls "just work" against OpenRouter
without any code change at the call sites.

Resolution order:

1. An explicit ``api_key`` argument (with optional ``base_url``).
2. ``OPENROUTER_API_KEY`` -> routed to OpenRouter; ``base_url`` defaults to
   ``$OPENROUTER_BASE_URL`` or ``https://openrouter.ai/api/v1``.
3. ``OPENAI_API_KEY`` -> default OpenAI endpoint (the ``openai`` SDK still
   honors ``$OPENAI_BASE_URL`` when ``base_url`` is ``None``).

Model names are NOT rewritten: when targeting OpenRouter, pass an OpenRouter
model id (e.g. ``"openai/gpt-4o"``, ``"anthropic/claude-3.5-sonnet"``).
"""

from __future__ import annotations

import os

import openai

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def make_vlm_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> openai.OpenAI:
    """Build an :class:`openai.OpenAI` client honoring ``OPENROUTER_API_KEY``.

    See the module docstring for the resolution order. Raises ``RuntimeError``
    when no key is available so the caller sees a clear, actionable message
    (unlike a bare ``KeyError`` on ``os.environ``).
    """
    if api_key is not None:
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return openai.OpenAI(
            api_key=openrouter_key,
            base_url=base_url
            or os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL),
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return openai.OpenAI(api_key=openai_key, base_url=base_url)

    raise RuntimeError(
        "No VLM API key found. Set OPENROUTER_API_KEY (routed to OpenRouter) "
        "or OPENAI_API_KEY, or pass api_key=... explicitly."
    )

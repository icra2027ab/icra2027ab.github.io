"""Build an OpenAI-compatible client, honoring ``OPENROUTER_API_KEY``.

OpenRouter (https://openrouter.ai/api/v1) speaks the OpenAI wire protocol, so
the same ``openai`` SDK can target either provider. Set ``OPENROUTER_API_KEY``
to route VLM calls through OpenRouter; otherwise ``OPENAI_API_KEY`` is used.

The ``openai`` package is an optional dependency (the ``vlm`` extra), so it is
imported lazily inside the functions — importing this module never requires it.
"""

from __future__ import annotations

import os
from typing import Any

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def vlm_api_key_present() -> bool:
    """True when either ``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY`` is set."""
    return bool(
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )


def make_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Return an ``openai.OpenAI`` client, preferring OpenRouter when configured.

    Resolution order:

    1. An explicit ``api_key`` argument (with optional ``base_url``).
    2. ``OPENROUTER_API_KEY`` -> OpenRouter (``base_url`` defaults to
       ``$OPENROUTER_BASE_URL`` or ``https://openrouter.ai/api/v1``).
    3. ``OPENAI_API_KEY`` -> default OpenAI endpoint.

    Model names are NOT rewritten: when targeting OpenRouter, pass an
    OpenRouter model id (e.g. ``"openai/gpt-4o"``).
    """
    from openai import OpenAI

    if api_key is not None:
        return OpenAI(api_key=api_key, base_url=base_url)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url=base_url
            or os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL),
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key, base_url=base_url)

    raise RuntimeError(
        "No VLM API key found. Set OPENROUTER_API_KEY (routed to OpenRouter) "
        "or OPENAI_API_KEY, or pass api_key=... explicitly."
    )

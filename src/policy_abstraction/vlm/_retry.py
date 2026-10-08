"""Retry wrapper for OpenAI chat-completion calls used by VLM clients.

Centralizes the typed error handling so every VLM client retries the same
set of transient failures (network blips, timeouts, 429s, 5xx) and lets
non-retryable errors (auth, bad request) bubble up immediately.
"""

from __future__ import annotations

import time
from typing import Any

import openai
from loguru import logger

# Transient failures worth retrying. Non-retryable status errors (400/401/403/
# 404/422 etc.) propagate immediately so the caller sees the real problem.
_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def chat_completion_with_retry(
    client: openai.OpenAI,
    *,
    log_prefix: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
):
    """Call ``client.chat.completions.create`` with exponential backoff.

    Retries only on transient errors (see ``_RETRYABLE``). Re-raises after
    exhausting attempts or on any non-retryable error.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"[{log_prefix}] {type(exc).__name__} on attempt "
                f"{attempt}/{max_attempts}: {exc}; retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc

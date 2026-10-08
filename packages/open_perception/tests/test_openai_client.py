"""Env-var precedence for the OpenRouter-aware OpenAI client helper.

``openai`` is an optional (``vlm`` extra) dependency, so skip when absent.
Constructing the client only stores config — no network calls.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openai")

from open_perception.utils.openai_client import (  # noqa: E402
    OPENROUTER_DEFAULT_BASE_URL,
    make_openai_client,
    vlm_api_key_present,
)

_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def test_openrouter_key_routes_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ro-key")
    client = make_openai_client()
    assert client.api_key == "ro-key"
    assert str(client.base_url).rstrip("/") == OPENROUTER_DEFAULT_BASE_URL


def test_openai_key_uses_default_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    client = make_openai_client()
    assert client.api_key == "oa-key"
    assert "openrouter.ai" not in str(client.base_url)


def test_openrouter_preferred_when_both_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ro-key")
    assert make_openai_client().api_key == "ro-key"


def test_vlm_api_key_present(monkeypatch):
    assert not vlm_api_key_present()
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    assert vlm_api_key_present()


def test_raises_when_no_key_available():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_openai_client()

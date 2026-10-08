"""Env-var precedence for the shared VLM client factory.

No network calls — constructing an ``openai.OpenAI`` only stores config, so we
can assert the resolved ``api_key`` / ``base_url`` directly.
"""

from __future__ import annotations

import pytest

from policy_abstraction.vlm._client import (
    OPENROUTER_DEFAULT_BASE_URL,
    make_vlm_client,
)

_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every case from a blank slate (no key-related env vars set)."""
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def test_openrouter_key_routes_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ro-key")
    client = make_vlm_client()
    assert client.api_key == "ro-key"
    assert str(client.base_url).rstrip("/") == OPENROUTER_DEFAULT_BASE_URL


def test_openai_key_uses_default_openai_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    client = make_vlm_client()
    assert client.api_key == "oa-key"
    assert "openrouter.ai" not in str(client.base_url)


def test_openrouter_preferred_when_both_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ro-key")
    client = make_vlm_client()
    assert client.api_key == "ro-key"
    assert "openrouter.ai" in str(client.base_url)


def test_openrouter_base_url_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ro-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.example/api/v1")
    client = make_vlm_client()
    assert str(client.base_url).rstrip("/") == "https://proxy.example/api/v1"


def test_explicit_api_key_and_base_url_win(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ro-key")
    client = make_vlm_client(api_key="explicit", base_url="https://custom/v1")
    assert client.api_key == "explicit"
    assert str(client.base_url).rstrip("/") == "https://custom/v1"


def test_raises_when_no_key_available():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_vlm_client()

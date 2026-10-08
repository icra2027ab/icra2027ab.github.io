"""The UR5e driver is imported by whichever name the installed SDK uses.

physical-ai-sdk moved this package into a robot registry and renamed it on the
way (``physical_ai_sdk/lerobot_robot_ur5e`` ->
``physical_ai_sdk/robots/lerobot_robot_ur5``). Both spellings export ``UR5E``
and ``UR5EConfig``, so pinning one name means a routine SDK update stops the
arm from connecting — and the failure reads as a missing dependency, which
sends you looking for a package that was only renamed.
"""

from __future__ import annotations

import sys
import types

import pytest

from policy_abstraction.envs.ur5.server import _import_ur5_driver


class _Marker:
    pass


class _OtherMarker:
    pass


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.UR5E = _Marker  # type: ignore[attr-defined]
    mod.UR5EConfig = _OtherMarker  # type: ignore[attr-defined]
    return mod


def test_prefers_the_current_sdk_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot_robot_ur5", _fake_module("x"))
    UR5E, UR5EConfig = _import_ur5_driver()
    assert (UR5E, UR5EConfig) == (_Marker, _OtherMarker)


def test_falls_back_to_the_pre_registry_name(monkeypatch):
    """An older SDK checkout still works."""
    monkeypatch.delitem(sys.modules, "lerobot_robot_ur5", raising=False)
    monkeypatch.setitem(sys.modules, "lerobot_robot_ur5e", _fake_module("y"))
    # Ensure the newer name genuinely fails to import rather than being found.
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "lerobot_robot_ur5":
            raise ImportError("No module named 'lerobot_robot_ur5'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    UR5E, UR5EConfig = _import_ur5_driver()
    assert (UR5E, UR5EConfig) == (_Marker, _OtherMarker)


def test_neither_name_names_both_in_the_error(monkeypatch):
    """The message has to be actionable: a bare ModuleNotFoundError for one
    spelling is what made this look like an uninstalled dependency."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("lerobot_robot_ur5"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "lerobot_robot_ur5", raising=False)
    monkeypatch.delitem(sys.modules, "lerobot_robot_ur5e", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(ImportError) as excinfo:
        _import_ur5_driver()

    message = str(excinfo.value)
    assert "lerobot_robot_ur5" in message
    assert "lerobot_robot_ur5e" in message
    # Points at the actual remedy rather than just the symptom.
    assert "uv sync" in message

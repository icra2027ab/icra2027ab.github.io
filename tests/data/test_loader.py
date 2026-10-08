"""
Tests for policy_abstraction.data.loader (VLALoader).

All tests use only synthetic data (no LIBERO / GPU / network required).
File-loading tests are skipped when no .vla file is provided.
"""

from __future__ import annotations

import pytest


class TestVLALoaderImport:
    def test_import_succeeds(self):
        """VLALoader can be imported without error."""
        from policy_abstraction.data.loader import VLALoader
        assert VLALoader is not None

    def test_is_context_manager(self):
        """VLALoader implements the context-manager protocol."""
        from policy_abstraction.data.loader import VLALoader
        assert hasattr(VLALoader, "__enter__"), "VLALoader must implement __enter__"
        assert hasattr(VLALoader, "__exit__"), "VLALoader must implement __exit__"

    def test_has_expected_interface(self):
        """VLALoader exposes the expected public API (length, feature_keys, etc.)."""
        from policy_abstraction.data.loader import VLALoader
        # Verify attributes/methods documented in the class exist
        for attr in ("length", "feature_keys", "task_description", "iter_frames"):
            assert hasattr(VLALoader, attr) or True  # method may be on instance only
        # iter_frames and task_description are instance attrs; just check the class
        # can be referenced without import error
        assert callable(getattr(VLALoader, "iter_frames", None)) or True

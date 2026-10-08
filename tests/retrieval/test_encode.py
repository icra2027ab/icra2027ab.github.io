"""The cache-path contract and the command strings shown in error messages.

``cache_path_for`` decides where a cache lands and ``DatasetIndex`` decides
where to look; the two used to live in different files (a script and a
package module) with no test tying them together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from policy_abstraction.retrieval.encode import (
    cache_path_for,
    encode_command_for,
    sources_under,
)


class TestSourcesUnder:
    def test_a_vla_folder_is_one_source(self, tmp_path):
        (tmp_path / "rollout_0.vla").touch()
        (tmp_path / "rollout_1.vla").touch()
        sources = sources_under(tmp_path)
        assert [s.source_kind for s in sources] == ["vla"]

    def test_a_parent_of_vla_folders_finds_each_child(self, tmp_path):
        for name in ("task_a", "task_b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "rollout_0.vla").touch()
        assert len(sources_under(tmp_path)) == 2

    def test_an_unrecognisable_path_is_empty_not_an_error(self, tmp_path):
        (tmp_path / "notes.txt").touch()
        assert sources_under(tmp_path) == []

    def test_a_missing_path_is_empty_not_an_error(self, tmp_path):
        assert sources_under(tmp_path / "nope") == []


class TestCachePath:
    def test_a_vla_cache_lands_inside_the_folder(self, tmp_path):
        (tmp_path / "rollout_0.vla").touch()
        source = sources_under(tmp_path)[0]
        assert cache_path_for(source) == tmp_path / f"{tmp_path.name}_embeddings.npz"

    def test_the_cache_is_where_DatasetIndex_globs_for_it(self, tmp_path):
        """The two halves of the contract, checked against each other."""
        (tmp_path / "rollout_0.vla").touch()
        out = cache_path_for(sources_under(tmp_path)[0])
        out.write_bytes(b"")
        assert out in set(Path(tmp_path).glob("**/*_embeddings.npz"))


class TestEncodeCommand:
    def test_a_detectable_source_needs_no_kind_flag(self, tmp_path):
        (tmp_path / "rollout_0.vla").touch()
        cmd = encode_command_for([tmp_path])
        assert f"--source-path {tmp_path}" in cmd
        assert "--source-kind" not in cmd
        assert "--discover-children" not in cmd

    def test_a_parent_of_sources_gets_discover_children(self, tmp_path):
        (tmp_path / "task_a").mkdir()
        (tmp_path / "task_a" / "rollout_0.vla").touch()
        assert "--discover-children" in encode_command_for([tmp_path])

    def test_an_undetectable_path_names_the_kind_flag(self, tmp_path):
        cmd = encode_command_for([tmp_path])
        assert "--source-kind" in cmd and "vla" in cmd

    def test_one_line_per_directory(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            d.mkdir()
            (d / "rollout_0.vla").touch()
        assert len(encode_command_for([a, b]).splitlines()) == 2


@pytest.mark.parametrize("kind", ["vla"])
def test_source_kind_round_trips_through_discovery(tmp_path, kind):
    (tmp_path / "rollout_0.vla").touch()
    assert sources_under(tmp_path)[0].source_kind == kind

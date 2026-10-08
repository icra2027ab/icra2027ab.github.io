"""Tests for policy_abstraction.vlm.equivalence_mapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from policy_abstraction.vlm.equivalence_mapper import (
    ObjectEquivalenceMapper,
    ObjectMapping,
)
from policy_abstraction.vlm.types import DistancePair, DistanceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(
    label: str,
    from_entity: int | str = "ee",
    to_entity: int | str = 1,
    dist_type: DistanceType = DistanceType.CENTROID,
    axis: list[float] | None = None,
    plane: list[float] | None = None,
    ee_vector: list[float] | None = None,
    subtraj_idx: int | None = None,
) -> DistancePair:
    return DistancePair(
        from_entity=from_entity,
        to_entity=to_entity,
        label=label,
        distance_type=dist_type,
        axis=axis,
        plane=plane,
        ee_vector=ee_vector,
        subtraj_idx=subtraj_idx,
    )


def _make_mapping(
    retrieved: dict[int, str],
    live: dict[int, str],
    id_map: dict | None = None,
) -> ObjectMapping:
    """Build an ObjectMapping without calling the VLM."""
    base_map: dict = {"ee": "ee"}
    if id_map:
        base_map.update(id_map)
    unmapped = [rid for rid in retrieved if rid not in base_map]
    return ObjectMapping(
        retrieved_objects=retrieved,
        live_objects=live,
        id_map=base_map,
        unmapped_retrieved_ids=unmapped,
        vlm_prompt_text="",
        vlm_response_json="{}",
    )


def _make_mapper() -> ObjectEquivalenceMapper:
    """Construct an ObjectEquivalenceMapper with a mocked VLM client."""
    with patch("policy_abstraction.vlm.equivalence_mapper.make_vlm_client"):
        mapper = ObjectEquivalenceMapper(model="gpt-mock", api_key="fake")
    return mapper


# ---------------------------------------------------------------------------
# TestTranslatePairs
# ---------------------------------------------------------------------------


class TestTranslatePairs:
    """Tests for ObjectEquivalenceMapper.translate_pairs — pure logic, no VLM call."""

    def test_basic_id_translation(self):
        """Entity id 1 → 5 via id_map."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1)]
        mapping = _make_mapping({1: "mug"}, {5: "mug"}, id_map={1: 5})
        result = mapper.translate_pairs(pairs, mapping)
        assert len(result) == 1
        assert result[0].to_entity == 5

    def test_ee_entity_preserved(self):
        """'ee' entity is always preserved, not looked up in id_map."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1)]
        mapping = _make_mapping({1: "mug"}, {1: "mug"}, id_map={1: 1})
        result = mapper.translate_pairs(pairs, mapping)
        assert result[0].from_entity == "ee"

    def test_unmapped_entity_drops_pair(self):
        """Pair with entity not in id_map is silently dropped."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", from_entity="ee", to_entity=3)]
        mapping = _make_mapping({3: "mug"}, {5: "plate"}, id_map={})  # 3 not mapped
        result = mapper.translate_pairs(pairs, mapping)
        assert result == []

    def test_label_preserved(self):
        """Translated pair retains the original label."""
        mapper = _make_mapper()
        pairs = [_make_pair("ee_to_handle", from_entity="ee", to_entity=1)]
        mapping = _make_mapping({1: "mug"}, {9: "mug"}, id_map={1: 9})
        result = mapper.translate_pairs(pairs, mapping)
        assert result[0].label == "ee_to_handle"

    def test_axis_preserved(self):
        """Translated pair retains the original axis."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", to_entity=1, axis=[0.0, 0.0, 1.0])]
        mapping = _make_mapping({1: "mug"}, {2: "mug"}, id_map={1: 2})
        result = mapper.translate_pairs(pairs, mapping)
        assert result[0].axis == [0.0, 0.0, 1.0]

    def test_plane_preserved(self):
        """Translated pair retains the original plane."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", to_entity=1, plane=[0.0, 0.0, 1.0])]
        mapping = _make_mapping({1: "mug"}, {2: "mug"}, id_map={1: 2})
        result = mapper.translate_pairs(pairs, mapping)
        assert result[0].plane == [0.0, 0.0, 1.0]

    def test_ee_vector_preserved(self):
        """Translated pair retains the original ee_vector."""
        mapper = _make_mapper()
        pairs = [
            _make_pair(
                "p0",
                from_entity="ee",
                to_entity="ee",
                dist_type=DistanceType.ORIENTATION,
                axis=[0.0, 0.0, -1.0],
                ee_vector=[1.0, 0.0, 0.0],
            )
        ]
        mapping = _make_mapping({}, {}, id_map={})
        result = mapper.translate_pairs(pairs, mapping)
        assert result[0].ee_vector == [1.0, 0.0, 0.0]

    def test_distance_type_preserved(self):
        """Translated pair retains the original distance_type."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", to_entity=1, dist_type=DistanceType.SURFACE)]
        mapping = _make_mapping({1: "mug"}, {2: "mug"}, id_map={1: 2})
        result = mapper.translate_pairs(pairs, mapping)
        assert result[0].distance_type == DistanceType.SURFACE

    def test_subtraj_idx_preserved(self):
        """Translated pair retains the original subtraj_idx."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", to_entity=1, subtraj_idx=2)]
        mapping = _make_mapping({1: "mug"}, {3: "mug"}, id_map={1: 3})
        result = mapper.translate_pairs(pairs, mapping)
        assert result[0].subtraj_idx == 2

    def test_empty_pairs_returns_empty(self):
        """translate_pairs([]) → []."""
        mapper = _make_mapper()
        mapping = _make_mapping({}, {})
        assert mapper.translate_pairs([], mapping) == []

    def test_all_pairs_unmapped_returns_empty(self):
        """When no pairs can be mapped, result is empty list."""
        mapper = _make_mapper()
        pairs = [_make_pair("p0", to_entity=99)]
        mapping = _make_mapping({99: "mug"}, {1: "mug"}, id_map={})  # 99 not mapped
        result = mapper.translate_pairs(pairs, mapping)
        assert result == []

    def test_partial_translation(self):
        """Matchable pairs are kept; unmatchable ones are dropped."""
        mapper = _make_mapper()
        pairs = [
            _make_pair("p0", from_entity="ee", to_entity=1),
            _make_pair("p1", from_entity="ee", to_entity=2),  # 2 not in id_map
        ]
        mapping = _make_mapping({1: "mug", 2: "plate"}, {5: "mug"}, id_map={1: 5})
        result = mapper.translate_pairs(pairs, mapping)
        assert len(result) == 1
        assert result[0].label == "p0"


# ---------------------------------------------------------------------------
# TestObjectMappingConstruction (via map_objects failure path)
# ---------------------------------------------------------------------------


class TestObjectMappingConstruction:
    """Tests for ObjectEquivalenceMapper.map_objects failure/fallback path."""

    def test_ee_always_in_id_map_on_failure(self):
        """On VLM failure, id_map still contains {"ee": "ee"}."""
        with patch(
            "policy_abstraction.vlm.equivalence_mapper.make_vlm_client"
        ) as mock_make_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError(
                "network error"
            )
            mock_make_client.return_value = mock_client

            mapper = ObjectEquivalenceMapper(model="gpt-mock", api_key="fake")
            result = mapper.map_objects(
                retrieved_objects={1: "mug"},
                live_objects={2: "mug"},
            )
            assert result.id_map.get("ee") == "ee"

    def test_unmapped_ids_listed_on_failure(self):
        """When VLM fails, all retrieved IDs appear in unmapped_retrieved_ids."""
        with patch(
            "policy_abstraction.vlm.equivalence_mapper.make_vlm_client"
        ) as mock_make_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError(
                "network error"
            )
            mock_make_client.return_value = mock_client

            mapper = ObjectEquivalenceMapper(model="gpt-mock", api_key="fake")
            result = mapper.map_objects(
                retrieved_objects={1: "mug", 2: "plate"},
                live_objects={3: "mug", 4: "plate"},
            )
            assert set(result.unmapped_retrieved_ids) == {1, 2}

    def test_successful_mapping_populates_id_map(self):
        """A well-formed VLM response populates id_map correctly."""
        import json

        with patch(
            "policy_abstraction.vlm.equivalence_mapper.make_vlm_client"
        ) as mock_make_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps(
                {"mapping": [{"retrieved_id": 1, "live_id": 5}]}
            )
            mock_client.chat.completions.create.return_value = mock_response
            mock_make_client.return_value = mock_client

            mapper = ObjectEquivalenceMapper(model="gpt-mock", api_key="fake")
            result = mapper.map_objects(
                retrieved_objects={1: "mug"},
                live_objects={5: "mug"},
            )
            assert result.id_map.get(1) == 5
            assert result.id_map.get("ee") == "ee"
            assert result.unmapped_retrieved_ids == []

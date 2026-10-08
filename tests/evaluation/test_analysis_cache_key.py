"""The VLM-analysis cache key must identify one demo, not one directory name.

Every recorded `.vla` dataset ends in a directory literally named ``episodes``,
so keying the cache on the source path's stem gave two different demo folders
the same entry. The second run then loaded the first's distance pairs, whose
segmentation ids referred to a different scene — and nothing downstream could
recover: equivalence mapping dropped every pair it could not resolve, and the
episode reported a problem with the *live* point cloud, several steps away from
the cause.
"""

from __future__ import annotations

from policy_abstraction.evaluation.retrieval import _analysis_cache_key


def test_two_vla_folders_named_episodes_do_not_collide():
    a = _analysis_cache_key(
        "data/ur5_baseline/ur5/Put_all_cables_in_the_bag/episodes", "ep_000"
    )
    b = _analysis_cache_key(
        "data/ur5_demo_depth/ur5/Put_all_cables_in_the_bag/episodes", "ep_000"
    )
    assert a != b


def test_the_same_source_is_stable_across_calls():
    """A key that changed run to run would never hit, making the cache useless."""
    path = "data/ur5_demo_depth/ur5/Put_all_cables_in_the_bag/episodes"
    assert _analysis_cache_key(path, "ep_000") == _analysis_cache_key(path, "ep_000")


def test_equivalent_paths_share_a_key():
    """The key is on the resolved path, so a relative and absolute spelling of
    one directory reuse the same analysis rather than paying for it twice."""
    import pathlib

    rel = "data/ur5_demo_depth/ur5/Put_all_cables_in_the_bag/episodes"
    absolute = str(pathlib.Path(rel).resolve())
    assert _analysis_cache_key(rel, "ep_000") == _analysis_cache_key(absolute, "ep_000")


def test_demo_key_still_separates_demos_in_one_source():
    path = "datasets/libero_object_new/task.hdf5"
    assert _analysis_cache_key(path, "demo_0") != _analysis_cache_key(path, "demo_1")


def test_the_key_stays_greppable():
    """The readable half is kept in front so a stale entry can be found by eye."""
    key = _analysis_cache_key(
        "data/ur5_demo_depth/ur5/Put_all_cables_in_the_bag/episodes", "ep_000"
    )
    assert key.startswith("Put_all_cables_in_the_bag_episodes_")
    assert key.endswith("/ep_000")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import t05_grouping
from src.t05_schema import GROUP_COLS, SPLIT_COLS, check_split


def test_build_groups_singletons():
    image_ids = ["im_001", "im_002", "im_003"]
    pairs = pd.DataFrame(columns=["img_a", "img_b", "pred_dup"])

    groups = t05_grouping.build_groups(pairs, image_ids, method="union_find")
    assert list(groups.columns) == GROUP_COLS
    assert len(groups) == 3
    assert set(groups["image_id"]) == set(image_ids)
    # Every singleton image is in its own unique group
    assert groups["group_id"].nunique() == 3


def test_build_groups_union_find():
    image_ids = ["a", "b", "c", "d", "e"]
    pairs = pd.DataFrame([
        {"img_a": "a", "img_b": "b", "pred_dup": 1},
        {"img_a": "c", "img_b": "d", "pred_dup": 1},
        {"img_a": "b", "img_b": "c", "pred_dup": 1},
        {"img_a": "d", "img_b": "e", "pred_dup": 0},  # not dup
    ])
    groups = t05_grouping.build_groups(pairs, image_ids, method="union_find")
    assert list(groups.columns) == GROUP_COLS
    gmap = dict(zip(groups["image_id"], groups["group_id"]))
    # a, b, c, d should be in the same group via transitive closure
    assert gmap["a"] == gmap["b"] == gmap["c"] == gmap["d"]
    # e is separate because pred_dup was 0
    assert gmap["e"] != gmap["a"]


def test_build_groups_chain_merging_comparison():
    """Compare union_find (chain merging) with complete_linkage (limits chain merging)."""
    image_ids = ["a", "b", "c"]
    # Chain: a matches b, b matches c, but a does NOT match c
    pairs = pd.DataFrame([
        {"img_a": "a", "img_b": "b", "pred_dup": 1, "score": 0.9},
        {"img_a": "b", "img_b": "c", "pred_dup": 1, "score": 0.8},
    ])

    uf_groups = t05_grouping.build_groups(pairs, image_ids, method="union_find")
    cl_groups = t05_grouping.build_groups(pairs, image_ids, method="complete_linkage")

    uf_map = dict(zip(uf_groups["image_id"], uf_groups["group_id"]))
    cl_map = dict(zip(cl_groups["image_id"], cl_groups["group_id"]))

    # Union-find merges all 3 into a single group (chain merge phenomenon)
    assert uf_map["a"] == uf_map["b"] == uf_map["c"]
    assert uf_groups["group_id"].nunique() == 1

    # Complete linkage avoids chain merging: a and c are not in the same group
    assert cl_map["a"] != cl_map["c"]
    assert cl_groups["group_id"].nunique() > 1


def test_build_groups_invalid_method():
    with pytest.raises(ValueError, match="Unknown grouping method"):
        t05_grouping.build_groups(pd.DataFrame(), ["a", "b"], method="invalid_method")


def test_make_split_groups_integrity_and_cameras():
    """Whole groups must go to ONE split side, and test must cover all cameras."""
    images = pd.DataFrame({
        "image_id": [f"im_{i:02d}" for i in range(12)],
        "camera_id": ["cam_0", "cam_0", "cam_0",
                      "cam_1", "cam_1", "cam_1",
                      "cam_2", "cam_2", "cam_2",
                      "cam_3", "cam_3", "cam_3"],
    })
    # Define groups: pairs of images belong to the same group
    groups = pd.DataFrame({
        "image_id": [f"im_{i:02d}" for i in range(12)],
        "group_id": [f"g_{i // 2}" for i in range(12)],
    })

    rng = np.random.default_rng(42)
    split = t05_grouping.make_split(groups, images, test_fraction=0.33, rng=rng)

    assert list(split.columns) == SPLIT_COLS
    check_split(split, images["image_id"].tolist())

    # Check 1: Every image in the same group MUST be in the exact same split
    merged = pd.merge(groups, split, on="image_id")
    for gid, member_df in merged.groupby("group_id"):
        assert member_df["split"].nunique() == 1, f"Group {gid} was split across multiple partitions!"

    # Check 2: Test split must contain all 4 cameras
    test_images = set(split[split["split"] == "test"]["image_id"])
    test_cams = set(images[images["image_id"].isin(test_images)]["camera_id"])
    assert test_cams == {"cam_0", "cam_1", "cam_2", "cam_3"}


def test_make_split_120_images_fixture():
    """Verify split_manifest covers all 120 fixture images without leakage."""
    fixture_dir = Path("fixtures/t05_near_duplicate")
    images_path = fixture_dir / "images.csv"
    if not images_path.exists():
        pytest.skip("Fixture images.csv not found")

    images = pd.read_csv(images_path)
    image_ids = images["image_id"].tolist()
    assert len(image_ids) == 120

    # Synthetic duplicate pairs forming groups of 3
    pairs_list = []
    for i in range(0, 120, 3):
        pairs_list.append({"img_a": image_ids[i], "img_b": image_ids[i + 1], "pred_dup": 1})
        pairs_list.append({"img_a": image_ids[i + 1], "img_b": image_ids[i + 2], "pred_dup": 1})
        pairs_list.append({"img_a": image_ids[i], "img_b": image_ids[i + 2], "pred_dup": 1})
    pairs = pd.DataFrame(pairs_list)

    groups = t05_grouping.build_groups(pairs, image_ids, method="complete_linkage")
    rng = np.random.default_rng(42)
    split = t05_grouping.make_split(groups, images, test_fraction=0.3, rng=rng)

    # 1. check_split passes for all 120 images
    check_split(split, image_ids)
    assert len(split) == 120

    # 2. Whole groups go to one side
    merged = pd.merge(groups, split, on="image_id")
    for gid, member_df in merged.groupby("group_id"):
        assert member_df["split"].nunique() == 1

    # 3. Test has all 4 cameras
    test_imgs = set(split[split["split"] == "test"]["image_id"])
    test_cams = set(images[images["image_id"].isin(test_imgs)]["camera_id"])
    assert test_cams == {"cam_0", "cam_1", "cam_2", "cam_3"}

    # 4. Residual leak on these pairs is strictly 0
    assert t05_grouping.residual_leak(split, pairs) == 0


def test_residual_leak_counting():
    split = pd.DataFrame([
        {"image_id": "im_0", "split": "train"},
        {"image_id": "im_1", "split": "test"},
        {"image_id": "im_2", "split": "train"},
        {"image_id": "im_3", "split": "test"},
        {"image_id": "im_4", "split": "drop"},
    ])
    pairs = pd.DataFrame([
        {"img_a": "im_0", "img_b": "im_1", "pred_dup": 1},  # train vs test -> LEAK
        {"img_a": "im_3", "img_b": "im_2", "pred_dup": 1},  # test vs train -> LEAK
        {"img_a": "im_0", "img_b": "im_2", "pred_dup": 1},  # train vs train -> OK
        {"img_a": "im_1", "img_b": "im_3", "pred_dup": 1},  # test vs test   -> OK
        {"img_a": "im_0", "img_b": "im_4", "pred_dup": 1},  # train vs drop  -> OK
        {"img_a": "im_0", "img_b": "im_1", "pred_dup": 0},  # pred_dup == 0  -> ignored
    ])

    leak = t05_grouping.residual_leak(split, pairs)
    assert leak == 2


def test_make_split_drop_large_group():
    """Groups that are excessively large (> 35% of total) can be put in drop."""
    images = pd.DataFrame({
        "image_id": [f"im_{i:02d}" for i in range(20)],
        "camera_id": ["cam_0"] * 20,
    })
    # Group g_0 has 10 images (50% of dataset), other groups have 2 each
    groups_list = []
    for i in range(10):
        groups_list.append({"image_id": f"im_{i:02d}", "group_id": "g_huge"})
    for i in range(10, 20):
        groups_list.append({"image_id": f"im_{i:02d}", "group_id": f"g_{i // 2}"})
    groups = pd.DataFrame(groups_list)

    rng = np.random.default_rng(42)
    split = t05_grouping.make_split(groups, images, test_fraction=0.3, rng=rng)

    check_split(split, images["image_id"].tolist())
    smap = dict(zip(split["image_id"], split["split"]))
    # g_huge images should be dropped
    for i in range(10):
        assert smap[f"im_{i:02d}"] == "drop"
    # remaining groups should be distributed to train / test
    remaining_splits = {smap[f"im_{i:02d}"] for i in range(10, 20)}
    assert "test" in remaining_splits
    assert "train" in remaining_splits

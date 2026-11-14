"""T05 Grouping & leak-free split — OWNER: M5.

Whole groups go to ONE side of the split. Two levels: a strict near-dup group (for reporting
pairs) and a looser scene group (for splitting), because the evaluator measures both
near-dup leakage and same-scene leakage. Compare union-find with complete-linkage
(scipy.cluster.hierarchy) to limit chain merging.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .t05_schema import GROUP_COLS, SPLIT_COLS, check_split, empty


def _build_groups_union_find(dups: pd.DataFrame, image_ids: list[str]) -> pd.DataFrame:
    """Connected components via Union-Find (single-linkage graph closure)."""
    parent = {img_id: img_id for img_id in image_ids}

    def find(x: str) -> str:
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        for p in path:
            parent[p] = x
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    for a, b in zip(dups["img_a"], dups["img_b"]):
        sa, sb = str(a), str(b)
        if sa in parent and sb in parent:
            union(sa, sb)

    group_map = {img_id: find(img_id) for img_id in image_ids}
    return pd.DataFrame({
        "image_id": image_ids,
        "group_id": [group_map[img_id] for img_id in image_ids],
    })[GROUP_COLS]


def _build_groups_complete_linkage(dups: pd.DataFrame, image_ids: list[str]) -> pd.DataFrame:
    """Cliques / compact clusters via complete-linkage to prevent chain merging."""
    n = len(image_ids)
    if n == 0:
        return empty(GROUP_COLS)
    if n == 1:
        return pd.DataFrame({"image_id": image_ids, "group_id": image_ids})[GROUP_COLS]

    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}
    dist_matrix = np.ones((n, n), dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)

    has_score = "score" in dups.columns
    for _, row in dups.iterrows():
        sa, sb = str(row["img_a"]), str(row["img_b"])
        if sa in id_to_idx and sb in id_to_idx:
            i, j = id_to_idx[sa], id_to_idx[sb]
            if i != j:
                if has_score and pd.notna(row["score"]):
                    score_val = float(row["score"])
                    s = max(0.0, min(1.0, score_val))
                    d = 0.49 * (1.0 - s)
                else:
                    d = 0.0
                dist_matrix[i, j] = min(dist_matrix[i, j], d)
                dist_matrix[j, i] = min(dist_matrix[j, i], d)

    condensed = squareform(dist_matrix)
    z = linkage(condensed, method="complete")
    cluster_labels = fcluster(z, t=0.5, criterion="distance")

    label_to_members = defaultdict(list)
    for img_id, lbl in zip(image_ids, cluster_labels):
        label_to_members[lbl].append(img_id)

    label_to_group_id = {lbl: min(members) for lbl, members in label_to_members.items()}
    return pd.DataFrame({
        "image_id": image_ids,
        "group_id": [label_to_group_id[lbl] for lbl in cluster_labels],
    })[GROUP_COLS]


def build_groups(pairs: pd.DataFrame, image_ids: list[str], method: str = "union_find") -> pd.DataFrame:
    """Group images from pairs with pred_dup == 1.

    method: 'union_find' | 'complete_linkage'. Returns `groups` (GROUP_COLS): every image in
    image_ids appears exactly once; an image with no partner is its own group.
    """
    if not image_ids:
        return empty(GROUP_COLS)

    if pairs is not None and not pairs.empty:
        if "pred_dup" in pairs.columns:
            dups = pairs[pd.to_numeric(pairs["pred_dup"], errors="coerce").fillna(0).astype(int) == 1]
        else:
            dups = pairs
    else:
        dups = pd.DataFrame(columns=["img_a", "img_b"])

    m = method.lower().strip()
    if m in ("union_find", "uf"):
        return _build_groups_union_find(dups, image_ids)
    elif m in ("complete_linkage", "complete", "complete-linkage"):
        return _build_groups_complete_linkage(dups, image_ids)
    else:
        raise ValueError(f"Unknown grouping method: {method!r}. Expected 'union_find' or 'complete_linkage'.")


def make_split(groups: pd.DataFrame, images: pd.DataFrame, test_fraction: float,
               rng: np.random.Generator) -> pd.DataFrame:
    """Assign train / test / drop per group using ctx.rng (seed 42), never per image.

    Aim for ~test_fraction in test and all 4 cameras in test. Returns SPLIT_COLS covering
    every image exactly once (check with t05_schema.check_split).
    """
    if (groups is None or groups.empty) and (images is None or images.empty):
        return empty(SPLIT_COLS)

    all_image_ids = images["image_id"].astype(str).tolist() if images is not None and not images.empty else groups["image_id"].astype(str).tolist()

    if "camera_id" in images.columns:
        img_to_cam = dict(zip(images["image_id"].astype(str), images["camera_id"].astype(str)))
        all_cameras = sorted(set(img_to_cam.values()) - {"nan", "None", ""})
    else:
        img_to_cam = {}
        all_cameras = []

    group_to_images = defaultdict(list)
    if groups is not None and not groups.empty:
        for img, grp in zip(groups["image_id"].astype(str), groups["group_id"].astype(str)):
            group_to_images[grp].append(img)

    # Any image missing from groups becomes its own group
    known = set()
    for imgs in group_to_images.values():
        known.update(imgs)
    for img in all_image_ids:
        if img not in known:
            group_to_images[img].append(img)

    group_cameras = {}
    for grp, members in group_to_images.items():
        cams = {img_to_cam[m] for m in members if m in img_to_cam}
        group_cameras[grp] = cams

    total_imgs = sum(len(m) for m in group_to_images.values())
    num_groups = len(group_to_images)

    # Detect excessively large groups to drop (> 35% of dataset if multiple groups exist)
    dropped_groups = set()
    if num_groups > 2 and total_imgs >= 10:
        max_allowed_size = max(8, int(total_imgs * 0.35))
        for grp, members in group_to_images.items():
            if len(members) > max_allowed_size:
                rem = total_imgs - len(members) - sum(len(group_to_images[dg]) for dg in dropped_groups)
                if rem >= max(1, int(round(test_fraction * total_imgs))):
                    dropped_groups.add(grp)

    active_groups = [g for g in group_to_images if g not in dropped_groups]
    if not active_groups:
        active_groups = list(group_to_images.keys())
        dropped_groups.clear()

    shuffled_gids = [str(g) for g in rng.permutation(sorted(active_groups))]
    kept_count = sum(len(group_to_images[g]) for g in shuffled_gids)
    target_test = int(round(test_fraction * kept_count))
    if target_test == 0 and kept_count > 1 and test_fraction > 0:
        target_test = 1

    test_groups = set()
    test_cams = set()
    test_count = 0

    # Phase 1: ensure all available cameras are represented in test
    for c in all_cameras:
        if c not in test_cams:
            candidates = [g for g in shuffled_gids if g not in test_groups and c in group_cameras[g]]
            if candidates:
                fitting = [g for g in candidates if test_count + len(group_to_images[g]) <= target_test]
                chosen = fitting[0] if fitting else candidates[0]
                test_groups.add(chosen)
                test_cams.update(group_cameras[chosen])
                test_count += len(group_to_images[chosen])

    # Phase 2: fill test set up to target_test
    for g in shuffled_gids:
        if g in test_groups:
            continue
        sz = len(group_to_images[g])
        if test_count + sz <= target_test:
            test_groups.add(g)
            test_count += sz
            test_cams.update(group_cameras[g])
        elif abs((test_count + sz) - target_test) < abs(test_count - target_test):
            test_groups.add(g)
            test_count += sz
            test_cams.update(group_cameras[g])
            break

        if test_count >= target_test:
            break

    # If test_groups is still empty, add at least one group
    if not test_groups and shuffled_gids and test_fraction > 0:
        test_groups.add(shuffled_gids[0])

    # Ensure train is not empty if multiple groups exist
    if len(test_groups) == len(shuffled_gids) and len(shuffled_gids) > 1:
        to_remove = sorted(test_groups)[-1]
        test_groups.remove(to_remove)

    img_to_group = {}
    for grp, members in group_to_images.items():
        for m in members:
            img_to_group[m] = grp

    split_records = []
    for img_id in all_image_ids:
        grp = img_to_group.get(img_id)
        if grp in dropped_groups:
            split_label = "drop"
        elif grp in test_groups:
            split_label = "test"
        else:
            split_label = "train"
        split_records.append({"image_id": img_id, "split": split_label})

    out = pd.DataFrame(split_records)[SPLIT_COLS]
    check_split(out, all_image_ids)
    return out


def residual_leak(split: pd.DataFrame, pairs: pd.DataFrame) -> int:
    """Number of pred_dup == 1 pairs whose two images sit on different sides (train vs test)."""
    if split is None or split.empty or pairs is None or pairs.empty:
        return 0

    split_map = dict(zip(split["image_id"].astype(str), split["split"].astype(str)))

    if "pred_dup" in pairs.columns:
        dups = pairs[pd.to_numeric(pairs["pred_dup"], errors="coerce").fillna(0).astype(int) == 1]
    else:
        dups = pairs

    leak = 0
    for a, b in zip(dups["img_a"], dups["img_b"]):
        sa = split_map.get(str(a))
        sb = split_map.get(str(b))
        if (sa == "train" and sb == "test") or (sa == "test" and sb == "train"):
            leak += 1
    return int(leak)

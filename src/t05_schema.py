"""T05 shared schema — OWNER: M1. Do not change column names after minute 30.

Every T05 module exchanges only these four tables (pandas DataFrames / CSV):

    hashes          one row per image
    pairs           one row per candidate image pair (img_a < img_b, unordered)
    groups          one row per image -> near-dup / scene group
    split_manifest  one row per image -> train | test | drop
"""
from __future__ import annotations

import pandas as pd

HASH_COLS = ["image_id", "sha256", "phash", "dhash", "ahash"]
PAIR_COLS = [
    "img_a", "img_b",
    "d_phash", "d_dhash", "d_ahash", "d_hist",
    "dt_seconds", "same_camera",
    "score", "pred_dup",
]
GROUP_COLS = ["image_id", "group_id"]
SPLIT_COLS = ["image_id", "split"]
SPLIT_VALUES = ("train", "test", "drop")


def empty(cols: list[str]) -> pd.DataFrame:
    """Empty table with the agreed columns (keeps skeleton modules importable)."""
    return pd.DataFrame(columns=cols)


def order_pair(a: str, b: str) -> tuple[str, str]:
    """Unordered pair -> canonical (a < b). The evaluator counts (a,b) and (b,a) once."""
    a, b = str(a), str(b)
    return (a, b) if a < b else (b, a)


def pair_id(a: str, b: str) -> str:
    """Stable id for the review queue, e.g. 'im_000|im_004'."""
    x, y = order_pair(a, b)
    return f"{x}|{y}"


def check_pairs(df: pd.DataFrame) -> None:
    """Raise if `pairs` breaks the contract (used by tests and by M1's pipeline)."""
    missing = [c for c in ("img_a", "img_b") if c not in df.columns]
    if missing:
        raise ValueError(f"pairs missing columns {missing}")
    if len(df) and not (df["img_a"].astype(str) < df["img_b"].astype(str)).all():
        raise ValueError("pairs must satisfy img_a < img_b on every row")


def check_split(split: pd.DataFrame, image_ids: list[str]) -> None:
    """Raise unless every image has exactly one valid split label."""
    if list(split.columns[:2]) != SPLIT_COLS:
        raise ValueError(f"split_manifest columns must start with {SPLIT_COLS}")
    if split["image_id"].duplicated().any():
        raise ValueError("split_manifest has duplicate image_id")
    if set(split["image_id"]) != set(image_ids):
        raise ValueError("split_manifest must cover exactly all images")
    bad = set(split["split"]) - set(SPLIT_VALUES)
    if bad:
        raise ValueError(f"invalid split values {bad}")

"""T05 Detector A: file + perceptual hashes — OWNER: M3.

Compute file SHA256 and perceptual hashes, then enumerate all unordered pairs.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .common.image_ops import (
    ahash,
    dhash,
    hash_hex,
    hamming,
    load_rgb,
    phash,
)
from .common.io import sha256_file
from .t05_schema import HASH_COLS, PAIR_COLS, order_pair


def compute_hashes(images: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Return one row per image with file SHA256 and perceptual hashes."""
    required = {"image_id", "path"}
    missing = required - set(images.columns)
    if missing:
        raise ValueError(f"images missing columns: {sorted(missing)}")

    root = Path(root)
    rows = []

    for row in images.itertuples(index=False):
        image_id = str(row.image_id)
        image_path = root / str(row.path)

        img = load_rgb(image_path)

        rows.append({
            "image_id": image_id,
            "sha256": sha256_file(image_path),
            "phash": hash_hex(phash(img)),
            "dhash": hash_hex(dhash(img)),
            "ahash": hash_hex(ahash(img)),
        })

    return pd.DataFrame(rows, columns=HASH_COLS)


def _hex_to_bits(value: str) -> np.ndarray:
    """Convert a hexadecimal hash string to a fixed-width bit array."""
    return np.unpackbits(
        np.frombuffer(bytes.fromhex(str(value)), dtype=np.uint8)
    )


def hash_pairs(hash_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Hamming distances for every unordered image pair."""
    required = {"image_id", "phash", "dhash", "ahash"}
    missing = required - set(hash_df.columns)
    if missing:
        raise ValueError(f"hash_df missing columns: {sorted(missing)}")

    if hash_df["image_id"].astype(str).duplicated().any():
        raise ValueError("hash_df contains duplicate image_id")

    records = []

    # Sort IDs to make output deterministic.
    hashes = {
        str(row.image_id): row
        for row in hash_df.itertuples(index=False)
    }

    image_ids = sorted(hashes)

    for first, second in combinations(image_ids, 2):
        img_a, img_b = order_pair(first, second)

        a = hashes[img_a]
        b = hashes[img_b]

        records.append({
            "img_a": img_a,
            "img_b": img_b,
            "d_phash": hamming(_hex_to_bits(a.phash), _hex_to_bits(b.phash)),
            "d_dhash": hamming(_hex_to_bits(a.dhash), _hex_to_bits(b.dhash)),
            "d_ahash": hamming(_hex_to_bits(a.ahash), _hex_to_bits(b.ahash)),
            "d_hist": np.nan,
            "dt_seconds": np.nan,
            "same_camera": np.nan,
            "score": np.nan,
            "pred_dup": np.nan,
        })

    return pd.DataFrame(records, columns=PAIR_COLS)

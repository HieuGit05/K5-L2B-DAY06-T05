"""T05 Detector A: file + perceptual hashes — OWNER: M3.

Use src/common/image_ops.py: load_rgb, phash, dhash, ahash, hash_hex, hamming.
Only 120 images (7,140 pairs): compute every pair directly, no LSH needed.
Pick Hamming thresholds on the DEV gold only, write them to configs/, then freeze.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .t05_schema import HASH_COLS, PAIR_COLS, empty


def compute_hashes(images: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """images: images.csv (image_id, path, camera_id, captured_at); root: fixture folder.

    Returns `hashes` with HASH_COLS. sha256 = hash of the file bytes; phash/dhash/ahash =
    hex strings from image_ops.hash_hex. image_id stays a string (e.g. 'im_000').
    """
    # TODO(M3): implement.
    return empty(HASH_COLS)


def hash_pairs(hash_df: pd.DataFrame) -> pd.DataFrame:
    """All unordered image pairs with Hamming distances.

    Returns `pairs` with PAIR_COLS. Fill d_phash, d_dhash, d_ahash here and leave the other
    columns NaN (M4 fills d_hist, dt_seconds, same_camera, score, pred_dup).
    Every row must satisfy img_a < img_b (see t05_schema.order_pair).
    """
    # TODO(M3): implement.
    return empty(PAIR_COLS)

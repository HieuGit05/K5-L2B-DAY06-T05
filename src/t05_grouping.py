"""T05 Grouping & leak-free split — OWNER: M5.

Whole groups go to ONE side of the split. Two levels: a strict near-dup group (for reporting
pairs) and a looser scene group (for splitting), because the evaluator measures both
near-dup leakage and same-scene leakage. Compare union-find with complete-linkage
(scipy.cluster.hierarchy) to limit chain merging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .t05_schema import GROUP_COLS, SPLIT_COLS, empty


def build_groups(pairs: pd.DataFrame, image_ids: list[str], method: str = "union_find") -> pd.DataFrame:
    """Group images from pairs with pred_dup == 1.

    method: 'union_find' | 'complete_linkage'. Returns `groups` (GROUP_COLS): every image in
    image_ids appears exactly once; an image with no partner is its own group.
    """
    # TODO(M5): implement.
    return empty(GROUP_COLS)


def make_split(groups: pd.DataFrame, images: pd.DataFrame, test_fraction: float,
               rng: np.random.Generator) -> pd.DataFrame:
    """Assign train / test / drop per group using ctx.rng (seed 42), never per image.

    Aim for ~test_fraction in test and all 4 cameras in test. Returns SPLIT_COLS covering
    every image exactly once (check with t05_schema.check_split).
    """
    # TODO(M5): implement.
    return empty(SPLIT_COLS)


def residual_leak(split: pd.DataFrame, pairs: pd.DataFrame) -> int:
    """Number of pred_dup == 1 pairs whose two images sit on different sides (train vs test)."""
    # TODO(M5): implement.
    return 0

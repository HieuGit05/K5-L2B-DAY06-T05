"""T05 Detector B: extra features, score combination, ablation — OWNER: M4.

Colour histogram (image_ops.color_histogram), capture time / camera, and the final
`score` + `pred_dup`. Time is a weak signal only: frames 40 s apart may be another scene.
Prefer a simple rule first; use logistic regression only if the dev gold is large enough.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .t05_schema import PAIR_COLS


def hist_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Distance between two L1-normalised colour histograms (L1 or Bhattacharyya)."""
    # TODO(M4): implement.
    raise NotImplementedError("hist_distance")


def add_hist_and_time(pairs: pd.DataFrame, images: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Fill d_hist, dt_seconds (abs, from ISO captured_at) and same_camera (0/1) on `pairs`."""
    # TODO(M4): implement.
    return pairs


def combine_score(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add `score` in [0, 1] (higher = more likely near-dup) and `pred_dup` (0/1).

    params come from configs/ (thresholds frozen on dev gold). Must return PAIR_COLS.
    """
    # TODO(M4): implement.
    out = pairs.copy()
    for c in PAIR_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[PAIR_COLS]

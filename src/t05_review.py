"""T05 Human review, gold check, failure analysis — OWNER: M6.

Review queue = pairs closest to the decision threshold (budget 20). Contact sheet shows
image A as `before` and image B as `after`:

    render_contact_sheet(queue, out_html, image_col="path_b", before_image_col="path_a",
                         image_root=ctx.input_dir, fields=("rank", "item_id", "score", "reason"))

Reviewer decisions: approve (true duplicate) / reject (different scene) / escalate.
"""
from __future__ import annotations

import pandas as pd


def make_queue(pairs: pd.DataFrame, images: pd.DataFrame, budget: int, threshold: float) -> pd.DataFrame:
    """Top-`budget` pairs with |score - threshold| smallest.

    Returns a queue compatible with src/common/review.py (REVIEW_COLUMNS) plus the extra
    columns `path_a` and `path_b` (image paths relative to the fixture folder).
    item_id = t05_schema.pair_id(img_a, img_b).
    """
    # TODO(M6): implement (see review.make_review_queue).
    raise NotImplementedError("make_queue")


def gold_check(pairs: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Precision / recall / F1 of pred_dup pairs against the hand-made gold (M2's gold_pairs.csv)."""
    # TODO(M6): implement or reuse src/metrics/pairs.py::pair_prf.
    raise NotImplementedError("gold_check")

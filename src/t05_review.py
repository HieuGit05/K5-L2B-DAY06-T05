"""T05 Human review, gold check, failure analysis — OWNER: M6.

Review queue = pairs closest to the decision threshold (budget 20). Contact sheet shows
image A as `before` and image B as `after`:

    render_contact_sheet(queue, out_html, image_col="path_b", before_image_col="path_a",
                         image_root=ctx.input_dir, fields=("rank", "item_id", "score", "reason"))

Reviewer decisions: approve (true duplicate) / reject (different scene) / escalate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common.review import REVIEW_COLUMNS, make_review_queue
from .metrics.pairs import pair_prf
from .t05_schema import order_pair, pair_id


def make_queue(pairs: pd.DataFrame, images: pd.DataFrame, budget: int, threshold: float) -> pd.DataFrame:
    """Top-`budget` pairs with |score - threshold| smallest.

    Returns a queue compatible with src/common/review.py (REVIEW_COLUMNS) plus the extra
    columns `path_a` and `path_b` (image paths relative to the fixture folder).
    item_id = t05_schema.pair_id(img_a, img_b).
    """
    required_pairs = {"img_a", "img_b", "score"}
    required_images = {"image_id", "path", "camera_id", "captured_at"}
    missing_pairs = required_pairs - set(pairs.columns)
    missing_images = required_images - set(images.columns)
    if missing_pairs:
        raise ValueError(f"pairs missing columns: {sorted(missing_pairs)}")
    if missing_images:
        raise ValueError(f"images missing columns: {sorted(missing_images)}")
    try:
        budget_value = int(budget)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("budget must be a non-negative integer") from exc
    if isinstance(budget, (bool, np.bool_)) or budget_value != budget or budget_value < 0:
        raise ValueError("budget must be a non-negative integer")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be finite") from exc
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite")
    if images["image_id"].isna().any() or images["image_id"].astype(str).str.strip().eq("").any():
        raise ValueError("images.image_id must contain non-empty values")
    if images["image_id"].astype(str).duplicated().any():
        raise ValueError("images has duplicate image_id values")
    if images["path"].isna().any() or images["path"].astype(str).str.strip().eq("").any():
        raise ValueError("images.path must contain non-empty values")
    if pairs[["img_a", "img_b"]].isna().any().any():
        raise ValueError("pairs endpoints must contain non-empty values")

    ranked = pairs.copy()
    if ranked["img_a"].astype(str).str.strip().eq("").any() or ranked["img_b"].astype(str).str.strip().eq("").any():
        raise ValueError("pairs endpoints must contain non-empty values")
    ranked["score"] = pd.to_numeric(ranked["score"], errors="raise")
    if not np.isfinite(ranked["score"].to_numpy(dtype=float)).all():
        raise ValueError("pairs.score must contain only finite numbers")

    ordered = [order_pair(a, b) for a, b in zip(ranked["img_a"], ranked["img_b"])]
    ranked["img_a"] = [a for a, _ in ordered]
    ranked["img_b"] = [b for _, b in ordered]
    if (ranked["img_a"] == ranked["img_b"]).any():
        raise ValueError("pairs must not contain self-pairs")
    ranked["item_id"] = [pair_id(a, b) for a, b in zip(ranked["img_a"], ranked["img_b"])]
    if ranked["item_id"].duplicated().any():
        raise ValueError("pairs contains duplicate unordered image pairs")

    path_of = dict(zip(images["image_id"].astype(str), images["path"].astype(str)))
    missing_ids = sorted((set(ranked["img_a"]) | set(ranked["img_b"])) - set(path_of))
    if missing_ids:
        raise ValueError(f"images has no path for pair endpoints: {missing_ids[:10]}")

    ranked["_threshold_distance"] = (ranked["score"] - threshold_value).abs()
    metadata_of = images.set_index(images["image_id"].astype(str))[["camera_id", "captured_at"]].to_dict("index")

    def evidence(value: object) -> str:
        if pd.isna(value) or not str(value).strip():
            return "unknown"
        return str(value).strip()

    image_metadata = {
        str(image_id): (evidence(row["camera_id"]), evidence(row["captured_at"]))
        for image_id, row in metadata_of.items()
    }
    ranked["_review_reason"] = [
        (
            f"Closest to threshold (distance={distance:.6f}); "
            f"camera_a={image_metadata[a][0]}, captured_at_a={image_metadata[a][1]}; "
            f"camera_b={image_metadata[b][0]}, captured_at_b={image_metadata[b][1]}"
        )
        for a, b, distance in zip(
            ranked["img_a"], ranked["img_b"], ranked["_threshold_distance"]
        )
    ]
    queue = make_review_queue(
        ranked,
        id_col="item_id",
        score_col="_threshold_distance",
        budget=int(budget),
        reason_col="_review_reason",
        ascending=True,
    )
    score_of = dict(zip(ranked["item_id"], ranked["score"]))
    queue["score"] = queue["item_id"].map(score_of).astype(float).round(6)
    endpoint_of = {
        pair_id(a, b): (a, b) for a, b in zip(ranked["img_a"], ranked["img_b"])
    }
    queue["path_a"] = queue["item_id"].map(
        lambda item: path_of[endpoint_of[item][0]]
    )
    queue["path_b"] = queue["item_id"].map(
        lambda item: path_of[endpoint_of[item][1]]
    )
    return queue[REVIEW_COLUMNS + ["path_a", "path_b"]]


def gold_check(pairs: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Precision / recall / F1 of pred_dup pairs against the hand-made gold (M2's gold_pairs.csv)."""
    required_pairs = {"img_a", "img_b", "pred_dup"}
    required_gold = {"img_a", "img_b", "is_dup"}
    missing_pairs = required_pairs - set(pairs.columns)
    missing_gold = required_gold - set(gold.columns)
    if missing_pairs:
        raise ValueError(f"pairs missing columns: {sorted(missing_pairs)}")
    if missing_gold:
        raise ValueError(f"gold missing columns: {sorted(missing_gold)}")

    gold_keys = [order_pair(a, b) for a, b in zip(gold["img_a"], gold["img_b"])]
    if any(a == b for a, b in gold_keys):
        raise ValueError("gold must not contain self-pairs")
    if len(set(gold_keys)) != len(gold_keys):
        raise ValueError("gold contains duplicate unordered image pairs")
    labels = pd.to_numeric(gold["is_dup"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("gold.is_dup must contain only 0 or 1")
    positive_gold = {key for key, label in zip(gold_keys, labels) if int(label) == 1}

    pair_keys = [order_pair(a, b) for a, b in zip(pairs["img_a"], pairs["img_b"])]
    if any(a == b for a, b in pair_keys):
        raise ValueError("pairs must not contain self-pairs")
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("pairs contains duplicate unordered image pairs")
    predictions = pd.to_numeric(pairs["pred_dup"], errors="raise")
    if predictions.isna().any() or not predictions.isin([0, 1]).all():
        raise ValueError("pairs.pred_dup must contain only 0 or 1")

    labeled_keys = set(gold_keys)
    predicted_positive = {
        key
        for key, prediction in zip(pair_keys, predictions)
        if key in labeled_keys and int(prediction) == 1
    }
    return pair_prf(predicted_positive, positive_gold)

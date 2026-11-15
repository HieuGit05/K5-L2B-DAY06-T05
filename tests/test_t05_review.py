import pandas as pd
import pytest

from src.common.review import REVIEW_COLUMNS
from src.t05_review import gold_check, make_queue


def test_make_queue_ranks_canonical_pairs_and_attaches_paths():
    pairs = pd.DataFrame([
        {"img_a": "b", "img_b": "a", "score": 0.51},
        {"img_a": "a", "img_b": "c", "score": 0.49},
        {"img_a": "c", "img_b": "d", "score": 0.9},
        {"img_a": "d", "img_b": "e", "score": 0.505},
    ])
    images = pd.DataFrame({
        "image_id": ["a", "b", "c", "d", "e"],
        "path": [f"images/{image_id}.jpg" for image_id in ["a", "b", "c", "d", "e"]],
        "camera_id": ["cam_0", "cam_0", "cam_1", "cam_1", "cam_2"],
        "captured_at": [
            "2026-10-01T08:00:00",
            "2026-10-01T08:00:01",
            "2026-10-01T09:00:00",
            "2026-10-01T09:00:01",
            "2026-10-01T10:00:00",
        ],
    })

    queue = make_queue(pairs, images, budget=3, threshold=0.5)

    assert list(queue.columns) == REVIEW_COLUMNS + ["path_a", "path_b"]
    assert list(queue["item_id"]) == ["d|e", "a|b", "a|c"]
    assert list(queue["score"]) == [0.505, 0.51, 0.49]
    assert list(queue["path_a"]) == ["images/d.jpg", "images/a.jpg", "images/a.jpg"]
    assert list(queue["path_b"]) == ["images/e.jpg", "images/b.jpg", "images/c.jpg"]
    assert queue["reviewer_decision"].eq("").all()
    assert "camera_a=cam_1" in queue.iloc[0]["reason"]
    assert "captured_at_b=2026-10-01T10:00:00" in queue.iloc[0]["reason"]


def test_make_queue_rejects_repeated_unordered_pairs_and_self_pairs():
    images = pd.DataFrame({
        "image_id": ["a", "b"],
        "path": ["a.jpg", "b.jpg"],
        "camera_id": ["cam_0", "cam_0"],
        "captured_at": ["2026-10-01T08:00:00", "2026-10-01T08:00:01"],
    })
    repeated = pd.DataFrame([
        {"img_a": "a", "img_b": "b", "score": 0.5},
        {"img_a": "b", "img_b": "a", "score": 0.6},
    ])
    self_pair = pd.DataFrame([{"img_a": "a", "img_b": "a", "score": 0.5}])

    with pytest.raises(ValueError, match="duplicate unordered"):
        make_queue(repeated, images, budget=20, threshold=0.5)
    with pytest.raises(ValueError, match="self-pairs"):
        make_queue(self_pair, images, budget=20, threshold=0.5)


def test_gold_check_matches_unordered_pairs_only_within_gold():
    pairs = pd.DataFrame([
        {"img_a": "a", "img_b": "b", "pred_dup": 1},
        {"img_a": "c", "img_b": "d", "pred_dup": 1},
        {"img_a": "x", "img_b": "y", "pred_dup": 1},
    ])
    gold = pd.DataFrame([
        {"img_a": "b", "img_b": "a", "is_dup": 1},
        {"img_a": "c", "img_b": "d", "is_dup": 0},
    ])

    result = gold_check(pairs, gold)

    assert result == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
    }


def test_gold_check_rejects_invalid_binary_labels():
    pairs = pd.DataFrame([{"img_a": "a", "img_b": "b", "pred_dup": 1}])
    gold = pd.DataFrame([{"img_a": "a", "img_b": "b", "is_dup": 2}])

    with pytest.raises(ValueError, match="gold.is_dup"):
        gold_check(pairs, gold)

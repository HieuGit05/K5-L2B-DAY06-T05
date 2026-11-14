import numpy as np
import pandas as pd
import pytest

from src import t05_features as f
from src.t05_schema import PAIR_COLS


def test_hist_distance_range():
    a = np.array([0.5, 0.5, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.5, 0.5])
    assert f.hist_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert f.hist_distance(a, b) == pytest.approx(1.0)
    assert 0.0 < f.hist_distance(a, np.array([0.25, 0.25, 0.25, 0.25])) < 1.0


def test_hist_distance_shape_mismatch():
    with pytest.raises(ValueError):
        f.hist_distance(np.ones(3), np.ones(4))


def _pairs(rows):
    df = pd.DataFrame(rows, columns=["img_a", "img_b", "d_phash", "d_hist"])
    for c in PAIR_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[PAIR_COLS]


def test_combine_score_contract_and_colour_trap():
    pairs = _pairs([
        ("im_000", "im_001", 0, 0.0),    # exact duplicate
        ("im_000", "im_003", 0, 0.6),    # brightness edit: hist far, pHash identical
        ("im_078", "im_096", 28, 0.05),  # colour trap: hist close, pHash far
    ])
    out = f.combine_score(pairs, {"w_phash": 0.8, "w_hist": 0.2, "threshold": 0.757})
    assert list(out.columns) == PAIR_COLS
    assert out["score"].between(0, 1).all()
    assert out["pred_dup"].tolist() == [1, 1, 0]


def test_combine_score_empty():
    out = f.combine_score(pd.DataFrame(columns=PAIR_COLS), {})
    assert list(out.columns) == PAIR_COLS and out.empty


def test_add_hist_and_time_mixed_iso(tmp_path):
    from PIL import Image

    for name, colour in [("a.png", (255, 0, 0)), ("b.png", (255, 0, 0)), ("c.png", (0, 0, 255))]:
        Image.new("RGB", (8, 8), colour).save(tmp_path / name)
    images = pd.DataFrame({
        "image_id": ["a", "b", "c"],
        "path": ["a.png", "b.png", "c.png"],
        "camera_id": ["cam_0", "cam_0", "cam_1"],
        "captured_at": ["2026-10-01T08:00:00", "2026-10-01T08:00:00.200000", "2026-10-01T08:01:00"],
    })
    pairs = _pairs([("a", "b", 0, np.nan), ("a", "c", 0, np.nan)])
    out = f.add_hist_and_time(pairs, images, tmp_path)
    assert out["d_hist"].iloc[0] == pytest.approx(0.0)
    assert out["d_hist"].iloc[1] > 0.5  # red vs blue (G channel identical, so < 1)
    assert out["dt_seconds"].tolist() == pytest.approx([0.2, 60.0])
    assert out["same_camera"].tolist() == [1, 0]


def test_select_threshold_prefers_best_f1():
    r = f.select_threshold(np.array([0.95, 0.9, 0.6, 0.5]), np.array([1, 1, 0, 0]))
    assert r["f1"] == 1.0 and 0.6 < r["threshold"] < 0.9

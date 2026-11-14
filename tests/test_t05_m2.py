"""OWNER: M2. Baselines and gold preparation; synthetic test labels only."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import main as runner
from src.common.manifest import build_manifest
from src.common.gt_guard import enable_gt_guard, disable_gt_guard
from src.t05_baseline import baseline_tables, load_images, prepare_gold, validate_gold
from src.t05_schema import PAIR_COLS, check_split


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    rows = []
    for i in range(12):
        # Six exact groups, each of two distinct files with identical content.
        array = np.random.default_rng(i // 2).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        name = f"im_{i:02d}"
        Image.fromarray(array).save(root / f"{name}.png")
        rows.append({"image_id": name, "path": f"{name}.png", "camera_id": f"cam_{i % 4}",
                     "captured_at": f"2026-10-01T08:00:{i:02d}"})
    images = pd.DataFrame(rows)
    images.to_csv(root / "images.csv", index=False)
    build_manifest(root, [*images.path, "images.csv"], "synthetic:test", "CC0").to_csv(root / "manifest.csv", index=False)
    return root, images


def test_random_and_exact_schema_seed_and_whole_groups(pack):
    root, images = pack
    for method in ("random", "exact"):
        pairs, split, summary = baseline_tables(images, root, method, .3, np.random.default_rng(42))
        p2, s2, _ = baseline_tables(images.iloc[::-1], root, method, .3, np.random.default_rng(42))
        pd.testing.assert_frame_equal(pairs, p2)
        pd.testing.assert_frame_equal(split, s2)
        assert list(pairs.columns) == PAIR_COLS
        check_split(split, list(images.image_id))
        assert not summary["quality_evaluated"]
        if method == "random":
            assert pairs.empty and int(split.split.eq("test").sum()) == 4
        else:
            assert len(pairs) == 6
            side = split.set_index("image_id").split
            assert all(side[r.img_a] == side[r.img_b] for r in pairs.itertuples(index=False))
    with pytest.raises(ValueError, match="test_fraction"):
        baseline_tables(images, root, "random", 1, np.random.default_rng(42))


def test_default_team_command_is_exact_and_runs_offline(pack, tmp_path):
    root, _ = pack
    out = tmp_path / "baseline"
    assert runner.main(["--config", "configs/lab.yaml", "--input", str(root), "--output", str(out),
                        "--run-name", "baseline", "--pipeline", "src.t05_baseline", "--offline", "--seed", "42"]) == 0
    assert json.loads((out / "baseline_summary.json").read_text(encoding="utf-8"))["method"] == "exact"
    assert json.loads((out / "decision.json").read_text(encoding="utf-8"))["status"] == "HOLD"
    assert (out / "suspect_pairs.csv").exists() and (out / "split_manifest.csv").exists()
    (root / "im_00.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="manifest"):
        load_images(root)


def test_empty_predictions_keep_csv_header(pack, tmp_path):
    root, images = pack
    pairs, _, _ = baseline_tables(images, root, "random", .3, np.random.default_rng(42))
    file = tmp_path / "suspect_pairs.csv"
    pairs.to_csv(file, index=False)
    assert list(pd.read_csv(file).columns) == PAIR_COLS


def test_gold_preparation_never_autolabels_or_overwrites(pack, tmp_path):
    root, _ = pack
    gold_path, page_path = tmp_path / "gold.csv", tmp_path / "gold.html"
    enable_gt_guard()
    try:
        prepare_gold(root, gold_path, page_path)
    finally:
        disable_gt_guard()
    gold = pd.read_csv(gold_path, dtype=str)
    assert 30 <= len(gold) <= 50
    assert gold.is_dup.isna().all() and gold.note.isna().all()
    assert not gold.duplicated(["img_a", "img_b"]).any()
    assert (gold.img_a < gold.img_b).all()
    gold.loc[0, "is_dup"] = "1"
    gold.loc[0, "note"] = "Test human-labelled example, not real lab gold"
    gold.to_csv(gold_path, index=False)
    before = gold_path.read_bytes()
    prepare_gold(root, gold_path, page_path)
    assert gold_path.read_bytes() == before
    page = page_path.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in page
    assert "gold_pairs_draft.csv" in page and "reviewer=" in page


def test_gold_check_rejects_drafts_duplicate_and_unknown_ids(pack, tmp_path):
    root, images = pack
    gold_path = tmp_path / "gold.csv"
    prepare_gold(root, gold_path, tmp_path / "gold.html")
    with pytest.raises(ValueError, match="unlabelled"):
        validate_gold(gold_path, images)
    gold = pd.read_csv(gold_path, dtype=str).fillna("")
    gold["is_dup"] = 0
    gold["note"] = "Synthetic test label; manual coverage still requires a human"
    gold.to_csv(gold_path, index=False)
    assert validate_gold(gold_path, images)["n_pairs"] == len(gold)
    gold.loc[1, ["img_a", "img_b"]] = gold.loc[0, ["img_a", "img_b"]]
    gold.to_csv(gold_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        validate_gold(gold_path, images)


@pytest.mark.parametrize("scenario", [
    test_default_team_command_is_exact_and_runs_offline,
    test_gold_preparation_never_autolabels_or_overwrites,
], ids=["baseline-json", "gold-html"])
def test_m2_text_reads_ignore_windows_default_encoding(pack, tmp_path, monkeypatch, scenario):
    """UTF-8 JSON/HTML must be readable even when Windows defaults to CP1252."""
    original = Path.read_text
    implicit_reads = []

    def read_with_windows_default(path, encoding=None, errors=None):
        if encoding is None:
            implicit_reads.append(path.name)
        return original(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", read_with_windows_default)
    scenario(pack, tmp_path)
    assert implicit_reads == [], f"M2 text reads must specify UTF-8: {implicit_reads}"

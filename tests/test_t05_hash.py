
import numpy as np
import pandas as pd
from PIL import Image

from src.t05_hash import compute_hashes, hash_pairs
from src.t05_schema import HASH_COLS, PAIR_COLS


def test_hashes_and_all_pairs(tmp_path):
    # Two different files containing identical image content.
    image = Image.fromarray(
        np.full((64, 64, 3), 120, dtype=np.uint8)
    )

    (tmp_path / "a.png").parent.mkdir(parents=True, exist_ok=True)
    image.save(tmp_path / "a.png")
    image.save(tmp_path / "b.png")

    images = pd.DataFrame([
        {"image_id": "im_001", "path": "a.png"},
        {"image_id": "im_000", "path": "b.png"},
        {"image_id": "im_002", "path": "a.png"},
    ])

    hashes = compute_hashes(images, tmp_path)

    assert list(hashes.columns) == HASH_COLS
    assert len(hashes) == 3

    pairs = hash_pairs(hashes)

    assert list(pairs.columns) == PAIR_COLS
    assert len(pairs) == 3  # 3 * (3 - 1) / 2

    assert (pairs["img_a"] < pairs["img_b"]).all()

    # Identical image content must have zero perceptual distance.
    same_pair = pairs[
        (pairs["img_a"] == "im_000") &
        (pairs["img_b"] == "im_001")
    ].iloc[0]

    assert same_pair["d_phash"] == 0
    assert same_pair["d_dhash"] == 0
    assert same_pair["d_ahash"] == 0

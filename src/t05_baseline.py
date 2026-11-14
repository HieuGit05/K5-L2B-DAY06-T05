"""T05 baselines — OWNER: M2. Entry point for `--pipeline src.t05_baseline`.

    python -m src.main --config configs/lab.yaml --track t05 --output out/baseline \
        --run-name baseline --pipeline src.t05_baseline --offline --seed 42

Baseline A: random split per image (ctx.rng, seed 42), no pair search.
Baseline B: group images with the same sha256 (exact duplicates), then split by group.
Write BOTH suspect_pairs.csv (columns img_a, img_b, ...) and split_manifest.csv
(image_id, split) into ctx.output_dir, even when no pair is found (header only).
"""
from __future__ import annotations

from .common.context import RunContext


def run(ctx: RunContext) -> None:
    # TODO(M2): implement baseline A and B.
    raise NotImplementedError("T05 baseline not implemented yet (M2)")

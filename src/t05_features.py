"""T05 Detector B: extra features, score combination, ablation — OWNER: M4.

Colour histogram (image_ops.color_histogram), capture time / camera, and the final
`score` + `pred_dup`. Time is a weak signal only: frames 40 s apart may be another scene.
Prefer a simple rule first; use logistic regression only if the dev gold is large enough.

Score rule (all knobs live in configs/lab.yaml `params`, frozen on dev gold):

    sim_phash = 1 - d_phash / 64            structure; robust to brightness / re-encode
    sim_hist  = 1 - d_hist                  colour; Bhattacharyya distance in [0, 1]
    score     = w_phash * sim_phash + w_hist * sim_hist      (w_phash + w_hist = 1)
    pred_dup  = score >= threshold

Colour alone falls into the "different scene, same colours" trap and brightness edits
move the histogram a lot, so pHash carries most of the weight. Time / camera are kept as
features for review and grouping but do not enter the score (weak signal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common.image_ops import color_histogram, load_rgb
from .t05_schema import PAIR_COLS, order_pair

HASH_BITS = 64
DEFAULTS = {"w_phash": 0.8, "w_hist": 0.2, "threshold": 0.5, "hist_bins": 16}


def hist_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Bhattacharyya (Hellinger) distance between two L1-normalised histograms, in [0, 1].

    0 = identical colour distribution, 1 = no overlap.
    """
    a = np.asarray(hist_a, dtype=np.float64)
    b = np.asarray(hist_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"histogram shapes differ: {a.shape} vs {b.shape}")
    a = a / max(a.sum(), 1e-12)
    b = b / max(b.sum(), 1e-12)
    bc = float(np.sqrt(a * b).sum())
    return float(np.sqrt(max(0.0, 1.0 - min(bc, 1.0))))


def add_hist_and_time(pairs: pd.DataFrame, images: pd.DataFrame, root: str | Path,
                      bins: int = DEFAULTS["hist_bins"]) -> pd.DataFrame:
    """Fill d_hist, dt_seconds (abs, from ISO captured_at) and same_camera (0/1) on `pairs`."""
    out = pairs.copy()
    if out.empty:
        return out
    root = Path(root)
    meta = images.assign(image_id=images["image_id"].astype(str)).set_index("image_id")

    hists = {iid: color_histogram(load_rgb(root / str(meta.at[iid, "path"])), bins=bins)
             for iid in pd.unique(pd.concat([out["img_a"], out["img_b"]]).astype(str))}
    a = out["img_a"].astype(str)
    b = out["img_b"].astype(str)
    out["d_hist"] = [hist_distance(hists[x], hists[y]) for x, y in zip(a, b)]

    if "captured_at" in meta.columns:
        # format="ISO8601": the fixture mixes "08:00:00" and "08:00:00.200000".
        ts = pd.to_datetime(meta["captured_at"], format="ISO8601", errors="coerce")
        dt = (ts.reindex(a).to_numpy() - ts.reindex(b).to_numpy()) / np.timedelta64(1, "s")
        out["dt_seconds"] = np.abs(dt.astype(np.float64))
    if "camera_id" in meta.columns:
        cam = meta["camera_id"].astype(str)
        out["same_camera"] = (cam.reindex(a).to_numpy() == cam.reindex(b).to_numpy()).astype(int)
    return out


def _params(params: dict | None) -> dict:
    p = {**DEFAULTS, **{k: v for k, v in (params or {}).items() if k in DEFAULTS}}
    total = float(p["w_phash"]) + float(p["w_hist"])
    if total <= 0:
        raise ValueError("w_phash + w_hist must be > 0")
    p["w_phash"], p["w_hist"] = float(p["w_phash"]) / total, float(p["w_hist"]) / total
    p["threshold"] = float(p["threshold"])
    return p


def raw_score(pairs: pd.DataFrame, w_phash: float, w_hist: float) -> pd.Series:
    """Weighted similarity in [0, 1]; missing features count as 'not similar' (0)."""
    sim_phash = (1.0 - pd.to_numeric(pairs["d_phash"], errors="coerce") / HASH_BITS).clip(0, 1).fillna(0.0)
    sim_hist = (1.0 - pd.to_numeric(pairs["d_hist"], errors="coerce")).clip(0, 1).fillna(0.0)
    return (w_phash * sim_phash + w_hist * sim_hist).clip(0, 1)


def combine_score(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add `score` in [0, 1] (higher = more likely near-dup) and `pred_dup` (0/1).

    params come from configs/ (thresholds frozen on dev gold). Must return PAIR_COLS.
    """
    p = _params(params)
    out = pairs.copy()
    for c in PAIR_COLS:
        if c not in out.columns:
            out[c] = np.nan
    if out.empty:
        return out[PAIR_COLS]
    out["score"] = raw_score(out, p["w_phash"], p["w_hist"]).round(6)
    out["pred_dup"] = (out["score"] >= p["threshold"]).astype(int)
    return out[PAIR_COLS]


# ------------------------------------------------------------------ dev-gold tuning
def _join_gold(pairs: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Gold pairs (img_a, img_b, is_dup) joined to their feature rows."""
    g = gold.copy()
    g[["img_a", "img_b"]] = [order_pair(x, y) for x, y in zip(g["img_a"], g["img_b"])]
    g = g.drop_duplicates(["img_a", "img_b"])
    m = g[["img_a", "img_b", "is_dup"]].merge(pairs, on=["img_a", "img_b"], how="left")
    if m["d_phash"].isna().any():
        missing = m.loc[m["d_phash"].isna(), ["img_a", "img_b"]].values.tolist()
        raise ValueError(f"gold pairs not found in pairs table: {missing}")
    return m


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def select_threshold(scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Best-F1 threshold on dev gold; ties -> higher precision, then higher threshold.

    The returned threshold is the midpoint between the chosen cut and the next lower
    score, so it does not sit exactly on a gold example.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    uniq = np.unique(scores)[::-1]
    best = None
    for i, t in enumerate(uniq):
        r = _prf(y_true, (scores >= t).astype(int))
        key = (r["f1"], r["precision"], t)
        if best is None or key > best[0]:
            lower = uniq[i + 1] if i + 1 < len(uniq) else max(t - 0.01, 0.0)
            best = (key, {**r, "threshold": round(float((t + lower) / 2), 4)})
    return best[1]


ABLATIONS = {
    "phash_only": {"w_phash": 1.0, "w_hist": 0.0},
    "hist_only": {"w_phash": 0.0, "w_hist": 1.0},
}


def ablation(pairs: pd.DataFrame, gold: pd.DataFrame, combined_grid=(0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    """F1 on dev gold for pHash only, histogram only and pHash+hist (weight grid).

    Each variant gets its own best threshold on the same gold, so the comparison is fair.
    """
    m = _join_gold(pairs, gold)
    y = m["is_dup"].astype(int).to_numpy()
    variants = dict(ABLATIONS)
    for w in combined_grid:
        variants[f"combined_w{w:.1f}"] = {"w_phash": w, "w_hist": round(1 - w, 4)}
    rows = []
    for name, w in variants.items():
        r = select_threshold(raw_score(m, w["w_phash"], w["w_hist"]).to_numpy(), y)
        rows.append({"variant": name, **w, **r})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """python -m src.t05_features --gold data/gold/gold_pairs.csv [--out reports/M4_ablation.csv]"""
    import argparse

    from .t05_hash import compute_hashes, hash_pairs

    ap = argparse.ArgumentParser(description="T05 M4 ablation + threshold selection on dev gold")
    ap.add_argument("--input", default="fixtures/t05_near_duplicate")
    ap.add_argument("--gold", default="data/gold/gold_pairs.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    images = pd.read_csv(Path(args.input) / "images.csv")
    pairs = add_hist_and_time(hash_pairs(compute_hashes(images, args.input)), images, args.input)
    table = ablation(pairs, pd.read_csv(args.gold))
    print(table.round(4).to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

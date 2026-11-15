"""T05 PoC PIPELINE — Leakage & Near-Duplicate Guardian. OWNER: M1 (integration).

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      images.csv + provenance check (manifest.csv)
    02 process   hashes (M3) -> hist/time features + score (M4)
    03 rank      review queue of image PAIRS, item_id = t05_schema.pair_id(img_a, img_b)
    04 review    apply reviewer decisions (approve -> dup, reject -> not dup) + audit log,
                 then groups + split (M5) on the reviewed pairs
    05 decide    GO / HOLD / REWORK with reason_code + owner

Outputs (in ctx.output_dir, e.g. out/poc/):
    suspect_pairs.csv       img_a, img_b, score, pred_dup (+ feature columns)   <- evaluated
    split_manifest.csv      image_id, split in {train, test, drop}             <- must sit next to it
    suspect_pairs_raw.csv   predictions before human review (kept for the audit diff)
    review_queue.csv, review_contact_sheet.html, decision.json, decision_log.csv, audit_log.jsonl

Rules: never read fixtures/*/gt/ here; seed via ctx.rng; no network. The similarity algorithms
live in the M3–M6 helper modules — this file only wires them together and enforces the contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track
from . import t05_features, t05_grouping, t05_hash, t05_review
from .t05_schema import PAIR_COLS, SPLIT_COLS, check_pairs, check_split, order_pair, pair_id

SPLIT_FILE = "split_manifest.csv"


def _warn(ctx: RunContext, code: str, msg: str) -> None:
    """Record an integration problem (e.g. a helper still returning a stub) for decide()."""
    print(f"[pipeline] WARNING {code}: {msg}")
    ctx.stats.setdefault("warnings", []).append(code)
    if ctx.audit:
        ctx.audit.log(actor="pipeline", action="integration_warning", item_id=code, reason=msg)


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    images_csv = ctx.input_dir / "images.csv"
    if not images_csv.exists():
        raise FileNotFoundError(f"T05 needs {images_csv}")
    images = pd.read_csv(images_csv, dtype={"image_id": str, "path": str, "camera_id": str, "captured_at": str})
    if images["image_id"].duplicated().any():
        raise ValueError("images.csv has duplicate image_id")
    ctx.stats["n_images"] = len(images)
    return {"manifest": pd.read_csv(manifest) if manifest.exists() else None, "images": images}


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    """Detectors in contract order. Returns scored `pairs` (PAIR_COLS).

    Grouping + split run later (group_and_split), AFTER human review, so that an approved
    or rejected pair changes the groups and the split stays leak-free.
    """
    images = data["images"]

    # Detector A (M3): file + perceptual hashes, all unordered pairs with Hamming distances.
    hashes = t05_hash.compute_hashes(images, ctx.input_dir)
    if hashes.empty:
        _warn(ctx, "HASH_NOT_IMPLEMENTED", "t05_hash.compute_hashes returned no rows")
    pairs = t05_hash.hash_pairs(hashes)

    # Detector B (M4): histogram + time/camera features, final score and pred_dup.
    pairs = t05_features.add_hist_and_time(pairs, images, ctx.input_dir)
    pairs = t05_features.combine_score(pairs, ctx.config.get("params") or {})
    pairs = _normalise_pairs(pairs)
    check_pairs(pairs)
    if pairs.empty:
        _warn(ctx, "NO_CANDIDATE_PAIRS", "no candidate pairs produced (detectors empty or stubbed)")
    ctx.stats["n_pairs"] = len(pairs)
    ctx.stats["n_pred_dup_raw"] = int(pairs["pred_dup"].sum()) if len(pairs) else 0
    return pairs


def group_and_split(ctx: RunContext, pairs: pd.DataFrame, images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouping + leak-free split (M5) on the FINAL (post-review) pairs: whole groups go to one side."""
    image_ids = images["image_id"].tolist()
    groups = t05_grouping.build_groups(pairs, image_ids, method=str(ctx.param("group_method", "union_find")))
    if groups.empty:
        _warn(ctx, "GROUPING_NOT_IMPLEMENTED", "build_groups returned no rows; using one group per image")
        groups = pd.DataFrame({"image_id": image_ids, "group_id": image_ids})
    split = t05_grouping.make_split(groups, images, float(ctx.param("test_fraction", 0.3)), ctx.rng)
    if split.empty:
        _warn(ctx, "SPLIT_NOT_IMPLEMENTED", "make_split returned no rows; using a seeded split per group")
        split = _fallback_split(groups, float(ctx.param("test_fraction", 0.3)), ctx.rng)
    ctx.stats["n_pred_dup"] = int(pairs["pred_dup"].sum()) if len(pairs) else 0
    ctx.stats["n_groups"] = int(groups["group_id"].nunique())
    return groups, split


def _normalise_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Enforce the shared schema: string ids, img_a < img_b, unique pairs, numeric score/pred_dup."""
    out = pairs.copy()
    for c in PAIR_COLS:
        if c not in out.columns:
            out[c] = np.nan
    if out.empty:
        return out[PAIR_COLS + [c for c in out.columns if c not in PAIR_COLS]]
    ordered = [order_pair(a, b) for a, b in zip(out["img_a"], out["img_b"])]
    out["img_a"] = [a for a, _ in ordered]
    out["img_b"] = [b for _, b in ordered]
    out = out[out["img_a"] != out["img_b"]].drop_duplicates(["img_a", "img_b"])
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0).astype(float)
    out["pred_dup"] = pd.to_numeric(out["pred_dup"], errors="coerce").fillna(0).astype(int)
    return out.reset_index(drop=True)


def _fallback_split(groups: pd.DataFrame, test_fraction: float, rng: np.random.Generator) -> pd.DataFrame:
    """Seeded group-level split, only used until M5's make_split is implemented."""
    gids = np.array(sorted(groups["group_id"].astype(str).unique()))
    test = set(rng.permutation(gids)[: int(round(test_fraction * len(gids)))])
    return pd.DataFrame({"image_id": groups["image_id"].astype(str),
                         "split": ["test" if str(g) in test else "train" for g in groups["group_id"]]})


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame, images: pd.DataFrame | None = None) -> pd.DataFrame:
    """Review queue of image PAIRS. item_id = 'img_a|img_b' (stable, order-independent)."""
    budget = int(ctx.param("budget", 20))
    threshold = float(ctx.param("threshold", 0.5))
    scored = scored.copy()
    scored["item_id"] = [pair_id(a, b) for a, b in zip(scored["img_a"], scored["img_b"])]
    try:
        q = t05_review.make_queue(scored, images, budget, threshold)   # M6: closest to threshold
    except NotImplementedError:
        _warn(ctx, "QUEUE_NOT_IMPLEMENTED", "t05_review.make_queue missing; using top-B by score")
        q = make_review_queue(scored, id_col="item_id", score_col="score", budget=budget,
                              reason_col="reason" if "reason" in scored.columns else None)
        q = _attach_paths(q, images)
    write_review_queue(q, ctx.out("review_queue.csv"))
    if len(q) and {"path_a", "path_b"} <= set(q.columns):
        from .common.contact_sheet import render_contact_sheet
        render_contact_sheet(q, ctx.out("review_contact_sheet.html"), image_col="path_b", before_image_col="path_a",
                             image_root=ctx.input_dir, fields=("rank", "item_id", "score", "reason"),
                             title="T05 suspect pairs — approve = duplicate, reject = different scene")
    ctx.stats["n_queue"] = len(q)
    return q


def _attach_paths(q: pd.DataFrame, images: pd.DataFrame | None) -> pd.DataFrame:
    if images is None or q.empty:
        return q
    path_of = dict(zip(images["image_id"].astype(str), images["path"]))
    ab = q["item_id"].str.split("|", n=1, expand=True)
    return q.assign(path_a=ab[0].map(path_of), path_b=ab[1].map(path_of))


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Apply out/.../review_queue_reviewed.csv: approve -> pred_dup=1, reject -> pred_dup=0,
    edit -> pred_dup=override_value. escalate/skip leave the prediction unchanged."""
    final = scored.copy()
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    ctx.stats["n_reviewed"] = 0
    if not reviewed_path.exists() or final.empty:
        return final
    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = len(reviewed)
    ctx.stats["review_counts"] = reviewed["reviewer_decision"].value_counts().to_dict()
    key = pd.Series([pair_id(a, b) for a, b in zip(final["img_a"], final["img_b"])], index=final.index)
    for r in reviewed.itertuples(index=False):
        if r.reviewer_decision == "approve":
            new = 1
        elif r.reviewer_decision == "reject":
            new = 0
        elif r.reviewer_decision == "edit" and str(r.override_value).strip() in {"0", "1"}:
            new = int(str(r.override_value).strip())
        else:
            continue
        mask = key == r.item_id
        if not mask.any():
            continue
        before = int(final.loc[mask, "pred_dup"].iloc[0])
        final.loc[mask, "pred_dup"] = new
        ctx.audit.log(actor=r.reviewer, action=f"review:{r.reviewer_decision}", item_id=r.item_id,
                      before=before, after=new, reason=r.review_note)
    return final


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame, split: pd.DataFrame) -> Decision:
    """Policy frozen before evaluation; uses only pipeline outputs, never ground truth."""
    owner = str(ctx.config.get("owner", "")).strip() or "TODO-owner"
    target = float(ctx.param("test_fraction", 0.3))
    kept = split[split["split"] != "drop"]
    test_share = float((kept["split"] == "test").mean()) if len(kept) else 0.0
    leak = t05_grouping.residual_leak(split, final)
    counts = ctx.stats.get("review_counts", {})
    n_rev = int(ctx.stats.get("n_reviewed", 0))
    reject_rate = counts.get("reject", 0) / n_rev if n_rev else 0.0
    metrics = {"n_images": len(split), "n_pairs": len(final), "n_pred_dup": int(final["pred_dup"].sum()) if len(final) else 0,
               "residual_leak_pairs": int(leak), "test_share": round(test_share, 3), "target_test_share": target,
               "kept_share": round(len(kept) / len(split), 3) if len(split) else 0.0,
               "n_reviewed": n_rev, "review_reject_rate": round(reject_rate, 3),
               "warnings": ctx.stats.get("warnings", [])}
    evidence = [str(ctx.out(get_track(ctx.track).pred_file)), str(ctx.out(SPLIT_FILE)),
                str(ctx.out("review_queue.csv")), str(ctx.output_dir / "audit_log.jsonl")]
    stub_codes = {"HASH_NOT_IMPLEMENTED", "GROUPING_NOT_IMPLEMENTED", "SPLIT_NOT_IMPLEMENTED"}
    stubs = sorted(stub_codes & set(metrics["warnings"]))

    if stubs:
        status, code, why = "REWORK", "DETECTOR_OR_SPLIT_INCOMPLETE", f"Helper modules still stubbed: {', '.join(stubs)}"
    elif leak > 0:
        status, code, why = "REWORK", "RESIDUAL_LEAK_AFTER_SPLIT", f"{leak} predicted duplicate pairs cross train/test"
    elif kept.empty or (kept["split"] == "test").sum() == 0:
        status, code, why = "REWORK", "EMPTY_TEST_SPLIT", "No image left in the test split"
    elif len(final) and n_rev == 0:
        status, code, why = "HOLD", "PENDING_HUMAN_REVIEW", "Review queue not reviewed yet (no review_queue_reviewed.csv)"
    elif reject_rate > float(ctx.param("max_review_reject_rate", 0.3)):
        status, code, why = "HOLD", "HIGH_FALSE_MERGE_IN_REVIEW", f"Reviewers rejected {reject_rate:.0%} of queued pairs (false merges)"
    elif abs(test_share - target) > float(ctx.param("split_tolerance", 0.1)):
        status, code, why = "HOLD", "SPLIT_RATIO_OFF_TARGET", f"Test share {test_share:.2f} vs target {target:.2f}"
    else:
        status, code, why = "GO", "SPLIT_LEAK_FREE", "No predicted duplicate crosses the split; review passed; ratio on target"
    return Decision(status=status, reason_code=code, reason=why, owner=owner, scope=f"{ctx.track}/{ctx.run_name}",
                    evidence=evidence, metrics=metrics)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    if spec.track != "t05":
        raise NotImplementedError(f"this pipeline implements t05 only, got {spec.track} ({spec.title})")
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    scored.to_csv(ctx.out("suspect_pairs_raw.csv"), index=False)
    queue = rank(ctx, scored, data["images"])
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)

    _, split = group_and_split(ctx, final, data["images"])   # after review: overrides reshape the groups
    check_split(split, data["images"]["image_id"].tolist())
    split[SPLIT_COLS].to_csv(ctx.out(SPLIT_FILE), index=False)   # next to suspect_pairs.csv (evaluator needs it)
    write_decision(decide(ctx, final, split), ctx.out("decision.json"), ctx.out("decision_log.csv"))

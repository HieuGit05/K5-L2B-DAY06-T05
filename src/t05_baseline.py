"""T05 baselines — OWNER: M2. Entry point for `--pipeline src.t05_baseline`.

    python -m src.main --config configs/lab.yaml --track t05 --output out/baseline \
        --run-name baseline --pipeline src.t05_baseline --offline --seed 42

Baseline A: random split per image (ctx.rng, seed 42), no pair search.
Baseline B: group images with the same sha256 (exact duplicates), then split by group.
Write BOTH suspect_pairs.csv (columns img_a, img_b, ...) and split_manifest.csv
(image_id, split) into ctx.output_dir, even when no pair is found (header only).
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from itertools import combinations
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.image_ops import color_histogram, dhash, hash_hex, load_rgb, phash
from .common.io import sha256_file, write_json
from .common.manifest import verify_manifest
from .t05_schema import PAIR_COLS, SPLIT_COLS, check_pairs, check_split, order_pair

GOLD_COLS = ["img_a", "img_b", "is_dup", "note"]


def load_images(root: Path) -> pd.DataFrame:
    root = Path(root).resolve()
    problems = verify_manifest(root / "manifest.csv")
    if problems:
        raise ValueError(f"M2 input manifest failed: {problems[:3]}")
    images = pd.read_csv(root / "images.csv", dtype=str)
    required = ["image_id", "path", "camera_id", "captured_at"]
    if not set(required) <= set(images) or images.empty:
        raise ValueError(f"images.csv requires {required} and at least one image")
    if images[required].isna().any().any() or images.image_id.duplicated().any():
        raise ValueError("missing image metadata or duplicate image_id")
    for row in images.itertuples(index=False):
        path = (root / row.path).resolve()
        if Path(row.path).is_absolute() or not path.is_relative_to(root):
            raise ValueError("image paths must stay inside the fixture directory")
        if not path.is_file():
            raise FileNotFoundError(path)
    return images.sort_values("image_id").reset_index(drop=True)


def baseline_tables(images, root, method, test_fraction, rng):
    """Baseline A has no search; B compares ONLY byte SHA-256, not M3/M5 code."""
    if method not in {"random", "exact"}:
        raise ValueError("baseline_method must be random or exact")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    images = images.sort_values("image_id").reset_index(drop=True)
    if len(images) < 2:
        raise ValueError("need at least two images for train/test")
    grouped = {}
    for row in images.itertuples(index=False):
        key = row.image_id if method == "random" else sha256_file(Path(root) / row.path)
        grouped.setdefault(key, []).append(row.image_id)
    records = []
    if method == "exact":
        for members in grouped.values():
            for a, b in combinations(sorted(members), 2):
                records.append({"img_a": a, "img_b": b, "score": 1.0, "pred_dup": 1})
    pairs = pd.DataFrame(records).reindex(columns=PAIR_COLS)
    check_pairs(pairs)
    keys = sorted(grouped)
    test_ids = set()
    target = round(len(images) * test_fraction)
    for position in rng.permutation(len(keys)):
        members = grouped[keys[position]]
        if abs(len(test_ids) + len(members) - target) < abs(len(test_ids) - target):
            test_ids.update(members)
    split = pd.DataFrame({"image_id": images.image_id,
                          "split": ["test" if i in test_ids else "train" for i in images.image_id]})
    check_split(split, list(images.image_id))
    summary = {"method": method, "n_images": len(images), "n_groups": len(grouped),
               "n_predicted_duplicate_pairs": len(pairs), "test_fraction_target": test_fraction,
               "test_fraction_actual": float(split.split.eq("test").mean()),
               "empty_partition": not {"train", "test"} <= set(split.split),
               "quality_evaluated": False,
               "limitation": "Random split is per image; exact baseline detects identical bytes only. No GT read."}
    return pairs, split[SPLIT_COLS], summary


def run(ctx: RunContext) -> None:
    if ctx.track != "t05":
        raise ValueError("M2 baseline supports t05 only")
    # Existing team command --run-name baseline stays valid and means exact.
    # Explicit run names select A/B without editing M1/M4's shared config.
    aliases = {"baseline_random": "random", "baseline_a": "random",
               "baseline_exact": "exact", "baseline_b": "exact"}
    method = aliases.get(ctx.run_name, ctx.param("baseline_method", "exact"))
    images = load_images(ctx.input_dir)
    pairs, split, summary = baseline_tables(images, ctx.input_dir, method,
                                           float(ctx.param("test_fraction", .3)), ctx.rng)
    pairs.to_csv(ctx.out("suspect_pairs.csv"), index=False)
    split.to_csv(ctx.out("split_manifest.csv"), index=False)
    summary.update({"seed": ctx.seed, "manifest_sha256": sha256_file(ctx.input_dir / "manifest.csv")})
    write_json(ctx.out("baseline_summary.json"), summary)
    write_decision(Decision(status="HOLD", reason_code="BASELINE_ONLY", owner=ctx.config.get("owner", "M2"),
                            reason="Reference baseline; not a release recommendation or evaluated quality claim.",
                            scope=f"t05/{ctx.run_name}", evidence=["suspect_pairs.csv", "split_manifest.csv"],
                            metrics=summary), ctx.out("decision.json"))
    ctx.stats.update({"n_items": len(images), "baseline_method": method, "n_groups": summary["n_groups"]})


def suggest_pairs(images, root):
    """48 diverse retrieval candidates. NEVER assign is_dup from descriptors."""
    descriptors = {}
    for row in images.itertuples(index=False):
        array = load_rgb(Path(root) / row.path)
        descriptors[row.image_id] = {
            "sha": sha256_file(Path(root) / row.path), "phash": int(hash_hex(phash(array)), 16),
            "dhash": int(hash_hex(dhash(array)), 16), "hist": color_histogram(array),
            "time": pd.to_datetime(row.captured_at, utc=True), "camera": row.camera_id}
    all_pairs = []
    for a, b in combinations(sorted(descriptors), 2):
        x, y = descriptors[a], descriptors[b]
        all_pairs.append({"img_a": a, "img_b": b, "exact": x["sha"] == y["sha"],
                          "d_hash": (x["phash"] ^ y["phash"]).bit_count() + (x["dhash"] ^ y["dhash"]).bit_count(),
                          "d_hist": float(abs(x["hist"] - y["hist"]).sum()),
                          "dt": abs((x["time"] - y["time"]).total_seconds()),
                          "same_camera": x["camera"] == y["camera"]})
    buckets = [
        ("byte_match_candidate", [p for p in all_pairs if p["exact"]], lambda p: (p["img_a"], p["img_b"])),
        ("visual_similarity_candidate", [p for p in all_pairs if not p["exact"]], lambda p: (p["d_hash"], p["d_hist"], p["img_a"], p["img_b"])),
        ("burst_candidate", [p for p in all_pairs if p["same_camera"] and p["dt"] <= 1 and not p["exact"]],
         lambda p: (p["dt"], p["d_hash"], p["img_a"], p["img_b"])),
        ("similar_color_candidate", [p for p in all_pairs if not p["exact"] and p["d_hash"] >= 35],
         lambda p: (p["d_hist"], -p["d_hash"], p["img_a"], p["img_b"]))]
    chosen, used, frequency = [], set(), Counter()

    def take(candidates, bucket, limit, cap):
        added = 0
        for pair in candidates:
            key = (pair["img_a"], pair["img_b"])
            if key in used or any(frequency[i] >= cap for i in key):
                continue
            chosen.append({"img_a": key[0], "img_b": key[1], "bucket": bucket})
            used.add(key)
            frequency.update(key)
            added += 1
            if added >= limit:
                break

    for bucket, candidates, sort_key in buckets:
        take(sorted(candidates, key=sort_key), bucket, 12, 5)
    if len(chosen) < 48:
        take(sorted(all_pairs, key=lambda p: (p["d_hash"], p["img_a"], p["img_b"])),
             "additional_candidate", 48 - len(chosen), 10)
    return chosen


def validate_gold(path, images):
    gold = pd.read_csv(path, dtype=str).fillna("")
    if list(gold.columns) != GOLD_COLS:
        raise ValueError(f"gold columns must be exactly {GOLD_COLS}")
    if not 30 <= len(gold) <= 50:
        raise ValueError("final gold must contain 30–50 manually labelled pairs")
    check_pairs(gold)
    if gold.duplicated(["img_a", "img_b"]).any():
        raise ValueError("duplicate gold pair")
    if not (set(gold.img_a) | set(gold.img_b)) <= set(images.image_id):
        raise ValueError("unknown image_id in gold")
    if not gold.is_dup.isin(["0", "1"]).all():
        raise ValueError("gold still contains unlabelled pairs; humans must assign is_dup=0/1")
    if gold.note.str.strip().eq("").any():
        raise ValueError("each manually labelled pair needs a note")
    return {"n_pairs": len(gold), "n_dup": int(gold.is_dup.eq("1").sum()),
            "n_not_dup": int(gold.is_dup.eq("0").sum()), "gold_sha256": sha256_file(path),
            "manual_coverage_check_required": "Confirm exact, transformed, burst and cross-scene color-trap cases by viewing images."}


def prepare_gold(root, gold_path, html_path):
    images = load_images(root)
    gold_path, html_path = Path(gold_path), Path(html_path)
    if gold_path.exists() and len(pd.read_csv(gold_path)):
        gold = pd.read_csv(gold_path, dtype=str).fillna("")
        if list(gold.columns) != GOLD_COLS:
            raise ValueError("existing gold uses a different schema; will not overwrite it")
        candidates = [dict(r, bucket="existing_pair") for r in gold.to_dict("records")]
    else:
        candidates = suggest_pairs(images, root)
        gold = pd.DataFrame(candidates)[["img_a", "img_b"]].assign(is_dup="", note="")
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold[GOLD_COLS].to_csv(gold_path, index=False)
    image_of = images.set_index("image_id").to_dict("index")

    def thumbnail(image_id):
        with Image.open(Path(root) / image_of[image_id]["path"]) as picture:
            picture = picture.convert("RGB")
            picture.thumbnail((640, 360))
            stream = io.BytesIO()
            picture.save(stream, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()

    cards = []
    for index, pair in enumerate(candidates):
        figures = []
        for key in ("img_a", "img_b"):
            image_id = pair[key]
            meta = image_of[image_id]
            caption = f"{image_id} · {meta['camera_id']} · {meta['captured_at']}"
            figures.append(f'<figure><img src="{thumbnail(image_id)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        cards.append(f'''<article data-index="{index}"><h2>{index + 1}. {html.escape(pair['img_a'])} / {html.escape(pair['img_b'])}</h2>
<div class="images">{''.join(figures)}</div><div class="choices">
<label>Nhãn bạn xác nhận <select class="label"><option value="">Chưa gán</option><option value="1">1 — Trùng / near-duplicate</option><option value="0">0 — Không trùng</option></select></label>
<label>Loại case đã kiểm tra <select class="kind"><option value="">Chọn loại</option><option value="exact">Trùng y hệt</option><option value="transformed">Trùng sau biến đổi</option><option value="burst">Ảnh liên tiếp / burst</option><option value="color_trap">Khác scene, cùng màu</option><option value="other">Khác / ghi rõ</option></select></label></div>
<label>Lý do quan sát trên ảnh <textarea class="note" placeholder="Ví dụ: cùng bố cục/đối tượng nhưng sáng hơn; hoặc khác vị trí đối tượng dù cùng màu..."></textarea></label></article>''')
    manifest_hash = sha256_file(Path(root) / "manifest.csv")
    payload = json.dumps(candidates, ensure_ascii=False).replace("<", "\\u003c")
    page = GOLD_HTML.replace("__CARDS__", "\n".join(cards)).replace("__PAIRS__", payload)
    page = page.replace("__STORAGE__", json.dumps(f"t05-m2-gold-{manifest_hash}"))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    return {"candidates": len(candidates), "gold_path": str(gold_path), "html_path": str(html_path),
            "status": "UNLABELLED: descriptor retrieval is not ground truth"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="M2: offline gold labelling helper; no automatic labels")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-gold", action="store_true")
    action.add_argument("--check-gold", type=Path)
    parser.add_argument("--input", type=Path, default=Path("fixtures/t05_near_duplicate"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold_pairs.csv"))
    parser.add_argument("--html", type=Path, default=Path("out/m2/label_gold.html"))
    args = parser.parse_args(argv)
    try:
        result = prepare_gold(args.input, args.gold, args.html) if args.prepare_gold else validate_gold(args.check_gold, load_images(args.input))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"M2: {error}\n")


GOLD_HTML = r'''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>T05 · M2 · Gán nhãn gold</title>
<style>body{font:16px system-ui;margin:0;background:#eef2f5;color:#182633}header{position:sticky;top:0;background:#fff;padding:14px 24px;border-bottom:1px solid #c9d3de;z-index:2}main{max-width:1150px;margin:24px auto;padding:0 16px}h1{font-size:22px;margin:0 0 8px}h2{font-size:18px}article{background:white;border:1px solid #c9d3de;padding:20px;margin:20px 0;border-radius:12px}.images,.choices{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:240px}img{width:100%;image-rendering:auto}figcaption{font-size:13px;padding:8px 0;color:#506477}label{display:block;flex:1;margin:8px 0}select,textarea,input{font:inherit;padding:8px;box-sizing:border-box}select,textarea{display:block;width:100%;margin-top:6px}textarea{min-height:64px}button{font:inherit;padding:8px 12px;margin:4px;cursor:pointer;background:#153d64;color:white;border:0;border-radius:6px}#progress{font-weight:600}p{line-height:1.5}.notice{background:#fff4d5;padding:14px;border-radius:8px}article.done{border-color:#32815b}small{color:#506477}</style>
<header><h1>T05 · Người 2: gán nhãn gold bằng mắt</h1>
<label>Tên người gán nhãn <input id="reviewer" placeholder="Nhập tên của bạn"></label>
<span id="progress"></span> <button id="draft">Tải bản nháp</button><button id="export">Xuất gold hoàn chỉnh</button>
<small id="storage"></small></header><main><p class="notice">Các cặp dưới đây chỉ là ứng viên tìm kiếm, CHƯA phải nhãn thật. Bạn tự xem cả hai ảnh, chọn nhãn và loại case, ghi lý do. Không chắc: để trống, không đoán. Gold dùng để chọn ngưỡng là tập dev, không phải holdout độc lập.</p>
<p>1 = cùng nội dung nguồn / near-duplicate dù crop, đổi sáng hoặc nén; 0 = ảnh khác nội dung. Cùng camera, gần thời gian hoặc cùng màu chưa đủ kết luận trùng. Cần ít nhất 30–50 cặp và đủ bốn loại case khi xuất.</p>
<p>Xuất xong, chép file vào <b>data/gold/gold_pairs.csv</b> trong checkout M2. Bản nháp chỉ lưu tiến độ, chưa bàn giao cho Người 4.</p>
__CARDS__</main><script>
const PAIRS = __PAIRS__;
const STORAGE = __STORAGE__;
const CARDS = [...document.querySelectorAll('article')];
let state = {reviewer:'', rows:{}};
try { state = JSON.parse(localStorage.getItem(STORAGE)) || state; } catch(e) { document.getElementById('storage').textContent='Trình duyệt không lưu được tiến độ: hãy tải bản nháp thường xuyên.'; }
const key = p => p.img_a+'|'+p.img_b;
document.getElementById('reviewer').value = state.reviewer || '';
function readRows(){return CARDS.map((c,i)=>({img_a:PAIRS[i].img_a,img_b:PAIRS[i].img_b,is_dup:c.querySelector('.label').value,kind:c.querySelector('.kind').value,note:c.querySelector('.note').value.trim()}));}
function save(){
 state = {reviewer:document.getElementById('reviewer').value.trim(),rows:{}};
 const rows=readRows(); rows.forEach(r=>state.rows[key(r)]={...r});
 try{localStorage.setItem(STORAGE,JSON.stringify(state));}catch(e){}
 const count=rows.filter(r=>r.is_dup!==''&&r.kind&&r.note).length;
 document.getElementById('progress').textContent=`${count}/${PAIRS.length} cặp đã gán đủ thông tin`;
 CARDS.forEach((c,i)=>c.classList.toggle('done',rows[i].is_dup!==''&&!!rows[i].kind&&!!rows[i].note));
}
CARDS.forEach((c,i)=>{
 const p=PAIRS[i], old=state.rows[key(p)];
 if(old){c.querySelector('.label').value=old.is_dup||'';c.querySelector('.kind').value=old.kind||'';c.querySelector('.note').value=old.note||'';}
 else if(p.is_dup==='0'||p.is_dup==='1'){c.querySelector('.label').value=p.is_dup;c.querySelector('.note').value=p.note||'';}
 c.addEventListener('input',save);c.addEventListener('change',save);
});
document.getElementById('reviewer').addEventListener('input',save);
function esc(v){v=String(v??'');return /[",\r\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;}
function exportGold(final){
 save(); const reviewer=state.reviewer, rows=readRows();
 if(!reviewer){alert('Nhập tên người gán nhãn.');return;}
 let output=final?rows.filter(r=>r.is_dup!==''):rows;
 if(final){
  if(output.length<30||output.length>50){alert('Cần 30–50 cặp có nhãn.');return;}
  if(output.some(r=>!r.kind||!r.note)){alert('Mỗi cặp đã gán phải có loại case và lý do.');return;}
  const kinds=new Set(output.map(r=>r.kind));
  if(['exact','transformed','burst','color_trap'].some(k=>!kinds.has(k))){alert('Cần đủ exact, transformed, burst và color_trap. Kiểm tra thủ công, không gán ép để đủ loại.');return;}
  if(output.some(r=>(r.kind==='color_trap'&&r.is_dup!=='0')||(['exact','transformed'].includes(r.kind)&&r.is_dup!=='1'))){alert('Nhãn mâu thuẫn với loại case. Kiểm tra lại ảnh.');return;}
 }
 const csv=[['img_a','img_b','is_dup','note'],...output.map(r=>[r.img_a,r.img_b,r.is_dup,`reviewer=${reviewer}; type=${r.kind||'unconfirmed'}; ${r.note}`])].map(row=>row.map(esc).join(',')).join('\r\n')+'\r\n';
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download=final?'gold_pairs.csv':'gold_pairs_draft.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.getElementById('draft').onclick=()=>exportGold(false);
document.getElementById('export').onclick=()=>exportGold(true);
save();
</script></html>'''


if __name__ == "__main__":
    raise SystemExit(main())

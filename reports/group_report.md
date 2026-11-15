# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).
>
> **Trạng thái bản nháp (M1, 2026-11-15):** đã điền 00, 01, 07, 08, 10 và phần số liệu của 02, 04, 05.
> Các chỗ `【Mx bổ sung】` là phần thành viên tương ứng viết thêm. Số liệu bên dưới lấy từ output script
> ở commit `38fe316` (`out/poc/decision.json`, `out/poc/run_info.json`, `out/baseline/baseline_summary.json`,
> `reports/M4_ablation.csv`, `reports/M2_GOLD_HANDOFF.txt`).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | Nhóm NhăngNhăng |
| Challenge | T05 · Leakage & Near-Duplicate Guardian |
| Thành viên (6) + vai trò | 1. Đặng Trung Hiếu — HieuGit05 (Lead / Integrator) · 2. Khải — kasdie (Baseline + gold set) · 3. Hoang Quang Truong (Detector A — hash) · 4. dqhuy11 (Detector B — feature + score + ablation) · 5. longnguyen3204 (Grouping + leak-free split) · 6. 【M6 bổ sung】 (Human review + failure analysis) |
| Git URL | https://github.com/HieuGit05/K5-L2B-DAY06-T05 |
| Tag `demo-180m` (commit) | 【M1 điền sau khi gắn tag】 |
| Tag `final-24h` (commit) | 【M1 điền sau khi gắn tag】 |
| Deadline (giờ VN) | 【M1 điền】 |

## 01 · Problem statement

- **Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai?** Đội huấn luyện và đánh giá mô hình thị giác dùng bộ ảnh camera này. Nếu ảnh trùng hoặc gần trùng (cùng khung hình, bản crop/chỉnh sáng/nén lại, hay ảnh chụp liên tiếp cùng scene) nằm ở cả train lẫn test, điểm test bị **thổi phồng** — mô hình được "xem trước đề". Người ra quyết định release tin vào con số sai và đưa mô hình kém ra thực tế. Ngược lại, nếu gộp nhầm hai scene khác nhau (false merge), ta loại bỏ hoặc dồn lệch dữ liệu không cần thiết.
- **Điểm nghẽn cụ thể:** `images.csv` không có `scene_id`; chỉ có `image_id, path, camera_id, captured_at`. Hash file (sha256) chỉ bắt được ảnh giống hệt từng byte, bỏ sót bản đã biến đổi. Chỉ dùng màu thì dính bẫy "khác scene nhưng màu giống". Thời gian chụp là tín hiệu yếu (hai frame cách nhau 40 s có thể là scene khác).
- **Đầu vào → đầu ra:** 120 ảnh (`fixtures/t05_near_duplicate`, 4 camera) + `manifest.csv` → `out/poc/suspect_pairs.csv` (`img_a, img_b, score, pred_dup` cho 7.140 cặp) + `out/poc/split_manifest.csv` (`image_id → train/test/drop`, cả nhóm trùng nằm cùng một phía) + hàng đợi review, audit log, `decision.json`.
- **Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval):** xem bảng trong `reports/metric_contract.md` và mục 07. Tóm tắt: REWORK nếu module còn thiếu, còn cặp nghi trùng cắt ngang train/test, hoặc test rỗng; HOLD nếu chưa có human review, reviewer reject > 30 %, hoặc tỉ lệ test lệch 0.3 quá ±0.1; còn lại GO.
- **Giả định của nhóm (cần DPO/legal/domain owner xác nhận):** (1) ảnh fixture là dữ liệu tổng hợp/được phép dùng cho lab, không chứa PII cần xử lý thêm; (2) "cùng near-dup group" là đơn vị leakage đúng cho bài toán downstream; (3) `test_fraction = 0.3` là tỉ lệ chấp nhận được với chủ dữ liệu; (4) giữ lại ảnh trùng trong cùng một split (thay vì xoá) là chấp nhận được.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev (gold tự gán) | 47 cặp ảnh (34 trùng / 13 không trùng) | `data/gold/gold_pairs.csv`, gán tay bởi Khải (M2); sha256 `edd4b669b42b829b04cca127aa2326fc7036fe95a16dd600cd95f0dffd26c86d` | 【M2 bổ sung theo `manifest.csv`】 | Nhãn do người gán; ảnh thuộc fixture |
| eval / heldout | 120 ảnh = 7.140 cặp, GT ẩn `gt/gt_groups.csv` | `fixtures/t05_near_duplicate` — GT do giảng viên giữ, không có trong repo sinh viên | 【M2 bổ sung】 | 【M2 bổ sung】 |

- **Schema & class mapping:** 4 bảng chung trong `src/t05_schema.py` — `hashes` (`image_id, sha256, phash, dhash, ahash`), `pairs` (`img_a < img_b`, `d_phash, d_dhash, d_ahash, d_hist, dt_seconds, same_camera, score, pred_dup`), `groups` (`image_id, group_id`), `split_manifest` (`image_id, split ∈ {train, test, drop}`). Gold: `img_a, img_b, is_dup (0/1), note`.
- **Seed: 42 · Cách chia split:** theo **nhóm** near-dup (union-find trên các cặp `pred_dup = 1`), không theo ảnh; cả nhóm vào một phía; mục tiêu 30 % test (`src/t05_grouping.py::make_split`, rng seed 42).
- **`manifest.csv` verified (sha256): có** — `python -m tools.verify_manifests` → 14 manifests, 0 failing (kiểm cả trên bản clone mới với `core.autocrlf=true`, sau khi thêm `.gitattributes` để Git không đổi LF→CRLF trong `fixtures/` và `data/`). Pipeline chạy với `strict_provenance: true`.
- **Gold set:** loại case — exact 16 · transformed 1 · burst 1 · color_trap 10 · other 19. Cặp `im_090/im_094` người gán không chắc nên loại khỏi file cuối. 【M2 bổ sung cách xem ảnh và tiêu chí gán】
- **Dữ liệu thật dùng (nếu có):** không dùng `data/real/`.

## 03 · Research & design

- 2–3 references (URL + ngày xem) và **vì sao** chọn kỹ thuật: 【M3 + M4 bổ sung — pHash/dHash/aHash, Bhattacharyya histogram, union-find vs complete-linkage】
- **Sơ đồ data flow:**

```
images.csv + manifest.csv ──(01 load: verify sha256)──▶ 120 ảnh
   └─(02 process)─▶ M3 hash: sha256 + phash/dhash/ahash ─▶ 7.140 cặp + Hamming
                    ─▶ M4 feature: d_hist, dt_seconds, same_camera ─▶ score = 0.8·sim_phash + 0.2·sim_hist
                    ─▶ pred_dup = score ≥ 0.757
                    ─▶ M5 group (union-find) ─▶ split theo nhóm (seed 42)
   └─(03 rank)─▶ review_queue.csv (B = 20, item_id = "img_a|img_b") + contact sheet HTML
   └─(04 review)─▶ approve/reject/edit → pred_dup, ghi audit_log.jsonl
   └─(05 decide)─▶ decision.json (GO/HOLD/REWORK + reason_code + owner) ─▶ report
```

- **Thành phần offline:** không dùng model hay weights; chỉ numpy / scipy / Pillow / pandas theo `requirements.txt`. Chạy với `--offline` (network bị chặn).
- **Trade-offs đã cân nhắc:** 120 ảnh → so toàn bộ 7.140 cặp, không cần LSH. pHash bền với đổi sáng/nén nhưng có thể yếu với crop mạnh; histogram màu bắt được biến đổi hình học nhẹ nhưng dính bẫy màu → chỉ cho trọng số 0.2. Union-find đơn giản nhưng có thể gộp dây chuyền; complete-linkage chặt hơn. 【M4 + M5 bổ sung】

## 04 · Baseline vs PoC

- **Baseline (M2, `src/t05_baseline.py`, commit `027fb16` / `bcfea2a`):**
  - A — random split theo từng ảnh (seed 42), không tìm cặp: 36 test / 84 train, `suspect_pairs.csv` chỉ có header.
  - B — exact sha256 (lệnh `--run-name baseline` mặc định): 100 nhóm, **20 cặp trùng y hệt**, 36 test / 84 train (`test_fraction_actual = 0.3`).
- **PoC (`src/pipeline.py` + M3/M4/M5, commit `38fe316`):** score = 0.8 · pHash + 0.2 · histogram, ngưỡng 0.757 (chốt trên gold dev), split theo nhóm union-find. Kết quả: **236 / 7.140 cặp `pred_dup = 1`**, 81 train / 39 test (test 0.325), giữ lại 100 % ảnh, **0 cặp nghi trùng cắt ngang train/test**.
- **Cùng budget / cùng heldout / cùng seed:** xác nhận — cùng 120 ảnh, seed 42, `test_fraction = 0.3`, budget B = 20, cùng lệnh evaluate.

**Bảng 1 · Baseline vs PoC & hard slices**

> ⚠️ Repo sinh viên **không có GT** (`src.evaluate` báo *"Ground truth is not distributed in the student repo; the instructor runs official evaluation"*), nên nhóm **không tự sinh được** `metrics_baseline.json`, `metrics_poc.json`, `table1_baseline_vs_poc.md`. Bảng dưới để trống ô chính thức chờ giảng viên chấm; cột "gold dev" là đánh giá nội bộ trên 47 cặp đã dùng để chọn ngưỡng — **là điểm dev, không phải holdout**, không được coi là số chính thức.

| Metric (unit) | Baseline (exact) | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Leakage Detection F1 (pair) — chính thức | 【GV chấm】 | 【GV chấm】 | higher is better |
| Primary trên gold dev (47 cặp, nội bộ) | 【M6 chạy gold_check】 | 0.986 (TP 34 / FP 1 / FN 0; P 0.971, R 1.000) | điểm dev, ngưỡng đã chọn trên chính tập này |
| Hard-slice: color_trap trên gold dev | 【M6 bổ sung】 | 【M6 bổ sung】 | N = 10 cặp |
| Review@B / miss / FP | — | 【sau review M6】 | B = 20 |
| Runtime / 100 items | 【M2: `out/baseline/run_info.json`】 | 1.85 s / 120 ảnh (≈ 1.5 s / 100 ảnh) | CPU Intel64 Family 6 Model 154, Windows 11, Python 3.13.5 |

## 05 · Evaluation protocol

- **Công thức primary metric:** đơn vị = cặp ảnh không thứ tự. Chỉ xét `pred_dup = 1`. TP = cặp dự đoán cùng `dup_group_id` trong GT; FP = không cùng nhóm; FN = cặp GT cùng nhóm mà không dự đoán. F1 = 2PR/(P+R). Chi tiết ở `reports/metric_contract.md`.
- **Threshold / budget B / K và lúc chốt:** M4 chọn trên gold dev 47 cặp, ghi vào `configs/lab.yaml` (`w_phash 0.8, w_hist 0.2, threshold 0.757`, F1 gold 0.986, ngưỡng = điểm giữa khe score 0.635..0.879), commit `bac0120`, khoá trước evaluate. B = 20. Metric contract chốt 2026-11-14 15:30.
- **Ablation (M4, `reports/M4_ablation.csv`, trên gold dev):**

| Variant | w_phash | w_hist | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| phash_only | 1.0 | 0.0 | 34 | 1 | 0 | 0.9714 | 1.0 | 0.9855 |
| hist_only | 0.0 | 1.0 | 34 | 13 | 0 | 0.7234 | 1.0 | 0.8395 |
| combined_w0.5 | 0.5 | 0.5 | 31 | 1 | 3 | 0.9688 | 0.9118 | 0.9394 |
| combined_w0.8 (chọn) | 0.8 | 0.2 | 34 | 1 | 0 | 0.9714 | 1.0 | 0.9855 |

  `hist_only` có 13 FP — đúng bằng số cặp không trùng trong gold, minh hoạ bẫy màu. 【M4 bổ sung: vì sao chọn 0.8 thay vì phash_only khi F1 bằng nhau】
- **Độ bất định (N nhỏ):** gold dev chỉ 34 cặp dương → Wilson CI 95 % cho recall 34/34 ≈ [0.90, 1.00], precision 34/35 ≈ [0.85, 0.99] (`src/metrics/estimation.py::wilson_ci`). 【M4 xác nhận lại bằng script】
- **Lệnh evaluate chính xác (giảng viên chạy):** `python -m src.evaluate --track t05 --pred out/poc/suspect_pairs.csv --run-name poc --out reports/metrics_poc.json`

## 06 · Result slices

| Slice | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| 【M5/M6 bổ sung — exact / transformed / burst】 | | | | |
| 【M5/M6 bổ sung — color_trap】 | 10 (gold) | | | bẫy khác scene màu giống |

- Failure case 1 (ảnh/metadata + root cause): 【M6 — bắt buộc: gộp nhầm hai scene vì màu giống】
- Failure case 2: 【M6 / M5 — ví dụ: leakage còn sót sau group split, hoặc 1 FP trên gold】
- **Latency / compute:** PoC 1.85 s cho 120 ảnh / 7.140 cặp trên Intel64 Family 6 Model 154, Windows 11, Python 3.13.5, CPU-only.

## 07 · Human & decision

- **Review queue:** `out/poc/review_queue.csv` + `out/poc/review_contact_sheet.html` (ảnh A bên trái, ảnh B bên phải, có nút Export CSV). B = 20 cặp/lần chạy. Mỗi dòng là một cặp, `item_id = "img_a|img_b"` (`t05_schema.pair_id`, không phụ thuộc thứ tự). Ưu tiên: cặp có score gần ngưỡng nhất (`t05_review.make_queue`, M6); khi module M6 chưa có, pipeline tạm xếp theo score cao nhất và ghi cảnh báo `QUEUE_NOT_IMPLEMENTED`.
- **Cách override + audit log:** reviewer điền `reviewer` (bắt buộc), `reviewer_decision`, `review_note`, lưu thành `out/poc/review_queue_reviewed.csv` rồi chạy lại pipeline. `approve` → `pred_dup = 1`; `reject` → `pred_dup = 0`; `edit` → `pred_dup = override_value` (0/1); `escalate`/`skip` giữ nguyên. Mỗi thay đổi ghi một dòng vào `out/poc/audit_log.jsonl` (ai, lúc nào, before/after, lý do). Bản trước review giữ ở `out/poc/suspect_pairs_raw.csv` để đối chiếu.
- **Decision policy + `reason_code` + owner** (`src/pipeline.py::decide`, không đọc GT; owner = Đặng Trung Hiếu (trưởng nhóm) từ `configs/lab.yaml`):

| Thứ tự | Điều kiện | Status | `reason_code` |
|---|---|---|---|
| 1 | Module hash / grouping / split còn trống | REWORK | `DETECTOR_OR_SPLIT_INCOMPLETE` |
| 2 | Còn cặp `pred_dup = 1` cắt ngang train/test | REWORK | `RESIDUAL_LEAK_AFTER_SPLIT` |
| 3 | Tập test rỗng | REWORK | `EMPTY_TEST_SPLIT` |
| 4 | Có cặp nghi trùng nhưng chưa review | HOLD | `PENDING_HUMAN_REVIEW` |
| 5 | Reviewer reject > 30 % | HOLD | `HIGH_FALSE_MERGE_IN_REVIEW` |
| 6 | Tỉ lệ test lệch 0.3 quá ±0.1 | HOLD | `SPLIT_RATIO_OFF_TARGET` |
| 7 | Còn lại | GO | `SPLIT_LEAK_FREE` |

- **Quyết định hiện tại (commit `38fe316`):** **HOLD · `PENDING_HUMAN_REVIEW`** — 236 cặp nghi trùng, 0 cặp cắt ngang split, test 0.325, giữ 100 % ảnh; chỉ còn chờ review. Baseline: HOLD · `BASELINE_ONLY` (chỉ tham chiếu, không phải khuyến nghị release). 【M1 cập nhật sau khi M6 review】
- **Khi nào HOLD/REWORK thay vì tự động release:** không bao giờ GO nếu chưa có người review hàng đợi; REWORK khi lỗi nằm ở code/split (phải sửa rồi chạy lại); HOLD khi lỗi là rủi ro cần người quyết (review chưa xong, reviewer thấy nhiều gộp nhầm, tỉ lệ split lệch).

## 08 · Reproduce & tests

```bash
pip install -r requirements.txt
python -m tools.verify_manifests                                   # 14 manifests, 0 failing
python -m src.main --config configs/lab.yaml --offline --seed 42   # PoC -> out/poc/
python -m src.main --config configs/lab.yaml --output out/baseline --run-name baseline --pipeline src.t05_baseline --offline --seed 42
python -m src.evaluate --track t05 --pred out/poc/suspect_pairs.csv --run-name poc --out reports/metrics_poc.json   # cần GT của giảng viên
python -m pytest -q
```

- **Output của `pytest -q`:** `69 passed, 39 skipped` (108 test; 39 skip là test của các track khác hoặc cần GT giảng viên). Test T05: `tests/test_t05_hash.py` (M3), `tests/test_t05_features.py` (M4), `tests/test_t05_grouping.py` (M5), `tests/test_t05_m2.py` (M2, có regression CP1252/UTF-8).
- **Máy / CPU / OS / Python:** Intel64 Family 6 Model 154 · Windows 11 (10.0.26300) · Python 3.13.5 · git sha: `38fe316` · config hash `c4c8b781a604`.
- **Sự cố tái lập đã sửa:** clone trên Windows (`core.autocrlf=true`) đổi LF→CRLF trong `fixtures/*.csv` → sha256 lệch manifest → pipeline dừng. Đã thêm `.gitattributes` (`fixtures/** -text`, `data/** -text`), kiểm lại trên bản checkout mới: 14 manifests OK, pipeline `status=0`.

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: 【cả nhóm bổ sung】
- **Điều chưa biết:** gold dev chỉ 47 cặp và **đã dùng để chọn ngưỡng** → F1 0.986 là điểm dev, có thể lạc quan; fixture không có heldout riêng cho sinh viên. Slice `transformed` và `burst` mỗi loại chỉ có 1 cặp trong gold → gần như chưa kiểm được. Nếu scale 10× (1.200 ảnh ≈ 720.000 cặp) vẫn so toàn bộ được nhưng chậm hơn ~100×; khi đó cần LSH/ANN. 【cả nhóm bổ sung】
- Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động; giả định cần DPO/legal xác nhận: xem mục 01.
- Kế hoạch cải tiến: 【cả nhóm bổ sung】

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Member 1 — Đặng Trung Hiếu | HieuGit05 / dangtrunghieu29.12.05@gmail.com | Integration / CLI: `src/pipeline.py`, `configs/lab.yaml`, `reports/metric_contract.md`, `.gitattributes`, tích hợp + review các nhánh | `2e35394`, `62bb36c`, `94aec32`, `d3bd028` (`1597c09` = import code gốc của lab) | 【】 |
| Member 2 — Khải | kasdie / lqkhai2512@gmail.com | Data + GT: `src/t05_baseline.py`, `data/gold/gold_pairs.csv` (47 cặp), `tests/test_t05_m2.py` | `027fb16`, `c1b537a`, `bcfea2a` | 【】 |
| Member 3 — Hoang Quang Truong | Hoang Quang Truong / truonghoangg25@gmail.com | Core algorithm (Detector A): `src/t05_hash.py`, `tests/test_t05_hash.py` | `4bdf355` | 【】 |
| Member 4 | dqhuy11 / duongquochuy1107@gmail.com | Metric + ablation (Detector B): `src/t05_features.py`, `tests/test_t05_features.py`, `reports/M4_ablation.csv`, ngưỡng trong `configs/lab.yaml` | `bac0120` (merge/revert PR: `1907ad2`, `010eabf`, `d84c9fc`, `47f6d07`) | 【】 |
| Member 5 | longnguyen3204 / longnguyen2403@gmail.com | Grouping + split: `src/t05_grouping.py`, `tests/test_t05_grouping.py` | `a32c85e` | 【】 |
| Member 6 | 【M6 bổ sung】 | Review queue + failure analysis: `src/t05_review.py`, `demo/` | 【】 | 【】 |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| 【M3/M4 bổ sung】 | | |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log.
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.

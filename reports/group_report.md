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
| Thành viên (6) + vai trò | 1. Đặng Trung Hiếu — HieuGit05 (Lead / Integrator) · 2. Khải — kasdie (Baseline + gold set) · 3. Hoang Quang Truong (Detector A — hash) · 4. dqhuy11 (Detector B — feature + score + ablation) · 5. longnguyen3204 (Grouping + leak-free split) · 6. Cheese1501 (Human review + failure analysis) |
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
| dev (gold tự gán) | 47 cặp ảnh (34 trùng / 13 không trùng) | `data/gold/gold_pairs.csv`, gán tay bởi Khải (M2); sha256 `edd4b669b42b829b04cca127aa2326fc7036fe95a16dd600cd95f0dffd26c86d` | Ảnh: CC0-1.0 theo manifest; nhãn gold do M2 gán cho lab, chưa có tuyên bố giấy phép riêng | Nhãn do người gán; ảnh thuộc fixture |
| eval / heldout | 120 ảnh = 7.140 cặp, GT ẩn `gt/gt_groups.csv` | `fixtures/t05_near_duplicate` — GT do giảng viên giữ, không có trong repo sinh viên | CC0-1.0 cho input synthetic theo manifest; GT ẩn không được phát hành trong repo | Có — ảnh tổng hợp của fixture, không dùng `data/real/` |

- **Schema & class mapping:** 4 bảng chung trong `src/t05_schema.py` — `hashes` (`image_id, sha256, phash, dhash, ahash`), `pairs` (`img_a < img_b`, `d_phash, d_dhash, d_ahash, d_hist, dt_seconds, same_camera, score, pred_dup`), `groups` (`image_id, group_id`), `split_manifest` (`image_id, split ∈ {train, test, drop}`). Gold: `img_a, img_b, is_dup (0/1), note`.
- **Seed: 42 · Cách chia split:** theo **nhóm** near-dup (union-find trên các cặp `pred_dup = 1`), không theo ảnh; cả nhóm vào một phía; mục tiêu 30 % test (`src/t05_grouping.py::make_split`, rng seed 42).
- **`manifest.csv` verified (sha256): có** — `python -m tools.verify_manifests` → 14 manifests, 0 failing (kiểm cả trên bản clone mới với `core.autocrlf=true`, sau khi thêm `.gitattributes` để Git không đổi LF→CRLF trong `fixtures/` và `data/`). Pipeline chạy với `strict_provenance: true`.
- **Gold set:** loại case — exact 16 · transformed 1 · burst 1 · color_trap 10 · other 19. Cặp `im_090/im_094` người gán không chắc nên loại khỏi file cuối. **Cách gán (M2):** Khải xem trực tiếp hai ảnh cạnh nhau trên trang HTML offline, chọn `is_dup`, loại case và ghi lý do quan sát kèm tên reviewer trong `note`. Nhãn `1` là cùng nội dung nguồn/near-duplicate dù crop, đổi sáng hoặc nén; nhãn `0` là khác nội dung, kể cả khi có màu nền giống nhau. Cùng camera hoặc gần thời gian chỉ gợi ý burst, không tự động là trùng. Ban đầu có 48 ứng viên; một cặp chưa chắc được để trống rồi bỏ khi xuất CSV. Các ứng viên được tìm bằng hash/histogram và thời gian/camera để người xem chọn, không dùng score làm nhãn và không suy nhãn từ tên file hay GT. File cuối được kiểm tra đúng schema, `img_a < img_b`, không lặp cặp, nhãn 0/1 và ghi chú đầy đủ; các nhãn/ghi chú người gán được giữ nguyên. License ảnh được chép từ cột `license` của `fixtures/t05_near_duplicate/manifest.csv` (`CC0-1.0 (synthetic, generated for AI20K Day06)`); nguồn ghi trong manifest là `synthetic:tools/make_fixtures.py (seed 42)`. Nhãn gold là dữ liệu người gán cho bài lab, không mặc định có cùng giấy phép với ảnh.
- **Dữ liệu thật dùng (nếu có):** không dùng `data/real/`.

## 03 · Research & design

- 2–3 references (URL + ngày xem) và **vì sao** chọn kỹ thuật:
  - **Union-Find (Disjoint-Set Forest)**: Cormen et al. (2009), *Introduction to Algorithms*. Khai thác tính đóng bắc cầu (transitive closure) của quan hệ tương đương: nếu $(A, B)$ và $(B, C)$ là near-dup thì $\{A, B, C\}$ bắt buộc phải vào chung một nhóm để triệt tiêu hoàn toàn nguy cơ rò rỉ chéo khi phân chia train/test.
  - **Complete-Linkage Clustering**: Defays (1977); SciPy documentation (`scipy.cluster.hierarchy.linkage`, method='complete') — https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html (xem 2026-11-14). Đo khoảng cách xa nhất giữa các cụm $D(u, v) = \max \text{dist}(u, v) \le 0.5$. Chọn làm phương pháp đối chứng nhằm khống chế đường kính cụm, kiểm chứng và ngăn ngừa hiện tượng "gộp dây chuyền" (chain merging).
  - **Detector A — perceptual hashing (M3):** sử dụng đồng thời ba hash 64-bit để phát hiện ảnh trùng và gần trùng. **pHash** áp dụng Discrete Cosine Transform (DCT), giữ các thành phần tần số thấp để mô tả cấu trúc tổng thể; chọn làm tín hiệu chính vì có khả năng chịu được nhiều biến đổi ảnh thông thường như nén và thay đổi cường độ sáng tốt hơn các hash đơn giản. **dHash** mã hoá quan hệ sáng/tối giữa các pixel lân cận theo chiều ngang, biểu diễn gradient và bổ sung thông tin cấu trúc cạnh. **aHash** so sánh từng pixel với độ sáng trung bình, rất nhanh nên được dùng làm tín hiệu đối chứng, dù độ bền với biến đổi ảnh thấp hơn. Khoảng cách Hamming đếm số bit khác nhau giữa hai hash; khoảng cách càng nhỏ thì hai ảnh càng có khả năng tương tự. Các hash toàn ảnh này không chuyên xử lý crop mạnh vì việc cắt ảnh có thể làm thay đổi cấu trúc tổng thể. Nguồn: Ofcom (2022), https://www.ofcom.org.uk/__data/assets/pdf_file/0036/247977/Perceptual-hashing-technology.pdf; Daft image_hash documentation, https://docs.daft.ai/en/stable/api/functions/image_hash/ (xem 2026-11-15).

  - **Detector B — histogram màu (M4):** khoảng cách **Bhattacharyya (dạng Hellinger)** `d = sqrt(1 − Σ sqrt(h_a·h_b))` trên `image_ops.color_histogram` (3 kênh × 16 bin, chuẩn hoá L1). Chọn Bhattacharyya thay vì L1 vì giá trị nằm sẵn trong [0, 1] nên ghép thẳng được vào score, và ít nhạy hơn với bin có tần suất nhỏ (xem [11]). Thời gian (`dt_seconds`) và camera (`same_camera`) chỉ lưu làm feature cho review/grouping, **không đưa vào score**: trong gold, cặp trùng có `dt` từ 0,2 s đến 5 ngày, còn bẫy màu có cặp chỉ cách 39 phút → thời gian không tách được hai nhóm.
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
- **Trade-offs đã cân nhắc:**
  - *Không gian tìm kiếm*: 120 ảnh → so toàn bộ 7.140 cặp, không cần LSH (Locality Sensitive Hashing) hay ANN.
  - *Trọng số đặc trưng*: pHash bền với đổi sáng/nén nhưng có thể yếu với crop mạnh; histogram màu bắt được biến đổi hình học nhẹ nhưng dính bẫy màu → chỉ cho trọng số 0.2 (kết hợp `0.8·sim_phash + 0.2·sim_hist`).
  - *Gom nhóm & chia split (Union-Find vs Complete-Linkage — M5)*:
    - **Union-Find (PoC chọn)**: Đảm bảo nguyên tắc bảo toàn nhóm tuyệt đối — toàn bộ thành phần liên thông đi về cùng một phía của split. Kết quả thực nghiệm: **0 cặp nghi trùng cắt ngang train/test (`residual_leak_pairs = 0`)**, giữ 100% dữ liệu. Đánh đổi: Dễ bị gộp dây chuyền (chain merging) nếu có cặp dương tính giả làm cầu nối (trên 120 ảnh gom thành 31 nhóm, nhóm lớn nhất có 11 ảnh).
    - **Complete-Linkage**: Ép mọi cặp trong cụm phải có khoảng cách $\le 0.5$ (gần như clique), giúp cụm rất tinh khiết (gom thành 47 nhóm, nhóm lớn nhất chỉ 5 ảnh). Đánh đổi: Bẻ gãy các chuỗi liên kết yếu, dẫn đến **30 cặp nghi trùng bị cắt đôi sang hai phía train/test (`residual_leak_pairs = 30`)**, vi phạm yêu cầu zero-leakage trừ phi chấp nhận drop các cụm biên. Vì mục tiêu hàng đầu của T05 là chống rò rỉ dữ liệu, nhóm chọn Union-Find cho pipeline chính thức.
  - **M4 — luật đơn giản thay vì logistic regression:** gold chỉ 47 cặp (13 âm) → fit 2–3 hệ số dễ overfit; luật tuyến tính 1 trọng số + 1 ngưỡng đủ đạt F1 0.986 trên gold và dễ giải thích khi review. **Rủi ro chấp nhận:** cặp `im_018/im_022` (burst, khác thời điểm, `d_phash = 2`, `d_hist = 0.013`) không feature nào tách được → 1 FP cố định, chuyển cho human review. Bug đã sửa: `captured_at` lẫn 2 định dạng (`08:00:00` và `08:00:00.200000`) làm `dt_seconds` bị NaN → parse bằng `format="ISO8601"`, hiện 7.140/7.140 cặp có `dt_seconds`.

## 04 · Baseline vs PoC

- **Baseline (M2, `src/t05_baseline.py`, commit `027fb16` / `bcfea2a`):**
  - A — random split theo từng ảnh (seed 42), không tìm cặp: 36 test / 84 train, `suspect_pairs.csv` chỉ có header.
  - B — exact sha256 (lệnh `--run-name baseline` mặc định): 100 nhóm, **20 cặp trùng y hệt**, 36 test / 84 train (`test_fraction_actual = 0.3`).
- **PoC (`src/pipeline.py` + M3/M4/M5/M6, đo ở commit `38fe316`, chạy lại sau review ở `afd1070`):** score = 0.8 · pHash + 0.2 · histogram, ngưỡng 0.757 (chốt trên gold dev), split theo nhóm union-find **sau** human review. Kết quả trước review: **236 / 7.140 cặp `pred_dup = 1`**, 81 train / 39 test (test 0.325), giữ lại 100 % ảnh, **0 cặp nghi trùng cắt ngang train/test**. Sau review 20 cặp (3 cặp bỏ gộp, 3 cặp thêm): vẫn 236 cặp `pred_dup = 1`, test 0.325, 0 cặp cắt ngang split.
- **Cùng budget / cùng heldout / cùng seed:** xác nhận — cùng 120 ảnh, seed 42, `test_fraction = 0.3`, budget B = 20, cùng lệnh evaluate.

**Bảng 1 · Baseline vs PoC & hard slices**

> ⚠️ Repo sinh viên **không có GT** (`src.evaluate` báo *"Ground truth is not distributed in the student repo; the instructor runs official evaluation"*), nên nhóm **không tự sinh được** `metrics_baseline.json`, `metrics_poc.json`, `table1_baseline_vs_poc.md`. Bảng dưới để trống ô chính thức chờ giảng viên chấm; cột "gold dev" là đánh giá nội bộ trên 47 cặp đã dùng để chọn ngưỡng — **là điểm dev, không phải holdout**, không được coi là số chính thức.

| Metric (unit) | Baseline (exact) | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Leakage Detection F1 (pair) — chính thức | 【GV chấm】 | 【GV chấm】 | higher is better |
| Primary trên gold dev (47 cặp, nội bộ) | 0.522 (TP 12 / FP 0 / FN 22; P 1.000, R 0.353) | 0.986 (TP 34 / FP 1 / FN 0; P 0.971, R 1.000) | điểm dev, ngưỡng đã chọn trên chính tập này |
| Hard-slice: color_trap trên gold dev | Acc 100% (TN 10 / FP 0) | Acc 100% (TN 10 / FP 0) | N = 10 cặp; cả 10 cặp đều là negative |
| Review@B / miss / FP | — | B = 20: 4 approve / 16 reject / 0 escalate; reject rate = 80%; post-review miss/FP chưa đo | Decision `HOLD` (`HIGH_FALSE_MERGE_IN_REVIEW`); không suy diễn metric |
| Runtime / 100 items | 0.0241 s / 120 ảnh (≈ 0.0201 s / 100 ảnh); máy M2 macOS ARM64, xem ghi chú bên dưới | 1.85 s / 120 ảnh (≈ 1.5 s / 100 ảnh) | CPU Intel64 Family 6 Model 154, Windows 11, Python 3.13.5 |

**Runtime bổ sung của M2 (đo ngày 15/11/2026, code `d168cec`):** lấy `runtime_seconds` và `stats.n_items` từ `out/baseline/run_info.json` (exact) và `out/baseline_random/run_info.json` (random); giữ nguyên bản ghi và kết quả quy đổi trong [`M2_baseline_runtime.json`](M2_baseline_runtime.json). Baseline B exact: 0.0241 s / 120 ảnh, quy đổi `runtime_seconds × 100 / n_items` = 0.020083 s / 100 ảnh. Baseline A random: 0.0578 s / 120 ảnh, quy đổi 0.048167 s / 100 ảnh. Cả hai là một lần chạy CPU-only, seed 42, trên macOS 26.5.2 ARM64, Python 3.13.18. Timer của `src.main` đo vùng import/gọi pipeline, không gồm khởi động interpreter và các import ban đầu của runner; chưa phải phân phối benchmark nhiều lần. Cột PoC vẫn là số đo Windows của M1, vì vậy không suy ra speedup trực tiếp giữa hai cột khác máy.

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

  `hist_only` có 13 FP — đúng bằng số cặp không trùng trong gold, minh hoạ bẫy màu.

  **Vì sao chọn w = 0.8 thay vì `phash_only` khi F1 bằng nhau (M4):** trên gold, `phash_only` và mọi `combined` w = 0.6–0.9 hoà nhau (F1 0.9855, cùng precision) → quy tắc "hoà thì precision cao hơn" trong metric contract không phân xử được; đây là **lựa chọn thiết kế, không phải do gold quyết định**. Lý do:
  1. **Trên toàn bộ 7.140 cặp hai phương án khác nhau:** `phash_only` gắn 260 cặp `pred_dup`, `combined_w0.8` gắn 236. Chênh lệch là 26 cặp chỉ pHash bắt (đều có `d_phash` 10–16, `d_hist` 0.24–0.87, 25/26 cách nhau ≥ 1 ngày, 16/26 khác camera — ví dụ `im_009` với `im_114/115/116/118`) và 2 cặp chỉ combined bắt. Histogram đóng vai trò **tín hiệu thứ hai chặn các cặp cấu trúc hơi giống nhưng màu rất khác**; với union-find (M5), mỗi cặp sai có thể kéo cả hai nhóm dính vào nhau, nên bớt cặp sai ở vùng giữa quan trọng hơn ở bước so cặp đơn lẻ.
  2. **Không dùng histogram nhiều hơn:** w = 0.5 đã mất 3 cặp trùng thật (cặp đổi sáng có `d_hist` 0.55–0.60, như `im_000/im_003`, `im_018/im_021`); w = 0.6 chỉ còn khe 0.052 giữa cặp trùng thấp nhất và bẫy màu cao nhất.
  3. **0.8 là giá trị mặc định đặt trước khi chạy gold**, sau đó gold xác nhận nằm trong vùng hoà 0.6–0.9 nên giữ nguyên, không dò tiếp.

  **Điều phải nói thật:** trên gold, `phash_only` có khe rộng hơn (cặp trùng thấp nhất 0.938 vs bẫy màu cao nhất 0.562, khe 0.375) so với `combined_w0.8` (0.879 vs 0.635, khe 0.245) — tức là histogram **không** giúp tách bẫy màu trong gold, chỉ là vẫn đủ an toàn. Gold không có cặp nào ở vùng `d_phash` 5–27, nên lợi ích ở mục 1 **chưa kiểm chứng được** bằng nhãn; cần GT chính thức hoặc gán thêm gold ở vùng này.
- **Độ bất định (N nhỏ):** gold dev chỉ 34 cặp dương → Wilson CI 95 % cho recall 34/34 ≈ [0.90, 1.00], precision 34/35 ≈ [0.85, 0.99] (`src/metrics/estimation.py::wilson_ci`). ✅ M4 đã chạy lại `wilson_ci(34, 34)` = [0.898, 1.000] và `wilson_ci(34, 35)` = [0.855, 0.995] — khớp số trên.
- **Lệnh evaluate chính xác (giảng viên chạy):** `python -m src.evaluate --track t05 --pred out/poc/suspect_pairs.csv --run-name poc --out reports/metrics_poc.json`

## 06 · Result slices

| Slice | N | Baseline (exact) | PoC | Ghi chú |
|---|---|---|---|---|
| **exact** | 16 | Acc: 75.0% (TP 12 / FN 4 / FP 0), F1: 0.857 | **Acc: 100.0% (TP 16 / FN 0 / FP 0), F1: 1.000** | Baseline bỏ sót 4 cặp do nén/re-encode làm đổi byte SHA-256; PoC bắt trọn nhờ pHash |
| **transformed** | 1 | Acc: 0.0% (TP 0 / FN 1 / FP 0), F1: 0.000 | **Acc: 100.0% (TP 1 / FN 0 / FP 0), F1: 1.000** | Cặp `im_105/im_106` đổi sáng; SHA-256 trượt hoàn toàn, PoC nhận diện chính xác |
| **burst** | 1 | Acc: 100.0% (TN 1 / FP 0) | **Acc: 0.0% (TN 0 / FP 1), F1: 0.000** | Cặp `im_018/im_022` (cách 0.2s, vật thể di chuyển vi mô); PoC dự đoán nhầm (1 FP duy nhất) |
| **color_trap** | 10 | Acc: 100.0% (TN 10 / FP 0) | **Acc: 100.0% (TN 10 / FP 0)** | Khác scene cùng màu; PoC trọng số pHash 0.8 lấn át histogram 0.2 nên tránh 100% bẫy màu |
| **other** | 19 | Acc: 10.5% (TP 0 / FN 17 / TN 2), F1: 0.000 | **Acc: 100.0% (TP 17 / FN 0 / TN 2), F1: 1.000** | 17 cặp near-dup đổi sáng và 2 cặp âm khác scene; PoC đạt độ chính xác tuyệt đối |
| **Toàn bộ gold dev** | **47** | **P: 1.000, R: 0.353, F1: 0.522** (TP 12 / FN 22) | **P: 0.971, R: 1.000, F1: 0.986** (TP 34 / FP 1) | Đánh giá nội bộ trên 47 cặp dev do M2 gán nhãn |

- **Failure case 1 — bẫy màu và gộp nhầm nếu chỉ dùng histogram (M6):**
  - Cặp `im_028` / `im_101` là hard negative do M2 gán nhãn (`is_dup=0`, `type=color_trap`): [im_028](../fixtures/t05_near_duplicate/images/im_028.jpg) và [im_101](../fixtures/t05_near_duplicate/images/im_101.jpg). Người gán ghi nhận hai scene khác nhau nhưng cùng màu mặt đường và dải cỏ hai bên. Cả hai thuộc `cam_0`; thời điểm lần lượt là `2026-10-01T08:52:00.200000` và `2026-10-01T11:28:40`.
  - **Ablation hist-only sẽ gộp nhầm:** `d_hist = 0.062916`, nên `sim_hist = 1 - d_hist = 0.937084`, cao hơn threshold hist-only `0.3919` trong `reports/M4_ablation.csv`; nếu đưa quyết định này vào Union-Find, hai scene sẽ thành cùng group. Root cause là histogram chỉ so phân bố màu toàn ảnh, bỏ qua vị trí và bố cục đối tượng.
  - Đây là lỗi của biến thể hist-only, **không phải lỗi đã quan sát ở PoC kết hợp**: với `w_phash=0.8`, `w_hist=0.2`, cặp có `score=0.612417 < 0.757`, `pred_dup=0`; baseline exact và PoC đều có `TN=10/10` trên slice color-trap của gold dev. Vì vậy không báo cáo đây là false merge của bản PoC cuối.
- Failure case 2 (M5 — phân tích 1 FP trên gold dev và nguy cơ rò rỉ khi split):
  - **1 FP duy nhất trên gold dev (`im_018` và `im_022`)**:
    - *Metadata & Đặc trưng*: Cùng camera `cam_3`, chụp cách nhau 0.2 giây (lúc 08:39:00 và 08:39:00.200). Đặc trưng trích xuất: `d_phash = 2`, `d_hist = 0.0135`, `same_camera = 1`, `score = 0.9723 > 0.757` → PoC phân loại `pred_dup = 1`.
    - *Ground Truth & Root Cause*: Nhãn gold là `0` (không trùng) vì người gán phát hiện xe màu xanh lá phía sau chạy nhanh hơn và đè lên vạch sơn trắng trên mặt đường. Do thời gian chụp cực ngắn (burst), toàn bộ bối cảnh không gian, góc camera và >95% diện tích ảnh giống hệt nhau; pHash (ảnh thu nhỏ $32 \times 32$ mức xám) cùng histogram màu không đủ độ phân giải không gian cục bộ để nhận diện sự dịch chuyển của một vật thể nhỏ ở hậu cảnh. Đây là giới hạn tự nhiên của perceptual hash cấp độ toàn khung hình (global descriptor).
  - **Nguy cơ rò rỉ khi phân chia tập dữ liệu (Split Leakage)**:
    - *Nếu chia split ngẫu nhiên theo ảnh độc lập (Baseline A)*: Các frame thuộc cùng một scene/burst bị phân tán ngẫu nhiên vào cả train và test, gây rò rỉ dữ liệu downstream nặng nề (mô hình học thuộc lòng góc máy và bối cảnh).
    - *Nếu chia theo Complete-Linkage*: Thuật toán khống chế đường kính cụm chặt chẽ (tạo ra 47 nhóm, max 5 ảnh/nhóm), nhưng bẻ gãy các chuỗi near-dup gián tiếp, dẫn đến **30 cặp nghi trùng bị cắt ngang giữa train và test** (`residual_leak = 30`).
    - *Giải pháp của PoC (M5)*: Áp dụng **Union-Find Grouping** kết hợp **Seeded Group-Aware Split** (`src/t05_grouping.py`). Đưa toàn bộ thành phần liên thông vào một phía của split, đồng thời bảo đảm phân bổ đủ 4 camera cho tập test (~32.5% test) với **0 cặp rò rỉ chéo (`residual_leak = 0`)** và giữ trọn 100% dữ liệu (`kept_share = 1.0`).
- **Failure case 3 — vùng quanh ngưỡng, phát hiện qua human review (M1 tổng hợp từ review của M6, `demo/review_queue_reviewed.csv`):**
  - Hàng đợi B = 20 là các cặp có `|score − 0.757|` nhỏ nhất (0.00001–0.0065): **4 cặp trên ngưỡng** (`pred_dup = 1`) và **16 cặp dưới ngưỡng** (`pred_dup = 0`). Reviewer đồng ý với PoC ở **14/20 cặp (70 %)** và sửa 6 lỗi:

    | PoC dự đoán | Reviewer approve | Reviewer reject | Ý nghĩa |
    |---|---|---|---|
    | Trùng (4) | 1 | **3** | 3/4 cặp trên ngưỡng là **false merge** |
    | Không trùng (16) | **3** | 13 | 3/16 cặp dưới ngưỡng là **miss** |

  - **3 false merge** (reviewer reject, PoC đã gộp): `im_010/im_114` (score 0.757871, khác camera, "vị trí xe/người và góc cảnh khác rõ"), `im_010/im_115` (0.757871, khác camera, cách hơn một ngày), `im_020/im_023` (0.761253, cùng camera nhưng cách ba ngày, "bố cục xe/người khác rõ").
  - **3 miss** (reviewer approve, PoC bỏ sót): `im_080/im_081` (0.75675, "cùng bố cục đường hoàng hôn, chỉ khác nhẹ màu/vị trí"), `im_008/im_009` (0.753537, "biến đổi màu/ánh sáng nhẹ"), `im_104/im_105` (0.752443, "cùng bố cục cảnh đêm").
  - **Root cause:** score chỉ dựa trên descriptor toàn ảnh (pHash 0.8 + histogram 0.2). Cảnh khác nhau nhưng cùng bố cục đường/bầu trời và cùng tông màu cho score ≈ 0.76, ngang với bản cùng cảnh bị đổi sáng/màu → hai nhóm **chồng lên nhau** ở quanh ngưỡng, không ngưỡng đơn nào tách được. `im_010` xuất hiện trong 2 false merge: nếu không có review, union-find sẽ dùng nó làm "cầu" nối `im_114` và `im_115` vào cùng một nhóm. Ngưỡng 0.757 được chọn trên gold dev 47 cặp — tập này có khe score rộng (0.635..0.879) nên không chứa cặp nào ở vùng khó này.
  - **Tác động:** review đã sửa cả 6 lỗi trước khi split (pipeline gom nhóm **sau** review), kết quả vẫn **0 cặp cắt ngang train/test**. Nhưng tỉ lệ lỗi cao ở vùng ngưỡng cho thấy F1 0.986 trên gold dev lạc quan so với thực tế — đây là lý do quyết định cuối là HOLD (mục 07).
- **Latency / compute:** PoC 1.85 s cho 120 ảnh / 7.140 cặp trên Intel64 Family 6 Model 154, Windows 11, Python 3.13.5, CPU-only.

## 07 · Human & decision

- **Bằng chứng review đã nộp:** `reports/review_queue.csv` (= `demo/review_queue_reviewed.csv`, 20 cặp, reviewer Cheese1501, mỗi cặp có `review_note`), `demo/audit_log.jsonl` (20 dòng `review:approve|reject`, before → after), `demo/decision.json`, `reports/decision_log.csv` (log append-only các lần chạy trên máy M1, từ REWORK khi module còn trống → HOLD chờ review → HOLD sau review). Chạy lại pipeline với file review này cho đúng `demo/decision.json` (M1 kiểm tra lại).
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

- **Quyết định cuối (sau review của M6, commit `afd1070`):** **HOLD · `HIGH_FALSE_MERGE_IN_REVIEW`** — owner Đặng Trung Hiếu (trưởng nhóm). Sau review: 20/20 cặp đã review (4 approve / 16 reject / 0 escalate), 236 cặp `pred_dup = 1`, **0 cặp cắt ngang train/test**, test 0.325, giữ 100 % ảnh. Luật 1–4 đều qua; dừng ở luật 5 vì `review_reject_rate = 0.80 > 0.30`. Baseline: HOLD · `BASELINE_ONLY` (chỉ tham chiếu, không phải khuyến nghị release).
- **Nhận xét trung thực về luật 5:** luật đang tính tỉ lệ reject trên **toàn bộ** cặp đã review. Vì hàng đợi lấy cả hai phía ngưỡng, 13/16 lần reject chỉ là **xác nhận** cặp PoC đã đoán không trùng (`0 → 0`), không phải false merge. Đo đúng nghĩa "false merge" (chỉ trên các cặp PoC đoán trùng) là **3/4 = 75 %**, vẫn > 30 % → **cùng kết luận HOLD**. Nhóm **không sửa luật hay ngưỡng** sau khi thấy kết quả, vì cả hai đã chốt trong `reports/metric_contract.md` (2026-11-14 15:30); việc tách chỉ số thành "false-merge rate" và "miss rate" trong review được ghi vào kế hoạch cải tiến (mục 09).
- **Ý nghĩa của HOLD:** split hiện tại **không leak** theo dự đoán đã review, nhưng reviewer cho thấy detector sai nhiều ở vùng quanh ngưỡng (6/20 cặp), nên chưa tin được chất lượng phần lớn 7.140 cặp không được người xem. Muốn GO cần: chọn lại ngưỡng/feature trên một gold mới có cặp ở vùng khó, chốt lại **trước** khi chạy, rồi review một hàng đợi mới.
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

- **Output của `pytest -q`:** `73 passed, 39 skipped` (112 test; 39 skip là test của các track khác hoặc cần GT giảng viên). Test T05: `tests/test_t05_hash.py` (M3), `tests/test_t05_features.py` (M4), `tests/test_t05_grouping.py` (M5), `tests/test_t05_m2.py` (M2, có regression CP1252/UTF-8), `tests/test_t05_review.py` (M6).
- **Máy / CPU / OS / Python:** Intel64 Family 6 Model 154 · Windows 11 (10.0.26300) · Python 3.13.5 · git sha: `afd1070` (kèm bằng chứng review) · config hash `c4c8b781a604`.
- **Kiểm tra clone sạch (M1, commit `95028d5`):** `git clone` mới từ GitHub trên Windows (`core.autocrlf=true`) → `verify_manifests` 14/14 OK, PoC `status=0` (7.140 cặp, 0 cảnh báo, `split_manifest.csv` đủ 120 ảnh), baseline `status=0`, `pytest` 73 passed / 39 skipped; repo không chứa GT.
- **Sự cố tái lập đã sửa:** clone trên Windows (`core.autocrlf=true`) đổi LF→CRLF trong `fixtures/*.csv` → sha256 lệch manifest → pipeline dừng. Đã thêm `.gitattributes` (`fixtures/** -text`, `data/** -text`), kiểm lại trên bản checkout mới: 14 manifests OK, pipeline `status=0`.

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: **Phần M2:** chỉ dùng ảnh synthetic đã phát trong fixture; README mô tả không có người/biển số thật, manifest khai báo CC0-1.0. Nhãn gold được người dùng tự gán; không suy diễn giấy phép dữ liệu thật hoặc GT ẩn từ giấy phép input. Kết quả này chưa kiểm chứng trên dữ liệu thực có PII. **Phần M1:** `fixtures/t05_near_duplicate/manifest.csv` khai báo cả 121 file (120 ảnh + `images.csv`) là `synthetic:tools/make_fixtures.py (seed 42)`, license `CC0-1.0`; pipeline chạy `--offline`, không tải dữ liệu hay model ngoài. Dữ liệu cá nhân duy nhất trong repo là **tên/handle người review** trong `review_queue_reviewed.csv`, `audit_log.jsonl` và cột `note` của gold — cần thiết cho trách nhiệm giải trình (ai quyết định gì), nhưng nếu áp dụng cho dữ liệu thật thì phải được người review đồng ý và giới hạn quyền xem log. Nếu chạy trên ảnh camera thật (người, biển số), cần thêm bước làm mờ/ẩn danh **trước** khi đưa ảnh vào contact sheet, vì contact sheet nhúng ảnh base64 vào một file HTML dễ bị chia sẻ ra ngoài.
- **Điều chưa biết:** gold dev chỉ 47 cặp và **đã dùng để chọn ngưỡng** → F1 0.986 là điểm dev, có thể lạc quan; fixture không có heldout riêng cho sinh viên. Slice `transformed` và `burst` mỗi loại chỉ có 1 cặp trong gold → gần như chưa kiểm được. Nếu scale 10× (1.200 ảnh ≈ 720.000 cặp) vẫn so toàn bộ được nhưng chậm hơn ~100×; khi đó cần LSH/ANN. **Phần M1:** (1) **Gộp dây chuyền:** union-find nối mọi cặp `pred_dup = 1`, nên chỉ một FP làm "cầu" là hai nhóm thật bị gộp (nhóm lớn nhất hiện có 11 ảnh) — không gây leak nhưng làm split lệch và có thể kéo cả cụm lớn vào test. (2) **Phụ thuộc người review:** override của reviewer được áp thẳng vào `pred_dup` rồi mới gom nhóm; reviewer sai (approve nhầm) sẽ gộp nhầm scene, và hiện chỉ có **một** người review, chưa đo độ đồng thuận. (3) **Ngưỡng nằm sát nhiều cặp:** hàng đợi B = 20 có các cặp chỉ cách ngưỡng 0.757 khoảng 0.00001–0.0065, tức kết quả nhạy với sai số nhỏ của score; 20 cặp là toàn bộ vùng "khó" mà người xem được. (4) **False alarm:** 236 / 7.140 cặp bị đánh dấu trùng nhưng gold chỉ phủ 47 cặp, nên tỉ lệ FP thật ngoài gold chưa biết — chờ điểm chính thức của giảng viên. (5) **Review xác nhận rủi ro (3):** trên 20 cặp gần ngưỡng, reviewer thấy 3/4 cặp PoC gộp là sai và 3/16 cặp PoC bỏ qua là trùng thật (failure case 3) — gold dev không có cặp nào ở vùng này nên không phát hiện được lúc chọn ngưỡng. (6) **Chỉ số review gộp hai loại lỗi:** luật 5 dùng tỉ lệ reject trên toàn hàng đợi, lẫn "xác nhận đúng" với "false merge" (mục 07); lần này không đổi kết luận, nhưng có thể đổi ở lần khác.
- **Giới hạn M2:** bộ cặp được chọn có chủ đích theo các tín hiệu giống nhau nên không phải mẫu ngẫu nhiên đại diện cho toàn bộ cặp ảnh; một người gán, chưa đo đồng thuận giữa người gán và chưa có holdout độc lập của M2. Exact SHA-256 chỉ phát hiện trùng từng byte, còn random split theo từng ảnh là đối chứng có nguy cơ leakage. Số cặp dự đoán/cắt ngang split chỉ là kiểm tra nội bộ khi chưa có GT ẩn; không thay thế F1 chính thức.
- Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động; giả định cần DPO/legal xác nhận: xem mục 01.
- Kế hoạch cải tiến: **Phần M2:** bổ sung gold gán tay cho biến đổi/crop/nén và burst, có người gán thứ hai để đối chiếu bất đồng; chốt protocol đánh giá độc lập theo nhóm ảnh/scene trước khi dùng nhãn giữ kín; đo baseline và PoC trên cùng máy/cùng phạm vi timer nếu muốn so tốc độ. Không thay nhãn, ngưỡng hoặc số liệu hiện có để cải thiện điểm sau khi xem evaluation. **Phần M1:** (1) tách gold thành **dev** (chọn ngưỡng) và **holdout** (chỉ đo một lần) để có số nội bộ không lạc quan; (2) giảm gộp dây chuyền: giữ union-find để đảm bảo zero-leak nhưng **cảnh báo/đưa vào review** các nhóm lớn bất thường (ví dụ > 6 ảnh) hoặc các cạnh có score gần ngưỡng làm "cầu" giữa hai cụm; (3) thêm feature bền với crop (ORB keypoint hoặc embedding ảnh offline có checksum) cho slice `transformed`; (4) hai người review độc lập cho các cặp trong hàng đợi, đo Cohen's kappa, cặp bất đồng chuyển `escalate`; (5) đưa kiểm tra clone sạch (`verify_manifests` + pipeline + `pytest`) vào CI để bắt sớm lỗi kiểu CRLF; (6) cho lần chốt policy sau: tách luật 5 thành **false-merge rate** (reject / cặp PoC đoán trùng) và **miss rate** (approve / cặp PoC đoán không trùng), mỗi chỉ số có ngưỡng riêng; (7) bổ sung vào gold các cặp ở vùng score 0.74–0.77 (lấy từ hàng đợi review này) rồi chọn lại ngưỡng **trước** lần chạy tiếp theo — không dùng nhãn review hiện tại để chỉnh ngưỡng rồi báo cáo trên chính các cặp đó.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Member 1 — Đặng Trung Hiếu | HieuGit05 / dangtrunghieu29.12.05@gmail.com | Integration / CLI: `src/pipeline.py`, `configs/lab.yaml`, `reports/metric_contract.md`, `.gitattributes`, `reports/group_report.md`, tích hợp + review các nhánh; sửa pipeline gom nhóm + split **sau** review | `2e35394`, `62bb36c`, `94aec32`, `d3bd028`, `d168cec`, `0886f6c` (`1597c09` = import code gốc của lab) | 【】 |
| Member 2 — Khải | kasdie / lqkhai2512@gmail.com | Data + GT: `src/t05_baseline.py`, `data/gold/gold_pairs.csv` (47 cặp), `tests/test_t05_m2.py` | `027fb16`, `c1b537a`, `bcfea2a` | 【】 |
| Member 3 — Hoang Quang Truong | Hoang Quang Truong / truonghoangg25@gmail.com | Core algorithm (Detector A): `src/t05_hash.py`, `tests/test_t05_hash.py` | `4bdf355` | 【】 |
| Member 4 | dqhuy11 / duongquochuy1107@gmail.com | Metric + ablation (Detector B): `src/t05_features.py`, `tests/test_t05_features.py`, `reports/M4_ablation.csv`, ngưỡng trong `configs/lab.yaml` | `bac0120` (merge/revert PR: `1907ad2`, `010eabf`, `d84c9fc`, `47f6d07`) | 【】 |
| Member 5 | longnguyen3204 / longnguyen2403@gmail.com | Grouping + split: `src/t05_grouping.py`, `tests/test_t05_grouping.py` | `a32c85e` | 【】 |
| Member 6 | Cheese1501 / hoangkimchi46@gmail.com | Review queue + `gold_check` + failure analysis: `src/t05_review.py`, `tests/test_t05_review.py`, `reports/group_report.md`, `reports/team_contributions.csv`; **human review 20 cặp** (`demo/`) | `2e10b20`, `5dd1d9b`, `afd1070` | 【】 |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html | 2026-11-14 | Tài liệu SciPy phân cụm phân cấp (complete linkage) để kiểm soát đường kính cụm và chặn gộp dây chuyền. |
| https://docs.voxel51.com/user_guide/brain.html#near-duplicates | 2026-11-14 | Hướng dẫn FiftyOne Brain về phát hiện near-duplicate và nguyên tắc chia train/test theo cụm/scene để tránh leakage. |
| https://www.ofcom.org.uk/__data/assets/pdf_file/0036/247977/Perceptual-hashing-technology.pdf | 2026-11-15 | Báo cáo tổng quan về perceptual hashing, giải thích DCT trong pHash, gradient trong dHash, ngưỡng trung bình trong aHash và cách dùng Hamming distance để so sánh hash. |
| https://docs.daft.ai/en/stable/api/functions/image_hash/ | 2026-11-15 | Tài liệu API mô tả các phương pháp pHash, dHash, aHash, kích thước hash và ứng dụng trong phát hiện near-duplicate; đồng thời phân biệt hash thông thường với crop-resistant hash. |
| https://en.wikipedia.org/wiki/Bhattacharyya_distance | 2026-11-15 | Định nghĩa hệ số và khoảng cách Bhattacharyya giữa hai phân phối; dạng Hellinger `sqrt(1 − BC)` nằm trong [0, 1] — cơ sở cho `hist_distance` (M4). |
| https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html | 2026-11-15 | So sánh các thước đo histogram (correlation, chi-square, intersection, Bhattacharyya); Bhattacharyya là lựa chọn phổ biến cho so khớp histogram màu (M4). |
| https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval | 2026-11-15 | Khoảng Wilson cho tỉ lệ khi N nhỏ, vẫn dùng được khi k = n (recall 34/34) — dùng cho CI ở mục 05 (M4). |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log.
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.

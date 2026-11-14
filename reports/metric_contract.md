# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T05 · Leakage & Near-Duplicate Guardian (`fixtures/t05_near_duplicate`, 120 ảnh, 4 camera) |
| Primary metric (tên đúng như evaluator in ra) | `Leakage Detection F1 (pair)` |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | **pair** — cặp ảnh không thứ tự `(img_a, img_b)`, luôn ghi `img_a < img_b` |
| Công thức + numerator / denominator | Chỉ xét các dòng `pred_dup == 1` trong `out/poc/suspect_pairs.csv`. TP = cặp dự đoán nằm cùng `dup_group_id` trong GT; FP = cặp dự đoán không cùng nhóm; FN = cặp GT cùng nhóm mà không được dự đoán. Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R) (`src/metrics/pairs.py::pair_prf`) |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Khớp chính xác theo `image_id` của cả hai ảnh, không phân biệt thứ tự; cặp `(a, a)` và cặp trùng lặp bị bỏ. Không có tolerance |
| Candidate pool cố định | Toàn bộ C(120, 2) = **7.140 cặp** do `t05_hash.hash_pairs` sinh ra; không lọc trước bằng GT. Baseline và PoC dùng cùng 120 ảnh trong `images.csv` |
| Review budget B / K | **B = 20 cặp**/lần chạy (`params.budget` trong `configs/tracks/t05.yaml`), ưu tiên cặp có `|score − threshold|` nhỏ nhất |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Ngưỡng `score` (và Hamming nếu dùng) do M4 chọn **chỉ trên gold dev tự gán nhãn tay** (`data/gold/gold_pairs.csv`, 30–50 cặp do M2 gán, gồm cả cặp trùng thật và cặp khác scene màu giống). Tiêu chí: F1 cao nhất trên gold dev; hoà thì chọn ngưỡng có precision cao hơn. Ghi vào `params.threshold` trong `configs/lab.yaml`, commit, rồi **đóng băng** trước lần evaluate đầu tiên |
| Eval split (không dùng để tune) | GT ẩn `fixtures/t05_near_duplicate/gt/gt_groups.csv` (`scene_id, dup_group_id, transform`) — chỉ `src.evaluate` đọc; pipeline bị GT guard chặn (exit 5). Lưu ý trung thực: gold dev được gán trên cùng pool 120 ảnh (fixture không có heldout riêng), nên chỉ tune ngưỡng, không tune theo số evaluator in ra |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | **Miss (FN)** là lỗi xấu nhất: bỏ sót cặp trùng ⇒ ảnh trùng có thể rơi vào cả train và test (leakage). Theo dõi kèm: `Residual cross-split leak (near-dup)` và `(scene)` — lower is better. **False alarm**: `False merge rate` (cặp dự đoán khác scene / số cặp dự đoán) — gộp nhầm làm mất dữ liệu hoặc lệch split |
| Slices sẽ báo cáo (≥2) | (1) theo `transform` (exact / biến đổi / burst…) — evaluator in `share_in_any_pred_pair` theo slice; (2) cặp **khác scene nhưng màu giống** (trap trong đề) qua `False merge rate` + failure case; (3) theo `camera_id` khi kiểm tra split (test có đủ 4 camera) |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Wilson CI 95% cho precision và recall (`src/metrics/estimation.py::wilson_ci`, dùng TP/FP/FN evaluator in ra); bootstrap theo nhóm ảnh (`src/metrics/correlation.py::bootstrap_ci`, seed 42) nếu cần CI cho F1. Báo cả N = số cặp GT |
| Ngày giờ chốt + người chốt | 2026-11-14 15:30 (giờ VN) · Đặng Trung Hiếu (trưởng nhóm) |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.

## Metric phụ (in cùng lần evaluate, không dùng để tune)

| Metric (tên evaluator) | Hướng | Ý nghĩa |
|---|---|---|
| `Pair precision` / `Pair recall` | higher | Tách F1 để thấy lỗi nằm ở FP hay FN |
| `False merge rate` | lower | Cặp dự đoán nối hai scene khác nhau |
| `Residual cross-split leak (near-dup)` | lower | Cặp GT near-dup bị chia ra hai phía train/test (cần `split_manifest.csv` cạnh `suspect_pairs.csv`) |
| `Residual cross-split leak (scene)` | lower | Cặp cùng scene bị chia ra hai phía |
| `% data kept` | higher | Ảnh không bị `drop` / tổng ảnh — chặn cách "drop hết cho hết leak" |

## So sánh công bằng Baseline vs PoC

- Baseline A (M2): chia train/test ngẫu nhiên theo từng ảnh. Baseline B: gom theo `sha256` trùng rồi chia theo nhóm.
- Cùng 120 ảnh, cùng `test_fraction = 0.3`, cùng seed 42, cùng budget B = 20, cùng lệnh evaluate.
- Báo đủ cả kết quả bằng 0 hoặc xấu hơn baseline.

## Decision policy (đóng băng trong `src/pipeline.py::decide`, không đọc GT)

Xét theo thứ tự, gặp điều kiện nào trước thì dừng:

| Thứ tự | Điều kiện | Status | `reason_code` |
|---|---|---|---|
| 1 | Còn helper chưa cài (hash / grouping / split) | REWORK | `DETECTOR_OR_SPLIT_INCOMPLETE` |
| 2 | Còn cặp `pred_dup = 1` nằm ở cả train và test | REWORK | `RESIDUAL_LEAK_AFTER_SPLIT` |
| 3 | Tập test rỗng | REWORK | `EMPTY_TEST_SPLIT` |
| 4 | Có cặp nghi trùng nhưng chưa có review | HOLD | `PENDING_HUMAN_REVIEW` |
| 5 | Reviewer reject > 30% số cặp đã review | HOLD | `HIGH_FALSE_MERGE_IN_REVIEW` |
| 6 | Tỉ lệ test lệch mục tiêu 0.3 quá ±0.1 | HOLD | `SPLIT_RATIO_OFF_TARGET` |
| 7 | Không rơi vào các điều kiện trên | GO | `SPLIT_LEAK_FREE` |

Owner ký quyết định: Đặng Trung Hiếu (trưởng nhóm) — lấy từ `owner` trong `configs/lab.yaml`.

# Baseline B0: ImmortalTracker gốc, nuScenes val (150 scene)

Kết quả tái lập khớp paper, dùng làm mốc so sánh. File `results.json` (khoảng 270 MB) không có trong repo; tạo lại bằng `repro/run_rest.sh`.

| Metric | Paper (Table 5) | B0 |
|---|---|---|
| AMOTA (%) | 70.2 | 70.5 |
| MOTA (%) | 60.1 | 60.0 |
| IDS | 385 | 383 |

Bổ sung: AMOTP 0.525 · Recall 0.745 · MOTAR 0.804 · TP 82.381 · FP 15.065 · FN 19.133 · FRAG 425 · MT 4.474 · ML 1.589 · FAF 52.4 · TID 0.38 s · LGD 0.57 s

- `results.json` (không có trong repo): kết quả tracking, định dạng submission của nuScenes, là đầu vào cho `offline/`
- `metrics_summary.json`: toàn bộ metric, có theo từng class
- `eval_log.txt`: log bộ chấm
- `detection_md5.txt`: md5 của file detection dùng (`detections/cp_val_noflip.json`)
- `ImmortalTracker_COMMIT.txt`: commit repo gốc

Thiết lập: repo gốc không sửa, config `configs/nu_configs/immortal.yaml`, nuscenes-devkit 1.1.11 (`pandas<2`), metadata `data/nuscenes_valslim`.
Kết quả với bộ detection có flip (AMOTA 70.5 / MOTA 60.4 / IDS 277) chỉ ghi trong `../RESULTS.md` để tham khảo.

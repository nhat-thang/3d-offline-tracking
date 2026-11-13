# offline/: xử lý offline trên track

Đầu vào là kết quả tracking của ImmortalTracker (`baseline_B0/results.json`, không có trong repo; tạo bằng `repro/`). Mọi lệnh chạy từ thư mục gốc.

## Chia dữ liệu

150 scene val chia xen kẽ theo tên (`split_dev_test.json`): 75 scene **dev** để thử nghiệm và chỉnh tham số, 75 scene **test** chỉ dùng cho báo cáo cuối.

## Script chính

| Script | Việc |
|---|---|
| `scripts/common.py` | Hàm dùng chung: đọc/ghi `results.json` theo track, chia dev/test |
| `scripts/baselines.py` | B1 (nội suy tuyến tính), B2 (kéo dài hai chiều kiểu CTRL) |
| `scripts/pipeline.py` | Các bước offline cộng dồn: nối track hai chiều, kéo dài có điều kiện, làm mượt track |
| `scripts/eval_split.py` | Chấm điểm nuScenes trên dev hoặc test |
| `scripts/measure_tracking.py` | Đo không lọc ngưỡng: recall, bỏ sót đầu/giữa/cuối track, đổi ID, box trùng, sai số tâm |

```bash
python offline/scripts/pipeline.py R3 dev            # tạo offline/results/R3/results_dev.json
bash offline/run_eval.sh R3 dev                      # chấm điểm nuScenes trên dev
python offline/scripts/measure_tracking.py offline/results/R3/results_dev.json dev R3
```

## Script phân tích

| Script | Việc |
|---|---|
| `scripts/analyze_b0.py` | Phân tích lỗi của B0 ở ngưỡng bộ chấm chọn |
| `scripts/diag_switch.py` | Phân loại nguyên nhân đổi ID |
| `scripts/link_precision.py` | Tỉ lệ nối track đúng, đối chiếu GT |
| `scripts/analyze_ext.py` | Tỉ lệ box kéo dài trùng vật thật |
| `scripts/check_yaw_size.py` | Sai số hướng và kích thước box so với GT |
| `scripts/lidar_check_mini.py` | Đếm điểm LiDAR trong box kéo dài (2 scene có LiDAR trong nuScenes mini) |
| `scripts/find_cases_mini.py` | Tìm thời điểm minh họa trong 2 scene mini |

## Trực quan hóa

| Script | Đầu ra |
|---|---|
| `scripts/viz_b0.py`, `scripts/viz_ctrl_cases.py` | Trang HTML nhìn từ trên xuống, có điểm LiDAR và GT |
| `scripts/render_video.py`, `scripts/render_ctrl_cases.py` | Video 6 camera + LiDAR nhìn từ trên xuống (cần nuScenes mini) |

## Khác

- `run_eval.sh`, `run_dev_eval.sh`: chấm điểm trong Docker.
- `download_val_lidar.sh`, `val_lidar_files.txt`: tải point cloud LiDAR keyframe của tập val từ bản công khai của nuScenes.
- `notes/`: ghi chú phân tích.
- Kết quả ghi vào `offline/results/` (không đưa lên repo).

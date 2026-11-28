# Waymo: tái lập CTRL và chạy phương pháp đề xuất

Kết quả và phân tích: [`offline/notes/03_waymo_tai_lap_ctrl.md`](../offline/notes/03_waymo_tai_lap_ctrl.md). Metric: [`docs/tracking_metrics.md`](../docs/tracking_metrics.md), mục 8.

Dữ liệu Waymo Open Dataset dùng theo giấy phép phi thương mại; dữ liệu, file kết quả và ảnh nằm trong `data/` và `offline/results/`, không đưa lên git. Mọi bước chạy trong Docker trên Windows qua các file `tools/waymo_*.bat` (chạy từ thư mục gốc của repo).

## Môi trường

| Image | Dockerfile | Dùng cho |
|---|---|---|
| `waymo_eval` | `Dockerfile.eval` | Đọc/ghi file `.bin`, chấm AP/APH, tải dữ liệu (gsutil, gdown) |
| `ctrl_env` | `Dockerfile.ctrl` | Code CTRL (tusen-ai/SST) và ImmortalTracker-for-CTRL: torch 1.8.1 + CUDA 11.1, mmcv-full 1.3.9, mmdet 2.14, spconv, torch-scatter, TorchEx |
| `ctrl_env2` | `Dockerfile.ctrl2` | `ctrl_env` + file proto Waymo, nvitop, matplotlib 3.3.4 |

## Thứ tự chạy

| Bước | File | Việc |
|---|---|---|
| 0 | `tools/waymo_0_tao_moi_truong.bat` | Dựng image `waymo_eval` |
| 1 | `tools/waymo_1_tai_ctrl.bat` → `tai_ctrl.sh` | Tải kết quả FSD, kết quả CTRL, checkpoint, `poses.pkl` của tác giả |
| 2 | `tools/waymo_2_dang_nhap_gcloud.bat` | Đăng nhập Google Cloud để tải Waymo (cần đăng ký Waymo Open Dataset trước) |
| 3 | `tools/waymo_3_tai_gt.bat` → `tai_gt.sh` | Tải GT `lidar_box` của 202 chuỗi validation |
| 4 | `tools/waymo_4_cham_ctrl.bat` → `eval_det.py` | Bước A: chấm kết quả của tác giả trên 202 chuỗi |
| 5 | `tools/waymo_5_tai_lidar.bat` → `tai_lidar_subset.sh` | Tải LiDAR các chuỗi trong `subset8.txt` |
| 6 | `tools/waymo_6_chuyen_diem.bat` → `v2_to_kitti_points.py` | Chuyển điểm LiDAR sang định dạng của mmdet3d |
| 7 | `tools/waymo_7_tao_moi_truong_ctrl.bat` | Dựng image `ctrl_env` |
| 8 | `tools/waymo_8_chuan_bi_tracker.bat` → `prep_immortal.py` | Chuẩn bị timestamp, pose, detection cho ImmortalTracker |
| 9 | `tools/waymo_9_bo_sung_moi_truong.bat` | Dựng image `ctrl_env2` |
| 10 | `tools/waymo_10_chay_ctrl.bat` → `run_ctrl.sh` | Bước B: tracker → kéo dài → point cropping → refine không TTA → refine Track TTA → empty-box removal (GPU) |
| 10b | `tools/waymo_10b_tta_xoa_rong.bat` | Chỉ chạy refine Track TTA và empty-box removal |
| 11 | `tools/waymo_11_cham_subset.bat` → `eval_subset.sh` | Chấm mọi kết quả trên các chuỗi trong `subset8.txt` |
| 12 | `tools/waymo_12_so_box.bat` → `compare_boxes.py` | So từng box của bản chạy lại với kết quả chính thức |
| 13 | `tools/waymo_13_ban_de_xuat.bat` → `run_ours_refine.sh`, `eval_ours.sh` | Bước C: đưa kết quả của phương pháp đề xuất qua kéo dài, point cropping, refine CTRL, empty-box removal; chấm cả hai bản ở từng giai đoạn |
| 14 | `tools/waymo_14_cham_tracking.bat` → `eval_track.sh`, `eval_track.py` | Chấm metric tracking của Waymo (MOTA, MOTP, miss, ID switch, false positive) cho CTRL và phương pháp đề xuất |

`ours.bin` dùng ở bước 13 được tạo bằng `ours_waymo.py` (cần numpy, scipy, pandas và file proto Waymo):

```
python waymo/export_boxes_npz.py --contexts waymo/subset8.txt --gt_dir data/waymo/v2/validation/lidar_box \
    --src fsd=data/waymo/ctrl_release/fsd_base_vehicle_val.bin \
    --src ctrl_tracker=data/waymo/ctrl_work/mot_results/waymo/validation/immortal_fsd/bin/pred_merged.bin \
    --src ctrl_extend=data/waymo/ctrl_work/mot_results/waymo/validation/immortal_fsd/bin/pred_merged_extend.bin \
    --src ctrl_refine=data/waymo/ctrl_work/ctrl_noTTA_wo_empty_right.bin \
    --out offline/results/waymo/subset8/npz
python waymo/ours_waymo.py --npz offline/results/waymo/subset8/npz --poses data/waymo/kitti_format/poses.pkl \
    --out data/waymo/ctrl_work/ours/ours.bin --name ours
```

## Phân tích lỗi và ảnh (chỉ cần numpy, scipy, matplotlib)

```
python waymo/analyze_ctrl_errors.py --npz offline/results/waymo/subset8/npz --out offline/results/waymo/subset8/loi_ctrl [--compare ours_refine]
python waymo/viz_ctrl_cases.py --npz offline/results/waymo/subset8/npz --cases offline/results/waymo/subset8/loi_ctrl/cac_ca.json \
    --kitti data/waymo/kitti_format --out offline/results/waymo/subset8/loi_ctrl/anh [--compare ours_refine]
```

Mở `offline/results/waymo/subset8/loi_ctrl/anh/index.html` để xem.

## Các file khác

| File | Việc |
|---|---|
| `boxgeo.py` | IoU 3D, đọc `.npz`, đọc điểm LiDAR (dùng chung) |
| `ctrl_test.py` | Gọi `tools/test.py` của SST với cấu hình sửa trong bộ nhớ (batch, Track TTA, file track đầu vào) |
| `make_ctx2ts.py` | Tạo `context2timestamp.pkl` cho bước track extension |
| `inspect_bin.py` | Xem nhanh một file `.bin` |
| `subset8.txt` | 8 chuỗi validation đầu tiên theo thứ tự tên |

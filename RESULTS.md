# Kết quả

- **Code tracking:** repo gốc ImmortalTracker, config `configs/nu_configs/immortal.yaml`, không sửa
- **Chấm điểm:** nuscenes-devkit 1.1.11
- **Detection:** CenterPoint, bản do tác giả CenterPoint công bố: `cp_val_noflip.json` (không TTA/flip) và `cp_val_flip.json` (có flip)
- **Chia dữ liệu:** 150 scene val chia xen kẽ theo tên: 75 scene **dev** (thử nghiệm, chỉnh tham số), 75 scene **test** (chỉ dùng báo cáo cuối)

## 1. Tái lập ImmortalTracker (150 scene val)

| Metric | Paper (Table 5, arXiv 2111.13672) | Không flip (**B0**) | Có flip |
|---|---|---|---|
| AMOTA (%) | 70.2 | **70.5** | 70.5 |
| MOTA (%) | 60.1 | **60.0** | 60.4 |
| IDS | 385 | **383** | 277 |

Bộ không flip khớp paper gần như hoàn toàn, nên được chọn làm mốc **B0**.

| | AMOTP (m) | Recall | MOTAR | TP | FP | FN | FRAG |
|---|---|---|---|---|---|---|---|
| Không flip (B0) | 0.525 | 0.745 | 0.804 | 82.381 | 15.065 | 19.133 | 425 |
| Có flip | 0.510 | 0.728 | 0.825 | 81.219 | 12.934 | 20.401 | 355 |

AMOTA theo class:

| | bus | car | pedestrian | motorcycle | truck | bicycle | trailer |
|---|---|---|---|---|---|---|---|
| Không flip (B0) | 0.875 | 0.830 | 0.795 | 0.684 | 0.690 | 0.564 | 0.497 |
| Có flip | 0.871 | 0.840 | 0.802 | 0.703 | 0.694 | 0.522 | 0.504 |

## 2. Các phiên bản

| Phiên bản | Gồm | Trạng thái |
|---|---|---|
| B0 | ImmortalTracker gốc | Mốc |
| B1 | B0 + nội suy tuyến tính khoảng trống | Xong |
| B2 | B0 + kéo dài hai chiều kiểu CTRL (không refine) | Xong |
| R1 | B0 + nối track hai chiều (mọi lớp) | Xong |
| R1b | B0 + nối track hai chiều (bỏ pedestrian, score track ≥ 0,15) | Xong |
| R2 | R1 + kéo dài có điều kiện | Đã thử, bỏ |
| R3 | R1 + làm mượt cả track | Xong |

## 3. Điểm nuScenes trên 75 scene dev

| Phiên bản | AMOTA | AMOTP | MOTA | Recall | IDS | FRAG | FP | FN | TID | LGD |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.730 | 0.520 | 0.637 | 0.770 | 211 | 197 | 6.842 | 9.346 | 0.318 | 0.465 |
| B1 | 0.730 | 0.512 | 0.634 | 0.762 | 188 | 188 | 6.802 | 9.450 | 0.333 | 0.487 |
| B2 | 0.630 | 0.483 | 0.520 | 0.667 | 127 | 160 | 7.357 | 13.173 | 0.281 | 0.453 |

R1, R1b, R3: đang chạy bộ chấm.

- B1 không đổi AMOTA vì bộ chấm đã tự nội suy tuyến tính.
- B2 giảm 10 điểm AMOTA: box kéo dài khi không có refine phần lớn là FP. B2 vượt giới hạn 500 box mỗi sample ở một số frame; 305 box score thấp nhất bị bỏ theo luật nuScenes.

## 4. Đo không lọc ngưỡng (75 scene dev)

Đo trên toàn bộ track, không lọc score, không nội suy (`offline/scripts/measure_tracking.py`). GT dev: 3.224 vật, 47.815 box.

| Phiên bản | Recall | FN đầu | FN giữa | FN cuối | Đổi ID | Box trùng | Box xa mọi vật | Sai số tâm (m) |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.953 | 753 | 784 | 362 | 786 | 8.690 | 198.630 | 0.306 |
| B1 | 0.962 | 677 | 461 | 321 | 674 | 11.128 | 295.043 | 0.314 |
| B2 | 0.963 | 473 | 663 | 270 | 742 | 14.996 | 482.732 | 0.316 |
| R1 | 0.953 | 755 | 784 | 362 | 741 | 8.672 | 198.405 | 0.306 |
| R2 | 0.956 | 630 | 765 | 352 | 741 | 8.685 | 199.730 | 0.308 |
| R3 | 0.955 | 775 | 706 | 329 | 709 | 8.719 | 198.333 | **0.289** |

## 5. Phân tích

### 5.1 Lỗi của B0 ở ngưỡng bộ chấm chọn

Chi tiết: `offline/notes/01_b0_analysis.md`. Trong 19.133 FN của B0, 75% là vật có track bám theo nhưng score trung bình của track thấp hơn ngưỡng nên cả track bị loại. IDS chỉ khoảng 1% tổng lỗi.

### 5.2 Nối track hai chiều

Phân loại 786 lần đổi ID của B0 (đo không lọc ngưỡng, `offline/scripts/diag_switch.py`):

- 106 lần: track đứt rồi mọc lại với ID mới. Bước nối track xử lý loại này; R1 giảm còn 54.
- 616 lần: track của vật bị hở frame, phép ghép 2 m nhận nhầm track của vật bên cạnh. Bộ chấm nuScenes đã tự nội suy nên phần lớn không tính vào IDS chính thức.
- 64 lần: trường hợp khác.

Độ chính xác khi nối, đối chiếu GT (`offline/scripts/link_precision.py`):

| Cấu hình | Tổng lần nối | Có track nhiễu | Nối đúng | Nối sai | Tỉ lệ đúng |
|---|---|---|---|---|---|
| R1 (mọi lớp) | 5.422 | 5.353 | 38 | 31 | 55% |
| R1b (bỏ pedestrian, score ≥ 0,15) | 992 | 965 | 20 | 7 | 74% |

Pedestrian: 15 đúng, 23 sai, do người đi theo nhóm dễ bị nối nhầm sang người bên cạnh. Trường hợp một vật thật bị cắt thành hai track rất hiếm, nên tác dụng của bước này nhỏ.

### 5.3 Kéo dài đầu và đuôi track

Kéo dài mọi track có ≥ 2 box thật, tối đa 2 s mỗi phía (104.554 box), đối chiếu GT (`offline/scripts/analyze_ext.py`): khoảng 1% box kéo dài trùng vật thật; với track có score trung bình 0,7 là 13%.

Nguyên nhân: nuScenes bỏ khỏi GT các vật không có điểm LiDAR. Detector mất dấu đầu/đuôi track chủ yếu đúng lúc vật không còn điểm LiDAR, nên box kéo dài ở đó bị tính là FP.

Kiểm tra trên 2 scene val có LiDAR (scene-0103, scene-0916, `offline/scripts/lidar_check_mini.py`): số điểm LiDAR trong box kéo dài không phân biệt được box đúng và sai (tỉ lệ đúng 0–1% ở mọi mức số điểm). Kết luận: không dùng bước kéo dài khi không có refine.

### 5.4 Làm mượt cả track

Áp cho track có ≥ 3 box thật: dùng chung kích thước (trung vị), sửa hướng lật 180° theo cửa sổ ±3 box, làm mượt vị trí bằng hồi quy tuyến tính có trọng số (cửa sổ ±2 box, Gauss σ = 1 s, nhân score). Tham số chọn trên dev: ±2 box bậc 1 cho sai số tâm 0,289 m; ±3 box bậc 2 cho 0,299 m; ±3 box bậc 1 σ 1,5 s cho 0,297 m.

| Phiên bản | Sai số tâm (m) | Box lệch hướng > 90° | Sai số kích thước (1 − IoU thể tích) |
|---|---|---|---|
| R1 | 0.306 | 7,7% | 0,204 |
| R3 | 0.289 | 6,5% | 0,195 |

Hướng và kích thước đo bằng `offline/scripts/check_yaw_size.py` (box score ≥ 0,3 ghép GT trong 2 m, khoảng 42.400 box). Bộ chấm nuScenes chỉ dùng tâm box, không chấm hướng và kích thước; hai cột sau có ý nghĩa cho chất lượng nhãn auto-labeling.

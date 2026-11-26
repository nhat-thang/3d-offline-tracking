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

R1, R1b, R3 chưa chấm devkit trên dev. Cấu hình gộp R1b + R3 (Ours) được chấm trên 75 scene test, xem mục 6.

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

### 5.5 Sửa luật lật hướng trong bước làm mượt (Ours_v2)

Đo riêng bước làm mượt trên dev (`offline/scripts/diag_smooth.py`: ghép box với GT, so cùng box trước và sau làm mượt):

| | Box lật hướng trước | Sau (luật cũ) | Sau (luật `vote`) | Sửa đúng / làm sai (luật cũ) | Sửa đúng / làm sai (`vote`) |
|---|---|---|---|---|---|
| Xe chạy bình thường (44.700 box) | 8,67% | 7,08% | **6,78%** | 1.509 / 798 | 1.114 / **270** |
| Xe đang lùi (283 box) | 25,8% | 27,6% | **22,6%** | 18 / 23 | 14 / **5** |

- Luật cũ lấy chiều chuyển động làm hướng đầu xe khi xe đi nhanh hơn 1 m/s, nên lật nhầm xe đang lùi. Luật `vote` (`--flip_mode vote --flip_vote 0.7`) chỉ lật box ngược hướng với ít nhất 70% số box còn lại trong cửa sổ ±3.
- Cùng phép đo cho thấy làm mượt không gây hại ở các tình huống dự đoán là khó: xe rẽ gắt (> 20°/s) sai số tâm 0,454 → 0,357 m; kích thước trailer 0,229 → 0,218, bus 0,187 → 0,174 (1 − IoU). Hai luật này giữ nguyên.
- Thử thêm cổng hướng (`--max_dyaw`) và cổng tốc độ (`--max_dv`) cho bước nối track: trên dev chỉ khoảng 25 cặp nối kiểm tra được với GT, hai cổng loại bỏ cặp đúng nhiều hơn cặp sai, nên để tắt mặc định.

Ours_v2 = Ours với `--flip_mode vote`. Trên 75 scene test, tỉ lệ box lệch hướng > 90° (theo lớp, `check_yaw_size.py`):

| | car | truck | trailer | bus | motorcycle | bicycle | pedestrian |
|---|---|---|---|---|---|---|---|
| Ours | 4,66% | 4,11% | 22,0% | 0,90% | 3,44% | 14,1% | 9,88% |
| Ours_v2 | **4,06%** | **2,81%** | **18,9%** | 1,13% | 4,40% | 14,2% | 9,83% |

Tính chung khoảng 5,9% (Ours: 6,44%). AMOTA của Ours_v2: chưa chấm.

## 6. Kết quả trên 75 scene test

Chấm một lần cho mọi cấu hình bằng nuscenes-devkit 1.1.11 (`offline/refine/run_eval_all.sh`). Refine: seed 0.

| Cấu hình | AMOTA | AMOTP (m) | MOTA | Recall | IDS | FRAG | FP | FN | Lệch hướng > 90° | 1 − IoU kích thước |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 (ImmortalTracker) | 0.685 | 0.546 | 0.590 | 0.744 | 214 | 246 | 8.843 | 8.815 | 7,44% | 0,204 |
| B0 + refine, chỉ đổi score | 0.690 | 0.553 | 0.604 | 0.750 | 167 | 251 | 7.852 | 9.485 | 8,19% | 0,204 |
| B0 + refine | 0.691 | 0.553 | 0.604 | 0.750 | 167 | 250 | 7.851 | 9.485 | 8,19% | 0,204 |
| **Ours** (nối track R1b + làm mượt R3) | **0.691** | **0.515** | 0.596 | 0.747 | 183 | **168** | 8.598 | 8.916 | **6,44%** | 0,193 |
| Ours + refine | 0.691 | 0.536 | 0.604 | 0.738 | 147 | 170 | 7.393 | 9.856 | 6,46% | 0,191 |
| CTRLext (kéo dài hai chiều) | 0.651 | 0.531 | 0.538 | 0.687 | 199 | 209 | 8.711 | 10.324 | 7,45% | 0,204 |
| CTRLext + refine (các bước kiểu CTRL) | 0.679 | 0.542 | 0.590 | 0.718 | 159 | 225 | 7.836 | 10.261 | 7,80% | 0,201 |

AMOTA theo lớp:

| | car | pedestrian | truck | bus | trailer | motorcycle | bicycle |
|---|---|---|---|---|---|---|---|
| B0 | 0.827 | 0.759 | 0.677 | 0.873 | 0.495 | 0.670 | 0.492 |
| B0 + refine | 0.827 | 0.763 | 0.674 | 0.877 | 0.486 | 0.692 | 0.514 |
| Ours | 0.830 | 0.757 | 0.683 | 0.878 | 0.508 | 0.687 | 0.493 |
| Ours + refine | 0.821 | 0.760 | 0.678 | 0.881 | 0.482 | 0.712 | 0.506 |
| CTRLext | 0.815 | 0.742 | 0.672 | 0.864 | 0.490 | 0.578 | 0.397 |
| CTRLext + refine | 0.814 | 0.751 | 0.670 | 0.869 | 0.479 | 0.680 | 0.488 |

Nhận xét:

- Ours đạt cùng AMOTA với B0 + refine mà không cần train; box chính xác hơn (AMOTP −0,031 m, lệch hướng 7,44% → 6,44%, kích thước 0,204 → 0,193), số track bị vỡ (FRAG) giảm 32%, IDS giảm 14%.
- Chênh AMOTA +0,006 là nhỏ và chưa kiểm định thống kê trên 75 scene.
- Refine tăng AMOTA nhờ chấm lại score (chỉ đổi score đã được 0,690); phần chỉnh box không cải thiện, AMOTP và lệch hướng còn tăng nhẹ. Áp lên Ours thì không tăng thêm.
- Kéo dài hai chiều làm giảm 0,034 AMOTA: box kéo dài mang score của box thật gần nhất (`ext_decay = 1`) nhưng nằm ngoài hai đầu track, nơi nuScenes không có GT, nên thành FP có score cao; ngưỡng cho MOTA tốt nhất bị đẩy lên và recall tại ngưỡng đó giảm 0,744 → 0,687. Refine bù lại được 0,028 nhưng vẫn dưới B0.
- AMOTA tối đa đạt được với recall của CenterPoint là khoảng 0,86, nên các bước sau tracker chỉ thay đổi được ít.

Chi tiết refine: [`docs/refine.md`](docs/refine.md). Thí nghiệm về cách bộ chấm phạt từng kiểu lỗi: [`docs/tracking_metrics.md`](docs/tracking_metrics.md), mục 7.

## 7. Waymo Open Dataset (vehicle, tập validation)

Chấm theo cách của paper CTRL: metric detection AP/APH, ghép theo IoU 3D ≥ 0,7 (định nghĩa: [`docs/tracking_metrics.md`](docs/tracking_metrics.md), mục 8). Chi tiết và phân tích: [`offline/notes/03_waymo_tai_lap_ctrl.md`](offline/notes/03_waymo_tai_lap_ctrl.md).

### 7.1 Kết quả của tác giả CTRL, 202 chuỗi

| Kết quả | L1 AP / APH | L2 AP / APH | Paper |
|---|---|---|---|
| FSD (detector nền) | 82,79 / 82,31 | 75,13 / 74,69 | Table 4: 74,9 / 74,4 |
| CTRL, Track TTA | 88,47 / 87,48 | 82,28 / 81,31 | Table 1: 82,3 / 81,3 |

### 7.2 Chạy lại CTRL và phương pháp đề xuất, 8 chuỗi

Cùng detection FSD, cùng ImmortalTracker-for-CTRL, cùng các bước kéo dài, refine (checkpoint của tác giả) và xóa box rỗng; phương pháp đề xuất chỉ thêm bước nối track + làm mượt sau tracker, tham số giữ nguyên từ nuScenes (đổi đơn vị sang giây). Paper không báo cáo số trên tập con này.

| Giai đoạn (L2 AP / APH) | CTRL | Đề xuất |
|---|---|---|
| Sau tracker | 76,37 / 75,88 | 77,30 / 76,82 |
| Sau kéo dài | 76,42 / 75,93 | 77,36 / 76,88 |
| Sau refine, không TTA | 82,63 / 81,92 | 83,38 / 82,74 |
| Sau xóa box rỗng, không TTA | 83,25 / 82,53 | **84,06 / 83,42** |
| Sau xóa box rỗng, Track TTA | 84,33 / 83,29 | chưa chạy |
| Kết quả chính thức của tác giả, Track TTA | 84,51 / 83,46 | |

- CTRL chạy lại thấp hơn kết quả chính thức 0,17 APH L2; so từng box: 98% box ghép được, lệch tâm trung vị 2,5 cm, 98,7% box thuộc đúng track tương ứng.
- Đề xuất hơn CTRL khoảng 0,8–0,95 APH L2 ở mọi giai đoạn. Bước nối track chỉ nối 3 cặp; phần tăng chủ yếu đến từ làm mượt. So từng box GT sau refine: CTRL sai mà đề xuất đúng 874 box, ngược lại 363 box.
- Hạn chế: 8/202 chuỗi, chưa kiểm định thống kê, chưa tách đóng góp từng thành phần, chưa chấm metric tracking của Waymo.

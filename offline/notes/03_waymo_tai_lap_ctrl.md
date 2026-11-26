# Tái lập CTRL và chạy phương pháp đề xuất trên Waymo Open Dataset

## Mục tiêu

Paper CTRL (arXiv 2304.12315) chỉ báo cáo kết quả trên Waymo Open Dataset và chấm theo IoU 3D, nên phần tái lập trước đây trên nuScenes chưa đối chiếu trực tiếp được với paper. Phần này làm hai việc trên Waymo, tập validation, lớp vehicle:

1. Kiểm chứng code và kết quả của CTRL bằng kết quả và checkpoint công bố của tác giả. Chỉ chạy inference, không train lại.
2. Chạy phương pháp đề xuất (nối track + làm mượt cả track) trên cùng dữ liệu và so sánh với CTRL.

Metric dùng trong phần này (AP/APH, LEVEL_1/LEVEL_2, IoU 3D) được giải thích ở [`docs/tracking_metrics.md`](../../docs/tracking_metrics.md), mục 8. Cách chạy từng bước: [`waymo/README.md`](../../waymo/README.md).

## Nguồn dữ liệu và kết quả của tác giả

- Hướng dẫn chính thức `docs/CTRL_instructions.md` trong repo tusen-ai/SST có link Google Drive chứa kết quả detector nền, kết quả CTRL và checkpoint.
- Dữ liệu Waymo dùng theo giấy phép phi thương mại của Waymo Open Dataset. Dữ liệu, file kết quả và ảnh không đưa lên git.
- File của tác giả (`data/waymo/ctrl_release/`):

| File | Dung lượng | Là gì |
|---|---|---|
| `fsd_base_vehicle_val.bin` | 195 MB | Kết quả detector nền FSD, lớp vehicle, tập validation |
| `best_val_in_paper.bin` | 609 MB | Kết quả cuối của CTRL trên validation, bản trong paper |
| `ctrl_veh_epoch_24.pth` | 246 MB | Checkpoint refine của CTRL cho vehicle |
| `poses.pkl` | 39 MB | Pose của xe từng frame |

- Nhãn GT: thành phần `lidar_box` của Waymo v2.0.1, tập validation, 202 file parquet, 155 MB.
- Điểm LiDAR: thành phần `lidar`, `lidar_calibration`, `lidar_pose`, `vehicle_pose` của v2.0.1, tải cho 8 chuỗi validation đầu tiên theo thứ tự tên (`waymo/subset8.txt`, 1.571 frame), chuyển sang định dạng điểm của mmdet3d bằng `waymo/v2_to_kitti_points.py`.

## Môi trường và các trục trặc kỹ thuật

| Vấn đề | Cách xử lý |
|---|---|
| Gói `waymo-open-dataset` đòi `jaxlib==0.4.13`, PyPI không còn | Cài với `--no-deps` và tự cài các phụ thuộc phần chấm cần (`waymo/Dockerfile.eval`) |
| Môi trường CTRL (Ubuntu 18.04, Python 3.8) không cài được gói Waymo (wheel manylinux_2_28) | Chỉ lấy các file `*_pb2.py` từ gói (`waymo/Dockerfile.ctrl2`) |
| `torch-scatter` biên dịch từ nguồn bị lỗi | Dùng bản dựng sẵn `torch-scatter==2.0.8` cho torch 1.8 + CUDA 11.1 (`waymo/Dockerfile.ctrl`) |
| matplotlib mới đòi numpy ≥ 1.20, mmdet3d cần numpy 1.19.5 | Hạ matplotlib về 3.3.4 |
| mmcv 1.3.9 lỗi khi ghi file cấu hình với yapf mới | Sửa cấu hình trong bộ nhớ rồi gọi `tools/test.py` (`waymo/ctrl_test.py`) |
| Máy có 1 GPU, script của tác giả gán GPU theo số thứ tự tiến trình (0–3), các tiến trình 1–3 lỗi không báo | Chạy tracker, cắt điểm và xóa box rỗng bằng 1 tiến trình |
| Nhánh 1 tiến trình của `remove_empty.py` truyền sai thứ tự tham số | Sửa trên bản sao khi chạy, không sửa code gốc |
| Thiếu `context2timestamp.pkl` cho bước kéo dài | Tạo từ dữ liệu đã chuyển đổi (`waymo/make_ctx2ts.py`) |
| Chấm 4,6 triệu box bị tắt do hết RAM | Chấm từng lớp, đọc thẳng vào mảng numpy |

## Bước A: chấm lại kết quả của tác giả (202 chuỗi)

| Cấu hình (vehicle, 202 chuỗi validation) | L1 AP/APH | L2 AP/APH | Trong paper |
|---|---|---|---|
| FSD nền | 82,79 / 82,31 | 75,13 / 74,69 | Table 4, Offline FSD: 74,9 / 74,4 |
| CTRL (có Track TTA) | 88,47 / 87,48 | 82,28 / 81,31 | Table 1: L1 88,5 / 87,5, L2 82,3 / 81,3 |

- CTRL khớp Table 1, nên bộ chấm và GT dùng ở đây cho cùng kết quả như cách tác giả chấm.
- FSD cao hơn Table 4 khoảng 0,2–0,3 điểm; file có thể là một cấu hình FSD offline gần nhưng không trùng hẳn cấu hình trong bảng.
- CTRL tăng khoảng 6,6 điểm APH L2 so với detector nền (74,69 → 81,31).
- GT: 956.767 box L1, 166.606 box L2, bỏ 121.683 box không có điểm LiDAR.

## Bước B: tự chạy lại CTRL trên 8 chuỗi

Pipeline: ImmortalTracker-for-CTRL (cấu hình `immortal_for_ctrl_keep10`) trên `fsd_base_vehicle_val.bin` → kéo dài về quá khứ → cắt điểm theo track → refine bằng `ctrl_veh_epoch_24.pth` → xóa box rỗng. GPU RTX 3050 4 GB. Thời gian trên 8 chuỗi: tracker khoảng 2 phút, cắt điểm khoảng 3,5 phút, refine không TTA khoảng 5 phút, có Track TTA khoảng 23 phút, xóa box rỗng khoảng 6 phút.

Chấm vehicle trên đúng 8 chuỗi (GT: 37.050 box L1, 7.447 box L2, bỏ 4.198 box không có điểm):

| Kết quả | Số box | L1 AP / APH | L2 AP / APH |
|---|---|---|---|
| FSD nền (file tác giả) | 58.232 | 83,57 / 83,06 | 76,91 / 76,42 |
| ImmortalTracker-for-CTRL | 98.543 | 82,92 / 82,41 | 76,37 / 75,88 |
| + kéo dài về quá khứ | 123.943 | 82,96 / 82,45 | 76,42 / 75,93 |
| + refine CTRL, không TTA | 123.825 | 87,36 / 86,66 | 82,63 / 81,92 |
| + xóa box rỗng | 93.351 | 88,04 / 87,33 | 83,25 / 82,53 |
| + refine CTRL, Track TTA | 123.825 | 88,35 / 87,25 | 83,71 / 82,65 |
| + xóa box rỗng | 93.443 | **89,01 / 87,94** | **84,33 / 83,29** |
| CTRL chính thức (`best_val_in_paper.bin`, cùng 8 chuỗi) | 93.589 | 89,15 / 88,06 | 84,51 / 83,46 |

- Bản chạy lại (Track TTA + xóa box rỗng) thấp hơn kết quả chính thức khoảng 0,17–0,18 điểm L2 trên cùng 8 chuỗi; số box gần như bằng nhau.
- Paper không báo cáo số trên tập con này. Số tuyệt đối trên 8 chuỗi cao hơn số toàn tập trong paper, nên mốc so sánh là kết quả chính thức chấm trên cùng chuỗi, không phải số trong paper.
- Mức tăng so với FSD nền: +7,4 AP / +6,9 APH L2 (kết quả chính thức: +7,6 / +7,0). Refine là phần tăng chính (khoảng 6 điểm); xóa box rỗng khoảng 0,6 điểm; Track TTA khoảng 0,8 điểm.

### So từng box với kết quả chính thức

`waymo/compare_boxes.py` ghép từng box của bản chạy lại với box của kết quả chính thức trong cùng frame (không dùng GT):

| Chỉ số (bản Track TTA + xóa box rỗng, 8 chuỗi) | Giá trị |
|---|---|
| Box ghép được | 91.673 / 93.443 (khoảng 98%) |
| Box chỉ một bên có | 98,5% có score dưới 0,1 |
| IoU 3D giữa hai box, trung vị | 0,965 (58,5% cặp ≥ 0,95) |
| Lệch tâm, trung vị | 2,5 cm |
| Lệch hướng, trung vị | 0,3° |
| Box thuộc đúng track tương ứng | 98,7% |

- Phần tracking gần như trùng với kết quả chính thức; phần refine cho box lệch vài cm.
- Giả thuyết (chưa kiểm chứng): điểm LiDAR đầu vào khác một chút, vì ở đây điểm được chuyển từ Waymo v2.0.1 bằng script riêng, còn tác giả dùng Waymo v1 với công cụ chuyển đổi của mmdet3d.

### Hạn chế

- Pipeline mới chạy lại trên 8/202 chuỗi.
- Dòng CTRL không TTA của Table 1 (81,7 / 81,0) chưa đối chiếu được vì không có file kết quả tương ứng.
- Mới có lớp vehicle.

## Phân tích lỗi còn lại của CTRL (8 chuỗi)

`waymo/analyze_ctrl_errors.py` ghép box GT có điểm LiDAR với box CTRL sau refine (không TTA, score ≥ 0,3; đúng khi IoU 3D ≥ 0,7 và lệch hướng < 30°), rồi gán mỗi box sai một nguyên nhân theo track đầu vào của refine. Trong 44.497 box GT, CTRL đúng 36.437, sai 8.060:

| Nguyên nhân | Box sai | Box đầu vào đã đúng (IoU ≥ 0,7) |
|---|---|---|
| Refine cho box sai dù track đầu vào sạch và xe đủ điểm | 2.246 | 554 |
| Refine hạ score dưới ngưỡng hoặc box bị xóa | 2.155 | 153 |
| Xe có ≤ 20 điểm LiDAR | 1.002 | 185 |
| Box đầu vào ngược hướng | 982 | 331 |
| Hai track đầu vào cho cùng một xe | 804 | 246 |
| Không có box đầu vào (lỗi từ detector/tracker) | 470 | 0 |
| Xe bị chia cho nhiều track nối tiếp | 276 | 36 |
| Frame do bước kéo dài tạo ra | 103 | 6 |
| Track đi qua hai xe khác nhau | 22 | 3 |

- Các nguyên nhân thuộc track đầu vào (ngược hướng, track trùng, track đứt, track nhảy vật) chiếm khoảng 26% số box sai.
- Phân loại dùng luật heuristic (khoảng cách, IoU, số frame), mỗi box chỉ gán một nguyên nhân theo thứ tự ưu tiên; dùng để thấy xu hướng.
- Ảnh từng ca: `waymo/viz_ctrl_cases.py`, xuất ra `offline/results/waymo/subset8/loi_ctrl/anh/index.html` (không đưa lên git).

## Bước C: phương pháp đề xuất trên Waymo (8 chuỗi)

`waymo/ours_waymo.py` chuyển Ours_v2 (nối track hai chiều + làm mượt cả track với luật lật hướng `vote`) sang Waymo:

- Giữ nguyên giá trị tham số đã chọn trên nuScenes; các tham số tính theo số box được đổi sang giây (2 Hz → 10 Hz). Không chỉnh tham số trên dữ liệu Waymo.
- Box được đổi sang hệ tọa độ thế giới bằng `poses.pkl` trước khi tính vận tốc và làm mượt, rồi đổi lại.
- Box thật của tracker được xác định bằng cách so với detection FSD cùng frame (tâm lệch < 0,3 m), vì score FSD thấp vẫn có thể là detection thật.
- Đầu vào là cùng kết quả ImmortalTracker-for-CTRL mà CTRL dùng; đầu ra đi qua đúng các bước sau tracker của CTRL (kéo dài, cắt điểm, refine bằng checkpoint CTRL không TTA, xóa box rỗng) bằng `waymo/run_ours_refine.sh`. Hai bản chỉ khác nhau ở bước nối track và làm mượt.

Vehicle L2 AP / APH trên 8 chuỗi:

| Giai đoạn | CTRL | Đề xuất | Chênh APH |
|---|---|---|---|
| Sau tracker | 76,37 / 75,88 | 77,30 / 76,82 | +0,94 |
| Sau kéo dài | 76,42 / 75,93 | 77,36 / 76,88 | +0,95 |
| Sau refine (không TTA) | 82,63 / 81,92 | 83,38 / 82,74 | +0,82 |
| Sau xóa box rỗng | 83,25 / 82,53 | 84,06 / 83,42 | +0,89 |

- Bước nối track chỉ nối được 3 cặp, vì tracker của CTRL giữ track lâu; phần tăng chủ yếu đến từ làm mượt (kích thước chung, sửa lật hướng 1.719 box, làm mượt vị trí).
- So từng box GT với CTRL sau refine: CTRL sai mà đề xuất đúng 874 box; CTRL đúng mà đề xuất sai 363 box. Phần sửa được nhiều nhất thuộc nhóm "refine cho box sai" (405) và "box đầu vào ngược hướng" (228). Nhiều ca sửa được là box có IoU sát ngưỡng 0,7.
- So với CTRL có Track TTA trên cùng 8 chuỗi (83,29 khi chạy lại, 83,46 kết quả chính thức), bản đề xuất không TTA đạt mức tương đương.

Hạn chế: mới 8 chuỗi, chưa kiểm định thống kê; chưa tách đóng góp của từng thành phần làm mượt; chưa có bước gộp track trùng; chưa chấm metric tracking của Waymo (MOTA, IDS).

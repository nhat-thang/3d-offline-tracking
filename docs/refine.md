# Refine rút gọn trên LiDAR keyframe

Thử bước refine theo track của CTRL (arXiv 2304.12315) trên nuScenes. Trọng số và code train của CTRL cho nuScenes không được công bố, nên refine được dựng lại ở dạng rút gọn, train trên 1 GPU 4 GB.

## 1. Dữ liệu

- **Điểm LiDAR:** chỉ LiDAR keyframe (2 Hz) của 150 scene val: 6.019 file, khoảng 4 GB (`offline/download_val_lidar_resume.sh`). Không dùng sweep.
- **Kiểm tra tọa độ:** đếm điểm trong box GT sau khi đổi tọa độ, so với `num_lidar_pts` của nuScenes; khớp 99,9% (`offline/scripts`, commit 5f52412).
- **Mẫu train:** mỗi box của track (kể cả box đoán) lấy 256 điểm trong box nới 1 m, đặt về hệ tọa độ của box. Ghép box với GT cùng frame bằng Hungarian, tâm < 2 m.
- **Nhãn:** real box/giả (có ghép GT), track thật/giả (track có ghép GT), độ lệch (dx, dy, dz, log kích thước, sin, cos góc) so với GT.

| Tập | Scene | Track | Box | Tỉ lệ box ghép được GT |
|---|---|---|---|---|
| B0 dev | 75 | 59.395 | 252.870 | 18,0% |
| B0 test | 75 | 58.613 | 248.096 | 17,6% |
| Ours dev | 75 | 58.574 | 252.870 | 18,1% |
| CTRLext dev | 75 | 59.778 | 287.375 | 15,9% |

Tỉ lệ box ghép GT thấp vì đầu ra ImmortalTracker giữ cả box score rất thấp; đây chính là phần refine cần học cách hạ score.

Script: `offline/refine/make_dataset.py` (chạy theo 3 phần `--chunk i --nchunks 3`), `offline/refine/merge_dataset.py`.

## 2. Model (`offline/refine/model.py`, khoảng 626 nghìn tham số)

1. **Theo từng box:** PointNet (7 → 64 → 128 → 256) trên 256 điểm, ghép với đặc trưng của box (score, real box hay đoán, log kích thước, độ cao tương đối, khoảng cách, số điểm trong box, vị trí trong track, vận tốc dọc/ngang).
2. **Dọc track:** 4 khối Conv1D giãn (dilation 1, 2, 4, 8), d = 128.
3. **Ba đầu ra:** chỉnh box (dx, dy, dz, dlog w/l/h, sin, cos); logit real box/giả; logit track thật/giả. Hai logit được cộng vào logit của score gốc; các đầu ra khởi tạo bằng 0, nên lúc đầu model trả lại đúng box và score gốc.
4. **Loss:** SmoothL1 (beta 0,2) cho phần chỉnh box, BCE cho box, BCE cho track có trọng số theo số box.
5. **Tăng cường dữ liệu:** bỏ ngẫu nhiên 0–80% điểm, nhiễu 2 cm, lật trục y.

So với CTRL: không dùng Sparse UNet, không gộp điểm nhiều frame theo track, không gán nhãn theo TIoU, không TTA, không TCO.

## 3. Train (`offline/refine/train.py`)

- Train trên 75 scene dev, giữ 1/7 số scene để kiểm tra; 12–20 epoch, batch 32.
- Chạy trong Docker `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (`tools/train_refine.bat`, `tools/train_refine2.bat`).
- Ba bản: refine trên đầu ra B0, trên Ours (track linking + smoothing) và trên CTRLext (kéo dài hai chiều).

Kết quả trên phần kiểm tra (bản B0, epoch tốt nhất 9; sau đó bắt đầu quá khớp):

| | Trước refine | Sau refine |
|---|---|---|
| AUC real box/giả | 0,956 | 0,983 |
| AUC track thật/giả | 0,973 | 0,979 |
| Sai số tâm (m) | 0,265 | 0,265 |
| Tỉ lệ heading flip | 9,4% | 9,4% |

Model học được cách chấm lại score, nhưng **không học được cách chỉnh vị trí, hướng, kích thước**. Nguyên nhân khả dĩ: keyframe thưa (khoảng 34 điểm mỗi box), track ngắn, dữ liệu ít.

## 4. Áp lên track và chấm điểm

- `offline/refine/predict.py`: dự đoán cho tập test.
- `offline/refine/apply.py`: áp phần chỉnh box và đặt score của mọi box trong track bằng xác suất track thật (`--no_box`: chỉ đổi score).
- `offline/refine/run_eval_all.sh` (qua `tools/eval_refine.bat`): chấm devkit cho mọi cấu hình, gom bảng vào `offline/results/refine/summary_test_<seed>.md`.

Kết quả: xem [`RESULTS.md`](../RESULTS.md), mục 6. Tóm tắt: refine tăng AMOTA của B0 từ 0,685 lên 0,691, gần như toàn bộ nhờ chấm lại score (chỉ đổi score: 0,690); áp lên Ours thì không tăng thêm.

## 5. Hạn chế và bước tiếp

- Mới train 1 seed; `tools/run_seeds.bat` train thêm seed 1, 2 để kiểm tra độ ổn định.
- Thử gộp sweep (20 Hz) để box có nhiều điểm hơn.
- Refine chỉ đổi score theo cả track, không bỏ được từng box sai; có thể thêm lựa chọn bỏ box có xác suất real box thấp.

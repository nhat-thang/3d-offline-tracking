# So sánh các bước: CTRL và cách làm hiện tại

## 1. Các bước của CTRL (arXiv 2304.12315)

| Bước | Nội dung | Cần train |
|---|---|---|
| 1. Detection | Detector FSD, dùng nhiều frame LiDAR | Có |
| 2. Tracking xuôi | ImmortalTracker: giữ track khi vật bị che | Không |
| 3. Kéo dài hai chiều | Kéo mọi track về quá khứ và tương lai: track > 100 frame kéo tới hết sequence, track ngắn hơn kéo thêm 20 frame | Không |
| 4. Refine theo track | Mạng dùng point cloud của cả track: sửa box, loại box kéo dài sai, chấm lại độ tin cậy | Có |

Điều kiện chạy: GPU, trọng số FSD và refine (tác giả không công khai do license Waymo), point cloud Waymo khoảng 1 TB.

## 2. Cách làm hiện tại (không dùng refine)

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1. Detection | CenterPoint (file do tác giả công bố, giữ cố định) | Có sẵn |
| 2. Tracking xuôi | ImmortalTracker gốc = B0 | Xong |
| 3a. Nối track bị cắt | Nối hai chiều, dùng frame trước và sau; chỉ áp cho xe | Xong (R1b) |
| 3b. Kéo dài đầu/đuôi | Không dùng: thiếu refine thì box kéo dài phần lớn sai | Đã thử, bỏ |
| 3c. Làm mượt cả track | Kích thước chung, sửa hướng lật, làm mượt vị trí | Xong (R3) |
| 3d. Chấm lại độ tin cậy | Score mới cho cả track từ đặc trưng track (độ dài, score các frame, độ mượt, độ ổn định kích thước) | Đang làm |
| 4. Refine | Không dùng | — |

Điều kiện chạy: file kết quả detector, pose xe, timestamp; chạy trên CPU.

## 3. Đối chiếu theo việc cần làm

| Việc | CTRL | Cách làm hiện tại |
|---|---|---|
| Giữ ID khi vật bị che | ImmortalTracker | ImmortalTracker |
| Track bị cắt thành hai ID | Không xử lý | Nối hai chiều sau khi tracking |
| Thiếu box ở đầu/cuối track | Kéo dài + refine lọc lại | Không xử lý |
| Box lệch, lật hướng, kích thước dao động | Refine sửa (dùng point cloud) | Làm mượt theo luật |
| Track nhiễu, độ tin cậy sai | Refine chấm lại | Chấm lại theo đặc trưng track |

## 4. Ưu và nhược

| | CTRL | Cách làm hiện tại |
|---|---|---|
| GPU, trọng số, point cloud | Cần | Không cần |
| Độ chính xác box | Cao hơn (nhờ refine) | Thấp hơn, chỉ dựa trên box có sẵn |
| Vật detector bỏ sót hoàn toàn | Lấy lại được một phần (kéo dài + refine) | Không lấy lại được |
| Gộp track bị cắt | Không | Có |
| Đổi detector | Phải train lại refine | Dùng ngay |

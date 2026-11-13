# Vị trí của bước sửa danh tính track trong pipeline CTRL

Ví dụ: một chiếc xe đi qua trong 10 giây, bị xe tải che 1 giây ở giữa.

## 1. Detector: "frame này có vật gì?"
Detector xử lý từng frame độc lập, không biết box ở các frame là cùng một vật.
```
Giây:   1  2  3  4  5  6  7  8  9  10
Box:    ■  ■  ■  ■  ·  ■  ■  ■  ■  ·      (giây 5 bị che, giây 10 ở xa)
```

## 2. Tracker (ImmortalTracker): "box nào là cùng một vật?"
Liên kết (association) box giữa các frame và gán ID. Sau đoạn bị che, tracker không nhận ra vật cũ và tạo ID mới.
```
ID:     1  1  1  1  ·  7  7  7  7  ·      → một xe nhưng hai ID
```

## 3. Rà soát danh tính track (offline, không cần train)
Xét toàn bộ đoạn (cả quá khứ và tương lai). ID 1 kết thúc ở giây 4, ID 7 bắt đầu ở giây 6; cùng lớp, cùng kích thước,
vị trí khớp với chuyển động → nối lại thành một ID.
```
ID:     1  1  1  1  ·  1  1  1  1  ·
```
Ba nhiệm vụ: (a) nối track bị cắt (đã thực hiện), gộp track trùng, tách track nhảy sang vật khác.
Rủi ro: nối nhầm hai vật khác nhau → cần đo tỉ lệ nối đúng trên dữ liệu có GT.

## 4. Kéo dài hai chiều (CTRL): "trước và sau đó vật ở đâu?"
Đoán thêm box ở đầu/cuối track theo vận tốc (□ = box đoán, chưa kiểm chứng).
```
ID 1:   ■  ■  ■  ■  ·  ■  ■  ■  ■  □
```

## 5. Refine (CTRL, mạng đã train): "sửa cho chính xác"
Nhận cả track, xem lại point cloud qua mọi frame: sửa vị trí, hướng, kích thước; lấp frame thiếu;
giữ và chỉnh box đoán nếu có vật thật, hạ điểm nếu không.

## Vì sao bước 3 đặt trước refine
Refine xử lý từng track riêng lẻ. Nếu track bị cắt đôi, refine sửa từng mảnh nhưng kết quả vẫn là hai vật,
và mỗi mảnh ngắn hơn nên refine có ít thông tin hơn. Paper CTRL ghi nhận vật bị bỏ sót tập trung ở các track ngắn (< 2,5 s, mục 5.3).

| Bước | Thành phần | Câu hỏi trả lời |
|---|---|---|
| 1 | Detector | Frame này có vật ở đâu? |
| 2 | Tracker | Box nào là cùng một vật? |
| 3 | Rà soát danh tính track | Tracker có liên kết sai, liên kết thiếu không? |
| 4 | CTRL kéo dài | Trước và sau đó vật ở đâu? |
| 5 | CTRL refine | Sửa box cho chính xác |

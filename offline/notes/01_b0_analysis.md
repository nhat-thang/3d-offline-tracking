# Phân tích quy mô vấn đề trên B0 (11/11/2026)

Script: `offline/scripts/analyze_b0.py analyze`.
Cách phân tích giống bộ chấm: score theo trung bình cả track, ngưỡng cho MOTA tốt nhất của từng class, ghép theo khoảng cách tâm ≤ 2 m, giữ cặp ghép cũ nếu còn trong 2 m.
Kiểm chứng: tổng IDS = 383, FRAG = 425, FN = 19.133, **khớp chính xác** với bộ chấm.

## FN 19.133 chia theo vị trí trong đời vật

| Loại bỏ sót | Số box | Tỷ lệ | Bước offline liên quan |
|---|---|---|---|
| Vật **không được bám frame nào** (1.547 vật) | 14.432 | 75% | chấm lại score cả track |
| Đầu đời vật (trước lần ghép đầu) | 2.706 | 14% | kéo dài về quá khứ |
| Cuối đời vật (sau lần ghép cuối) | 1.044 | 5% | kéo dài về tương lai |
| Giữa track (425 khoảng = FRAG) | 951 | 5% | lấp đoạn bị che (bộ chấm đã tự nội suy tuyến tính) |

- IDS 383, xảy ra ở 326 vật (car 120, pedestrian 237).
- Trong 1.547 vật không được bám: **1.371 vật thực ra có track bám theo** (phủ 11.216 box GT), nhưng score trung bình của track thấp hơn ngưỡng nên cả track bị loại.
  - Chỉ 369 trong số đó vượt ngưỡng nếu bỏ các box dự đoán khi bị che (score ×0.01) khỏi trung bình.
  - Vậy phần lớn là track có detection score thấp thật, không phải do box dự đoán kéo xuống.

## Kết luận
Phần lớn lỗi nằm ở **score của track**, không nằm ở nối track hay lấp đoạn che. Hướng có tiềm năng lớn nhất là **chấm lại score cho cả track** (offline, dùng toàn bộ thông tin track). Kéo dài đầu và đuôi là hướng thứ hai. Nối track và lấp đoạn che có dư địa nhỏ.

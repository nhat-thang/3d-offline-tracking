# Sửa luật heading flip trong bước smoothing (19/11/2026)

## Tôi muốn sửa gì

Sau tuần 3, tôi liệt kê các tình huống mà track linking (R1b) và smoothing (R3) có thể làm sai: xe lùi bị heading flip nhầm, xe rẽ gắt bị smoothing trễ, trailer/bus không phải vật cứng nên median kích thước có thể sai, link nhầm hai vật khác nhau (linking precision 74%). Trước khi sửa, tôi đo xem từng lỗi có thật sự xảy ra không, để chỉ sửa đúng chỗ có lỗi.

## Tôi đo thế nào

Tôi viết `offline/scripts/diag_smooth.py`: ghép box của B0 với GT trên 75 scene dev (cùng lớp, tâm < 2 m), rồi so cùng box trước và sau bước smoothing:

- heading flip: box đúng hướng bị làm lật (làm sai) và box lật được sửa lại (sửa đúng), tách riêng xe đang lùi (GT có vận tốc > 0,5 m/s ngược hướng đầu xe);
- xe rẽ gắt (GT đổi hướng > 20°/s): sai số tâm;
- trailer, bus: sai số kích thước (1 − IoU thể tích).

## Tôi thấy gì

| | Trước | Sau smoothing (luật cũ) |
|---|---|---|
| Heading flip, xe bình thường | 8,67% | 7,08% (sửa đúng 1.509, làm sai 798) |
| Heading flip, xe lùi (283 box) | 25,8% | 27,6% (sửa đúng 18, làm sai 23) |
| Sai số tâm khi rẽ gắt (377 box) | 0,454 m | 0,357 m |
| Kích thước trailer | 0,229 | 0,218 |
| Kích thước bus | 0,187 | 0,174 |

Chỉ có lỗi heading flip là thật: luật cũ lấy chiều chuyển động làm hướng đầu xe khi xe đi nhanh hơn 1 m/s, nên với xe đang lùi nó lật sai nhiều hơn sửa đúng, và với xe thường vẫn làm sai 798 box. Rẽ gắt, trailer, bus đều tốt lên sau smoothing, nên tôi không sửa.

## Tôi sửa gì

Tôi thêm luật `vote` vào `smooth()` trong `offline/scripts/pipeline.py` (`--flip_mode vote --flip_vote 0.7`): chỉ lật một box khi ít nhất 70% số box còn lại trong cửa sổ ±3 ngược hướng với nó. Luật này không dùng chiều chuyển động. Luật cũ vẫn là mặc định để kết quả cũ tái lập được.

Kết quả trên dev:

| | Luật cũ | Luật vote |
|---|---|---|
| Heading flip, xe bình thường | 7,08% (làm sai 798) | **6,78%** (làm sai **270**) |
| Heading flip, xe lùi | 27,6% (làm sai 23) | **22,6%** (làm sai **5**) |

Tôi cũng thử thêm heading gate (`--max_dyaw`) và velocity gate (`--max_dv`) cho bước track linking, đo bằng `offline/scripts/link_precision.py`. Trên dev chỉ khoảng 25 cặp link kiểm tra được với GT, và hai gate loại bỏ cặp đúng nhiều hơn cặp sai, nên tôi để tắt mặc định.

## Kết quả trên test

Tôi tạo `Ours_v2` = Ours với `--flip_mode vote` trên 75 scene test. Tỉ lệ box heading error > 90°: car 4,66% → 4,06%, truck 4,11% → 2,81%, trailer 22,0% → 18,9%; tính chung khoảng 6,44% → 5,9%. Bus và motorcycle tăng nhẹ nhưng số box rất ít (887 và 523 box).

AMOTA của Ours_v2 chưa chấm (đã thêm vào `offline/refine/run_eval_all.sh`). Vì evaluator không xét hướng box, tôi không kỳ vọng AMOTA thay đổi.

## Còn lại

- Bước smoothing chỉ sửa lật 180°, chưa smoothing góc, nên box bị xiên 20–45° vẫn giữ nguyên. Tôi cần thêm sai số góc trung bình vào `check_yaw_size.py` để đo lỗi này trước khi sửa.

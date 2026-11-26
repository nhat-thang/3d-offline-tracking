# Metric đánh giá 3D: tracking trên nuScenes và detection trên Waymo

Định nghĩa các metric, công thức, cách bộ chấm `nuscenes-devkit` tính điểm, và ví dụ trên kết quả thực nghiệm (mục 1–7). Mục 8 trình bày metric của Waymo Open Dataset (AP/APH), cách paper CTRL chấm, và so sánh với nuScenes.

- **Bộ chấm:** `nuscenes-devkit` 1.1.11, cấu hình `tracking_nips_2019`
- **Kết quả dùng làm ví dụ:** mốc B0 (ImmortalTracker gốc + detection CenterPoint không flip, 150 scene val), lưu tại `baseline_B0/metrics_summary.json`

---

## 1. Khái niệm

| Khái niệm | Nghĩa | Ví dụ |
|---|---|---|
| **Benchmark** | Bộ đề chuẩn: dữ liệu, cách chia tập, quy trình chấm | nuScenes tracking, tập val 150 scene |
| **Metric** | Công thức biến kết quả thành con số | AMOTA, MOTA, IDS |
| **Score** | Giá trị cụ thể của một metric trên một kết quả | AMOTA = 0.705 |

Cần phân biệt với **detection score**, là độ tin cậy của từng box do detector trả ra (0–1). Bộ chấm dùng detection score (cụ thể là trường `tracking_score`) để quét ngưỡng.

---

## 2. Cách bộ chấm làm việc

1. **Đầu vào:** file `results.json`. Mỗi box gồm `sample_token`, vị trí, kích thước, góc, `tracking_name` (class), `tracking_id` và `tracking_score`.
2. **Lọc box** (áp dụng cho cả GT và box dự đoán):
   - bỏ box ở xa hơn giới hạn của class: 50 m với car, truck, bus, trailer; 40 m với pedestrian, motorcycle, bicycle;
   - bỏ box GT **không có điểm LiDAR hoặc radar nào bên trong** (vật hoàn toàn bị che);
   - bỏ xe đạp nằm trong bãi để xe.
3. **Score theo cả track:** trước khi quét ngưỡng, bộ chấm **thay score của mọi box bằng score trung bình của cả track** (cùng `tracking_id` trong một scene). Vì vậy khi lọc theo ngưỡng, **cả track được giữ hoặc bị bỏ cùng lúc**, không lọc từng box.
4. **Nội suy track:** khi một track bị trống ở giữa, bộ chấm **tự nội suy tuyến tính** box cho các frame trống. Việc này áp dụng cho cả GT lẫn track dự đoán.
5. **Ghép box dự đoán với GT:** theo **khoảng cách tâm** (mặt phẳng ngang), ghép được khi khoảng cách ≤ **2 m**. Mỗi frame được ghép một-một.
6. **Quét ngưỡng:** với mỗi class, bộ chấm thử nhiều ngưỡng `tracking_score`, tương ứng **40 mức recall** đều nhau từ 0.1 đến 1.0. Ở mỗi ngưỡng, track có score trung bình thấp hơn bị bỏ, rồi tính lại toàn bộ metric. Với B0, ngưỡng tốt nhất khoảng 0.35–0.43 tuỳ class.
7. **Tổng hợp:**
   - AMOTA và AMOTP: lấy trung bình qua 40 mức recall.
   - Các metric còn lại (MOTA, FP, FN, IDS…): lấy tại **ngưỡng cho MOTA cao nhất** của từng class.
   - Kết quả tổng: các metric tỷ lệ (AMOTA, MOTA, recall…) là **trung bình của 7 class**. Các metric đếm (TP, FP, FN, IDS, FRAG…) là **tổng của 7 class**.

---

## 3. Định nghĩa và công thức

### 3.1 Số đếm cơ bản (tại một ngưỡng)

| Ký hiệu | Nghĩa |
|---|---|
| **GT** | Số box thật trong một class (trong bảng tổng: số track thật) |
| **TP** | Box dự đoán ghép được với một box thật |
| **FP** | Box dự đoán không ghép được với box thật nào (báo nhầm) |
| **FN** | Box thật không được ghép với box dự đoán nào (bỏ sót) |
| **IDS** | Đổi ID: một vật thật đang được bám bởi ID A, sang frame sau lại được ghép với ID B |
| **FRAG** | Đứt quỹ đạo: track của một vật đang được bám thì bị mất, sau đó bám lại |

### 3.2 Metric chính

**MOTA** (Multi-Object Tracking Accuracy):

$$\text{MOTA} = \max\left(0,\ 1 - \frac{\text{FN} + \text{FP} + \text{IDS}}{\text{GT}}\right)$$

Tỷ lệ box thật được bám đúng, sau khi trừ mọi loại lỗi.

**MOTAR** (MOTA đã hiệu chỉnh theo recall *r*):

$$\text{MOTAR} = \max\left(0,\ 1 - \frac{\text{IDS} + \text{FP} + \text{FN} - (1 - r)\cdot \text{GT}}{r \cdot \text{GT}}\right), \qquad r = \frac{\text{TP}}{\text{GT}}$$

Ở mức recall *r*, có (1 − r)·GT box thật được phép bỏ sót mà không bị phạt. Chỉ phần FN vượt quá mức đó mới bị trừ. Nhờ vậy, ngưỡng có recall thấp không bị phạt quá nặng chỉ vì bỏ sót.

**AMOTA** (Average MOTA, metric xếp hạng chính của nuScenes):

$$\text{AMOTA} = \frac{1}{40}\sum_{r \in \{0.1, \dots, 1.0\}} \text{MOTAR}_r$$

- Không phụ thuộc vào việc chọn một ngưỡng score, nên phản ánh chất lượng tracker trên toàn dải độ tin cậy.
- Mức recall nào tracker không đạt tới được thì MOTAR ở mức đó bằng 0.

**AMOTP:** sai số vị trí trung bình (m) giữa box ghép đúng và box thật, lấy trung bình qua các mức recall. Mức không đạt được tính là 2 m.

### 3.3 Metric phụ

| Metric | Định nghĩa |
|---|---|
| **Recall** | TP / GT |
| **MOTP** | Sai số vị trí trung bình (m) của các box ghép đúng, tại ngưỡng tốt nhất |
| **MT** (Mostly Tracked) | Số vật thật được bám ≥ 80% thời gian xuất hiện |
| **ML** (Mostly Lost) | Số vật thật chỉ được bám ≤ 20% thời gian xuất hiện |
| **FAF** | Số box báo nhầm trung bình mỗi frame |
| **TID** (Track Initialization Duration) | Thời gian trung bình (s) từ lúc vật xuất hiện đến lúc được bám lần đầu |
| **LGD** (Longest Gap Duration) | Khoảng đứt dài nhất trung bình (s) trong mỗi track |

---

## 4. Ví dụ trên kết quả thực nghiệm (mốc B0)

### 4.1 Tính tay MOTA cho class car

| GT | TP | FP | FN | IDS |
|---|---|---|---|---|
| 58.317 | 47.858 | 7.728 | 10.339 | 120 |

$$\text{MOTA}_{car} = 1 - \frac{10.339 + 7.728 + 120}{58.317} = 1 - \frac{18.187}{58.317} = 0.688$$

Kết quả khớp với số bộ chấm in ra (0.6881). Trong 18.187 lỗi bị trừ, FN và FP chiếm 18.067, còn IDS chỉ có 120. Như vậy, **giảm bỏ sót và báo nhầm tác động lên MOTA nhiều hơn hẳn giảm đổi ID**.

### 4.2 Toàn bộ metric của B0

| Metric | Giá trị | Metric | Giá trị |
|---|---|---|---|
| **AMOTA** ↑ | **0.705** | **IDS** ↓ | **383** |
| AMOTP ↓ | 0.525 m | FRAG ↓ | 425 |
| **MOTA** ↑ | **0.600** | GT (số track) | 14.556 |
| MOTAR ↑ | 0.804 | MT ↑ | 4.474 |
| MOTP ↓ | 0.355 m | ML ↓ | 1.589 |
| Recall ↑ | 0.745 | FAF ↓ | 52.4 |
| TP | 82.381 | TID ↓ | 0.38 s |
| FP ↓ | 15.065 | LGD ↓ | 0.57 s |
| FN ↓ | 19.133 | | |

**AMOTA theo class:** bus 0.875 · car 0.830 · pedestrian 0.795 · truck 0.690 · motorcycle 0.684 · bicycle 0.564 · trailer 0.497

### 4.3 So với paper ImmortalTracker (Table 5, nuScenes val)

| Metric | Paper | B0 |
|---|---|---|
| AMOTA | 0.702 | 0.705 |
| MOTA | 0.601 | 0.600 |
| IDS | 385 | 383 |

---

## 5. Lưu ý khi đọc và so sánh

1. **Không cộng FP, FN tổng vào công thức MOTA tổng.**
   - MOTA tổng là trung bình MOTA của 7 class, mỗi class ở ngưỡng tốt nhất của riêng nó.
   - FP và FN trong bảng tổng là tổng cộng 7 class.
2. **FP, FN phụ thuộc ngưỡng.** Đây là giá trị tại ngưỡng cho MOTA cao nhất, không phải trên toàn bộ box. Vì vậy hai phương pháp có thể có FP/FN khác nhau chỉ vì ngưỡng tốt nhất khác nhau.
3. **IDS phụ thuộc mạnh vào detection.**
   - Cùng code ImmortalTracker, detection CenterPoint không flip cho IDS 383, bản có flip cho IDS 277.
   - Do đó, so sánh giữa các phương pháp tracking **bắt buộc dùng cùng một file detection**.
4. **Bộ chấm tự nội suy khoảng trống trong track.** Một phương pháp tự lấp khoảng trống bằng nội suy tuyến tính sẽ gần như không đổi điểm. Muốn đánh giá cách lấp tốt hơn tuyến tính, cần thêm thí nghiệm khác (ví dụ pseudo-occlusion theo arXiv 2310.01288).
5. **Box GT không có điểm LiDAR bị loại.** Vật bị che hoàn toàn không được tính. Box dự đoán ở các đoạn đó nếu không ghép được với GT nào thì có thể bị tính là FP.
6. **Score trung bình theo track ảnh hưởng trực tiếp tới điểm.** Thêm box có score thấp vào một track (ví dụ box dự đoán khi vật bị che) sẽ kéo score trung bình xuống, có thể làm cả track bị loại ở ngưỡng tốt nhất. Ngược lại, chấm lại score cho cả track là một cách cải thiện điểm hợp lệ trong bài toán offline.
7. **Không so điểm giữa các benchmark khác nhau.** MOTA trên Waymo và MOTA trên nuScenes khác dữ liệu, khác cách ghép (Waymo dùng IoU, nuScenes dùng khoảng cách tâm), nên không so trực tiếp được.

---

## 6. Metric theo dõi cho phần tracking offline

| Cải tiến | Metric theo dõi chính | Metric cần canh |
|---|---|---|
| Nối track bị cắt | IDS, FRAG, LGD | FP (nối nhầm) |
| Lấp đoạn bị che | FN, Recall, ML, LGD | FP |
| Kéo dài đầu và đuôi track | TID, FN | FP, FAF |
| Tổng thể | **AMOTA**, MOTA | toàn bộ bảng 4.2 |

Mọi phiên bản được chấm bằng cùng bộ chấm, cùng file detection (`detections/cp_val_noflip.json`) và cùng 150 scene. Kết quả ghi vào bảng theo dõi trong `RESULTS.md`.

## 7. Thí nghiệm: bộ chấm phạt từng kiểu lỗi thế nào

Lấy chính box GT (sau bộ lọc của bộ chấm) của 2 scene dev làm kết quả "hoàn hảo", mỗi lần làm hỏng theo đúng một kiểu rồi chấm bằng devkit (`offline/scripts/metric_experiments.py`, `offline/scripts/run_metric_experiments.sh`).

| Thí nghiệm | Cách làm hỏng | AMOTA | AMOTP (m) | MOTA | Recall | IDS | FRAG | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Hoàn hảo | GT, score 0,9 | 1.000 | 0.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| Lệch 0,5 m | Dời tâm 0,5 m theo hướng ngẫu nhiên | 0.995 | 0.484 | 1.000 | 1.000 | 0 | 0 | 0 | 2 |
| Lệch 1,5 m | Như trên, 1,5 m | 0.962 | 1.448 | 0.973 | 0.979 | 4 | 3 | 5 | 11 |
| Lệch 1,99 m | Như trên, 1,99 m | 0.977 | 1.934 | 0.989 | 0.993 | 8 | 2 | 2 | 22 |
| Lệch 2,5 m | Như trên, 2,5 m | 0.002 | 1.946 | 0.003 | 0.132 | 35 | 72 | 981 | 1.095 |
| Ngược hướng, sai kích thước | Quay 180°, kích thước ×1,5 | 1.000 | 0.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| Đổi ID giữa track | Nửa sau mỗi track (≥ 4 box) mang ID mới | 0.925 | 0.150 | 0.947 | 0.992 | 47 | 4 | 0 | 14 |
| Bỏ 1 trong 2 frame | Xóa box ở frame lẻ (giữ box cuối) | 0.984 | 0.087 | 0.992 | 0.996 | 0 | 9 | 8 | 11 |
| Kéo dài 2 frame | Thêm 2 box ở mỗi đầu track theo vận tốc | 0.927 | 0.000 | 0.923 | 1.000 | 0 | 0 | 80 | 0 |
| Track giả, score thấp | Mỗi track thêm một bản sao lệch 5 m, score 0,3 | 1.000 | 0.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| Track giả, score cao | Như trên, score 0,95 | 0.054 | 0.038 | 0.066 | 1.000 | 0 | 0 | 1.244 | 0 |
| Mất 40% track | Bỏ ngẫu nhiên 40% số track | 0.555 | 0.890 | 0.576 | 0.576 | 0 | 2 | 0 | 410 |

Kết luận:

1. **Chỉ tâm box được chấm, với ngưỡng cứng 2 m.** Lệch 1,99 m vẫn gần như đủ điểm, lệch 2,5 m mất gần hết. Hướng và kích thước không ảnh hưởng (quay 180° vẫn 1,000), nên cần đo riêng nếu quan tâm chất lượng box.
2. **Bỏ frame trong track gần như không bị phạt**, vì bộ chấm tự nội suy tuyến tính khoảng trống.
3. **Box thêm ở ngoài hai đầu track bị tính FP**: ngoài đời vật không có GT (vật không còn điểm LiDAR hoặc ra khỏi tầm). Đây là lý do bước kéo dài hai chiều làm giảm AMOTA.
4. **Thứ tự score quan trọng hơn số lượng box sai**: cùng số track giả, score thấp hơn track thật thì không mất điểm (bị cắt ở các ngưỡng recall thấp), score cao thì AMOTA gần 0.
5. **Bỏ sót vật bị phạt theo tỉ lệ**: mất 40% track làm AMOTA còn 0,555, vì các mức recall trên 0,6 không đạt được và bị tính 0.
6. **Đổi ID bị phạt vừa phải** (0,925 với 47 lần đổi ID trên 2 scene).

---

## 8. Metric trên Waymo Open Dataset

Paper CTRL báo cáo trên Waymo bằng **metric detection** (AP/APH), không dùng metric tracking. Phần Waymo của repo ([`offline/notes/03_waymo_tai_lap_ctrl.md`](../offline/notes/03_waymo_tai_lap_ctrl.md)) chấm theo đúng cách này để đối chiếu với paper.

- **Bộ chấm:** `waymo-open-dataset` 1.6.7, bản Python (`WODDetectionEvaluator`), cùng cấu hình với chương trình chấm chính thức `compute_detection_metrics_main`. Script: `waymo/eval_det.py`.
- **GT:** thành phần `lidar_box` của Waymo v2.0.1, tập validation (202 chuỗi, mỗi chuỗi khoảng 200 frame, 10 Hz).

### 8.1 Cách bộ chấm làm việc

1. **Lọc GT:** bỏ box GT không có điểm LiDAR nào bên trong.
2. **Chia mức khó:** box GT thuộc **LEVEL_2** nếu nhãn ghi LEVEL_2 hoặc box có ≤ 5 điểm LiDAR; còn lại là **LEVEL_1**. Kết quả LEVEL_2 tính trên toàn bộ GT (cả LEVEL_1 và LEVEL_2), nên luôn thấp hơn LEVEL_1.
3. **Ghép box theo từng frame:** box dự đoán và box GT cùng lớp được ghép một-một bằng Hungarian theo **IoU 3D**. Ghép được khi IoU ≥ **0,7** với vehicle, ≥ **0,5** với pedestrian và cyclist.
4. **Xếp hạng theo score từng box:** box dự đoán được sắp theo score giảm dần; với mỗi ngưỡng score tính precision và recall, ra đường precision–recall.
5. **Tính điểm cho từng lớp và từng mức khó.** ID track không được dùng: mỗi frame được chấm độc lập.

### 8.2 Định nghĩa và công thức

**IoU 3D** của hai box xoay quanh trục đứng:

$$
\text{IoU} = \frac{V_{\text{giao}}}{V_A + V_B - V_{\text{giao}}}, \qquad V_{\text{giao}} = S_{\text{giao, nhìn từ trên}} \times h_{\text{giao}}
$$

$S_{\text{giao}}$ là diện tích phần chồng nhau của hai hình chữ nhật xoay khi nhìn từ trên xuống; $h_{\text{giao}}$ là đoạn chồng nhau theo chiều cao.

**AP** (Average Precision): diện tích dưới đường precision–recall, với precision được lấy giá trị lớn nhất ở các mức recall cao hơn:

$$
\text{AP} = 100 \int_0^1 \max\{p(r') \mid r' \ge r\}\, dr
$$

**APH** (AP weighted by Heading): tính như AP, nhưng mỗi box đúng (TP) chỉ được tính theo độ chính xác hướng:

$$
w = 1 - \frac{\min(|\tilde\theta - \theta|,\ 2\pi - |\tilde\theta - \theta|)}{\pi}
$$

Box đúng hướng có $w = 1$; lệch 90° có $w = 0{,}5$; ngược hướng 180° có $w = 0$, tức vẫn là TP với AP nhưng không được điểm với APH. Metric chính của paper CTRL và bảng xếp hạng Waymo là **APH LEVEL_2**.

### 8.3 Ví dụ: box lệch bao nhiêu thì bị tính sai

Hai box xe giống nhau, dài 4 m, rộng 2 m, cao 1,5 m (thể tích 12 m³), lệch nhau 1 m theo chiều dài xe:

$$
V_{\text{giao}} = 3 \times 2 \times 1{,}5 = 9, \qquad \text{IoU} = \frac{9}{12 + 12 - 9} = 0{,}6 < 0{,}7
$$

Lệch 1 m đã đủ để box bị tính sai. Cùng mức lệch này trên nuScenes vẫn được tính đúng (ngưỡng 2 m, xem mục 7).

### 8.4 Ví dụ trên kết quả thực nghiệm (vehicle, 202 chuỗi validation)

| Kết quả | L1 AP / APH | L2 AP / APH | Trong paper |
|---|---|---|---|
| FSD (detector nền, file của tác giả CTRL) | 82,79 / 82,31 | 75,13 / 74,69 | Table 4: 74,9 / 74,4 |
| CTRL (file kết quả của tác giả, có Track TTA) | 88,47 / 87,48 | 82,28 / 81,31 | Table 1: 82,3 / 81,3 |

GT sau lọc: 956.767 box LEVEL_1, 166.606 box LEVEL_2; bỏ 121.683 box không có điểm LiDAR. Khoảng cách giữa AP và APH (1,0 điểm với CTRL ở LEVEL_2) là phần mất do sai hướng.

### 8.5 So sánh với nuScenes

| | nuScenes (mục 1–7) | Waymo (cách paper CTRL chấm) |
|---|---|---|
| Loại metric | Tracking | Detection |
| Metric chính | AMOTA | APH LEVEL_2 |
| Ghép box với GT | Khoảng cách tâm ≤ 2 m | IoU 3D ≥ 0,7 (vehicle), ≥ 0,5 (pedestrian, cyclist) |
| Kích thước, hướng | Không ảnh hưởng ghép; hướng không được chấm | Kích thước ảnh hưởng IoU; sai hướng bị phạt trong APH |
| ID track | Có tính (IDS, FRAG) | Không tính |
| Score dùng để xếp hạng | Score trung bình cả track | Score từng box |
| Mức khó | Không chia | LEVEL_1, LEVEL_2 |
| GT bị loại | Không có điểm LiDAR/radar; ngoài 40–50 m | Không có điểm LiDAR |
| Tần số nhãn | 2 Hz (keyframe) | 10 Hz (mọi frame) |
| Kết quả tổng | Trung bình 7 lớp | Từng lớp riêng |

Hệ quả:

- Trên nuScenes, cải thiện ở bước hậu xử lý track thể hiện qua IDS, FRAG và AMOTP; AMOTA ít thay đổi vì box lệch vài chục cm không bị phạt.
- Trên Waymo, ID không được chấm; điểm chỉ thay đổi khi box chính xác hơn (vị trí, kích thước, hướng). Vì vậy bước refine của CTRL tăng khoảng 6 điểm APH, còn bước nối track gần như không ảnh hưởng.
- Điểm của hai bộ chấm không so trực tiếp với nhau được.

### 8.6 Metric tracking của Waymo

Waymo cũng có bộ chấm tracking riêng (dùng cho bảng xếp hạng tracking và trong paper ImmortalTracker): **MOTA, MOTP, mismatch (đổi ID), FP, miss** theo từng lớp và mức khó, ghép theo IoU 3D như detection. Paper CTRL không báo cáo các metric này. Repo hiện chưa chấm metric tracking trên Waymo.

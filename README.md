# 3D offline tracking trên nuScenes và Waymo

Tái lập kết quả [ImmortalTracker](https://github.com/ImmortalTracker/ImmortalTracker) trên tập val của nuScenes, các bước xử lý offline trên track sau khi tracking (không dùng mạng refine), và tái lập [CTRL](https://github.com/tusen-ai/SST) trên Waymo Open Dataset để so sánh với phương pháp đề xuất.

## Kết quả chính

Tái lập ImmortalTracker trên nuScenes val (150 scene), detection CenterPoint không flip, code gốc không sửa:

| Metric | Paper (Table 5) | Tái lập (B0) |
|---|---|---|
| AMOTA (%) | 70.2 | 70.5 |
| MOTA (%) | 60.1 | 60.0 |
| IDS | 385 | 383 |

Xử lý offline trên track, 75 scene test (nửa của val, không dùng để chỉnh tham số):

| Cấu hình | AMOTA | AMOTP (m) | FRAG | IDS | Box lệch hướng > 90° |
|---|---|---|---|---|---|
| B0 (ImmortalTracker) | 0.685 | 0.546 | 246 | 214 | 7,44% |
| B0 + refine rút gọn (train trên 75 scene dev) | 0.691 | 0.553 | 250 | 167 | 8,19% |
| Nối track + làm mượt (không train) | 0.691 | 0.515 | 168 | 183 | 6,44% |
| Kéo dài hai chiều + refine (các bước kiểu CTRL) | 0.679 | 0.542 | 225 | 159 | 7,80% |

Waymo, tập validation, lớp vehicle, APH LEVEL_2 (cách paper CTRL chấm):

| Cấu hình | Số chuỗi | APH L2 | Đối chiếu |
|---|---|---|---|
| CTRL, file kết quả của tác giả (Track TTA) | 202 | 81,31 | Paper, Table 1: 81,3 |
| CTRL chạy lại bằng checkpoint của tác giả (Track TTA) | 8 | 83,29 | Kết quả của tác giả trên cùng 8 chuỗi: 83,46 |
| CTRL chạy lại (không TTA) | 8 | 82,53 | |
| Nối track + làm mượt, rồi refine của CTRL (không TTA) | 8 | 83,42 | +0,89 so với CTRL không TTA |

Bảng đầy đủ và phân tích: [`RESULTS.md`](RESULTS.md), [`offline/notes/03_waymo_tai_lap_ctrl.md`](offline/notes/03_waymo_tai_lap_ctrl.md).

## Cấu trúc thư mục

```
.
├── repro/                 tái lập ImmortalTracker
│   ├── setup_env.sh       tạo môi trường Python 3.10 (~/imm_env) và 2 bản vá môi trường
│   ├── make_val_slim.py   tạo metadata rút gọn chỉ gồm 150 scene val
│   ├── run_chunk.py       chạy tracking theo đợt cho một class, bỏ qua scene đã xong
│   ├── run_rest.sh        chạy tracking cả 7 class → ghép kết quả → chấm điểm
│   └── compare_paper.py   in bảng so với paper từ metrics_summary.json
├── baseline_B0/           kết quả tái lập dùng làm mốc B0 (metric, log, md5 file detection)
├── offline/               các bước xử lý offline trên track, xem offline/README.md
├── waymo/                 tái lập CTRL và chạy phương pháp đề xuất trên Waymo, xem waymo/README.md
├── tools/                 file .bat chạy các bước trong Docker trên Windows
├── docs/                  tài liệu
└── RESULTS.md             bảng kết quả
```

Không có trong repo (tự chuẩn bị):

| Thư mục | Nội dung | Nguồn |
|---|---|---|
| `ImmortalTracker/` | Repo gốc, commit ghi trong `ImmortalTracker_COMMIT.txt` | github.com/ImmortalTracker/ImmortalTracker |
| `data/nuscenes/` | nuScenes v1.0-trainval, phần metadata và maps | nuscenes.org |
| `data/nuscenes_valslim/` | Metadata rút gọn 150 scene val | `python repro/make_val_slim.py <step>` |
| `detections/` | `cp_val_noflip.json`, `cp_val_flip.json`: detection CenterPoint trên val | Bản do tác giả CenterPoint công bố |
| `data/waymo/` | GT, điểm LiDAR, kết quả và checkpoint CTRL | Waymo Open Dataset v2.0.1; hướng dẫn CTRL trong repo tusen-ai/SST, xem `waymo/README.md` |

## Môi trường

Chỉ cần CPU. Chạy trong Docker:

```powershell
docker run -it --rm -v <thư mục project>:/repro -w /repro python:3.10 bash
```

Trong container:

```bash
bash repro/setup_env.sh && source ~/imm_env/bin/activate && pip install ijson
```

Bản vá môi trường (không sửa code repo gốc):

- **PyYAML ≥ 6:** code gốc gọi `yaml.load(f)` không truyền Loader; xử lý bằng `_yamlshim.py` và `zz_yamlshim.pth` trong site-packages.
- **motmetrics 1.1.3 trên Python 3.10:** đổi `collections.Iterable` thành `collections.abc.Iterable`.
- **pandas < 2:** nuscenes-devkit 1.1.11 lỗi khi gộp scene với pandas 2.x.

## Tái lập ImmortalTracker

```bash
cd /repro/ImmortalTracker
bash preparedata/nuscenes/nu_convert_detection.sh ../detections/cp_val_noflip.json cp
cd /repro
PROC=$(nproc) bash repro/run_rest.sh immortal_noflip
```

- `run_chunk.py` đọc detection ở `ImmortalTracker/data/nuscenes/validation_2hz/detection/cp`; muốn đổi bộ detection thì convert lại vào `cp` trước khi chạy.
- Thời gian tham khảo: tracking 30–40 phút với 12 nhân CPU; chấm điểm khoảng 1 giờ (một nhân).
- Bị ngắt giữa chừng: dọn các file `.npz` hỏng rồi chạy lại `run_rest.sh`; script bỏ qua scene đã xong.

```bash
python - <<'P'
import glob, numpy as np, os
for f in glob.glob('ImmortalTracker/mot_results/nuscenes/validation_2hz/*/summary/*/*.npz'):
    try: np.load(f, allow_pickle=True)['ids']
    except Exception: print('hong:', f); os.rename(f, f + '.broken')
P
```

Chỉ chạy lại phần chấm điểm:

```bash
OUT=ImmortalTracker/mot_results/nuscenes/validation_2hz/<tên>/results
python -m nuscenes.eval.tracking.evaluate $OUT/results.json --output_dir $OUT --eval_set val \
  --dataroot data/nuscenes_valslim --version v1.0-trainval
python repro/compare_paper.py $OUT/metrics_summary.json
```

`nuscenes_valslim` giữ nguyên mọi bản ghi của 150 scene val (sample, sample_annotation, keyframe sample_data, ego_pose), nên điểm không đổi so với dùng metadata đầy đủ.

## Tài liệu

| File | Nội dung |
|---|---|
| [`docs/tracking_metrics.md`](docs/tracking_metrics.md) | Metric tracking của nuScenes (định nghĩa, công thức, cách bộ chấm tính, thí nghiệm) và metric detection của Waymo (AP/APH) |
| [`docs/giai_thich_pipeline.md`](docs/giai_thich_pipeline.md) | Pipeline 3D auto-labeling offline: detection, tracking, refine |
| [`docs/so_sanh_ctrl.md`](docs/so_sanh_ctrl.md) | So sánh các bước của CTRL với cách làm trong repo |
| [`docs/refine.md`](docs/refine.md) | Refine rút gọn trên LiDAR keyframe: dữ liệu, model, train, kết quả |
| [`RESULTS.md`](RESULTS.md) | Kết quả tái lập và các thử nghiệm offline |
| [`offline/notes/`](offline/notes/) | Ghi chép quá trình: phân tích lỗi B0, sửa luật lật hướng, tái lập CTRL trên Waymo |
| [`waymo/README.md`](waymo/README.md) | Các bước chạy trên Waymo |

## Tham khảo

| Paper | Ghi chú |
|---|---|
| ImmortalTracker, arXiv 2111.13672 | nuScenes val (Table 5): AMOTA 70.2 / MOTA 60.1 / IDS 385 |
| CTRL, arXiv 2304.12315 | ImmortalTracker + kéo dài track về quá khứ + mạng refine theo track; chỉ báo cáo trên Waymo (Table 1, vehicle L2 APH: 81,0 không TTA, 81,3 Track TTA) |
| Offline Tracking with Object Permanence, arXiv 2310.01288 | Re-ID và hoàn thiện track bằng mạng học |

Dữ liệu nuScenes và Waymo Open Dataset chỉ dùng cho mục đích phi thương mại theo license của từng bộ dữ liệu; repo không chứa dữ liệu, file kết quả hay checkpoint của hai bộ dữ liệu này.

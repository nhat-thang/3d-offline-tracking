# 3D offline tracking trên nuScenes

Tái lập kết quả [ImmortalTracker](https://github.com/ImmortalTracker/ImmortalTracker) trên tập val của nuScenes, và các bước xử lý offline trên track sau khi tracking (không dùng mạng refine).

## Kết quả chính

Tái lập ImmortalTracker trên nuScenes val (150 scene), detection CenterPoint không flip, code gốc không sửa:

| Metric | Paper (Table 5) | Tái lập (B0) |
|---|---|---|
| AMOTA (%) | 70.2 | 70.5 |
| MOTA (%) | 60.1 | 60.0 |
| IDS | 385 | 383 |

Kết quả các bước offline: [`RESULTS.md`](RESULTS.md).

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
| [`docs/tracking_metrics.md`](docs/tracking_metrics.md) | Metric tracking của nuScenes: định nghĩa, công thức, cách bộ chấm tính |
| [`docs/giai_thich_pipeline.md`](docs/giai_thich_pipeline.md) | Pipeline 3D auto-labeling offline: detection, tracking, refine |
| [`docs/so_sanh_ctrl.md`](docs/so_sanh_ctrl.md) | So sánh các bước của CTRL với cách làm trong repo |
| [`RESULTS.md`](RESULTS.md) | Kết quả tái lập và các thử nghiệm offline |

## Tham khảo

| Paper | Ghi chú |
|---|---|
| ImmortalTracker, arXiv 2111.13672 | nuScenes val (Table 5): AMOTA 70.2 / MOTA 60.1 / IDS 385 |
| CTRL, arXiv 2304.12315 | ImmortalTracker + kéo dài track hai chiều + mạng refine theo track; chỉ báo cáo trên Waymo |
| Offline Tracking with Object Permanence, arXiv 2310.01288 | Re-ID và hoàn thiện track bằng mạng học |

Dữ liệu nuScenes chỉ dùng cho mục đích phi thương mại theo license của nuScenes; repo không chứa dữ liệu nuScenes.

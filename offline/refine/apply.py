"""Bước 7a (phần ghi kết quả, cần nuscenes-devkit, không cần PyTorch): áp kết quả model lên track rồi ghi results.json.
  - Chỉnh box: tâm += độ lệch đoán (đổi từ hệ box ra toàn cục), kích thước *= exp(dlog), hướng += dyaw.
  - Chấm lại score: mọi box của track nhận score = xác suất track thật do model đoán
    (evaluator nuScenes dùng score trung bình của cả track, nên score theo track là thứ quyết định).
  - Box nằm ngoài tầm chấm (không có trong dữ liệu) giữ nguyên hình học, nhưng vẫn nhận score mới của track.
Chỉ ghi các scene có trong file dữ liệu, để mọi phiên bản được chấm trên cùng tập scene.
Chạy: python offline/refine/apply.py B0 test --data offline/results/refine/data/B0_test.npz \
          --pred offline/results/refine/pred/B0_s0__B0_test.npz --out_tag B0_refine_s0 [--no_box] [--no_score]
Chấm điểm: python offline/scripts/eval_split.py offline/results/B0_refine_s0/results_test.json \
          offline/results/refine/data/B0_test_meta.json offline/results/eval/B0_refine_s0_test"""
import os, sys, json, argparse
import numpy as np
from pyquaternion import Quaternion
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts')); sys.path.insert(0, HERE)
import common
from make_dataset import load_input_tracks, yaw_of, wrap

ap = argparse.ArgumentParser()
ap.add_argument('src', help="'B0' hoặc results_<part>.json đã dùng để tạo dữ liệu")
ap.add_argument('part', choices=['dev', 'test'])
ap.add_argument('--data', required=True); ap.add_argument('--pred', required=True); ap.add_argument('--out_tag', required=True)
ap.add_argument('--no_box', action='store_true', help='không chỉnh box, chỉ chấm lại score')
ap.add_argument('--no_score', action='store_true', help='không chấm lại score, chỉ chỉnh box')
p = ap.parse_args()

from nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
meta = json.load(open(p.data.replace('.npz', '_meta.json')))
d, pr = np.load(p.data), np.load(p.pred)
REG, PTRK = pr['reg'], pr['ptrk']   # đọc một lần: mỗi lần gọi pr['reg'] numpy đọc lại cả mảng từ file .npz (rất chậm)
keep = set(meta['scenes'])
tracks = load_input_tracks(p.src, p.part, where, set(common.load_split()[p.part]))
tracks = {sc: v for sc, v in tracks.items() if sc in keep}
ptr, frame, tsc = d['ptr'], d['frame'], d['tscene']
n_box = 0
for t, tid in enumerate(meta['track_ids']):
    sc = meta['scenes'][tsc[t]]
    seq = tracks[sc].get(tid)
    if seq is None: seq = tracks[sc][type(next(iter(tracks[sc])))(tid)]   # id gốc có thể là số
    by_frame = {f: b for f, _, b in seq}
    if not p.no_score:
        for _, _, b in seq: b['tracking_score'] = float(PTRK[t])
    if p.no_box: continue
    for i in range(ptr[t], ptr[t + 1]):
        b = by_frame[int(frame[i])]; r = REG[i]
        c = np.array(b['translation'], float); yaw = yaw_of(b['rotation'])
        cy, sy = np.cos(yaw), np.sin(yaw)
        c += [r[0] * cy - r[1] * sy, r[0] * sy + r[1] * cy, r[2]]
        b['translation'] = [float(v) for v in c]
        b['size'] = [float(v) for v in np.array(b['size']) * np.exp(r[3:6])]
        b['rotation'] = list(Quaternion(axis=[0, 0, 1], angle=wrap(yaw + np.arctan2(r[6], r[7]))).q)
        n_box += 1
for sc in tracks:
    for tid, seq in tracks[sc].items():
        for f, t_, b in seq: b['sample_token'] = scenes[sc][f]
out = os.path.join(common.ROOT, 'offline', 'results', p.out_tag, f'results_{p.part}.json')
n = common.save_tracks(tracks, scenes, out, keep)
print(f'xong: {out} | {len(keep)} scene, {n} box ghi ra, {n_box} box duoc chinh')

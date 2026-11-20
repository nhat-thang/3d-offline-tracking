"""Đo xem bước làm mượt (d) sửa đúng hay làm sai ở những tình huống khó, so với GT (chỉ dùng tập dev).
Ghép box thật của B0 với GT cùng lớp mỗi frame (Hungarian, tâm < 2 m), rồi so cùng box TRƯỚC và SAU khi làm mượt:
  - lật hướng: box đúng hướng bị làm lật (làm sai) / box lật được sửa đúng; tách riêng xe đang LÙI
    (GT có vận tốc > 0,5 m/s ngược hướng đầu xe);
  - xe RẼ GẮT (GT đổi hướng > 20 độ/giây): sai số tâm trước/sau;
  - TRAILER, BUS: sai số kích thước (1 - IoU thể tích căn giữa) trước/sau, so với các lớp khác.
Chạy: python offline/scripts/diag_smooth.py [tham số làm mượt, ví dụ --flip_mode majority --turn_rate 20 --local_size trailer,bus]"""
import sys, os, copy, pickle, argparse, collections
import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(__file__)); import common, pipeline
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingBox

ap = argparse.ArgumentParser()
for k, v in dict(sw=2, sdeg=1, ssig=1.0, flip_mode='motion', flip_vote=0.7).items(): ap.add_argument('--' + k, type=type(v), default=v)
p = ap.parse_args()
cfg = config_factory('tracking_nips_2019')
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
before = pickle.load(open(os.path.expanduser('~/scr/b0_tracks_dev.pkl'), 'rb'))
after = copy.deepcopy(before)
st = pipeline.smooth(after, p)
yaw = lambda q: Quaternion(q).yaw_pitch_roll[0]
wrap = pipeline.wrap
# GT: (instance) -> {frame: yaw} để tính tốc độ đổi hướng
gyaw = collections.defaultdict(dict)
for tok in gt.sample_tokens:
    sc, fi, ts = where[tok]
    for g in gt.boxes[tok]: gyaw[g.tracking_id][fi] = (ts, yaw(g.rotation))
def yaw_rate(g, fi):
    d = gyaw[g.tracking_id]; r = []
    for f2 in (fi - 1, fi + 1):
        if f2 in d: r.append(abs(wrap(d[f2][1] - d[fi][1])) / abs(d[f2][0] - d[fi][0]))
    return max(r) if r else 0.0
S = collections.Counter(); E = collections.defaultdict(list)
for sc in before:
    toks = scenes[sc]
    per = collections.defaultdict(list)
    for tid, seq in before[sc].items():
        for k, (f, t, b) in enumerate(seq):
            if b['tracking_score'] >= common.PRED_SCORE: per[f].append((tid, k, b))
    for f, items in per.items():
        G = gt.boxes.get(toks[f], [])
        for cls in common.CLASSES:
            P = [x for x in items if x[2]['tracking_name'] == cls]; Gc = [g for g in G if g.tracking_name == cls]
            if not P or not Gc: continue
            D = np.array([[np.linalg.norm(np.array(x[2]['translation'][:2]) - g.translation[:2]) for g in Gc] for x in P])
            for i, j in zip(*linear_sum_assignment(np.where(D < 2, D, 1e6))):
                if D[i, j] >= 2: continue
                tid, k, b0 = P[i]; b1 = after[sc][tid][k][2]; g = Gc[j]; gy = yaw(g.rotation)
                f0 = abs(wrap(yaw(b0['rotation']) - gy)) > np.pi / 2; f1 = abs(wrap(yaw(b1['rotation']) - gy)) > np.pi / 2
                v = np.array(g.velocity[:2]); rev = np.linalg.norm(v) > 0.5 and np.cos(np.arctan2(v[1], v[0]) - gy) < 0
                grp = 'xe_lui' if rev else 'thuong'
                S[(grp, 'n')] += 1; S[(grp, 'lat_truoc')] += f0; S[(grp, 'lat_sau')] += f1
                S[(grp, 'lam_sai')] += (not f0) and f1; S[(grp, 'sua_dung')] += f0 and not f1
                if cls in ('car', 'truck', 'bus', 'trailer') and yaw_rate(g, f) > np.radians(20):
                    E['re_gat_truoc'].append(D[i, j]); E['re_gat_sau'].append(np.linalg.norm(np.array(b1['translation'][:2]) - g.translation[:2]))
                def sz(b):
                    inter = np.prod(np.minimum(b['size'], g.size)); return 1 - inter / (np.prod(b['size']) + np.prod(g.size) - inter)
                key = cls if cls in ('trailer', 'bus') else 'khac'
                E[f'kt_{key}_truoc'].append(sz(b0)); E[f'kt_{key}_sau'].append(sz(b1))
print('tham so:', vars(p))
for grp in ('thuong', 'xe_lui'):
    n = S[(grp, 'n')]
    print(f"{grp:7s} box {n:6d} | lat truoc {S[(grp,'lat_truoc')]/max(n,1):.2%} sau {S[(grp,'lat_sau')]/max(n,1):.2%} | sua dung {S[(grp,'sua_dung')]} lam sai {S[(grp,'lam_sai')]}")
print(f"re gat (>20 do/s): {len(E['re_gat_truoc'])} box, sai so tam truoc {np.mean(E['re_gat_truoc']):.3f} sau {np.mean(E['re_gat_sau']):.3f} m")
for key in ('trailer', 'bus', 'khac'):
    print(f"kich thuoc {key:7s}: {len(E[f'kt_{key}_truoc'])} box, 1-IoU truoc {np.mean(E[f'kt_{key}_truoc']):.3f} sau {np.mean(E[f'kt_{key}_sau']):.3f}")

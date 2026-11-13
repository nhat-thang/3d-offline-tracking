"""Đo độ chính xác của bước (a) nối track: mỗi lần nối track A (trước) với track B (sau), xem A và B bám vật GT nào.
Ghép box thật (score >= PRED_SCORE) với GT cùng lớp mỗi frame bằng Hungarian, ngưỡng tâm 2 m (như bộ chấm nuScenes).
Vật GT của A = vật ghép nhiều nhất trong 3 box thật cuối của A; của B = trong 3 box thật đầu của B.
Phân loại: đúng (cùng vật), SAI (hai vật khác nhau), một phía là nhiễu (không ghép GT).
Chạy: python offline/scripts/link_precision.py dev [tham số pipeline, ví dụ --d0 0.8]"""
import sys, os, pickle, collections, argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(__file__)); import common, pipeline
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingBox
ap = argparse.ArgumentParser(); ap.add_argument('part')
for k, v in dict(k=3, gap=3.0, d0=1.0, d1=1.0, d0p=0.6, d1p=0.6, skip_cls='', min_link_score=0.0).items(): ap.add_argument('--' + k, type=type(v), default=v)
p = ap.parse_args()
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
tracks = pickle.load(open(os.path.expanduser(f'~/scr/b0_tracks_{p.part}.pkl'), 'rb'))
cfg = config_factory('tracking_nips_2019')
gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
# (scene, tid, frame) -> GT instance, tính trên track gốc B0 (trước khi nối)
inst = {}
for sc, trs in tracks.items():
    toks = scenes[sc]
    by = collections.defaultdict(list)
    for tid, seq in trs.items():
        for f, t, b in seq:
            if b['tracking_score'] >= common.PRED_SCORE: by[(f, b['tracking_name'])].append((tid, np.array(b['translation'][:2])))
    for (f, c), lst in by.items():
        g = [x for x in gt.boxes.get(toks[f], []) if x.tracking_name == c]
        if not g: continue
        D = np.linalg.norm(np.array([x.translation[:2] for x in g])[:, None] - np.array([y for _, y in lst])[None], axis=2)
        C = np.where(D <= 2, D, 1e6)
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < 1e6: inst[(sc, lst[j][0], f)] = g[i].tracking_id
def who(sc, tid, end):
    r = pipeline.real_boxes(tracks[sc][tid]); r = r[-3:] if end else r[:3]
    v = [inst.get((sc, tid, f)) for f, _, _ in r]; v = [x for x in v if x]
    return collections.Counter(v).most_common(1)[0][0] if v else None
score = {(sc, tid): np.mean([b['tracking_score'] for _, _, b in pipeline.real_boxes(seq)]) if pipeline.real_boxes(seq) else 0
         for sc, trs in tracks.items() for tid, seq in trs.items()}
ia = {k: who(*k, True) for k in score}; ib = {k: who(*k, False) for k in score}
pipeline.LINKS.clear()
pipeline.reassociate(tracks, p)
R = collections.Counter(); bycls = collections.defaultdict(collections.Counter); bysc = collections.defaultdict(collections.Counter)
for sc, c, a, b, cost, dt in pipeline.LINKS:
    A, B = ia[(sc, a)], ib[(sc, b)]
    k = 'nhieu' if A is None or B is None else ('dung' if A == B else 'SAI')
    R[k] += 1; bycls[c][k] += 1
    s = min(score[(sc, a)], score[(sc, b)]); bysc[min(int(s * 10) / 10, 0.6)][k] += 1
tot = sum(R.values())
print('tong lan noi', tot, dict(R), ' ti le dung tren cac lan noi giua 2 vat that: %.3f' % (R['dung'] / max(1, R['dung'] + R['SAI'])))
for c in sorted(bycls): x = bycls[c]; print('  ', c, dict(x), ' dung/(dung+sai)=%.2f' % (x['dung'] / max(1, x['dung'] + x['SAI'])))
print(' theo score nho hon cua 2 track:')
for s in sorted(bysc): x = bysc[s]; print('   >=%.1f' % s, dict(x), ' dung/(dung+sai)=%.2f' % (x['dung'] / max(1, x['dung'] + x['SAI'])))

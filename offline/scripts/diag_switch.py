"""Đo 5 nhiệm vụ của tracking theo góc nhìn của bước refine: KHÔNG lọc ngưỡng score, KHÔNG nội suy
(giữ nguyên track như tracker xuất ra). Lọc GT giống bộ chấm nuScenes (tầm 40/50 m, bỏ box không có điểm LiDAR/radar,
bỏ xe đạp trong bãi); box dự đoán cũng lọc theo tầm. Ghép theo khoảng cách tâm <= 2 m, giữ cặp ghép cũ nếu còn hợp lệ.
Chạy (từ thư mục immortal_repro):  python offline/scripts/measure_tracking.py <results.json> <dev|test|val> [tên]
Đầu ra: offline/results/measure/<tên>_<phần>.json và in một dòng bảng."""
import sys, os, json, collections
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(__file__))
import common
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingBox

res_path, part = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.dirname(os.path.abspath(res_path)))
cfg = config_factory('tracking_nips_2019')  # gọi trước load_gt
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
keep = set(scenes) if part == 'val' else set(common.load_split()[part])
gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
ego = {}
for sc in keep:
    for tok in scenes[sc]:
        sd = nusc.get('sample_data', nusc.get('sample', tok)['data']['LIDAR_TOP'])
        ego[tok] = np.array(nusc.get('ego_pose', sd['ego_pose_token'])['translation'][:2])
tracks = common.load_tracks(res_path, where, keep)
S = collections.Counter(); err = []; EV = []
for sc in sorted(keep):
    toks = scenes[sc]
    pf = [collections.defaultdict(list) for _ in toks]
    for tid, seq in tracks.get(sc, {}).items():
        for fi, ts, b in seq:
            c = b['tracking_name']; xy = np.array(b['translation'][:2])
            if np.linalg.norm(xy - ego[toks[fi]]) > cfg.class_range[c]: continue
            pf[fi][c].append((tid, xy))
    for c in common.CLASSES:
        seq = collections.defaultdict(list); last = {}
        for fi, tok in enumerate(toks):
            g = [b for b in gt.boxes.get(tok, []) if b.tracking_name == c]
            p = pf[fi][c]
            S['pred_boxes'] += len(p)
            matched = {}
            if g and p:
                G = np.array([b.translation[:2] for b in g]); P = np.array([x for _, x in p])
                D = np.linalg.norm(G[:, None] - P[None], axis=2)
                pidx = {tid: j for j, (tid, _) in enumerate(p)}; used = set()
                for i, b in enumerate(g):
                    j = pidx.get(last.get(b.tracking_id))
                    if j is not None and j not in used and D[i, j] <= 2.0:
                        matched[i] = j; used.add(j)
                rg = [i for i in range(len(g)) if i not in matched]; rp = [j for j in range(len(p)) if j not in used]
                if rg and rp:
                    C = np.where(D[np.ix_(rg, rp)] <= 2.0, D[np.ix_(rg, rp)], 1e6)
                    for i, j in zip(*linear_sum_assignment(C)):
                        if C[i, j] < 1e6: matched[rg[i]] = rp[j]
                for i, j in matched.items(): err.append(D[i, j])
                um = [j for j in range(len(p)) if j not in set(matched.values())]
                for j in um:
                    if D[:, j].min() <= 2.0: S['dup_boxes'] += 1      # box thừa sát một vật đã có track bám
                    else: S['fp_far'] += 1                             # box không gần vật thật nào
            else:
                S['fp_far'] += len(p)
            for i, b in enumerate(g):
                pid = p[matched[i]][0] if i in matched else None
                if pid is not None: last[b.tracking_id] = pid
                seq[b.tracking_id].append(pid)
        for inst, s in seq.items():
            prev=None
            for fi2,x in enumerate(s):
                if x is not None:
                    if prev is not None and x!=prev: EV.append((sc,c,fi2,prev,x))
                    prev=x
            S['gt_instances'] += 1; S['gt_boxes'] += len(s)
            hit = [k for k, x in enumerate(s) if x is not None]
            if not hit: S['inst_no_track'] += 1; S['fn_no_track'] += len(s); continue
            S['tp'] += len(hit)
            ids = [x for x in s if x is not None]
            sw = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
            S['id_switches'] += sw; S['inst_multi_id'] += int(len(set(ids)) > 1)
            S['fn_head'] += hit[0]; S['fn_tail'] += len(s) - 1 - hit[-1]
            S['fn_mid'] += sum(b - a - 1 for a, b in zip(hit, hit[1:]))
            S['gaps_mid'] += sum(1 for a, b in zip(hit, hit[1:]) if b - a > 1)

import pickle
info = {}
for sc in keep:
    for tid, seq in tracks.get(sc, {}).items():
        r = [(f, b['tracking_score']) for f, t, b in seq if b['tracking_score'] >= common.PRED_SCORE]
        allf = [f for f, t, b in seq]
        info[(sc, tid)] = (r[0][0] if r else None, r[-1][0] if r else None, allf[0], allf[-1], np.mean([s for _, s in r]) if r else 0, len(r))
cat = collections.Counter(); gaps = []
rows = []
for sc, c, fi, a, b in EV:
    A = info[(sc, a)]; B = info[(sc, b)]
    if A[1] is None or B[0] is None: k = 'pred_only'
    elif B[0] > A[1]: k = 'fragment'; gaps.append(B[0] - A[1])
    elif B[0] <= A[1] and B[1] >= A[0] and A[1] >= fi: k = 'overlap_old_alive'
    else: k = 'overlap_other'
    cat[k] += 1; rows.append((sc, c, fi, a, b, k, A, B))
print(cat)
print('gap frames hist', collections.Counter(gaps).most_common(20))
pickle.dump(rows, open('/tmp/sw_' + name + '.pkl', 'wb'))

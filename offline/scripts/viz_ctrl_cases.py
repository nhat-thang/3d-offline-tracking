"""Trang HTML minh họa các trường hợp của khâu tracking kiểu CTRL trên 2 scene val có LiDAR (scene-0103, scene-0916, từ nuscenes_mini).
Track = B0 (ImmortalTracker gốc) + kéo dài hai chiều kiểu CTRL (như B2), KHÔNG có refine.
Mỗi box được gắn loại:
  - box thật (có detection), box đoán của ImmortalTracker khi vật bị che (score < PRED_SCORE);
  - box kéo dài: 'same' trùng đúng vật của track | 'other' trùng vật khác | 'zeropt' gần vật thật không có điểm LiDAR
    (bộ chấm nuScenes không tính) | 'empty' chỗ không có vật.
Sự kiện: giữ ID qua đoạn bị che, đổi ID (vật GT bị đổi track).
Chạy (từ thư mục immortal_repro):  python offline/scripts/viz_ctrl_cases.py
Đầu ra: offline/results/viz/ctrl_cases_mini.html (tự chứa; dữ liệu nuScenes, chỉ dùng nội bộ)."""
import sys, os, json, copy, collections
import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(__file__))
import common, baselines
from analyze_b0 import best_thresholds
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.eval.tracking.utils import category_to_tracking_name

SC = ['scene-0103', 'scene-0916']
MINI = os.path.join(common.ROOT, 'data', 'nuscenes_mini')
NPTS = 9000
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
thr = best_thresholds()
b0 = {sc: dict(v) for sc, v in common.load_tracks(common.B0, where, set(SC)).items()}
ext = copy.deepcopy(b0)
orig = {(sc, tid, f) for sc in b0 for tid, seq in b0[sc].items() for f, _, _ in seq}
baselines.b2(ext, scenes, where)
yaw = lambda q: float(Quaternion(q).yaw_pitch_roll[0])
r2 = lambda v: round(float(v), 2)
out = {'cls': common.CLASSES, 'scenes': []}
RAW = {}  # box hệ toàn cục cho video 6 camera: {sample_token: [[translation, size, rotation, kind, id], ...]}
for sc in SC:
    toks = scenes[sc]
    mean = {tid: np.mean([b['tracking_score'] for *_, b in seq]) for tid, seq in b0[sc].items()}
    cls_of = {tid: collections.Counter(b['tracking_name'] for *_, b in seq).most_common(1)[0][0] for tid, seq in b0[sc].items()}
    shown = {tid for tid in b0[sc] if mean[tid] >= thr[cls_of[tid]]}
    # GT mỗi frame: (inst, cls, xy, size, yaw, có điểm?)
    G = []
    for tok in toks:
        g = []
        for a in nusc.get('sample', tok)['anns']:
            ann = nusc.get('sample_annotation', a)
            c = category_to_tracking_name(ann['category_name'])
            if c is None: continue
            g.append((ann['instance_token'], c, np.array(ann['translation'][:2]), ann['size'], yaw(ann['rotation']), ann['num_lidar_pts'] + ann['num_radar_pts'] > 0))
        G.append(g)
    # ghép box của B0 (thật + đoán) với GT có điểm, Hungarian 2 m theo lớp
    match = {}  # (tid, f) -> inst
    for f in range(len(toks)):
        for c in common.CLASSES:
            g = [x for x in G[f] if x[1] == c and x[5]]
            p = [(tid, np.array(b['translation'][:2])) for tid, seq in b0[sc].items() if tid in shown for ff, _, b in seq if ff == f and b['tracking_name'] == c]
            if not g or not p: continue
            D = np.linalg.norm(np.array([x[2] for x in g])[:, None] - np.array([y for _, y in p])[None], axis=2)
            C = np.where(D <= 2, D, 1e6)
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6: match[(p[j][0], f)] = g[i][0]
    main = {}
    for tid in shown:
        v = [match.get((tid, f)) for f, _, b in b0[sc][tid] if b['tracking_score'] >= common.PRED_SCORE]
        v = [x for x in v if x]
        main[tid] = collections.Counter(v).most_common(1)[0][0] if v else None
    ids = {tid: k for k, tid in enumerate(sorted(shown, key=lambda t: (b0[sc][t][0][0], t)))}
    inst_ids = {}
    cnt = collections.Counter(); events = []
    frames = []
    for f, tok in enumerate(toks):
        sd = nusc.get('sample_data', nusc.get('sample', tok)['data']['LIDAR_TOP'])
        ep = nusc.get('ego_pose', sd['ego_pose_token']); cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        E = np.array(ep['translation'][:2]); ey = yaw(ep['rotation']); R = np.array([[np.cos(-ey), -np.sin(-ey)], [np.sin(-ey), np.cos(-ey)]])
        loc = lambda xy: R @ (np.asarray(xy) - E)
        pc = LidarPointCloud.from_file(os.path.join(MINI, sd['filename']))
        pc.rotate(Quaternion(cs['rotation']).rotation_matrix); pc.translate(np.array(cs['translation']))
        P = pc.points[:3]; keep = (np.abs(P[0]) < 55) & (np.abs(P[1]) < 55) & (P[2] > 0.3) & (P[2] < 2.5)
        P = P[:2, keep]; idx = np.random.RandomState(f).choice(P.shape[1], min(NPTS, P.shape[1]), replace=False)
        pts = np.round(P[:, idx].T, 1).flatten().tolist()
        gt = [[*map(r2, loc(x[2])), r2(x[3][1]), r2(x[3][0]), r2(x[4] - ey), common.CLASSES.index(x[1]), inst_ids.setdefault(x[0], len(inst_ids)), int(x[5])] for x in G[f]]
        tr = []
        for tid in shown:
            for ff, _, b in ext[sc][tid]:
                if ff != f: continue
                xy = np.array(b['translation'][:2]); c = b['tracking_name']
                if (sc, tid, f) in orig:
                    kind = 'real' if b['tracking_score'] >= common.PRED_SCORE else 'pred'
                else:
                    near = [(np.linalg.norm(x[2] - xy), x) for x in G[f] if x[1] == c]
                    vis = [(d, x) for d, x in near if x[5] and d <= 2]; zero = [(d, x) for d, x in near if not x[5] and d <= 2]
                    if vis:
                        x = min(vis, key=lambda z: z[0])[1]; kind = 'same' if x[0] == main[tid] else 'other'
                    else: kind = 'zeropt' if zero else 'empty'
                    cnt[kind] += 1
                RAW.setdefault(tok, []).append([b['translation'], b['size'], b['rotation'], kind, ids[tid]])
                tr.append([*map(r2, loc(xy)), r2(b['size'][1]), r2(b['size'][0]), r2(yaw(b['rotation']) - ey), common.CLASSES.index(c), ids[tid], kind])
        frames.append({'pts': pts, 'gt': gt, 'tr': tr})
    # sự kiện: giữ ID qua đoạn bị che
    for tid in shown:
        real = [f for f, _, b in b0[sc][tid] if b['tracking_score'] >= common.PRED_SCORE]
        for a, b in zip(real, real[1:]):
            if b - a > 1 and match.get((tid, a)) and match.get((tid, a)) == match.get((tid, b)):
                events.append({'type': 'keep', 'f': a + 1, 'f2': b, 'tid': ids[tid], 'txt': f'Track {ids[tid]} giữ nguyên ID qua {b - a - 1} frame bị mất detection (frame {a + 1}–{b - 1})'})
    # sự kiện: đổi ID
    seqs = collections.defaultdict(list)
    for (tid, f), inst in match.items(): seqs[inst].append((f, tid))
    for inst, s in seqs.items():
        s.sort()
        for (f1, t1), (f2, t2) in zip(s, s[1:]):
            if t1 != t2:
                events.append({'type': 'switch', 'f': f2, 'tid': ids[t2], 'tid0': ids[t1], 'txt': f'Vật GT {inst_ids.get(inst, "?")} đổi từ track {ids[t1]} sang track {ids[t2]} ở frame {f2}'})
    for e in [e for e in events]: pass
    ex_ev = collections.defaultdict(list)
    for f, fr in enumerate(frames):
        for b in fr['tr']:
            if b[7] in ('same', 'other', 'zeropt', 'empty'): ex_ev[(b[6], b[7])].append(f)
    lab = {'same': 'đúng vật của track', 'other': 'trùng VẬT KHÁC', 'zeropt': 'gần vật thật không có điểm LiDAR (bộ chấm không tính)', 'empty': 'chỗ không có vật'}
    for (tid, k), fs in ex_ev.items():
        events.append({'type': 'ext_' + k, 'f': min(fs), 'tid': tid, 'txt': f'Track {tid}: {len(fs)} box kéo dài {lab[k]} (frame {min(fs)}–{max(fs)})'})
    events.sort(key=lambda e: (e['type'], e['f']))
    out['scenes'].append({'name': sc, 'desc': nusc.get('scene', nusc.field2token('scene', 'name', sc)[0])['description'],
                          'frames': frames, 'events': events, 'ext': dict(cnt), 'ntracks': len(shown), 'ninst': len(inst_ids)})
    print(sc, len(frames), 'frame', len(shown), 'track hien thi', dict(cnt), collections.Counter(e['type'] for e in events))
html = open(os.path.join(os.path.dirname(__file__), 'viz_ctrl_template.html'), encoding='utf-8').read().replace('__DATA__', json.dumps(out, ensure_ascii=False, separators=(',', ':')))
p = os.path.join(common.ROOT, 'offline', 'results', 'viz', 'ctrl_cases_mini.html'); open(p, 'w', encoding='utf-8').write(html)
print(p, len(html) // 1024, 'KB')
json.dump(RAW, open(os.path.join(common.ROOT, 'offline', 'results', 'viz', 'ctrl_cases_boxes.json'), 'w'))

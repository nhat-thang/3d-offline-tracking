"""Tìm thời điểm cụ thể trong 2 scene mini (scene-0103, scene-0916) cho 3 điểm yếu mà các bước đề xuất nhắm tới:
 1) track bị cắt (vật GT chuyển sang track mới sau khi track cũ dừng) -> bước nối track; kiểm tra R1b có nối được không
 2) box lật hướng > 90° hoặc kích thước nhảy trong cùng track -> bước làm mượt
 3) track hiện (qua ngưỡng) mà không bám vật nào / vật thật mà track bị ẩn (dưới ngưỡng) -> bước chấm lại độ tin cậy
Thời điểm: frame chỉ số 0 = 0,0 s; video chạy 2 frame/s nên giây trong video = giây trong scene; ô "frame k/N" trên video = chỉ số + 1."""
import sys, os, json, collections, argparse
import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(__file__)); import common, pipeline
from analyze_b0 import best_thresholds
from nuscenes import NuScenes
from nuscenes.eval.tracking.utils import category_to_tracking_name
SC = ['scene-0103', 'scene-0916']
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
thr = best_thresholds()
b0 = {sc: dict(v) for sc, v in common.load_tracks(common.B0, where, set(SC)).items()}
r1 = {sc: dict(v) for sc, v in common.load_tracks(os.path.join(common.ROOT, 'offline/results/R1b/results_dev.json'), where, set(SC)).items()} if False else None
p = argparse.Namespace(k=3, gap=3.0, d0=1.0, d1=1.0, d0p=0.6, d1p=0.6, skip_cls='pedestrian', min_link_score=0.15)
import copy
r1 = copy.deepcopy(b0); pipeline.LINKS.clear(); pipeline.reassociate(r1, p)
links = {(a, b) for sc, c, a, b, *_ in pipeline.LINKS}
yaw = lambda q: Quaternion(q).yaw_pitch_roll[0]
for sc in SC:
    toks = scenes[sc]; trs = b0[sc]
    mean = {t: np.mean([b['tracking_score'] for *_, b in s]) for t, s in trs.items()}
    cls = {t: collections.Counter(b['tracking_name'] for *_, b in s).most_common(1)[0][0] for t, s in trs.items()}
    shown = {t for t in trs if mean[t] >= thr[cls[t]]}
    # số track hiển thị: lấy đúng số đã in trên video/HTML (ctrl_cases_boxes.json), ghép theo vị trí box
    raw = json.load(open(os.path.join(common.ROOT, 'offline/results/viz/ctrl_cases_boxes.json')))
    disp = {}
    for t in shown:
        f, _, b = trs[t][0]
        for tr, *_r, tid in raw.get(toks[f], []):
            if np.allclose(tr, b['translation'], atol=1e-3): disp[t] = tid; break
    G = []
    for tok in toks:
        g = []
        for a in nusc.get('sample', tok)['anns']:
            ann = nusc.get('sample_annotation', a); c = category_to_tracking_name(ann['category_name'])
            if c and ann['num_lidar_pts'] + ann['num_radar_pts'] > 0: g.append((ann['instance_token'], c, np.array(ann['translation'][:2]), yaw(ann['rotation']), ann['size']))
        G.append(g)
    # ghép mọi track (cả track ẩn) với GT, theo frame
    m = {}; gtof = {}
    for f in range(len(toks)):
        for c in common.CLASSES:
            g = [x for x in G[f] if x[1] == c]
            pp = [(t, b) for t, s in trs.items() for ff, _, b in s if ff == f and b['tracking_name'] == c and b['tracking_score'] >= common.PRED_SCORE]
            if not g or not pp: continue
            D = np.linalg.norm(np.array([x[2] for x in g])[:, None] - np.array([b['translation'][:2] for _, b in pp])[None], axis=2)
            # ưu tiên track hiện: cộng phạt nhỏ cho track ẩn
            C = np.where(D <= 2, D + np.array([0 if t in shown else 5 for t, _ in pp])[None], 1e6)
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6: m[(pp[j][0], f)] = g[i][0]; gtof[(pp[j][0], f)] = g[i]
    print(f'\n===== {sc} ({len(toks)} frame, {len(toks) / 2:.1f} s) =====')
    # 1) track bị cắt
    print('[1] TRACK BI CAT (vat GT chuyen sang track moi sau khi track cu dung) -> buoc noi track')
    seq = collections.defaultdict(list)
    for (t, f), inst in m.items():
        if t in shown: seq[inst].append((f, t))
    for inst, s in seq.items():
        s.sort()
        for (f1, t1), (f2, t2) in zip(s, s[1:]):
            if t1 != t2:
                last1 = max(f for f, _, b in trs[t1] if b['tracking_score'] >= common.PRED_SCORE); first2 = min(f for f, _, b in trs[t2] if b['tracking_score'] >= common.PRED_SCORE)
                kind = 'CAT (track cu dung truoc)' if first2 > last1 else 'nhay sang track song song'
                if first2 > last1:
                    ok = 'R1b NOI DUOC' if (t1, t2) in links else 'R1b khong noi'
                    print(f'   {cls[t1]:10s} track {disp[t1]} -> track {disp[t2]}: frame {f1}->{f2} (giay {f1/2:.1f}->{f2/2:.1f}), {kind}, {ok}')
    # 2) lật hướng / nhảy kích thước
    print('[2] LAT HUONG >90 do so voi GT / KICH THUOC NHAY -> buoc lam muot')
    flips = collections.defaultdict(list); sizej = []
    for t in shown:
        rb = [(f, b) for f, _, b in trs[t] if b['tracking_score'] >= common.PRED_SCORE and (t, f) in gtof]
        for f, b in rb:
            d = abs((yaw(b['rotation']) - gtof[(t, f)][3] + np.pi) % (2 * np.pi) - np.pi)
            if d > np.pi / 2 and cls[t] != 'pedestrian': flips[t].append(f)
        vol = [np.prod(b['size']) for f, b in rb]
        if len(vol) > 3:
            for (f1, b1), (f2, b2) in zip(rb, rb[1:]):
                r = np.prod(b2['size']) / np.prod(b1['size'])
                if (r > 1.4 or r < 1 / 1.4) and cls[t] != 'pedestrian': sizej.append((t, f1, f2, r))
    for t, fs in sorted(flips.items(), key=lambda x: -len(x[1]))[:6]:
        print(f'   {cls[t]:10s} track {disp[t]}: lat huong o frame {fs} (giay {[f/2 for f in fs]})')
    for t, f1, f2, r in sorted(sizej, key=lambda x: -abs(np.log(x[3])))[:6]:
        print(f'   {cls[t]:10s} track {disp[t]}: the tich box doi x{r:.2f} giua frame {f1}->{f2} (giay {f1/2:.1f}->{f2/2:.1f})')
    # 3) độ tin cậy
    print('[3] DO TIN CAY SAI -> buoc cham lai do tin cay')
    fp = [(t, len([1 for f, _, b in trs[t] if b['tracking_score'] >= common.PRED_SCORE])) for t in shown if not any((t, f) in m for f, _, b in trs[t])]
    for t, n in sorted(fp, key=lambda x: -x[1])[:6]:
        fs = [f for f, _, b in trs[t]]
        print(f'   HIEN NHUNG SAI: {cls[t]:10s} track {disp[t]} score tb {mean[t]:.2f} (nguong {thr[cls[t]]:.2f}), {n} box, khong bam vat nao, frame {min(fs)}-{max(fs)} (giay {min(fs)/2:.1f}-{max(fs)/2:.1f})')
    hid = []
    for t in trs:
        if t in shown: continue
        mf = [f for f, _, b in trs[t] if (t, f) in m]
        if len(mf) >= 5: hid.append((t, mf))
    for t, mf in sorted(hid, key=lambda x: -len(x[1]))[:6]:
        print(f'   VAT THAT BI AN: {cls[t]:10s} (track an, score tb {mean[t]:.2f} < nguong {thr[cls[t]]:.2f}) bam dung vat {len(mf)} frame, frame {min(mf)}-{max(mf)} (giay {min(mf)/2:.1f}-{max(mf)/2:.1f})')

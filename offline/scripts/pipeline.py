"""Pipeline xử lý offline (hướng của dự án), cộng dồn từng thành phần lên B0:
  R1 = B0 + (a) nối track hai chiều (bidirectional re-association)
  R2 = R1 + (b) kéo dài có điều kiện        [đã thử, bỏ: box kéo dài gần như toàn FP, xem RESULTS.md]
  R3 = R1 + (d) làm mượt cả track (không gồm (b); (c) để sau)
  R3 = R2 + (c) lấp khoảng trống Hermite      [chưa làm]
  R4 = R3 + (d) làm mượt cả track             [chưa làm]
  R5 = R4 + (e) chấm lại score box đoán       [chưa làm]
Chạy (từ thư mục immortal_repro):  python offline/scripts/pipeline.py R1 dev [--tag ten] [--gap 3 --d0 1.0 --d1 1.0]
Đầu ra: offline/results/<tag>/results_<dev|test>.json (hoặc results.json nếu val) + stats_<phần>.json"""
import sys, os, json, argparse, collections
import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(__file__))
import common

STEPS = {'R1': ['a'], 'R2': ['a', 'b'], 'R3': ['a', 'd']}
RANGE = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40}


def velocity(pts):
    """pts: [(t, xy)] -> vận tốc xy (m/s), bình phương tối thiểu; <2 điểm thì 0."""
    if len(pts) < 2: return np.zeros(2)
    t = np.array([p[0] for p in pts]); x = np.array([p[1] for p in pts])
    t = t - t.mean(); den = (t ** 2).sum()
    return (t[:, None] * (x - x.mean(0))).sum(0) / den if den > 0 else np.zeros(2)


def real_boxes(seq):
    return [(f, t, b) for f, t, b in seq if b['tracking_score'] >= common.PRED_SCORE]


# ---------------- (a) nối track hai chiều ----------------
def reassociate(tracks, p):
    """Mỗi track có một 'đuôi' (box thật cuối) và một 'đầu' (box thật đầu).
    Cặp (A, B) là ứng viên nếu: cùng lớp, B bắt đầu sau khi A kết thúc, khoảng trống <= gap giây.
    Chi phí hai chiều: giá trị nhỏ hơn trong hai lỗi
      - lỗi đoán xuôi: vị trí cuối A + v_A * dt  so với  vị trí đầu B
      - lỗi đoán ngược: vị trí đầu B - v_B * dt  so với  vị trí cuối A
    Cổng: chi phí <= d0 + d1 * dt (người đi bộ dùng d0, d1 nhỏ hơn) và tỉ lệ thể tích box trong [1/2, 2].
    Ghép cặp bằng Hungarian (mỗi đuôi nối tối đa 1 đầu, mỗi đầu tối đa 1 đuôi), rồi gộp thành chuỗi ID."""
    st = collections.Counter()
    for sc, trs in tracks.items():
        info = {}
        for tid, seq in trs.items():
            r = real_boxes(seq)
            if not r: continue
            cls = collections.Counter(b['tracking_name'] for _, _, b in r).most_common(1)[0][0]
            xy = lambda e: np.array(e[2]['translation'][:2])
            vol = np.median([np.prod(b['size']) for _, _, b in r])
            head, tail = r[:p.k], r[-p.k:]
            info[tid] = dict(cls=cls, vol=vol, score=np.mean([b['tracking_score'] for _, _, b in r]),
                             f0=r[0][0], t0=r[0][1], x0=xy(r[0]), v0=velocity([(t, xy(e)) for e in head for t in [e[1]]]),
                             f1=r[-1][0], t1=r[-1][1], x1=xy(r[-1]), v1=velocity([(t, xy(e)) for e in tail for t in [e[1]]]))
        for cls in common.CLASSES:
            if cls in p.skip_cls.split(','): continue
            ids = [t for t, i in info.items() if i['cls'] == cls and i['score'] >= p.min_link_score]
            if len(ids) < 2: continue
            ped = cls == 'pedestrian'
            d0, d1 = (p.d0p, p.d1p) if ped else (p.d0, p.d1)
            C = np.full((len(ids), len(ids)), 1e6)
            for i, a in enumerate(ids):
                A = info[a]
                for j, b in enumerate(ids):
                    if a == b: continue
                    B = info[b]
                    if B['f0'] <= A['f1']: continue
                    dt = B['t0'] - A['t1']
                    if dt > p.gap: continue
                    if not (0.5 <= B['vol'] / max(A['vol'], 1e-6) <= 2.0): continue
                    ef = np.linalg.norm(A['x1'] + A['v1'] * dt - B['x0'])
                    eb = np.linalg.norm(B['x0'] - B['v0'] * dt - A['x1'])
                    c = min(ef, eb)  # lấy chiều đoán tốt hơn: track ngắn (1-2 box) ước lượng vận tốc kém
                    if c <= d0 + d1 * dt: C[i, j] = c
            st['candidates'] += int((C < 1e6).sum())
            nxt = {}
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6: nxt[ids[i]] = ids[j]; LINKS.append((sc, cls, ids[i], ids[j], float(C[i, j]), info[ids[j]]['t0'] - info[ids[i]]['t1']))
            has_prev = set(nxt.values())
            for start in [t for t in nxt if t not in has_prev]:
                chain, cur = [start], start
                while cur in nxt: cur = nxt[cur]; chain.append(cur)
                merged = {}
                for k, tid in enumerate(chain):
                    nf0 = info[chain[k + 1]]['f0'] if k + 1 < len(chain) else None
                    for f, t, b in trs[tid]:
                        if nf0 is not None and f >= nf0: continue  # bỏ box đoán của track trước chồng lên track sau
                        if f not in merged or b['tracking_score'] > merged[f][2]['tracking_score']:
                            merged[f] = (f, t, b)
                for f, t, b in merged.values(): b['tracking_id'] = start
                trs[start] = sorted(merged.values(), key=lambda x: x[0])
                for tid in chain[1:]: del trs[tid]
                st['links'] += len(chain) - 1; st['chains'] += 1
    return dict(st)


ADDED = []  # ghi lại box kéo dài để phân tích
LINKS = []  # ghi lại các cặp track được nối (scene, lớp, track trước, track sau, chi phí, khoảng trống s)

# ---------------- (b) kéo dài có điều kiện ----------------
def extend(tracks, p, scenes, where, ego):
    """Khác B2 (kéo mọi track): chỉ kéo đầu/đuôi khi đủ tin cậy.
    Điều kiện track: >= min_len box thật và score trung bình box thật >= min_score (loại track nhiễu).
    Kéo theo vận tốc không đổi (bình phương tối thiểu trên k box thật ở đầu/đuôi), tối đa ext giây.
    Dừng kéo tại frame đầu tiên mà:
      - vị trí đoán ra ngoài tầm chấm của lớp (40/50 m quanh xe) -> vật đã ra khỏi vùng quan sát;
      - đã có box thật cùng lớp của track khác trong vòng 2 m -> vật đã được track khác bám, kéo thêm sẽ thành box trùng.
    Box kéo dài thay cho box dự đoán sẵn có (score < PRED_SCORE) ở cùng frame. Score = score box thật gần nhất x ext_decay."""
    st = collections.Counter()
    for sc, trs in tracks.items():
        toks = scenes[sc]; T = [where[t][2] for t in toks]
        occ = collections.defaultdict(list)  # (frame, lớp) -> [(tid, xy)] các box thật
        for tid, seq in trs.items():
            for f, t, b in seq:
                if b['tracking_score'] >= common.PRED_SCORE:
                    occ[(f, b['tracking_name'])].append((tid, np.array(b['translation'][:2])))
        for tid, seq in trs.items():
            r = real_boxes(seq)
            if len(r) < p.min_len or np.mean([b['tracking_score'] for _, _, b in r]) < p.min_score:
                st['skip_unreliable'] += 1; continue
            have = {f: (f, t, b) for f, t, b in seq if b['tracking_score'] >= common.PRED_SCORE}
            new = {}
            for side in ('head', 'tail'):
                ref = r[:p.k] if side == 'head' else r[-p.k:]
                v = velocity([(t, np.array(b['translation'][:2])) for _, t, b in ref])
                f_a, t_a, b_a = ref[0] if side == 'head' else ref[-1]
                x_a = np.array(b_a['translation'][:2]); cls = b_a['tracking_name']
                frames = range(f_a - 1, -1, -1) if side == 'head' else range(f_a + 1, len(toks))
                lim = p.ext if side == 'head' else p.ext_tail
                for f in frames:
                    dt = T[f] - t_a
                    if abs(dt) > lim + 1e-6: break
                    x = x_a + v * dt
                    if np.linalg.norm(x - ego[toks[f]]) > RANGE[cls]: st['stop_range'] += 1; break
                    if any(o != tid and np.linalg.norm(x - y) <= 2.0 for o, y in occ.get((f, cls), [])): st['stop_occupied'] += 1; break
                    if f in have: break
                    n = dict(b_a); n['translation'] = [float(x[0]), float(x[1]), b_a['translation'][2]]
                    n['tracking_score'] = b_a['tracking_score'] * p.ext_decay
                    new[f] = (f, T[f], n)
                    ADDED.append(dict(sc=sc, tok=toks[f], cls=cls, x=x, side=side, speed=float(np.linalg.norm(v)), steps=abs(f - f_a),
                                      dego=float(np.linalg.norm(x - ego[toks[f]])), score=float(np.mean([b['tracking_score'] for _, _, b in r])), nreal=len(r)))
            if new:
                kept = [e for e in seq if e[0] not in new]
                trs[tid] = sorted(kept + list(new.values()), key=lambda x: x[0])
                st['boxes_added'] += len(new); st['tracks_extended'] += 1
    return dict(st)


# ---------------- (d) làm mượt cả track ----------------
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def smooth(tracks, p):
    """Với mỗi track có >= 3 box thật:
    1) Kích thước: vật cứng không đổi kích thước -> mọi box của track dùng trung vị kích thước các box thật.
    2) Sửa hướng lật 180°: trong cửa sổ ±3 box thật, lấy trục hướng chung (trung bình góc nhân đôi, không phân biệt đầu/đuôi);
       chiều đúng = chiều di chuyển nếu vật đi nhanh hơn 1 m/s, ngược lại theo đa số các box trong cửa sổ.
       Box nào lệch chiều đúng hơn 90° thì xoay 180°.
    3) Vị trí: hồi quy đa thức bậc sdeg theo thời gian trên cửa sổ ±sw box thật (trọng số Gauss theo thời gian x score),
       thay x, y bằng giá trị hồi quy; z thay bằng trung vị cửa sổ. Box dự đoán (score < PRED_SCORE) giữ nguyên vị trí.
    Lưu ý: bộ chấm nuScenes chỉ dùng tâm box (khoảng cách 2 m, AMOTP), nên chỉ bước 3 ảnh hưởng điểm số;
    bước 1-2 cải thiện chất lượng nhãn cho bài toán auto-labeling."""
    st = collections.Counter()
    for sc, trs in tracks.items():
        for tid, seq in trs.items():
            r = real_boxes(seq)
            if len(r) < 3: continue
            st['tracks'] += 1
            size = list(np.median([b['size'] for _, _, b in r], axis=0))
            for _, _, b in seq: b['size'] = size
            t = np.array([x[1] for x in r]); X = np.array([b['translation'] for _, _, b in r])
            w_s = np.array([b['tracking_score'] for _, _, b in r])
            yaw = np.array([Quaternion(b['rotation']).yaw_pitch_roll[0] for _, _, b in r])
            n = len(r); newX = X.copy(); newyaw = yaw.copy()
            for i in range(n):
                lo, hi = max(0, i - 3), min(n, i + 4)
                axis = 0.5 * np.arctan2(np.sin(2 * yaw[lo:hi]).sum(), np.cos(2 * yaw[lo:hi]).sum())
                disp = X[hi - 1, :2] - X[lo, :2]; dur = t[hi - 1] - t[lo]
                if dur > 0 and np.linalg.norm(disp) / dur > 1.0:
                    sign = 1 if np.cos(np.arctan2(disp[1], disp[0]) - axis) >= 0 else -1
                else:
                    sign = 1 if (np.cos(yaw[lo:hi] - axis) >= 0).sum() * 2 >= (hi - lo) else -1
                ref = axis if sign > 0 else wrap(axis + np.pi)
                if abs(wrap(yaw[i] - ref)) > np.pi / 2: newyaw[i] = wrap(yaw[i] + np.pi); st['flips'] += 1
                lo, hi = max(0, i - p.sw), min(n, i + p.sw + 1)
                tt = t[lo:hi] - t[i]
                if hi - lo >= p.sdeg + 2:
                    w = np.exp(-0.5 * (tt / p.ssig) ** 2) * w_s[lo:hi]
                    A = np.vander(tt, p.sdeg + 1)
                    W = np.sqrt(w)[:, None]
                    coef, *_ = np.linalg.lstsq(A * W, X[lo:hi, :2] * W, rcond=None)
                    newX[i, :2] = coef[-1]
                    newX[i, 2] = np.median(X[lo:hi, 2])
            for k, (_, _, b) in enumerate(r):
                b['translation'] = [float(v) for v in newX[k]]
                if newyaw[k] != yaw[k]: b['rotation'] = list(Quaternion(axis=[0, 0, 1], angle=newyaw[k]).q)
            st['moved_m'] += float(np.linalg.norm(newX[:, :2] - X[:, :2], axis=1).sum())
    return dict(st)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('version'); ap.add_argument('part', nargs='?', default='dev')
    ap.add_argument('--tag'); ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--gap', type=float, default=3.0)
    ap.add_argument('--d0', type=float, default=1.0); ap.add_argument('--d1', type=float, default=1.0)
    ap.add_argument('--d0p', type=float, default=0.6); ap.add_argument('--d1p', type=float, default=0.6)
    ap.add_argument('--skip_cls', default=''); ap.add_argument('--min_link_score', type=float, default=0.0)
    ap.add_argument('--min_len', type=int, default=4); ap.add_argument('--min_score', type=float, default=0.3)
    ap.add_argument('--ext', type=float, default=1.0); ap.add_argument('--ext_tail', type=float, default=1.0); ap.add_argument('--ext_decay', type=float, default=1.0)
    ap.add_argument('--sw', type=int, default=2); ap.add_argument('--sdeg', type=int, default=1)
    ap.add_argument('--ssig', type=float, default=1.0)
    p = ap.parse_args()
    tag = p.tag or p.version
    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
    scenes, where = common.scene_index(nusc)
    keep = None if p.part == 'val' else set(common.load_split()[p.part])
    import pickle
    cache = os.path.join(os.path.expanduser('~'), 'scr', f'b0_tracks_{p.part}.pkl')  # đọc results.json rất chậm -> lưu tạm
    if os.path.exists(cache): tracks = pickle.load(open(cache, 'rb'))
    else:
        tracks = {sc: dict(v) for sc, v in common.load_tracks(common.B0, where, keep).items()}
        os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(tracks, open(cache, 'wb'))
    n_before = sum(len(v) for v in tracks.values())
    stats = {'params': vars(p), 'tracks_before': n_before}
    for s in STEPS[p.version]:
        if s == 'a': stats['a'] = reassociate(tracks, p)
        if s == 'd': stats['d'] = smooth(tracks, p)
        if s == 'b':
            ego = {}
            for sc in tracks:
                for tok in scenes[sc]:
                    sd = nusc.get('sample_data', nusc.get('sample', tok)['data']['LIDAR_TOP'])
                    ego[tok] = np.array(nusc.get('ego_pose', sd['ego_pose_token'])['translation'][:2])
            stats['b'] = extend(tracks, p, scenes, where, ego)
    stats['tracks_after'] = sum(len(v) for v in tracks.values())
    for sc in tracks:
        for tid, seq in tracks[sc].items():
            for f, t, b in seq: b['sample_token'] = scenes[sc][f]
    od = os.path.join(common.ROOT, 'offline', 'results', tag)
    out = os.path.join(od, 'results.json' if p.part == 'val' else f'results_{p.part}.json')
    stats['boxes'] = common.save_tracks(tracks, scenes, out, keep)
    json.dump(stats, open(os.path.join(od, f'stats_{p.part}.json'), 'w'), indent=1)
    if ADDED: pickle.dump(ADDED, open(os.path.join(os.path.expanduser('~'), 'scr', f'added_{tag}_{p.part}.pkl'), 'wb'))
    print(json.dumps(stats))

"""Phuong phap de xuat (Ours_v2 tren nuScenes) chuyen sang Waymo, lop xe:
  (a) noi track bi dut hai chieu (re-association) + (d) lam muot ca track (kich thuoc chung, sua lat huong kieu 'vote',
  lam muot vi tri). Giu nguyen y tuong va gia tri tham so da chon tren nuScenes; chi doi cac tham so tinh theo SO BOX
  sang GIAY (nuScenes 2 Hz -> Waymo 10 Hz), KHONG chinh lai tham so tren du lieu Waymo.

  nuScenes (2 Hz)                          -> Waymo (10 Hz)
  k = 3 box that o dau/duoi (~1 s)         -> box that trong 1,0 s o dau/duoi        (--k_sec 1.0)
  cua so lat huong +-3 box (~+-1,5 s)      -> +-1,5 s                                 (--flip_sec 1.5)
  cua so lam muot +-2 box (~+-1 s)         -> +-1,0 s                                 (--sw_sec 1.0)
  gap 3 s, d0 = d1 = 1 m, sigma 1 s, bac 1, vote 0,7, score track >= 0,15: giu nguyen

Khac biet ky thuat so voi ban nuScenes:
  - Box Waymo o he toa do xe (thay doi moi frame) -> doi sang he the gioi bang poses.pkl truoc khi tinh van toc / lam muot, roi doi lai.
  - "Box that" (co detection) khong nhan ra duoc bang nguong score (score FSD rat thap van la detection that) ->
    box cua tracker la box that neu trung (tam lech < 0,3 m) voi mot box FSD cung frame; con lai la box du doan cua tracker.

Dau vao: boxes_<tracker>.npz va boxes_fsd.npz (export_boxes_npz.py), poses.pkl.
Dau ra: <out>.bin (dinh dang Waymo, de dua vao keo dai / refine / cham cua CTRL) va boxes_<ten>.npz.

Chay: python waymo/ours_waymo.py --npz offline/results/waymo/subset8/npz --poses data/waymo/kitti_format/poses.pkl
        --out data/waymo/ctrl_work/ours/ours.bin --name ours [--no_link] [--no_smooth]"""
import argparse, collections, json, os, pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from boxgeo import Source

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--poses', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--name', default='ours'); ap.add_argument('--tracker', default='ctrl_tracker'); ap.add_argument('--det', default='fsd')
ap.add_argument('--k_sec', type=float, default=1.0); ap.add_argument('--gap', type=float, default=3.0)
ap.add_argument('--d0', type=float, default=1.0); ap.add_argument('--d1', type=float, default=1.0)
ap.add_argument('--min_link_score', type=float, default=0.15)
ap.add_argument('--flip_sec', type=float, default=1.5); ap.add_argument('--flip_vote', type=float, default=0.7)
ap.add_argument('--sw_sec', type=float, default=1.0); ap.add_argument('--sdeg', type=int, default=1); ap.add_argument('--ssig', type=float, default=1.0)
ap.add_argument('--real_dist', type=float, default=0.3)
ap.add_argument('--no_link', action='store_true'); ap.add_argument('--no_smooth', action='store_true')
p = ap.parse_args()

S = Source(os.path.join(p.npz, f'boxes_{p.tracker}.npz')); D = Source(os.path.join(p.npz, f'boxes_{p.det}.npz'))
ctxs = list(np.load(os.path.join(p.npz, 'contexts.npy')))
poses = pickle.load(open(p.poses, 'rb'))
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

# ---- box o he the gioi + co "box that"
N = len(S.ts); W = S.box.copy(); real = np.zeros(N, bool)
for i in range(N):
    P = np.asarray(poses[int(S.ts[i])], np.float64)
    W[i, :3] = P[:3, :3] @ S.box[i, :3] + P[:3, 3]
    W[i, 6] = wrap(S.box[i, 6] + np.arctan2(P[1, 0], P[0, 0]))
    dj = D.in_frame(S.ctx[i], S.ts[i])
    if dj: real[i] = np.linalg.norm(D.box[dj, :2] - S.box[i, :2], axis=1).min() < p.real_dist
score = S.score.astype(np.float64).copy()
tracks = {k: list(v) for k, v in S.tracks.items()}  # (ctx, id) -> [chi so box], theo thoi gian
T = lambda i: S.ts[i] / 1e6
stats = {'box': N, 'box_that': int(real.sum()), 'track_truoc': len(tracks)}


def velocity(idx):
    if len(idx) < 2: return np.zeros(2)
    t = np.array([T(i) for i in idx]); x = W[idx, :2]; t = t - t.mean(); den = (t ** 2).sum()
    return (t[:, None] * (x - x.mean(0))).sum(0) / den if den > 0 else np.zeros(2)


# ---- (a) noi track hai chieu
def reassociate():
    st = collections.Counter(); info = {}
    for key, idx in tracks.items():
        r = [i for i in idx if real[i]]
        if not r: continue
        head = [i for i in r if T(i) - T(r[0]) <= p.k_sec]; tail = [i for i in r if T(r[-1]) - T(i) <= p.k_sec]
        info[key] = dict(vol=np.median(np.prod(W[r, 3:6], axis=1)), score=float(np.mean(score[r])),
                         t0=T(r[0]), x0=W[r[0], :2], v0=velocity(head), t1=T(r[-1]), x1=W[r[-1], :2], v1=velocity(tail))
    for c in range(len(ctxs)):
        ids = [k for k in info if k[0] == c and info[k]['score'] >= p.min_link_score]
        if len(ids) < 2: continue
        C = np.full((len(ids), len(ids)), 1e6)
        for a, A_ in enumerate(ids):
            A = info[A_]
            for b, B_ in enumerate(ids):
                if a == b: continue
                B = info[B_]
                if B['t0'] <= A['t1']: continue
                dt = B['t0'] - A['t1']
                if dt > p.gap or not (0.5 <= B['vol'] / max(A['vol'], 1e-6) <= 2.0): continue
                cst = min(np.linalg.norm(A['x1'] + A['v1'] * dt - B['x0']), np.linalg.norm(B['x0'] - B['v0'] * dt - A['x1']))
                if cst <= p.d0 + p.d1 * dt: C[a, b] = cst
        st['ung_vien'] += int((C < 1e6).sum())
        nxt = {ids[a]: ids[b] for a, b in zip(*linear_sum_assignment(C)) if C[a, b] < 1e6}
        has_prev = set(nxt.values())
        for start in [k for k in nxt if k not in has_prev]:
            chain, cur = [start], start
            while cur in nxt: cur = nxt[cur]; chain.append(cur)
            merged = {}
            for n, key in enumerate(chain):
                nt0 = info[chain[n + 1]]['t0'] if n + 1 < len(chain) else None
                for i in tracks[key]:
                    if nt0 is not None and T(i) >= nt0 - 1e-6: continue  # bo box du doan cua track truoc chong len track sau
                    ts = int(S.ts[i])
                    if ts not in merged or score[i] > score[merged[ts]]: merged[ts] = i
            tracks[start] = sorted(merged.values(), key=lambda i: S.ts[i])
            for key in chain[1:]: del tracks[key]
            st['noi'] += len(chain) - 1; st['chuoi'] += 1
    return dict(st)


# ---- (d) lam muot ca track
def smooth():
    st = collections.Counter()
    for key, idx in tracks.items():
        r = [i for i in idx if real[i]]
        if len(r) < 3: continue
        st['track'] += 1
        size = np.median(W[r, 3:6], axis=0)
        for i in idx: W[i, 3:6] = size
        t = np.array([T(i) for i in r]); X = W[r, :3].copy(); yaw = W[r, 6].copy(); ws = score[r]
        newX, newyaw = X.copy(), yaw.copy()
        for n in range(len(r)):
            win = np.nonzero(np.abs(t - t[n]) <= p.flip_sec + 1e-6)[0]
            others = yaw[win[win != n]]
            if len(others) and (np.cos(others - yaw[n]) < 0).mean() >= p.flip_vote:
                newyaw[n] = wrap(yaw[n] + np.pi); st['lat'] += 1
            win = np.nonzero(np.abs(t - t[n]) <= p.sw_sec + 1e-6)[0]
            if len(win) >= p.sdeg + 2:
                tt = t[win] - t[n]
                w = np.exp(-0.5 * (tt / p.ssig) ** 2) * ws[win]
                if w.sum() <= 0: continue
                A = np.vander(tt, p.sdeg + 1); sw = np.sqrt(w)[:, None]
                coef, *_ = np.linalg.lstsq(A * sw, X[win, :2] * sw, rcond=None)
                newX[n, :2] = coef[-1]; newX[n, 2] = np.median(X[win, 2])
        st['dich_m'] += float(np.linalg.norm(newX[:, :2] - X[:, :2], axis=1).sum())
        W[r, :3] = newX; W[r, 6] = newyaw
    return dict(st)


if not p.no_link: stats['a_noi_track'] = reassociate()
if not p.no_smooth: stats['d_lam_muot'] = smooth()
stats['track_sau'] = len(tracks)

# ---- doi ve he toa do xe, ghi .bin + .npz
rows = []
for (c, key), idx in tracks.items():
    for i in idx:
        P = np.asarray(poses[int(S.ts[i])], np.float64); R = P[:3, :3]
        b = W[i].copy(); b[:3] = R.T @ (W[i, :3] - P[:3, 3]); b[6] = wrap(W[i, 6] - np.arctan2(P[1, 0], P[0, 0]))
        rows.append((c, int(S.ts[i]), str(key), b, float(score[i])))
os.makedirs(os.path.dirname(os.path.abspath(p.out)), exist_ok=True)
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2
objs = metrics_pb2.Objects()
for c, ts, key, b, s in rows:
    o = objs.objects.add(); o.context_name = ctxs[c]; o.frame_timestamp_micros = ts; o.score = s
    o.object.id = key; o.object.type = label_pb2.Label.TYPE_VEHICLE
    bx = o.object.box; bx.center_x, bx.center_y, bx.center_z, bx.length, bx.width, bx.height, bx.heading = [float(v) for v in b]
open(p.out, 'wb').write(objs.SerializeToString())
np.savez_compressed(os.path.join(p.npz, f'boxes_{p.name}.npz'), ctx=np.array([r[0] for r in rows], np.int16),
                    ts=np.array([r[1] for r in rows], np.int64), id=np.array([r[2] for r in rows]),
                    box=np.array([r[3] for r in rows], np.float32), score=np.array([r[4] for r in rows], np.float32))
stats['box_ra'] = len(rows); stats['params'] = vars(p)
json.dump(stats, open(os.path.splitext(p.out)[0] + '_stats.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != 'params'}, ensure_ascii=False, indent=1))

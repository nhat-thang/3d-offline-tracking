"""Tim cac doan (su kien) trong mot canh Waymo ma CTRL sai con ban de xuat dung (va nguoc lai), theo tung xe GT.
Dung = co box (score >= 0,3, tam cach GT <= 4 m) voi IoU 3D >= 0,7 va lech huong < 30 do.
Su kien = chuoi >= min_len frame lien tiep (cho phep ho 1 frame) cung mot xe GT.
Nguyen nhan loi CTRL o doan do: lat_huong (lech > 150 do), thieu_box, sai_kich_thuoc, lech_vi_tri.

Chay: python waymo/find_events.py --npz offline/results/waymo/subset8/npz --out offline/results/waymo/subset8/video_canh/events.json
        [--ctx 10448102132863604198_472_000_492_000 ...]"""
import argparse, json, os
import numpy as np
from boxgeo import Source, iou3d, yaw_diff

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--ctrl', default='ctrl_refine'); ap.add_argument('--ours', default='ours_refine')
ap.add_argument('--ctx', nargs='*', default=None); ap.add_argument('--score_thr', type=float, default=0.3)
ap.add_argument('--min_len', type=int, default=5); ap.add_argument('--max_range', type=float, default=50.0)
p = ap.parse_args()
L = lambda n: Source(os.path.join(p.npz, f'boxes_{n}.npz'))
G, C, X = L('gt'), L(p.ctrl), L(p.ours)
ctxs = list(np.load(os.path.join(p.npz, 'contexts.npy')))
use = [ctxs.index(c) for c in p.ctx] if p.ctx else list(range(len(ctxs)))


def best(S, c, t, gb):
    bi, bv = -1, 0.0
    for j in S.frames.get((c, t), []):
        if S.score[j] < p.score_thr or np.linalg.norm(S.box[j, :2] - gb[:2]) > 4: continue
        v = iou3d(gb, S.box[j])
        if v > bv: bv, bi = v, j
    return bi, bv


def info(S, j, gb, v):
    if j < 0: return dict(ok=False, iou=0.0, yaw=None, id=None, dpos=None, dlen=None)
    y = np.degrees(yaw_diff(S.box[j, 6], gb[6]))
    return dict(ok=bool(v >= 0.7 and y < 30), iou=round(v, 3), yaw=round(y, 1), id=str(S.id[j]),
                dpos=round(float(np.linalg.norm(S.box[j, :2] - gb[:2])), 2), dlen=round(float(S.box[j, 3] - gb[3]), 2))


out = {}
for c in use:
    ts = sorted({t for (cc, t) in G.frames if cc == c}); t0 = ts[0]
    per = {}  # gt id -> list of frame records
    for k, t in enumerate(ts):
        for gi in G.frames[(c, t)]:
            gb = G.box[gi]
            if G.npts is not None and G.npts[gi] == 0: continue
            jc, vc = best(C, c, t, gb); jx, vx = best(X, c, t, gb)
            per.setdefault(str(G.id[gi]), []).append(dict(k=k, sec=round((t - t0) / 1e6, 1), dist=round(float(np.linalg.norm(gb[:2])), 1),
                                                       npts=int(G.npts[gi]) if G.npts is not None else -1, glen=round(float(gb[3]), 2),
                                                       c=info(C, jc, gb, vc), x=info(X, jx, gb, vx)))
    ev = []
    for gid, recs in per.items():
        for kind, a, b in (('sua_duoc', 'c', 'x'), ('lam_sai', 'x', 'c')):
            flag = [(not r[a]['ok']) and r[b]['ok'] for r in recs]
            i = 0
            while i < len(recs):
                if not flag[i]: i += 1; continue
                j = i; last = i
                while j + 1 < len(recs) and (flag[j + 1] or (j + 2 < len(recs) and flag[j + 2] and recs[j + 2]['k'] - recs[last]['k'] <= 2)):
                    j += 1
                    if flag[j]: last = j
                seg = [recs[q] for q in range(i, last + 1) if flag[q]]
                if len(seg) >= p.min_len:
                    w = [r[a] for r in seg]
                    nflip = sum(1 for r in w if r['yaw'] is not None and r['yaw'] > 150)
                    nmiss = sum(1 for r in w if r['id'] is None)
                    dl = [abs(r['dlen']) for r in w if r['dlen'] is not None]
                    if nflip >= len(w) / 2: cause = 'lat_huong'
                    elif nmiss >= len(w) / 2: cause = 'thieu_box'
                    elif dl and np.median(dl) > 0.4: cause = 'sai_kich_thuoc'
                    else: cause = 'lech_vi_tri'
                    ev.append(dict(kind=kind, gt=gid, t_from=seg[0]['sec'], t_to=seg[-1]['sec'], n=len(seg), cause=cause,
                                   dist=round(float(np.median([r['dist'] for r in seg])), 1), npts=int(np.median([r['npts'] for r in seg])),
                                   glen=seg[0]['glen'],
                                   ids_ctrl=sorted({r['c']['id'] for r in seg if r['c']['id']}), ids_ours=sorted({r['x']['id'] for r in seg if r['x']['id']}),
                                   iou_ctrl=round(float(np.mean([r['c']['iou'] for r in seg])), 2), iou_ours=round(float(np.mean([r['x']['iou'] for r in seg])), 2),
                                   yaw_ctrl=round(float(np.median([r['c']['yaw'] for r in seg if r['c']['yaw'] is not None] or [-1])), 1),
                                   dlen_ctrl=round(float(np.median([r['c']['dlen'] for r in seg if r['c']['dlen'] is not None] or [0])), 2),
                                   dlen_ours=round(float(np.median([r['x']['dlen'] for r in seg if r['x']['dlen'] is not None] or [0])), 2),
                                   dpos_ctrl=round(float(np.median([r['c']['dpos'] for r in seg if r['c']['dpos'] is not None] or [-1])), 2),
                                   dpos_ours=round(float(np.median([r['x']['dpos'] for r in seg if r['x']['dpos'] is not None] or [-1])), 2)))
                i = last + 1
    ev.sort(key=lambda e: (e['kind'], e['t_from']))
    out[ctxs[c]] = dict(n_frames=len(ts), events=ev)
    good = [e for e in ev if e['kind'] == 'sua_duoc']; bad = [e for e in ev if e['kind'] == 'lam_sai']
    near = [e for e in good if e['dist'] <= p.max_range]
    print(ctxs[c], 'frames', len(ts), '| sua_duoc', len(good), '(gan', len(near), ', frame', sum(e['n'] for e in good), ') | lam_sai', len(bad),
          '(frame', sum(e['n'] for e in bad), ')')
os.makedirs(os.path.dirname(os.path.abspath(p.out)), exist_ok=True)
json.dump(out, open(p.out, 'w'), ensure_ascii=False, indent=1)

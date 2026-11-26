"""So tung box giua ket qua CTRL tu chay va file chinh thuc cua tac gia, tren cung cac chuoi (khong can GT).

Ghep box theo tung frame (cung context, cung timestamp, cung lop) bang Hungarian tren khoang cach tam BEV (cong 2 m),
roi do tren cac cap ghep: IoU 3D, lech tam, lech kich thuoc, lech huong, lech score; dem box chi mot ben co;
so ID track: moi track cua ban nay ung voi bao nhieu track cua ban kia.

Chay: python waymo/compare_boxes.py <mine.bin> <ref.bin> --contexts waymo/subset8.txt [--out ket_qua.json]"""
import argparse, collections, gc, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from waymo_open_dataset.protos import metrics_pb2

ap = argparse.ArgumentParser()
ap.add_argument('mine'); ap.add_argument('ref')
ap.add_argument('--contexts', required=True); ap.add_argument('--types', default='1')
ap.add_argument('--gate', type=float, default=2.0); ap.add_argument('--out', default='')
p = ap.parse_args()
ctxs = set(open(p.contexts).read().split()); types = {int(t) for t in p.types.split(',')}


def load(path):
    o = metrics_pb2.Objects(); o.ParseFromString(open(path, 'rb').read())
    rows, ids, keys = [], [], []
    for x in o.objects:
        if x.context_name not in ctxs or x.object.type not in types: continue
        b = x.object.box
        rows.append((b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading, x.score))
        ids.append(x.context_name + '/' + x.object.id); keys.append((x.context_name, x.frame_timestamp_micros))
    del o; gc.collect()
    frames = collections.defaultdict(list)
    for i, k in enumerate(keys): frames[k].append(i)
    return np.array(rows, np.float64), ids, keys, frames


def corners(b):  # 4 goc BEV, nguoc chieu kim dong ho
    c, s = np.cos(b[6]), np.sin(b[6]); l, w = b[3] / 2, b[4] / 2
    pts = np.array([[l, w], [-l, w], [-l, -w], [l, -w]])
    return pts @ np.array([[c, s], [-s, c]]) + b[:2]


def clip(poly, a, b):  # Sutherland-Hodgman: giu phan ben trai canh a->b
    out = []
    side = lambda q: (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]; sp, sq = side(P), side(Q)
        if sp >= 0: out.append(P)
        if sp * sq < 0: out.append(P + (Q - P) * (sp / (sp - sq)))
    return out


def area(poly):
    if len(poly) < 3: return 0.0
    x = np.array([q[0] for q in poly]); y = np.array([q[1] for q in poly])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def iou3d(a, b):
    pa, pb = corners(a), corners(b); poly = list(pa)
    for i in range(4):
        poly = clip(poly, pb[i], pb[(i + 1) % 4])
        if not poly: return 0.0
    inter_bev = area(poly)
    zo = max(0.0, min(a[2] + a[5] / 2, b[2] + b[5] / 2) - max(a[2] - a[5] / 2, b[2] - b[5] / 2))
    inter = inter_bev * zo; va, vb = a[3] * a[4] * a[5], b[3] * b[4] * b[5]
    return inter / (va + vb - inter + 1e-9)


A, idA, kA, fA = load(p.mine)
B, idB, kB, fB = load(p.ref)
print(f'ban tu chay: {len(A)} box, {len(fA)} frame | tac gia: {len(B)} box, {len(fB)} frame')

pairs = []
for k in set(fA) | set(fB):
    ia, ib = fA.get(k, []), fB.get(k, [])
    if not ia or not ib: continue
    d = np.linalg.norm(A[ia][:, None, :2] - B[ib][None, :, :2], axis=2)
    r, c = linear_sum_assignment(np.where(d < p.gate, d, 1e6))
    pairs += [(ia[i], ib[j]) for i, j in zip(r, c) if d[i, j] < p.gate]
pa = np.array([x for x, _ in pairs]); pb = np.array([y for _, y in pairs])
a, b = A[pa], B[pb]
iou = np.array([iou3d(x, y) for x, y in zip(a, b)])
dc = np.linalg.norm(a[:, :3] - b[:, :3], axis=1)
ds = np.abs(a[:, 3:6] - b[:, 3:6]).max(1)
dh = np.abs((a[:, 6] - b[:, 6] + np.pi) % (2 * np.pi) - np.pi)
dsc = np.abs(a[:, 7] - b[:, 7])

res = {}
res['so_box'] = dict(tu_chay=len(A), tac_gia=len(B), ghep_duoc=len(pairs),
                     chi_tu_chay=len(A) - len(pairs), chi_tac_gia=len(B) - len(pairs))
q = lambda v, ts: {f'<= {t}': round(100 * float((v <= t).mean()), 2) for t in ts}
res['iou3d'] = dict(trung_vi=round(float(np.median(iou)), 4), trung_binh=round(float(iou.mean()), 4),
                    **{f'% >= {t}': round(100 * float((iou >= t).mean()), 2) for t in (0.99, 0.95, 0.9, 0.7)})
res['lech_tam_m'] = dict(trung_vi=round(float(np.median(dc)), 4), **q(dc, (0.01, 0.05, 0.1, 0.3)))
res['lech_kich_thuoc_max_m'] = dict(trung_vi=round(float(np.median(ds)), 4), **q(ds, (0.01, 0.05, 0.1)))
res['lech_huong_do'] = dict(trung_vi=round(float(np.degrees(np.median(dh))), 3), **q(np.degrees(dh), (0.5, 2, 10)),
                            **{'% lech > 90 do (lat huong)': round(100 * float((dh > np.pi / 2).mean()), 2)})
res['lech_score'] = dict(trung_vi=round(float(np.median(dsc)), 4), **q(dsc, (0.001, 0.01, 0.05)))

# box chi mot ben co: score, khoang cach toi xe
for name, X, used in (('chi_tu_chay', A, set(pa.tolist())), ('chi_tac_gia', B, set(pb.tolist()))):
    rest = np.array([i for i in range(len(X)) if i not in used], int)
    if len(rest):
        res[name + '_dac_diem'] = {'score_trung_vi': round(float(np.median(X[rest, 7])), 3),
                                    '% score < 0.1': round(100 * float((X[rest, 7] < 0.1).mean()), 1),
                                    'khoang_cach_trung_vi_m': round(float(np.median(np.linalg.norm(X[rest, :2], axis=1))), 1)}

# theo tung chuoi
per = {}
for c in sorted(ctxs):
    m = np.array([kA[i][0] == c for i in pa]) if len(pa) else np.array([], bool)
    na = sum(1 for k in kA if k[0] == c); nb = sum(1 for k in kB if k[0] == c)
    per[c] = dict(tu_chay=na, tac_gia=nb, ghep=int(m.sum()),
                  iou_trung_vi=round(float(np.median(iou[m])), 4) if m.any() else None,
                  pct_iou_095=round(100 * float((iou[m] >= 0.95).mean()), 2) if m.any() else None)
res['theo_chuoi'] = per

# ID track: track tu chay -> track tac gia chiem da so
ma = collections.defaultdict(collections.Counter)
for i, j in pairs: ma[idA[i]][idB[j]] += 1
sizeA = collections.Counter(idA)
pure = sum(1 for t, cnt in ma.items() if cnt.most_common(1)[0][1] == sum(cnt.values()))
cov = sum(cnt.most_common(1)[0][1] for cnt in ma.values()) / max(1, len(pairs))
mb = collections.defaultdict(set)
for t, cnt in ma.items(): mb[cnt.most_common(1)[0][0]].add(t)
res['track'] = dict(so_track_tu_chay=len(sizeA), so_track_tac_gia=len(set(idB)),
                    track_tu_chay_chi_ung_1_track_tac_gia_pct=round(100 * pure / max(1, len(ma)), 2),
                    box_ghep_dung_track_da_so_pct=round(100 * cov, 2),
                    track_tac_gia_bi_chia_thanh_nhieu_track=sum(1 for v in mb.values() if len(v) > 1))

print(json.dumps(res, ensure_ascii=False, indent=1))
if p.out: json.dump(res, open(p.out, 'w'), ensure_ascii=False, indent=1)

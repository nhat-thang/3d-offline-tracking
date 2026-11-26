"""Tim va phan loai cac loi con lai sau buoc refine cua CTRL tren Waymo (lop xe), theo nguyen nhan o track dau vao.

Voi moi box GT co diem LiDAR, ghep voi box sau refine (score >= --score_thr) theo tung frame (Hungarian tren IoU 3D):
  dung       : IoU >= 0.7 va lech huong < 30 do
  lat_huong  : IoU >= 0.7 nhung lech huong > 90 do (AP tinh dung, APH = 0)
  lech_huong : IoU >= 0.7, lech huong 30-90 do
  sai_box    : co box gan (tam < 4 m) nhung IoU < 0.7
  bo_sot     : khong co box nao gan
Moi GT sai duoc gan mot nguyen nhan (theo thu tu uu tien):
  khong_co_dau_vao  : track dau vao (sau keo dai) khong co box nao gan GT -> loi tu detector/tracker, refine khong co gi de sua
  refine_ha_score   : dau vao co box, nhung sau refine score < nguong hoac bi xoa (vd. buoc xoa box rong)
  track_nhay_vat    : track dau vao di qua >= 2 vat GT khac nhau (moi vat >= 3 frame, vat chinh < 85% so frame)
  khung_keo_dai     : box dau vao o frame nay do buoc keo dai ve qua khu tao ra (khong co trong ket qua tracker)
  track_trung       : o cung frame co track dau vao khac cung om xe nay (hai track cho mot xe)
  track_dut         : xe bi chia cho >= 2 track dau vao (moi track >= 3 frame), track o frame nay phu < 70% doi xe
  lat_huong_dau_vao : box dau vao da lech huong > 90 do so voi GT
  it_diem           : track sach, nhung GT co <= 20 diem LiDAR (vat xa / bi che)
  refine_sai        : track dau vao sach, GT du diem, refine van cho box sai
Neu co --compare <ten> (vd. ban cua ban sau refine), dem them: CTRL sai ma ban dung, CTRL dung ma ban sai.

Chay: python waymo/analyze_ctrl_errors.py --npz offline/results/waymo/subset8/npz --out offline/results/waymo/subset8/loi_ctrl
        [--refine ctrl_refine --input ctrl_extend --tracker ctrl_tracker] [--compare ten_nguon]"""
import argparse, collections, json, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from boxgeo import Source, iou3d, yaw_diff

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--refine', default='ctrl_refine'); ap.add_argument('--input', default='ctrl_extend')
ap.add_argument('--tracker', default='ctrl_tracker'); ap.add_argument('--compare', default='')
ap.add_argument('--score_thr', type=float, default=0.3)
p = ap.parse_args()
os.makedirs(p.out, exist_ok=True)
L = lambda n: Source(os.path.join(p.npz, f'boxes_{n}.npz'))
G, R, I, T = L('gt'), L(p.refine), L(p.input), L(p.tracker)
X = L(p.compare) if p.compare else None
ctxs = list(np.load(os.path.join(p.npz, 'contexts.npy')))
D2R = np.pi / 180
CAUSES = ['khong_co_dau_vao', 'refine_ha_score', 'track_nhay_vat', 'khung_keo_dai', 'track_trung', 'track_dut',
          'lat_huong_dau_vao', 'it_diem', 'refine_sai']


def match_frame(S, c, t, g_idx, thr):
    """Ghep GT trong frame voi box cua nguon S (score >= thr). Tra ve dict g -> (trang_thai, iou, chi_so_box hoac -1)."""
    s_idx = [i for i in S.in_frame(c, t) if S.score[i] >= thr]
    res = {g: ('bo_sot', 0.0, -1) for g in g_idx}
    if not s_idx: return res
    gb, sb = G.box[g_idx], S.box[s_idx]
    d = np.linalg.norm(gb[:, None, :2] - sb[None, :, :2], axis=2)
    iou = np.zeros(d.shape)
    for a, b in zip(*np.nonzero(d < 4)): iou[a, b] = iou3d(gb[a], sb[b])
    r, k = linear_sum_assignment(-iou)
    best = {}
    for a, b in zip(r, k):
        if iou[a, b] > 0: best[a] = b
    for a, g in enumerate(g_idx):
        own = a in best
        if own:
            b = best[a]; v = iou[a, b]; j = s_idx[b]
        elif (d[a] < 4).any():  # box gan nhat da thuoc GT khac -> tinh la sai
            b = int(np.argmin(d[a])); v = iou[a, b]; j = s_idx[b]
        else:
            continue
        if own and v >= 0.7:
            y = yaw_diff(G.box[g, 6], S.box[j, 6])
            st = 'dung' if y < 30 * D2R else ('lat_huong' if y > 90 * D2R else 'lech_huong')
        else:
            st = 'sai_box'
        res[g] = (st, float(v), j)
    return res


def nearest(S, c, t, xy, maxd, thr=-1.0):
    idx = [i for i in S.in_frame(c, t) if S.score[i] >= thr]
    if not idx: return -1
    d = np.linalg.norm(S.box[idx, :2] - xy, axis=1); k = int(np.argmin(d))
    return idx[k] if d[k] < maxd else -1


# GT id theo tung box cua track dau vao (de biet track nhay vat / vat bi chia nhieu track)
def track_gt_ids(c, key):
    out = []
    for i in I.track(c, key):
        g = nearest(G, c, I.ts[i], I.box[i, :2], 2.5)
        out.append(str(G.id[g]) if g >= 0 else None)
    return out


track_cache, gt_tracks = {}, collections.defaultdict(lambda: collections.defaultdict(set))
for (c, key), idx in I.tracks.items():
    ids = track_gt_ids(c, key); track_cache[(c, key)] = ids
    for g, i in zip(ids, idx):
        if g: gt_tracks[(c, g)][key].add(int(I.ts[i]))
tracker_keys = {(int(T.ctx[i]), str(T.id[i]), int(T.ts[i])) for i in range(len(T.id))}

rows = []  # moi GT co diem LiDAR: trang thai CTRL (+ ban so sanh) va nguyen nhan
for (c, t), g_all in sorted(G.frames.items()):
    g_idx = [g for g in g_all if G.npts[g] > 0]
    if not g_idx: continue
    mr = match_frame(R, c, t, g_idx, p.score_thr)
    mx = match_frame(X, c, t, g_idx, p.score_thr) if X else {}
    for g in g_idx:
        st, v, j = mr[g]
        row = dict(ctx=c, ts=t, gt=str(G.id[g]), npts=int(G.npts[g]), level=int(G.level[g]), st=st, iou=round(v, 3),
                   ref_track=str(R.id[j]) if j >= 0 else '')
        if X: xs, xv, xj = mx[g]; row.update(st_x=xs, iou_x=round(xv, 3), x_track=str(X.id[xj]) if xj >= 0 else '')
        if st != 'dung':
            ii = nearest(I, c, t, G.box[g, :2], 3.0)
            row['in_track'] = str(I.id[ii]) if ii >= 0 else ''
            row['iou_in'] = round(iou3d(G.box[g], I.box[ii]), 3) if ii >= 0 else 0.0
            if ii < 0:
                cause = 'khong_co_dau_vao'
            elif j < 0:
                cause = 'refine_ha_score'
            else:
                key = str(R.id[j]); ids = [x for x in track_cache.get((c, key), []) if x]
                cnt = collections.Counter(ids)
                big = [k for k, n in cnt.items() if n >= 3]
                share = cnt.most_common(1)[0][1] / len(ids) if ids else 1.0
                ib = nearest(I, c, t, G.box[g, :2], 3.0)
                tr = {k: v for k, v in gt_tracks[(c, row['gt'])].items() if len(v) >= 3}
                life = {int(G.ts[x]) for x in G.track(c, row['gt'])}
                dup = [k for k, v in tr.items() if k != key and t in v]          # track khac cung om xe nay o cung frame
                own = len(tr.get(key, set()) & life) / max(1, len(life))       # track nay phu bao nhieu phan doi xe
                if len(big) >= 2 and share < 0.85: cause = 'track_nhay_vat'
                elif (c, str(I.id[ib]), t) not in tracker_keys: cause = 'khung_keo_dai'
                elif dup: cause = 'track_trung'
                elif own < 0.7 and len(tr) >= 2: cause = 'track_dut'
                elif yaw_diff(I.box[ib, 6], G.box[g, 6]) > 90 * D2R: cause = 'lat_huong_dau_vao'
                elif G.npts[g] <= 20: cause = 'it_diem'
                else: cause = 'refine_sai'
            row['cause'] = cause
        rows.append(row)

# ---- tong hop
N = len(rows); err = [r for r in rows if r['st'] != 'dung']
summ = dict(so_gt_co_diem=N, ctrl_dung=N - len(err),
            trang_thai=dict(collections.Counter(r['st'] for r in rows)),
            nguyen_nhan={k: collections.Counter(r['cause'] for r in err).get(k, 0) for k in CAUSES},
            nguyen_nhan_level2={k: sum(1 for r in err if r['cause'] == k and r['level'] == 2) for k in CAUSES},
            dau_vao_da_dung_iou07={k: sum(1 for r in err if r['cause'] == k and r.get('iou_in', 0) >= 0.7) for k in CAUSES})
if X:
    ok = lambda s: s == 'dung'
    summ['so_sanh'] = dict(
        ca_hai_dung=sum(ok(r['st']) and ok(r['st_x']) for r in rows),
        ctrl_sai_ban_dung=sum((not ok(r['st'])) and ok(r['st_x']) for r in rows),
        ctrl_dung_ban_sai=sum(ok(r['st']) and not ok(r['st_x']) for r in rows),
        ca_hai_sai=sum((not ok(r['st'])) and not ok(r['st_x']) for r in rows),
        ban_sua_duoc_theo_nguyen_nhan={k: sum(1 for r in err if r['cause'] == k and ok(r['st_x'])) for k in CAUSES})

# ---- gom thanh "ca": cung vat GT, cung nguyen nhan; xep theo so frame sai
cases = collections.defaultdict(list)
for r in err: cases[(r['ctx'], r['gt'], r['cause'])].append(r)
case_list = []
for (c, g, cause), rs in cases.items():
    rs.sort(key=lambda r: r['ts'])
    life = [i for i in G.track(c, g) if G.npts[i] > 0]
    item = dict(ctx=int(c), context=ctxs[c], gt=g, cause=cause, n_err=len(rs), n_life=len(life),
                ts=[r['ts'] for r in rs], ref_tracks=sorted({r['ref_track'] for r in rs if r['ref_track']}),
                in_tracks=sorted({r.get('in_track', '') for r in rs} - {''}),
                iou_tb=round(float(np.mean([r['iou'] for r in rs])), 3))
    if X: item['ban_dung'] = sum(r['st_x'] == 'dung' for r in rs)
    case_list.append(item)
case_list.sort(key=lambda x: (CAUSES.index(x['cause']), -x['n_err']))
if X:  # them cac ca so sanh: ban sua duoc / ban lam hong
    for tag, cond in (('ban_sua_duoc', lambda r: r['st'] != 'dung' and r['st_x'] == 'dung'),
                      ('ban_lam_hong', lambda r: r['st'] == 'dung' and r['st_x'] != 'dung')):
        grp = collections.defaultdict(list)
        for r in rows:
            if cond(r): grp[(r['ctx'], r['gt'])].append(r)
        for (c, g), rs in sorted(grp.items(), key=lambda kv: -len(kv[1])):
            rs.sort(key=lambda r: r['ts'])
            case_list.append(dict(ctx=int(c), context=ctxs[c], gt=g, cause=tag, n_err=len(rs),
                                  n_life=len(G.track(c, g)), ts=[r['ts'] for r in rs],
                                  ref_tracks=sorted({r['ref_track'] for r in rs} - {''}),
                                  in_tracks=[], iou_tb=round(float(np.mean([r['iou'] for r in rs])), 3)))

json.dump(summ, open(os.path.join(p.out, 'tong_hop.json'), 'w'), ensure_ascii=False, indent=1)
json.dump(case_list, open(os.path.join(p.out, 'cac_ca.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))
print(f'{len(case_list)} ca -> {p.out}/cac_ca.json')

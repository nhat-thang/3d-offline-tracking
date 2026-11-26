"""Ve cac ca loi cua CTRL (tu cac_ca.json cua analyze_ctrl_errors.py): nhin tu tren xuong quanh mot xe GT, qua nhieu frame,
kem duong IoU theo thoi gian. Tat ca nguon dung chung mot du lieu (cung canh, frame, diem LiDAR, GT); chi khac lop box.

  diem LiDAR: xam      GT: xanh la (xe dang xet dam, xe khac nhat)
  CTRL truoc refine (sau keo dai): cam, net dut      CTRL sau refine: do      nguon so sanh (--compare): xanh duong
  Vach tu tam ra mui xe = huong dau xe.

Chay (trong VS Code, chi can numpy + matplotlib):
  python waymo/viz_ctrl_cases.py --npz offline/results/waymo/subset8/npz --cases offline/results/waymo/subset8/loi_ctrl/cac_ca.json
         --kitti data/waymo/kitti_format --out offline/results/waymo/subset8/loi_ctrl/anh [--compare ten_nguon] [--max_per_cause 6]
Mo <out>/index.html de xem."""
import argparse, collections, html, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from boxgeo import Source, corners, iou3d, load_points, load_ts2idx

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--cases', required=True)
ap.add_argument('--kitti', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--refine', default='ctrl_refine'); ap.add_argument('--input', default='ctrl_extend')
ap.add_argument('--compare', default=''); ap.add_argument('--compare_label', default='Bản của bạn')
ap.add_argument('--max_per_cause', type=int, default=6); ap.add_argument('--n_frames', type=int, default=6)
ap.add_argument('--half', type=float, default=12.0, help='nua canh vung ve quanh xe (m)')
ap.add_argument('--score_thr', type=float, default=0.3)
p = ap.parse_args()
os.makedirs(p.out, exist_ok=True)
L = lambda n: Source(os.path.join(p.npz, f'boxes_{n}.npz'))
G, R, I = L('gt'), L(p.refine), L(p.input)
X = L(p.compare) if p.compare else None
ts2idx = load_ts2idx(p.kitti)

LAYERS = [('input', I, '#f28e2b', '--', 'CTRL trước refine'), ('refine', R, '#d62728', '-', 'CTRL sau refine')]
if X: LAYERS.append(('compare', X, '#1f77b4', '-', p.compare_label))

MO_TA = {
    'khong_co_dau_vao': 'Track đầu vào không có box nào gần xe: lỗi từ detector/tracker, refine không có gì để sửa.',
    'refine_ha_score': 'Đầu vào có box, nhưng sau refine score thấp hơn ngưỡng hoặc box bị xóa.',
    'track_nhay_vat': 'Track đầu vào đi qua hai xe khác nhau; refine gom điểm của cả hai xe.',
    'khung_keo_dai': 'Box đầu vào ở frame này do bước kéo dài về quá khứ tạo ra.',
    'track_trung': 'Cùng frame có hai track đầu vào cho cùng một xe.',
    'track_dut': 'Xe bị chia cho nhiều track đầu vào nối tiếp nhau; track ở frame này phủ dưới 70% đời xe.',
    'lat_huong_dau_vao': 'Box đầu vào đã ngược hướng so với GT.',
    'it_diem': 'Track sạch nhưng xe có rất ít điểm LiDAR (xa hoặc bị che).',
    'refine_sai': 'Track đầu vào sạch, xe đủ điểm, refine vẫn cho box sai.',
    'ban_sua_duoc': 'CTRL sai, bản so sánh đúng.',
    'ban_lam_hong': 'CTRL đúng, bản so sánh sai.',
}


def draw_box(ax, b, color, ls='-', lw=1.5, alpha=1.0, label=None):
    c = corners(b)
    ax.add_patch(Polygon(c, closed=True, fill=False, ec=color, ls=ls, lw=lw, alpha=alpha))
    f = b[:2] + np.array([np.cos(b[6]), np.sin(b[6])]) * b[3] / 2
    ax.plot([b[0], f[0]], [b[1], f[1]], color=color, lw=lw, alpha=alpha)
    if label: ax.text(c[:, 0].max(), c[:, 1].max(), label, color=color, fontsize=6, alpha=alpha)


def best_iou(S, c, t, gb, thr):
    best, bj = 0.0, -1
    for j in S.in_frame(c, t):
        if S.score[j] < thr or np.linalg.norm(S.box[j, :2] - gb[:2]) > 4: continue
        v = iou3d(gb, S.box[j])
        if v > best: best, bj = v, j
    return best, bj


pts_cache = {}
def points(t):
    k = ts2idx.get(int(t))
    if k is None: return np.zeros((0, 6), np.float32)
    if k not in pts_cache:
        if len(pts_cache) > 30: pts_cache.clear()
        pts_cache[k] = load_points(p.kitti, k)
    return pts_cache[k]


def render(case, fname):
    c, g = case['ctx'], case['gt']
    life = G.track(c, g); lt = np.array([G.ts[i] for i in life])
    t0 = lt[0]; err = set(case['ts'])
    # chon frame: trai deu tren doan sai, them 1 frame truoc va sau neu co
    e = sorted(err); lo, hi = np.searchsorted(lt, e[0]), np.searchsorted(lt, e[-1])
    span = list(range(max(0, lo - 3), min(len(lt), hi + 4)))
    pick = [span[int(round(k))] for k in np.linspace(0, len(span) - 1, min(p.n_frames, len(span)))]
    thr = {'input': -1.0, 'refine': p.score_thr, 'compare': p.score_thr}

    fig = plt.figure(figsize=(3.2 * len(pick), 6.4))
    gs = fig.add_gridspec(2, len(pick), height_ratios=[3, 1.3])
    for k, li in enumerate(pick):
        ax = fig.add_subplot(gs[0, k]); i = life[li]; t = int(G.ts[i]); gb = G.box[i]; cx, cy = gb[:2]
        P = points(t)
        m = (np.abs(P[:, 0] - cx) < p.half) & (np.abs(P[:, 1] - cy) < p.half) & (P[:, 2] > gb[2] - gb[5] / 2 + 0.2)
        ax.scatter(P[m, 0], P[m, 1], s=0.3, c='#9a9a9a', linewidths=0)
        for j in G.in_frame(c, t):  # GT
            if np.abs(G.box[j, :2] - [cx, cy]).max() > p.half + 3: continue
            me = j == i
            draw_box(ax, G.box[j], '#2ca02c', lw=2.2 if me else 0.8, alpha=1 if me else 0.5)
        txt = []
        for name, S, col, ls, _ in LAYERS:
            for j in S.in_frame(c, t):
                if S.score[j] < thr[name] or np.abs(S.box[j, :2] - [cx, cy]).max() > p.half + 3: continue
                near = np.linalg.norm(S.box[j, :2] - [cx, cy]) < 4
                draw_box(ax, S.box[j], col, ls, lw=1.6 if near else 0.7, alpha=1 if near else 0.45,
                         label=('#' + str(S.id[j])[-4:]) if near else None)
            v, _ = best_iou(S, c, t, gb, thr[name]); txt.append(f'{v:.2f}')
        ax.set_xlim(cx - p.half, cx + p.half); ax.set_ylim(cy - p.half, cy + p.half); ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        mark = ' (SAI)' if t in err else ''
        ax.set_title(f't = {(t - t0) / 1e6:.1f} s{mark}\nIoU: ' + ' / '.join(txt) + f'\nđiểm trong GT: {int(G.npts[i])}',
                     fontsize=7, color='#d62728' if mark else 'black')
    # duong IoU theo thoi gian
    ax = fig.add_subplot(gs[1, :])
    tt = (lt - t0) / 1e6
    for name, S, col, ls, lab in LAYERS:
        ax.plot(tt, [best_iou(S, c, int(G.ts[i]), G.box[i], thr[name])[0] for i in life], color=col, ls=ls, lw=1.2, label=lab)
    ax.axhline(0.7, color='k', lw=0.6, ls=':'); ax.set_ylim(0, 1.02); ax.set_xlim(tt[0], tt[-1])
    for e_ in err: ax.axvspan((e_ - t0) / 1e6 - 0.05, (e_ - t0) / 1e6 + 0.05, color='#d62728', alpha=0.08, lw=0)
    for li in pick: ax.axvline(tt[li], color='gray', lw=0.5)
    ax.set_xlabel('thời gian trong đời xe (s); nền đỏ = frame CTRL sai; đường chấm = ngưỡng IoU 0,7', fontsize=7)
    ax.set_ylabel('IoU với GT', fontsize=7); ax.tick_params(labelsize=6); ax.legend(fontsize=6, loc='lower left', ncol=3)
    fig.suptitle(f"{case['cause']} | cảnh {case['context'][:20]}… | xe GT {g[:8]} | sai {case['n_err']}/{case['n_life']} frame"
                 f" | track CTRL {', '.join('#' + x[-4:] for x in case['ref_tracks'][:4])}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(p.out, fname), dpi=110); plt.close(fig)


cases = json.load(open(p.cases))
by = collections.defaultdict(list)
for cs in cases: by[cs['cause']].append(cs)
summ_path = os.path.join(os.path.dirname(p.cases), 'tong_hop.json')
summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}

parts = ['<html><head><meta charset="utf-8"><title>Lỗi CTRL sau refine</title><style>body{font-family:sans-serif;margin:20px}'
         'img{width:100%;border:1px solid #ccc;margin:6px 0}h2{margin-top:28px}td,th{padding:2px 8px}</style></head><body>',
         '<h1>Các ca lỗi của CTRL sau refine (Waymo, 8 cảnh, lớp xe)</h1>',
         '<p>Xám: điểm LiDAR. Xanh lá: GT. Cam nét đứt: CTRL trước refine (sau kéo dài). Đỏ: CTRL sau refine.'
         + (f' Xanh dương: {html.escape(p.compare_label)}.' if X else '') + ' Vạch từ tâm ra = hướng đầu xe. '
         'Số IoU trên mỗi ô theo thứ tự các lớp box trên.</p>']
if summ:
    parts.append('<h3>Tổng hợp</h3><table border=1 style="border-collapse:collapse"><tr><th>Nguyên nhân</th><th>Số box GT sai</th>'
                 '<th>Trong đó LEVEL_2</th><th>Box đầu vào đã đúng (IoU ≥ 0,7)</th></tr>')
    for k, v in summ.get('nguyen_nhan', {}).items():
        parts.append(f'<tr><td>{k}</td><td>{v}</td><td>{summ["nguyen_nhan_level2"][k]}</td>'
                     f'<td>{summ.get("dau_vao_da_dung_iou07", {}).get(k, "")}</td></tr>')
    parts.append(f'</table><p>GT có điểm LiDAR: {summ["so_gt_co_diem"]}, CTRL đúng: {summ["ctrl_dung"]}. '
                 f'Trạng thái: {html.escape(json.dumps(summ["trang_thai"], ensure_ascii=False))}</p>')
    if 'so_sanh' in summ:
        parts.append(f'<p>So sánh: {html.escape(json.dumps(summ["so_sanh"], ensure_ascii=False))}</p>')
for cause, lst in by.items():
    parts.append(f'<h2>{cause} ({len(lst)} ca)</h2><p>{MO_TA.get(cause, "")}</p>')
    for n, cs in enumerate(lst[:p.max_per_cause]):
        fn = f'{cause}_{n:02d}.png'
        render(cs, fn); print('ve', fn)
        parts.append(f'<img src="{fn}">')
parts.append('</body></html>')
open(os.path.join(p.out, 'index.html'), 'w', encoding='utf-8').write('\n'.join(parts))
print('xong:', os.path.join(p.out, 'index.html'))

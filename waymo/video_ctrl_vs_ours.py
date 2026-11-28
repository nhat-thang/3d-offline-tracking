"""Video so sanh CTRL va phuong phap de xuat tren cac ca cu the (Waymo, lop xe, 8 chuoi).

Hai video co CUNG dong thoi gian (cung ca, cung frame, cung thu tu), chi khac lop box:
  video_ctrl.mp4 : diem LiDAR + GT (xanh la) + box CTRL sau refine (do)
  video_ours.mp4 : diem LiDAR + GT (xanh la) + box ban de xuat sau refine (xanh duong)
Moi khung ghi IoU voi GT, lech huong, trang thai DUNG/SAI (dung = IoU 3D >= 0,7 va lech huong < 30 do, giong
analyze_ctrl_errors.py). Video ban de xuat co dai bao "SUA DUOC" (CTRL sai, ban dung) va "LAM SAI" (CTRL dung, ban sai);
thanh thoi gian o duoi danh dau doan sua duoc (xanh) / lam sai (do) cua ca video. File moc_thoi_gian.md ghi giay cu the.

Cac buoc (moi buoc chay rieng duoc):
  --plan            chon ca va doan frame -> <out>/plan.json, moc_thoi_gian.md
  --render ctrl|ours [--part i/n]   ve khung hinh -> <out>/frames_<ten>/
  --encode          ghep video -> <out>/video_ctrl.mp4, video_ours.mp4 (ffmpeg)

Chay: python waymo/video_ctrl_vs_ours.py --npz offline/results/waymo/subset8/npz --kitti data/waymo/kitti_format
        --out offline/results/waymo/subset8/video --plan | --render ctrl | --render ours | --encode"""
import argparse, json, os, subprocess
import numpy as np
from boxgeo import Source, corners, iou3d, load_points, load_ts2idx, yaw_diff

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--kitti', default='data/waymo/kitti_format'); ap.add_argument('--out', required=True)
ap.add_argument('--ctrl', default='ctrl_refine'); ap.add_argument('--ours', default='ours_refine')
ap.add_argument('--plan', action='store_true'); ap.add_argument('--render', default=''); ap.add_argument('--part', default='0/1')
ap.add_argument('--encode', action='store_true')
ap.add_argument('--n_fix', type=int, default=4); ap.add_argument('--n_typical', type=int, default=1); ap.add_argument('--n_break', type=int, default=2)
ap.add_argument('--fps', type=int, default=5); ap.add_argument('--pad', type=int, default=8); ap.add_argument('--max_len', type=int, default=50)
ap.add_argument('--card_frames', type=int, default=12); ap.add_argument('--half', type=float, default=14.0)
ap.add_argument('--score_thr', type=float, default=0.3)
p = ap.parse_args()
os.makedirs(p.out, exist_ok=True)
D2R = np.pi / 180
L = lambda n: Source(os.path.join(p.npz, f'boxes_{n}.npz'))
ctxs = list(np.load(os.path.join(p.npz, 'contexts.npy')))
TEN = {'ctrl': 'CTRL (refine không TTA + empty-box removal)', 'ours': 'Bản đề xuất (track linking + smoothing → refine CTRL không TTA + empty-box removal)'}
MAU = {'ctrl': '#d62728', 'ours': '#1f77b4'}


def best(S, c, t, gb):
    """Box cua S gan GT nhat theo IoU (score >= nguong, tam < 4 m): (iou, lech huong do, chi so) hoac (0, None, -1)."""
    bi, bv = -1, 0.0
    js = np.asarray(S.in_frame(c, t), int)
    if len(js):
        js = js[(S.score[js] >= p.score_thr) & (np.linalg.norm(S.box[js, :2] - gb[:2], axis=1) <= 4)]
    for j in js:
        v = iou3d(gb, S.box[j])
        if v > bv: bv, bi = v, int(j)
    return bv, (yaw_diff(gb[6], S.box[bi, 6]) / D2R if bi >= 0 else None), bi


ok = lambda v, y: v >= 0.7 and y is not None and y < 30


def plan():
    G, C, X = L('gt'), L(p.ctrl), L(p.ours)
    cand = []
    for (c, g), idx in G.tracks.items():
        idx = [i for i in idx if G.npts[i] > 0]
        if len(idx) < 10: continue
        fr = []
        for i in idx:
            t = int(G.ts[i]); vc, yc, _ = best(C, c, t, G.box[i]); vx, yx, _ = best(X, c, t, G.box[i])
            fr.append(dict(ts=t, ic=vc, yc=yc, ix=vx, yx=yx, okc=ok(vc, yc), okx=ok(vx, yx), npts=int(G.npts[i])))
        fix = [k for k, f in enumerate(fr) if not f['okc'] and f['okx']]
        brk = [k for k, f in enumerate(fr) if f['okc'] and not f['okx']]
        if not fix and not brk: continue
        gain = sum(fr[k]['ix'] - fr[k]['ic'] + (0.5 if fr[k]['yc'] is not None and fr[k]['yc'] > 90 else 0) for k in fix)
        loss = sum(fr[k]['ic'] - fr[k]['ix'] + (0.5 if fr[k]['yx'] is not None and fr[k]['yx'] > 90 else 0) for k in brk)
        cand.append(dict(ctx=int(c), gt=str(g), frames=fr, fix=fix, brk=brk, gain=gain, loss=loss,
                         margin_fix=float(np.mean([fr[k]['ix'] - fr[k]['ic'] for k in fix])) if fix else 0.0))

    def segment(fr, ev):
        # doan lien tuc dai nhat cua cac frame su kien (cho phep ho <= 3 frame), them pad hai dau, cat max_len
        runs, s = [], [ev[0], ev[0]]
        for k in ev[1:]:
            if k - s[1] <= 4: s[1] = k
            else: runs.append(s); s = [k, k]
        runs.append(s)
        a, b = max(runs, key=lambda r: r[1] - r[0])
        a, b = max(0, a - p.pad), min(len(fr) - 1, b + p.pad)
        if b - a + 1 > p.max_len: b = a + p.max_len - 1
        return a, b

    used, chosen = set(), []
    def take(lst, kind, n, key):
        for cnd in sorted(lst, key=key, reverse=True):
            if n == 0: break
            if (cnd['ctx'], cnd['gt']) in used: continue
            ev = cnd['fix'] if kind != 'lam_sai' else cnd['brk']
            if len(ev) < 3: continue
            a, b = segment(cnd['frames'], ev)
            used.add((cnd['ctx'], cnd['gt'])); n -= 1
            chosen.append(dict(kind=kind, ctx=cnd['ctx'], context=ctxs[cnd['ctx']], gt=cnd['gt'], frames=cnd['frames'][a:b + 1],
                               t0_life=cnd['frames'][0]['ts']))
    # ca sua duoc ro rang: chenh IoU lon / sua lat huong; ca dien hinh: nhieu frame sua nhat (thuong sat nguong 0,7); ca lam sai
    take([x for x in cand if x['fix']], 'sua_duoc', p.n_fix, lambda x: x['gain'])
    take([x for x in cand if x['fix']], 'sua_duoc_dien_hinh', p.n_typical, lambda x: len(x['fix']))
    take([x for x in cand if x['brk']], 'lam_sai', p.n_break, lambda x: x['loss'])

    # dong thoi gian chung: the tieu de + frame cua tung ca
    timeline, moc = [], []
    for n, cs in enumerate(chosen):
        start = len(timeline) / p.fps
        timeline += [dict(card=n)] * p.card_frames
        for k, f in enumerate(cs['frames']):
            timeline.append(dict(case=n, k=k))
        cs['video_start_s'] = round(start, 1)
        # doan su kien theo giay video
        segs, cur = [], None
        for k, f in enumerate(cs['frames']):
            ev = (not f['okc'] and f['okx']) if cs['kind'] != 'lam_sai' else (f['okc'] and not f['okx'])
            sec = (start * p.fps + p.card_frames + k) / p.fps
            if ev and cur is None: cur = [sec, sec]
            elif ev: cur[1] = sec
            elif cur is not None: segs.append(cur); cur = None
        if cur is not None: segs.append(cur)
        cs['video_events_s'] = [[round(a, 1), round(b + 1 / p.fps, 1)] for a, b in segs]
    json.dump(dict(fps=p.fps, card_frames=p.card_frames, cases=chosen, timeline=timeline), open(os.path.join(p.out, 'plan.json'), 'w'), indent=1)

    TIEU = {'sua_duoc': 'Bản đề xuất sửa được lỗi của CTRL', 'sua_duoc_dien_hinh': 'Bản đề xuất sửa được lỗi của CTRL (ca điển hình, sát threshold IoU 0,7)',
            'lam_sai': 'Bản đề xuất làm sai chỗ CTRL đúng'}
    lines = ['# Mốc thời gian video so sánh CTRL và bản đề xuất', '',
             f'Hai video `video_ctrl.mp4` và `video_ours.mp4` có cùng dòng thời gian ({len(timeline) / p.fps:.1f} s, {p.fps} khung/giây, '
             'mỗi khung là một frame LiDAR 0,1 s). Đúng = IoU 3D với GT ≥ 0,7 và heading error < 30°.', '',
             '| Ca | Bắt đầu (giây video) | Loại | Cảnh | Đoạn sự kiện (giây video) | IoU CTRL → bản đề xuất (trung bình trên đoạn sự kiện) |',
             '|---|---|---|---|---|---|']
    for n, cs in enumerate(chosen):
        evk = [k for k, f in enumerate(cs['frames']) if ((not f['okc'] and f['okx']) if cs['kind'] != 'lam_sai' else (f['okc'] and not f['okx']))]
        ic = np.mean([cs['frames'][k]['ic'] for k in evk]); ix = np.mean([cs['frames'][k]['ix'] for k in evk])
        flips = sum(1 for k in evk if (cs['frames'][k]['yc'] or 0) > 90 or (cs['frames'][k]['yx'] or 0) > 90)
        ev = ', '.join(f'{a:.1f}–{b:.1f}' for a, b in cs['video_events_s'])
        lines.append(f"| {n + 1} | {cs['video_start_s']:.1f} | {TIEU[cs['kind']]}{f' ({flips} frame ngược hướng)' if flips else ''} | "
                     f"{cs['context'][:20]}… | {ev} | {ic:.2f} → {ix:.2f} |")
    open(os.path.join(p.out, 'moc_thoi_gian.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def render(name):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    pl = json.load(open(os.path.join(p.out, 'plan.json'))); cases, tl = pl['cases'], pl['timeline']
    G = L('gt'); S = L(p.ctrl if name == 'ctrl' else p.ours); ts2idx = load_ts2idx(p.kitti)
    od = os.path.join(p.out, f'frames_{name}'); os.makedirs(od, exist_ok=True)
    i0, nparts = map(int, p.part.split('/'))
    N = len(tl); T = N / pl['fps']
    # thanh thoi gian: mau tung khung
    bar = []
    for e in tl:
        if 'card' in e: bar.append('#bbbbbb'); continue
        cs = cases[e['case']]; f = cs['frames'][e['k']]
        if cs['kind'] != 'lam_sai' and not f['okc'] and f['okx']: bar.append('#2ca02c')
        elif cs['kind'] == 'lam_sai' and f['okc'] and not f['okx']: bar.append('#d62728')
        else: bar.append('#e8e8e8')
    TIEU = {'sua_duoc': 'Bản đề xuất sửa được lỗi của CTRL', 'sua_duoc_dien_hinh': 'Bản đề xuất sửa được lỗi của CTRL (điển hình, sát threshold IoU 0,7)',
            'lam_sai': 'Bản đề xuất làm sai chỗ CTRL đúng'}
    for fi in range(N):
        if fi % nparts != i0: continue
        fn = os.path.join(od, f'{fi:05d}.png')
        if os.path.exists(fn): continue
        e = tl[fi]
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[12, 1], hspace=0.12, wspace=0.05, left=0.03, right=0.98, top=0.93, bottom=0.04)
        fig.suptitle(TEN[name], fontsize=13, color=MAU[name], fontweight='bold')
        axb = fig.add_subplot(gs[1, :])  # thanh thoi gian
        for k, col in enumerate(bar): axb.add_patch(Rectangle((k / pl['fps'], 0), 1 / pl['fps'], 1, color=col, lw=0))
        axb.axvline(fi / pl['fps'], color='k', lw=2); axb.set_xlim(0, T); axb.set_ylim(0, 1); axb.set_yticks([])
        axb.set_xticks(np.arange(0, T + 0.01, 5)); axb.tick_params(labelsize=7)
        axb.set_title('xanh = đoạn bản đề xuất sửa được lỗi của CTRL, đỏ = đoạn bản đề xuất làm sai, xám = thẻ tiêu đề; vạch đen = vị trí hiện tại (giây)',
                      fontsize=7, loc='left')
        if 'card' in e:
            cs = cases[e['card']]; ax = fig.add_subplot(gs[0, :]); ax.axis('off')
            ax.text(0.5, 0.62, f"Ca {e['card'] + 1}/{len(cases)}", ha='center', fontsize=26, fontweight='bold')
            ax.text(0.5, 0.45, TIEU[cs['kind']], ha='center', fontsize=17,
                    color='#2ca02c' if cs['kind'] != 'lam_sai' else '#d62728')
            ax.text(0.5, 0.32, f"Cảnh {cs['context']} | xe GT {cs['gt'][:10]} | {len(cs['frames'])} frame", ha='center', fontsize=11, color='#555')
            fig.savefig(fn); plt.close(fig); continue
        cs = cases[e['case']]; f = cs['frames'][e['k']]; c = cs['ctx']; t = f['ts']
        gi = [i for i in G.in_frame(c, t) if str(G.id[i]) == cs['gt']][0]; gb = G.box[gi]; cx, cy = gb[:2]
        ax = fig.add_subplot(gs[0, 0])
        k = ts2idx.get(t)
        if k is not None:
            P = load_points(p.kitti, k)
            m = (np.abs(P[:, 0] - cx) < p.half) & (np.abs(P[:, 1] - cy) < p.half) & (P[:, 2] > gb[2] - gb[5] / 2 + 0.2)
            ax.scatter(P[m, 0], P[m, 1], s=0.6, c='#8a8a8a', linewidths=0)
        def draw(b, col, lw, a=1.0, ls='-'):
            q = corners(b); ax.add_patch(Polygon(q, closed=True, fill=False, ec=col, lw=lw, alpha=a, ls=ls))
            fr = b[:2] + np.array([np.cos(b[6]), np.sin(b[6])]) * b[3] / 2
            ax.plot([b[0], fr[0]], [b[1], fr[1]], color=col, lw=lw, alpha=a)
        for j in G.in_frame(c, t):
            if np.abs(G.box[j, :2] - [cx, cy]).max() > p.half + 3: continue
            draw(G.box[j], '#2ca02c', 3.0 if j == gi else 1.0, 1 if j == gi else 0.5)
        _, _, bj = best(S, c, t, gb)
        for j in S.in_frame(c, t):
            if S.score[j] < p.score_thr or np.abs(S.box[j, :2] - [cx, cy]).max() > p.half + 3: continue
            draw(S.box[j], MAU[name], 2.2 if j == bj else 1.0, 1 if j == bj else 0.5)
        ax.set_xlim(cx - p.half, cx + p.half); ax.set_ylim(cy - p.half, cy + p.half); ax.set_aspect('equal'); ax.tick_params(labelsize=7)
        ax.set_title('Nhìn từ trên xuống (m), xe đang xét ở giữa. Vạch từ tâm ra = hướng đầu xe', fontsize=9)
        # bang thong tin
        axt = fig.add_subplot(gs[0, 1]); axt.axis('off')
        v, y, okk = (f['ic'], f['yc'], f['okc']) if name == 'ctrl' else (f['ix'], f['yx'], f['okx'])
        info = [(f"Ca {e['case'] + 1}/{len(cases)}: {TIEU[cs['kind']]}", 10, 'black'),
                (f"Cảnh {cs['context'][:24]}…", 9, '#555'),
                (f"Thời điểm trong đời xe: {(t - cs['t0_life']) / 1e6:.1f} s", 10, 'black'),
                (f"Điểm LiDAR trong box GT: {f['npts']}", 10, 'black'), ('', 6, 'black'),
                (f"IoU 3D với GT: {v:.2f}" if y is not None else 'Không có box (score ≥ 0,3) gần xe', 14, 'black'),
                (f"Heading error: {y:.0f}°" if y is not None else '', 14, 'black'),
                ('ĐÚNG' if okk else 'SAI', 26, '#2ca02c' if okk else '#d62728')]
        yy = 0.97
        for txt, fs, col in info:
            axt.text(0.02, yy, txt, fontsize=fs, color=col, va='top', fontweight='bold' if fs >= 14 else 'normal', wrap=True); yy -= 0.006 * fs + 0.03
        if name == 'ours':
            if not f['okc'] and f['okx']: axt.text(0.02, 0.13, 'SỬA ĐƯỢC LỖI CỦA CTRL', fontsize=17, color='white', fontweight='bold', bbox=dict(fc='#2ca02c', ec='none', pad=8))
            elif f['okc'] and not f['okx']: axt.text(0.02, 0.13, 'SAI (CTRL ĐÚNG Ở FRAME NÀY)', fontsize=15, color='white', fontweight='bold', bbox=dict(fc='#d62728', ec='none', pad=8))
        else:
            if not f['okc']: axt.text(0.02, 0.13, 'CTRL SAI Ở FRAME NÀY', fontsize=17, color='white', fontweight='bold', bbox=dict(fc='#d62728', ec='none', pad=8))
        axt.text(0.02, 0.04, 'Xanh lá: GT (xe đang xét đậm)   ' + ('Đỏ: CTRL' if name == 'ctrl' else 'Xanh dương: bản đề xuất') + '   Xám: điểm LiDAR',
                 fontsize=8, color='#444')
        fig.savefig(fn); plt.close(fig)
    print('xong', name, p.part)


def encode():
    for name in ('ctrl', 'ours'):
        od = os.path.join(p.out, f'frames_{name}'); out = os.path.join(p.out, f'video_{name}.mp4')
        fps = json.load(open(os.path.join(p.out, 'plan.json')))['fps']
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', os.path.join(od, '%05d.png'),
                        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', out], check=True)
        print(out, os.path.getsize(out) // 1024, 'KB')


if p.plan: plan()
if p.render: render(p.render)
if p.encode: encode()

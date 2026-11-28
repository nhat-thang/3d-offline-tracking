"""Video 5 camera + BEV LiDAR so sanh CTRL va phuong phap de xuat (Waymo, lop xe), cung cac ca va cung dong thoi gian
voi video_ctrl_vs_ours.py (doc plan.json cua no). Box 3D duoc chieu len anh camera bang thong so camera cua Waymo v2:
  extrinsic (camera -> he toa do xe, ma tran 4x4) va intrinsic (f_u, f_v, c_u, c_v); he toa do camera Waymo:
  x huong ve truoc, y sang trai, z len tren -> u = c_u - f_u * y / x, v = c_v - f_v * z / x (bo qua meo ong kinh).
Hang tren: truoc-trai, truoc, truoc-phai; hang duoi: ben trai, ben phai, bang thong tin; ben phai: BEV quanh xe tu lai.
  GT: xanh la (xe dang xet dam)   CTRL: do   ban de xuat: xanh duong   diem LiDAR: trang (BEV)
Chi can numpy, pandas, pyarrow, Pillow, ffmpeg.

Chay: python waymo/video_cam_bev.py --npz offline/results/waymo/subset8/npz --video_dir offline/results/waymo/subset8/video
        --render ctrl | --render ours | --encode"""
import argparse, io, json, os, subprocess, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from boxgeo import Source, iou3d, load_points, load_ts2idx, yaw_diff

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--video_dir', required=True)
ap.add_argument('--kitti', default='data/waymo/kitti_format'); ap.add_argument('--v2', default='data/waymo/v2/validation')
ap.add_argument('--ctrl', default='ctrl_refine'); ap.add_argument('--ours', default='ours_refine')
ap.add_argument('--render', default=''); ap.add_argument('--encode', action='store_true'); ap.add_argument('--part', default='0/1')
ap.add_argument('--ids', action='store_true', help='(voi --clean) ghi ID track canh moi box cua phuong phap');
ap.add_argument('--clean', action='store_true', help='chi camera + BEV, khong chu, khong the tieu de, khong thanh thoi gian');
ap.add_argument('--score_thr', type=float, default=0.3); ap.add_argument('--cam_range', type=float, default=75.0)
p = ap.parse_args()
L = lambda n: Source(os.path.join(p.npz, f'boxes_{n}.npz'))
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F = lambda s, b=False: ImageFont.truetype(FONTB if b else FONT, s)
CAMS = [(2, 'TRƯỚC-TRÁI'), (1, 'TRƯỚC'), (3, 'TRƯỚC-PHẢI'), (4, 'BÊN TRÁI'), (5, 'BÊN PHẢI')]  # camera_name cua Waymo
GREEN, RED, BLUE, GREY = (40, 190, 60), (220, 40, 40), (40, 120, 230), (150, 150, 150)
MAU = {'ctrl': RED, 'ours': BLUE}
TEN = {'ctrl': 'CTRL (refine không TTA + empty-box removal)', 'ours': 'Bản đề xuất (track linking + smoothing → refine CTRL không TTA + empty-box removal)'}
TIEU = {'sua_duoc': 'Bản đề xuất sửa được lỗi của CTRL', 'sua_duoc_dien_hinh': 'Bản đề xuất sửa được lỗi của CTRL (điển hình, sát threshold IoU 0,7)',
        'lam_sai': 'Bản đề xuất làm sai chỗ CTRL đúng'}
W, H = 1760, 790          # khung video
CW, CH1, CH2 = 400, 267, 185  # o camera: hang tren 1920x1280 -> 400x267, hang duoi 1920x886 -> 400x185
BEV = 540


def col(df, suffix):
    m = [c for c in df.columns if c.endswith(suffix)]; assert len(m) == 1, (suffix, m); return df[m[0]]


def load_calib(ctx):
    df = pd.read_parquet(os.path.join(p.v2, 'camera_calibration', ctx + '.parquet'))
    cal = {}
    for i in range(len(df)):
        r = df.iloc[[i]]
        cal[int(col(r, 'key.camera_name').iloc[0])] = dict(
            E=np.array(col(r, 'extrinsic.transform').iloc[0], np.float64).reshape(4, 4),
            f=(float(col(r, 'intrinsic.f_u').iloc[0]), float(col(r, 'intrinsic.f_v').iloc[0])),
            c=(float(col(r, 'intrinsic.c_u').iloc[0]), float(col(r, 'intrinsic.c_v').iloc[0])),
            wh=(int(col(r, 'width').iloc[0]), int(col(r, 'height').iloc[0])))
    return cal


def load_images(ctx, ts_list):
    path = os.path.join(p.v2, 'camera_image', ctx + '.parquet')
    names = pq.read_schema(path).names
    k_ts = [n for n in names if n.endswith('key.frame_timestamp_micros')][0]
    k_cam = [n for n in names if n.endswith('key.camera_name')][0]
    k_img = [n for n in names if n.endswith('CameraImageComponent].image')][0]
    t = pq.read_table(path, columns=[k_ts, k_cam, k_img], filters=[(k_ts, 'in', [int(x) for x in ts_list])]).to_pandas()
    return {(int(a), int(b)): c for a, b, c in zip(t[k_ts], t[k_cam], t[k_img])}


def box_corners3d(b):
    cx, cy, cz, l, w, h, yaw = b
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2; y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y, cz + z], 1)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7), (0, 5), (1, 4)]  # 2 duong cheo = mat truoc


def project(cal, pts):
    Einv = np.linalg.inv(cal['E']); q = (Einv[:3, :3] @ pts.T + Einv[:3, 3:4]).T  # he toa do camera: x truoc, y trai, z len
    u = cal['c'][0] - cal['f'][0] * q[:, 1] / np.maximum(q[:, 0], 1e-6); v = cal['c'][1] - cal['f'][1] * q[:, 2] / np.maximum(q[:, 0], 1e-6)
    return np.stack([u, v], 1), q[:, 0]


NEAR = 1.0


def to_cam(cal, pts):
    Einv = np.linalg.inv(cal['E']); return (Einv[:3, :3] @ pts.T + Einv[:3, 3:4]).T


def uv_of(cal, q):
    return np.array([cal['c'][0] - cal['f'][0] * q[1] / q[0], cal['c'][1] - cal['f'][1] * q[2] / q[0]])


def draw_box_cam(d, cal, b, color, width, scale, off=(0, 0)):
    q = to_cam(cal, box_corners3d(b))
    if (q[:, 0] > NEAR).sum() == 0: return
    W_, H_ = cal['wh']
    for a, c in EDGES:
        A, C = q[a], q[c]
        if A[0] <= NEAR and C[0] <= NEAR: continue
        if A[0] <= NEAR: A = A + (C - A) * (NEAR - A[0]) / (C[0] - A[0])  # cat canh tai mat phang gan (x = NEAR)
        if C[0] <= NEAR: C = C + (A - C) * (NEAR - C[0]) / (A[0] - C[0])
        ua, uc = uv_of(cal, A), uv_of(cal, C)
        if max(ua[0], uc[0]) < -W_ or min(ua[0], uc[0]) > 2 * W_ or max(ua[1], uc[1]) < -H_ or min(ua[1], uc[1]) > 2 * H_: continue
        d.line([((ua[0] - off[0]) * scale, (ua[1] - off[1]) * scale), ((uc[0] - off[0]) * scale, (uc[1] - off[1]) * scale)], fill=color, width=width)


def render(name):
    pl = json.load(open(os.path.join(p.video_dir, 'plan.json'))); cases, tl, fps = pl['cases'], pl['timeline'], pl['fps']
    G = L('gt'); S = L(p.ctrl if name == 'ctrl' else p.ours); ts2idx = load_ts2idx(p.kitti)
    od = os.path.join(p.video_dir, f'frames_cam_{name}'); os.makedirs(od, exist_ok=True)
    i0, nparts = map(int, p.part.split('/'))
    N = len(tl); T = N / fps
    bar = []
    for e in tl:
        if 'card' in e: bar.append((190, 190, 190)); continue
        cs = cases[e['case']]; f = cs['frames'][e['k']]
        if cs['kind'] != 'lam_sai' and not f['okc'] and f['okx']: bar.append(GREEN)
        elif cs['kind'] == 'lam_sai' and f['okc'] and not f['okx']: bar.append(RED)
        else: bar.append((232, 232, 232))
    cache = {}

    def case_data(n):
        if n not in cache:
            cache.clear(); cs = cases[n]
            cache[n] = (load_calib(cs['context']), load_images(cs['context'], [f['ts'] for f in cs['frames']]))
        return cache[n]

    def best(c, t, gb):
        bi, bv = -1, 0.0
        for j in S.in_frame(c, t):
            if S.score[j] < p.score_thr or np.linalg.norm(S.box[j, :2] - gb[:2]) > 4: continue
            v = iou3d(gb, S.box[j])
            if v > bv: bv, bi = v, j
        return bi

    for fi in range(N):
        if fi % nparts != i0: continue
        fn = os.path.join(od, f'{fi:05d}.png')
        if os.path.exists(fn): continue
        e = tl[fi]
        img = Image.new('RGB', (W, H), (255, 255, 255)); d = ImageDraw.Draw(img)
        d.text((W // 2, 18), TEN[name], fill=MAU[name], font=F(22, True), anchor='mm')
        # thanh thoi gian
        y0, bw = H - 46, (W - 40) / N
        for k, cc in enumerate(bar): d.rectangle([20 + k * bw, y0, 20 + (k + 1) * bw, y0 + 22], fill=cc)
        d.line([20 + fi * bw, y0 - 4, 20 + fi * bw, y0 + 26], fill=(0, 0, 0), width=3)
        for s_ in range(0, int(T) + 1, 5): d.text((20 + s_ * fps * bw, y0 + 26), f'{s_}s', fill=(60, 60, 60), font=F(11))
        d.text((20, y0 - 18), 'Thanh thời gian: xanh = đoạn bản đề xuất sửa được lỗi của CTRL, đỏ = đoạn bản đề xuất làm sai, xám = thẻ tiêu đề; vạch đen = vị trí hiện tại',
               fill=(60, 60, 60), font=F(12))
        if 'card' in e:
            cs = cases[e['card']]
            d.text((W // 2, 300), f"Ca {e['card'] + 1}/{len(cases)}", fill=(0, 0, 0), font=F(44, True), anchor='mm')
            d.text((W // 2, 370), TIEU[cs['kind']], fill=GREEN if cs['kind'] != 'lam_sai' else RED, font=F(30, True), anchor='mm')
            d.text((W // 2, 425), f"Cảnh {cs['context']} | xe GT {cs['gt'][:10]} | {len(cs['frames'])} frame", fill=(90, 90, 90), font=F(18), anchor='mm')
            img.save(fn); continue
        cs = cases[e['case']]; f = cs['frames'][e['k']]; c = cs['ctx']; t = f['ts']
        cal, ims = case_data(e['case'])
        gi = [i for i in G.in_frame(c, t) if str(G.id[i]) == cs['gt']][0]; gb = G.box[gi]; bj = best(c, t, gb)
        gts = [j for j in G.in_frame(c, t) if np.linalg.norm(G.box[j, :2]) < p.cam_range]
        prs = [j for j in S.in_frame(c, t) if S.score[j] >= p.score_thr and np.linalg.norm(S.box[j, :2]) < p.cam_range]
        # camera
        for n, (cam, lab) in enumerate(CAMS):
            x0 = 20 + (n % 3) * CW if n < 3 else 20 + (n - 3) * CW
            yb = 40 if n < 3 else 40 + CH1 + 6
            ch = CH1 if n < 3 else CH2
            if (t, cam) in ims and cam in cal:
                im = Image.open(io.BytesIO(ims[(t, cam)])).convert('RGB'); sc = CW / im.width
                im = im.resize((CW, int(round(im.height * sc)))); di = ImageDraw.Draw(im)
                for j in gts: draw_box_cam(di, cal[cam], G.box[j], GREEN, 3 if j == gi else 1, sc)
                for j in prs: draw_box_cam(di, cal[cam], S.box[j], MAU[name], 3 if j == bj else 1, sc)
                img.paste(im.crop((0, 0, CW, ch)), (x0, yb))
            else:
                d.rectangle([x0, yb, x0 + CW, yb + ch], fill=(30, 30, 30))
            d.rectangle([x0, yb, x0 + 118, yb + 18], fill=(0, 0, 0)); d.text((x0 + 4, yb + 2), lab, fill=(255, 255, 255), font=F(12))
        # BEV quanh xe tu lai (x huong len tren)
        bx, by = 20 + 3 * CW + 20, 40
        bev = Image.new('RGB', (BEV, BEV), (0, 0, 0)); db = ImageDraw.Draw(bev)
        R = float(np.clip(np.linalg.norm(gb[:2]) + 12, 40, 80)); sc = BEV / (2 * R)
        tp = lambda x, y: (BEV / 2 - y * sc, BEV / 2 - x * sc)
        k = ts2idx.get(t)
        if k is not None:
            P = load_points(p.kitti, k); m = (np.abs(P[:, 0]) < R) & (np.abs(P[:, 1]) < R) & (P[:, 2] > 0.3)
            P = P[m][::2]; u = (BEV / 2 - P[:, 1] * sc).astype(int); v = (BEV / 2 - P[:, 0] * sc).astype(int)
            a = np.asarray(bev).copy(); ok = (u >= 0) & (u < BEV) & (v >= 0) & (v < BEV); a[v[ok], u[ok]] = (170, 170, 170); bev = Image.fromarray(a); db = ImageDraw.Draw(bev)
        def bevbox(b, colr, wdt):
            q = box_corners3d(b)[:4]; pts = [tp(x, y) for x, y in q[:, :2]]
            db.polygon(pts, outline=colr, width=wdt); fr = (q[0, :2] + q[1, :2]) / 2
            db.line([tp(*b[:2]), tp(*fr)], fill=colr, width=wdt)
        for j in G.in_frame(c, t): bevbox(G.box[j], GREEN, 3 if j == gi else 1)
        for j in S.in_frame(c, t):
            if S.score[j] >= p.score_thr: bevbox(S.box[j], MAU[name], 3 if j == bj else 1)
        cx, cy = tp(*gb[:2]); rr = max(14, 0.5 * gb[3] * sc + 8); db.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 210, 0), width=2)
        db.polygon([tp(2.4, 0), tp(-2.4, 1), tp(-2.4, -1)], fill=(255, 255, 255))
        img.paste(bev, (bx, by)); d.text((bx + 6, by + 4), f'BEV quanh xe tự lái (±{R:.0f} m), vòng vàng = xe đang xét', fill=(255, 255, 255), font=F(12))
        # bang thong tin (o duoi ben phai camera hang duoi)
        ix, iy = 20 + 2 * CW + 10, 40 + CH1 + 6
        v_, y_, ok_ = (f['ic'], f['yc'], f['okc']) if name == 'ctrl' else (f['ix'], f['yx'], f['okx'])
        d.text((ix, iy), f"Ca {e['case'] + 1}/{len(cases)}", fill=(0, 0, 0), font=F(16, True))
        d.text((ix, iy + 22), TIEU[cs['kind']], fill=(60, 60, 60), font=F(11))
        d.text((ix, iy + 40), f"Thời điểm trong đời xe: {(t - cs['t0_life']) / 1e6:.1f} s | điểm LiDAR trong GT: {f['npts']}", fill=(0, 0, 0), font=F(12))
        d.text((ix, iy + 62), (f"IoU 3D với GT: {v_:.2f}   heading error: {y_:.0f}°" if y_ is not None else 'Không có box (score ≥ 0,3) gần xe'), fill=(0, 0, 0), font=F(15, True))
        d.text((ix, iy + 88), 'ĐÚNG' if ok_ else 'SAI', fill=GREEN if ok_ else RED, font=F(30, True))
        ban = None
        if name == 'ours' and not f['okc'] and f['okx']: ban = ('SỬA ĐƯỢC LỖI CỦA CTRL', GREEN)
        elif name == 'ours' and f['okc'] and not f['okx']: ban = ('SAI (CTRL ĐÚNG Ở FRAME NÀY)', RED)
        elif name == 'ctrl' and not f['okc']: ban = ('CTRL SAI Ở FRAME NÀY', RED)
        if ban:
            d.rectangle([ix, iy + 132, ix + 378, iy + 172], fill=ban[1]); d.text((ix + 189, iy + 152), ban[0], fill=(255, 255, 255), font=F(17, True), anchor='mm')
        # hang 3: phong to xe dang xet tren camera + BEV phong to
        zy = 40 + CH1 + 6 + CH2 + 8; ZW, ZH = 400, 220
        bestcam, bd = None, 1e9
        for cam, _ in CAMS:
            if cam not in cal or (t, cam) not in ims: continue
            q = to_cam(cal[cam], gb[None, :3])[0]
            if q[0] <= NEAR: continue
            u = uv_of(cal[cam], q); W_, H_ = cal[cam]['wh']
            if 0 <= u[0] < W_ and 0 <= u[1] < H_:
                dd = abs(u[0] - W_ / 2) / W_
                if dd < bd: bd, bestcam = dd, cam
        if bestcam is not None:
            cc_ = cal[bestcam]; full = Image.open(io.BytesIO(ims[(t, bestcam)])).convert('RGB')
            qc = to_cam(cc_, box_corners3d(gb)); qc = qc[qc[:, 0] > NEAR]
            uvs = np.array([uv_of(cc_, x) for x in qc])
            u0, v0 = uvs.min(0); u1, v1 = uvs.max(0); cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
            ww = max((u1 - u0) * 2.2, (v1 - v0) * 2.2 * ZW / ZH, 160); hh = ww * ZH / ZW
            off = (cu - ww / 2, cv - hh / 2); sc2 = ZW / ww
            crop = full.crop((int(off[0]), int(off[1]), int(off[0] + ww), int(off[1] + hh))).resize((ZW, ZH)); dz = ImageDraw.Draw(crop)
            for j in gts:
                if np.linalg.norm(G.box[j, :2] - gb[:2]) < 15: draw_box_cam(dz, cc_, G.box[j], GREEN, 3 if j == gi else 1, sc2, off)
            for j in prs:
                if np.linalg.norm(S.box[j, :2] - gb[:2]) < 15: draw_box_cam(dz, cc_, S.box[j], MAU[name], 3 if j == bj else 1, sc2, off)
            img.paste(crop, (20, zy))
            lab2 = 'PHÓNG TO XE ĐANG XÉT (camera ' + dict(CAMS)[bestcam] + ')'
        else:
            d.rectangle([20, zy, 20 + ZW, zy + ZH], fill=(30, 30, 30)); lab2 = 'XE ĐANG XÉT KHÔNG NẰM TRONG CAMERA NÀO'
        d.rectangle([20, zy, 20 + 8 * len(lab2), zy + 18], fill=(0, 0, 0)); d.text((24, zy + 2), lab2, fill=(255, 255, 255), font=F(12))
        # BEV phong to quanh xe dang xet
        Z, Rz = 220, 12.0; scz = Z / (2 * Rz); zb = Image.new('RGB', (Z, Z), (0, 0, 0))
        tz = lambda x, y: (Z / 2 - (y - gb[1]) * scz, Z / 2 - (x - gb[0]) * scz)
        if k is not None:
            P = load_points(p.kitti, k); m = (np.abs(P[:, 0] - gb[0]) < Rz) & (np.abs(P[:, 1] - gb[1]) < Rz) & (P[:, 2] > gb[2] - gb[5] / 2 + 0.2)
            u = (Z / 2 - (P[m, 1] - gb[1]) * scz).astype(int); v = (Z / 2 - (P[m, 0] - gb[0]) * scz).astype(int)
            a = np.asarray(zb).copy(); ok = (u >= 0) & (u < Z) & (v >= 0) & (v < Z); a[v[ok], u[ok]] = (200, 200, 200); zb = Image.fromarray(a)
        dzb = ImageDraw.Draw(zb)
        def zbox(b, colr, wdt):
            q = box_corners3d(b)[:4]; dzb.polygon([tz(x, y) for x, y in q[:, :2]], outline=colr, width=wdt)
            dzb.line([tz(*b[:2]), tz(*((q[0, :2] + q[1, :2]) / 2))], fill=colr, width=wdt)
        for j in G.in_frame(c, t):
            if np.abs(G.box[j, :2] - gb[:2]).max() < Rz + 4: zbox(G.box[j], GREEN, 3 if j == gi else 1)
        for j in S.in_frame(c, t):
            if S.score[j] >= p.score_thr and np.abs(S.box[j, :2] - gb[:2]).max() < Rz + 4: zbox(S.box[j], MAU[name], 2 if j == bj else 1)
        img.paste(zb, (20 + ZW + 10, zy)); d.text((20 + ZW + 14, zy + 2), f'BEV phóng to (±{Rz:.0f} m)', fill=(255, 255, 255), font=F(11))
        # chu thich
        lx = 20 + ZW + 10 + Z + 20
        for n_, line in enumerate(['Chú thích', 'Xanh lá: GT, xe đang xét nét đậm',
                                   ('Đỏ: CTRL' if name == 'ctrl' else 'Xanh dương: bản đề xuất') + ', box khớp xe đang xét nét đậm',
                                   'Box 3D chiếu lên ảnh bằng intrinsic + extrinsic', 'của từng camera (bỏ qua méo ống kính)',
                                   'Hai đường chéo trên box = mặt đầu xe', 'BEV: điểm LiDAR (xám), xe tự lái = tam giác trắng',
                                   'Đúng = IoU 3D với GT ≥ 0,7 và heading error < 30°']):
            d.text((lx, zy + 4 + 24 * n_), line, fill=(0, 0, 0) if n_ == 0 else (60, 60, 60), font=F(14, n_ == 0))
        img.save(fn)
    print('xong', name, p.part)


def encode():
    fps = json.load(open(os.path.join(p.video_dir, 'plan.json')))['fps']
    for name in ('ctrl', 'ours'):
        od = os.path.join(p.video_dir, f'frames_cam_{name}'); out = os.path.join(p.video_dir, f'video_cam_{name}.mp4')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', os.path.join(od, '%05d.png'),
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out], check=True)
        print(out, os.path.getsize(out) // 1024, 'KB')


def tid(k):
    k = str(k); return '#' + (k if len(k) <= 6 else k[-6:])


def render_clean(name):
    """Bo cuc gon nhu video nuScenes: 3 camera tren, 2 camera ben o duoi (can giua), BEV ben phai, nen den, khong chu."""
    pl = json.load(open(os.path.join(p.video_dir, 'plan.json'))); cases = pl['cases']
    G = L('gt'); S = L(p.ctrl if name == 'ctrl' else p.ours); ts2idx = load_ts2idx(p.kitti)
    tag = 'cam_id' if p.ids else 'cam_clean'
    od = os.path.join(p.video_dir, f'frames_{tag}_{name}'); os.makedirs(od, exist_ok=True)
    cw, h1, h2 = 480, 320, 221; Bv = h1 + h2; Wc, Hc = 3 * cw + Bv + 1, Bv + 1
    Wc += Wc % 2; Hc += Hc % 2
    seq = [(n, k) for n, cs in enumerate(cases) for k in range(len(cs['frames']))]
    for fi, (n, k) in enumerate(seq):
        fn = os.path.join(od, f'{fi:05d}.png')
        if os.path.exists(fn): continue
        cs = cases[n]; f = cs['frames'][k]; c = cs['ctx']; t = f['ts']
        if not hasattr(render_clean, 'cache') or render_clean.cache[0] != n:
            render_clean.cache = (n, load_calib(cs['context']), load_images(cs['context'], [x['ts'] for x in cs['frames']]))
        _, cal, ims = render_clean.cache
        gi = [i for i in G.in_frame(c, t) if str(G.id[i]) == cs['gt']][0]; gb = G.box[gi]
        bj, bv = -1, 0.0
        for j in S.in_frame(c, t):
            if S.score[j] >= p.score_thr and np.linalg.norm(S.box[j, :2] - gb[:2]) <= 4:
                v = iou3d(gb, S.box[j])
                if v > bv: bv, bj = v, j
        gts = [j for j in G.in_frame(c, t) if np.linalg.norm(G.box[j, :2]) < p.cam_range]
        prs = [j for j in S.in_frame(c, t) if S.score[j] >= p.score_thr and np.linalg.norm(S.box[j, :2]) < p.cam_range]
        img = Image.new('RGB', (Wc, Hc), (0, 0, 0))
        pos = {2: (0, 0), 1: (cw, 0), 3: (2 * cw, 0), 4: (cw // 2, h1), 5: (cw // 2 + cw, h1)}
        for cam, _ in CAMS:
            if (t, cam) not in ims or cam not in cal: continue
            im = Image.open(io.BytesIO(ims[(t, cam)])).convert('RGB'); sc = cw / im.width
            im = im.resize((cw, int(round(im.height * sc)))); di = ImageDraw.Draw(im)
            for j in gts: draw_box_cam(di, cal[cam], G.box[j], GREEN, 3 if j == gi else 1, sc)
            for j in prs: draw_box_cam(di, cal[cam], S.box[j], MAU[name], 3 if j == bj else 1, sc)
            if p.ids:
                for j in prs:
                    q = to_cam(cal[cam], box_corners3d(S.box[j])); qv = q[q[:, 0] > NEAR]
                    if len(qv) < 8: continue
                    uv = np.array([uv_of(cal[cam], x) for x in qv]) * sc
                    if uv[:, 0].max() < 0 or uv[:, 0].min() > cw or uv[:, 1].max() < 0 or uv[:, 1].min() > im.height: continue
                    if uv[:, 1].max() - uv[:, 1].min() < 12: continue
                    lab = tid(S.id[j]); x_, y_ = float(np.clip(uv[:, 0].min(), 0, cw - 40)), float(max(uv[:, 1].min() - 13, 0))
                    fnt = F(11, j == bj); tw = di.textlength(lab, font=fnt)
                    di.rectangle([x_, y_, x_ + tw + 4, y_ + 13], fill=MAU[name] if j == bj else (0, 0, 0)); di.text((x_ + 2, y_), lab, fill=(255, 255, 255), font=fnt)
            img.paste(im.crop((0, 0, cw, h1 if cam in (1, 2, 3) else h2)), pos[cam])
        # BEV quanh xe tu lai
        R = float(np.clip(np.linalg.norm(gb[:2]) + 12, 40, 80)); scb = Bv / (2 * R)
        tp = lambda x, y: (Bv / 2 - y * scb, Bv / 2 - x * scb)
        bev = Image.new('RGB', (Bv, Bv), (0, 0, 0)); kk = ts2idx.get(t)
        if kk is not None:
            P = load_points(p.kitti, kk); m = (np.abs(P[:, 0]) < R) & (np.abs(P[:, 1]) < R) & (P[:, 2] > 0.3)
            u = (Bv / 2 - P[m, 1] * scb).astype(int); v = (Bv / 2 - P[m, 0] * scb).astype(int)
            a = np.asarray(bev).copy(); ok = (u >= 0) & (u < Bv) & (v >= 0) & (v < Bv); a[v[ok], u[ok]] = (225, 225, 225); bev = Image.fromarray(a)
        db = ImageDraw.Draw(bev)
        def bb(b, colr, wdt):
            q = box_corners3d(b)[:4]; db.polygon([tp(x, y) for x, y in q[:, :2]], outline=colr, width=wdt)
            db.line([tp(*b[:2]), tp(*((q[0, :2] + q[1, :2]) / 2))], fill=colr, width=wdt)
        for j in G.in_frame(c, t): bb(G.box[j], GREEN, 3 if j == gi else 1)
        for j in S.in_frame(c, t):
            if S.score[j] >= p.score_thr:
                bb(S.box[j], MAU[name], 3 if j == bj else 1)
                if p.ids:
                    x_, y_ = tp(*S.box[j, :2])
                    if 0 <= x_ < Bv and 0 <= y_ < Bv:
                        db.text((x_ + 6, y_ - 6), tid(S.id[j]), fill=(255, 210, 0) if j == bj else (200, 200, 200), font=F(12 if j == bj else 9, j == bj))
        cx, cy = tp(*gb[:2]); rr = max(14, 0.5 * gb[3] * scb + 8); db.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 210, 0), width=2)
        img.paste(bev, (3 * cw + 1, 0))
        img.save(fn)
    print('xong', name, len(seq), 'khung')


def encode_clean():
    fps = json.load(open(os.path.join(p.video_dir, 'plan.json')))['fps']
    for name in ('ctrl', 'ours'):
        tag = 'cam_id' if p.ids else 'cam_clean'
        od = os.path.join(p.video_dir, f'frames_{tag}_{name}'); out = os.path.join(p.video_dir, f'video_{tag}_{name}.mp4')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', os.path.join(od, '%05d.png'),
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out], check=True)
        print(out, os.path.getsize(out) // 1024, 'KB')


if p.render: (render_clean if p.clean else render)(p.render)
if p.encode: (encode_clean if p.clean else encode)()

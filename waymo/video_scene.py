"""Video tron mot canh Waymo (khoang 20 s, 10 khung/giay = thoi gian that) cho mot mo hinh: 5 camera + BEV LiDAR.
Khong chu thich, khong danh dau xe nao; chi co dong ho (giay trong canh) va ID track canh moi box de tra cuu.
  GT: xanh la   CTRL: do   ban de xuat: xanh duong   diem LiDAR: xam/trang (BEV)
  Tren BEV, doan thang tu tam box ra mat truoc = huong dau xe.
Mot box du doan chi duoc ve khi score >= 0,3.

Chay (tu thu muc goc repo):
  python waymo/video_scene.py --ctx <context> --name ctrl --part 0/5   (lap lai cho tung phan, roi cho ours)
  python waymo/video_scene.py --ctx <context> --encode"""
import argparse, io, os, subprocess
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from boxgeo import Source, load_points, load_ts2idx

ap = argparse.ArgumentParser()
ap.add_argument('--ctx', required=True); ap.add_argument('--name', default='')
ap.add_argument('--npz', default='offline/results/waymo/subset8/npz'); ap.add_argument('--out', default='offline/results/waymo/subset8/video_canh')
ap.add_argument('--kitti', default='data/waymo/kitti_format'); ap.add_argument('--v2', default='data/waymo/v2/validation')
ap.add_argument('--part', default='0/1'); ap.add_argument('--frames', default='', help='thu muc chua khung anh (mac dinh: canh video)'); ap.add_argument('--encode', action='store_true')
ap.add_argument('--score_thr', type=float, default=0.3); ap.add_argument('--R', type=float, default=60.0)
p = ap.parse_args()
SRC = {'ctrl': 'ctrl_refine', 'ours': 'ours_refine'}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F = lambda s, b=False: ImageFont.truetype(FONTB if b else FONT, s)
GREEN, RED, BLUE = (40, 200, 60), (235, 45, 45), (50, 130, 255)
MAU = {'ctrl': RED, 'ours': BLUE}
CAMS = [2, 1, 3, 4, 5]  # truoc-trai, truoc, truoc-phai, ben trai, ben phai
cw, h1 = 400, 267          # 3 camera truoc: 1920x1280 -> 400x267
sw, h2 = 600, 277          # 2 camera ben: 1920x886 -> 600x277
Bv = 960                   # BEV 960x960
Wc, Hc = 3 * cw + Bv, Bv
NEAR = 1.0
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7), (0, 5), (1, 4)]
ctxs = list(np.load(os.path.join(p.npz, 'contexts.npy'))); C = ctxs.index(p.ctx)
od_root = os.path.join(p.out, p.ctx[:20])


def col(df, suffix):
    m = [c for c in df.columns if c.endswith(suffix)]; return df[m[0]]


def load_calib():
    df = pd.read_parquet(os.path.join(p.v2, 'camera_calibration', p.ctx + '.parquet')); cal = {}
    for i in range(len(df)):
        r = df.iloc[[i]]
        cal[int(col(r, 'key.camera_name').iloc[0])] = dict(
            E=np.array(col(r, 'extrinsic.transform').iloc[0], np.float64).reshape(4, 4),
            f=(float(col(r, 'intrinsic.f_u').iloc[0]), float(col(r, 'intrinsic.f_v').iloc[0])),
            c=(float(col(r, 'intrinsic.c_u').iloc[0]), float(col(r, 'intrinsic.c_v').iloc[0])),
            wh=(int(col(r, 'width').iloc[0]), int(col(r, 'height').iloc[0])))
    return cal


def load_images(ts_list):
    path = os.path.join(p.v2, 'camera_image', p.ctx + '.parquet'); names = pq.read_schema(path).names
    k_ts = [n for n in names if n.endswith('key.frame_timestamp_micros')][0]; k_cam = [n for n in names if n.endswith('key.camera_name')][0]
    k_img = [n for n in names if n.endswith('CameraImageComponent].image')][0]
    t = pq.read_table(path, columns=[k_ts, k_cam, k_img], filters=[(k_ts, 'in', [int(x) for x in ts_list])]).to_pandas()
    return {(int(a), int(b)): c for a, b, c in zip(t[k_ts], t[k_cam], t[k_img])}


def corners3d(b):
    cx, cy, cz, l, w, h, yaw = b
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2; y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2; c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y, cz + z], 1)


def to_cam(cal, pts):
    Ei = np.linalg.inv(cal['E']); return (Ei[:3, :3] @ pts.T + Ei[:3, 3:4]).T


def uv_of(cal, q):
    return np.array([cal['c'][0] - cal['f'][0] * q[1] / q[0], cal['c'][1] - cal['f'][1] * q[2] / q[0]])


def draw_cam(d, cal, b, color, width, sc):
    q = to_cam(cal, corners3d(b))
    if (q[:, 0] > NEAR).sum() == 0: return None
    W_, H_ = cal['wh']; pts = []
    for a, c in EDGES:
        A, B = q[a], q[c]
        if A[0] <= NEAR and B[0] <= NEAR: continue
        if A[0] <= NEAR: A = A + (B - A) * (NEAR - A[0]) / (B[0] - A[0])
        if B[0] <= NEAR: B = B + (A - B) * (NEAR - B[0]) / (A[0] - B[0])
        ua, ub = uv_of(cal, A), uv_of(cal, B)
        if max(ua[0], ub[0]) < -W_ or min(ua[0], ub[0]) > 2 * W_ or max(ua[1], ub[1]) < -H_ or min(ua[1], ub[1]) > 2 * H_: continue
        d.line([tuple(ua * sc), tuple(ub * sc)], fill=color, width=width); pts += [ua * sc, ub * sc]
    return np.array(pts) if pts else None


def tid(k):
    k = str(k); return k.split('_')[-1]


def free(boxes, r):
    """r khong chong len nhan nao da ve thi them vao va tra True."""
    for b in boxes:
        if r[0] < b[2] and b[0] < r[2] and r[1] < b[3] and b[1] < r[3]: return False
    boxes.append(r); return True


def render():
    G = Source(os.path.join(p.npz, 'boxes_gt.npz')); S = Source(os.path.join(p.npz, f'boxes_{SRC[p.name]}.npz')); ts2idx = load_ts2idx(p.kitti)
    ts = sorted({t for (c, t) in G.frames if c == C}); t0 = ts[0]
    i0, n = map(int, p.part.split('/')); mine = [k for k in range(len(ts)) if k % n == i0]
    od = os.path.join(p.frames or od_root, f'khung_{p.name}'); os.makedirs(od, exist_ok=True)
    mine = [k for k in mine if not os.path.exists(os.path.join(od, f'{k:05d}.png'))]
    if not mine: print('da co du', p.name, p.part); return
    cal = load_calib(); ims = load_images([ts[k] for k in mine])
    pos = {2: (0, 0), 1: (cw, 0), 3: (2 * cw, 0), 4: (0, h1 + 4), 5: (sw, h1 + 4)}
    for k in mine:
        t = ts[k]; img = Image.new('RGB', (Wc, Hc), (0, 0, 0))
        gts = [j for j in G.in_frame(C, t) if np.linalg.norm(G.box[j, :2]) < 75]
        prs = [j for j in S.in_frame(C, t) if S.score[j] >= p.score_thr and np.linalg.norm(S.box[j, :2]) < 75]
        for cam in CAMS:
            if (t, cam) not in ims or cam not in cal: continue
            ww = cw if cam in (1, 2, 3) else sw
            im = Image.open(io.BytesIO(ims[(t, cam)])).convert('RGB'); sc = ww / im.width
            im = im.resize((ww, int(round(im.height * sc)))); di = ImageDraw.Draw(im)
            for j in gts: draw_cam(di, cal[cam], G.box[j], GREEN, 1, sc)
            used = []
            for j in prs:
                uv = draw_cam(di, cal[cam], S.box[j], MAU[p.name], 2, sc)
                if uv is None or len(uv) < 16: continue
                if uv[:, 0].max() < 0 or uv[:, 0].min() > ww or uv[:, 1].max() < 0 or uv[:, 1].min() > im.height: continue
                if uv[:, 1].max() - uv[:, 1].min() < 10: continue
                lab = tid(S.id[j]); fnt = F(11); x_, y_ = float(np.clip(uv[:, 0].min(), 0, ww - 50)), float(max(uv[:, 1].min() - 14, 0))
                rr = [x_, y_, x_ + di.textlength(lab, font=fnt) + 4, y_ + 13]
                if not free(used, rr): continue
                di.rectangle(rr, fill=(0, 0, 0)); di.text((x_ + 2, y_), lab, fill=(255, 255, 255), font=fnt)
            img.paste(im.crop((0, 0, ww, h1 if cam in (1, 2, 3) else h2)), pos[cam])
        # BEV quanh xe tu lai, truc x (phia truoc) huong len tren
        R = p.R; sb = Bv / (2 * R); tp = lambda x, y: (Bv / 2 - y * sb, Bv / 2 - x * sb)
        bev = Image.new('RGB', (Bv, Bv), (0, 0, 0)); kk = ts2idx.get(int(t))
        if kk is not None:
            P = load_points(p.kitti, kk); m = (np.abs(P[:, 0]) < R) & (np.abs(P[:, 1]) < R) & (P[:, 2] > 0.3)
            u = (Bv / 2 - P[m, 1] * sb).astype(int); v = (Bv / 2 - P[m, 0] * sb).astype(int)
            a = np.asarray(bev).copy(); ok = (u >= 0) & (u < Bv) & (v >= 0) & (v < Bv); a[v[ok], u[ok]] = (150, 150, 150); bev = Image.fromarray(a)
        db = ImageDraw.Draw(bev)

        def bb(b, colr, wdt):
            q = corners3d(b)[:4]; db.polygon([tp(x, y) for x, y in q[:, :2]], outline=colr, width=wdt)
            db.line([tp(*b[:2]), tp(*((q[0, :2] + q[1, :2]) / 2))], fill=colr, width=wdt)
            db.line([tp(*q[0, :2]), tp(*q[1, :2])], fill=colr, width=5 if wdt > 1 else 3)  # canh dau xe ve dam
        for j in G.in_frame(C, t): bb(G.box[j], GREEN, 1)
        used = []
        for j in S.in_frame(C, t):
            if S.score[j] < p.score_thr: continue
            bb(S.box[j], MAU[p.name], 2)
        for j in S.in_frame(C, t):
            if S.score[j] < p.score_thr: continue
            x_, y_ = tp(*S.box[j, :2]); lab = tid(S.id[j]); fnt = F(13); w_ = db.textlength(lab, font=fnt) + 2
            if not (0 <= x_ < Bv and 0 <= y_ < Bv): continue
            for ox, oy in ((10, -8), (-10 - w_, -8), (-w_ / 2, -26), (-w_ / 2, 10), (10, -24), (10, 8), (-10 - w_, -24), (-10 - w_, 8)):
                rr = [x_ + ox, y_ + oy, x_ + ox + w_, y_ + oy + 16]
                if rr[0] >= 0 and rr[2] < Bv and free(used, rr):
                    db.text((rr[0] + 1, rr[1]), lab, fill=(255, 220, 60), font=fnt); break
        db.polygon([tp(2.4, 0), tp(-2.4, 1.1), tp(-2.4, -1.1)], fill=(255, 255, 255))
        for r_ in (20, 40):
            db.ellipse([Bv / 2 - r_ * sb, Bv / 2 - r_ * sb, Bv / 2 + r_ * sb, Bv / 2 + r_ * sb], outline=(60, 60, 60))
            db.text((Bv / 2 + 3, Bv / 2 - r_ * sb + 2), f'{r_} m', fill=(110, 110, 110), font=F(10))
        db.rectangle([0, 0, 150, 40], fill=(0, 0, 0)); db.text((8, 4), f't = {(t - t0) / 1e6:4.1f} s', fill=(255, 255, 255), font=F(26, True))
        img.paste(bev, (3 * cw, 0))
        dl = ImageDraw.Draw(img); ly = h1 + h2 + 40
        dl.text((30, ly), {'ctrl': 'CTRL', 'ours': 'Bản đề xuất'}[p.name], fill=MAU[p.name], font=F(34, True))
        for n_, (txt, cc) in enumerate([('Xanh lá: GT (nhãn thật)', GREEN),
                                       (('Đỏ' if p.name == 'ctrl' else 'Xanh dương') + ': box của mô hình (score ≥ 0,3), kèm ID track', MAU[p.name]),
                                       ('BEV: nhìn từ trên xuống, xe tự lái = tam giác trắng, phía trước hướng lên trên', (200, 200, 200)),
                                       ('BEV: cạnh vẽ đậm của box = đầu xe; số vàng = ID track (bỏ tiền tố 1_)', (200, 200, 200)),
                                       ('Đồng hồ t = giây trong cảnh (trùng giây của video)', (200, 200, 200))]):
            dl.text((30, ly + 60 + 34 * n_), txt, fill=cc, font=F(20))
        img.save(os.path.join(od, f'{k:05d}.png'))
    print('xong', p.name, p.part, len(mine), 'khung')


def encode():
    for name in ('ctrl', 'ours'):
        od = os.path.join(p.frames or od_root, f'khung_{name}'); os.makedirs(od_root, exist_ok=True); out = os.path.join(od_root, f'video_{name}.mp4')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '10', '-i', os.path.join(od, '%05d.png'),
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', out], check=True)
        print(out, os.path.getsize(out) // 1024, 'KB')


if p.name: render()
if p.encode: encode()

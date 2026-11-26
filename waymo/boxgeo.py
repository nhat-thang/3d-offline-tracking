"""Ham hinh hoc va doc du lieu dung chung cho phan tich / ve loi CTRL tren Waymo (chi can numpy).
Box: (cx, cy, cz, dai, rong, cao, huong), he toa do xe Waymo, huong tinh tu truc x, nguoc chieu kim dong ho."""
import os, pickle
import numpy as np


def corners(b):
    """4 goc BEV cua box, nguoc chieu kim dong ho."""
    c, s = np.cos(b[6]), np.sin(b[6]); l, w = b[3] / 2, b[4] / 2
    pts = np.array([[l, w], [-l, w], [-l, -w], [l, -w]])
    return pts @ np.array([[c, s], [-s, c]]) + b[:2]


def _clip(poly, a, b):
    out = []
    side = lambda q: (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]; sp, sq = side(P), side(Q)
        if sp >= 0: out.append(P)
        if sp * sq < 0: out.append(P + (Q - P) * (sp / (sp - sq)))
    return out


def _area(poly):
    if len(poly) < 3: return 0.0
    x = np.array([q[0] for q in poly]); y = np.array([q[1] for q in poly])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def iou3d(a, b):
    """IoU 3D cua hai box xoay quanh truc dung."""
    pa, pb = corners(a), corners(b); poly = list(pa)
    for i in range(4):
        poly = _clip(poly, pb[i], pb[(i + 1) % 4])
        if not poly: return 0.0
    zo = max(0.0, min(a[2] + a[5] / 2, b[2] + b[5] / 2) - max(a[2] - a[5] / 2, b[2] - b[5] / 2))
    inter = _area(poly) * zo
    return float(inter / (a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - inter + 1e-9))


def yaw_diff(a, b):
    """Lech huong tuyet doi, radian, trong [0, pi]."""
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


class Source:
    """Box cua mot nguon (GT, CTRL truoc/sau refine, ban cua ban...) doc tu boxes_<ten>.npz, tra cuu theo frame va theo track."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.ctx, self.ts, self.id = d['ctx'].astype(int), d['ts'], d['id']
        self.box, self.score = d['box'].astype(np.float64), d['score']
        self.npts = d['npts'] if 'npts' in d else None
        self.level = d['level'] if 'level' in d else None
        self.frames, self.tracks = {}, {}
        for i, (c, t, k) in enumerate(zip(self.ctx, self.ts, self.id)):
            self.frames.setdefault((c, int(t)), []).append(i)
            self.tracks.setdefault((c, str(k)), []).append(i)
        for k, v in self.tracks.items():
            v.sort(key=lambda i: self.ts[i])

    def in_frame(self, c, t):
        return self.frames.get((c, int(t)), [])

    def track(self, c, k):
        return self.tracks.get((c, str(k)), [])


def load_sources(npz_dir, names):
    return {n: Source(os.path.join(npz_dir, f'boxes_{n}.npz')) for n in names}


def load_ts2idx(kitti_dir):
    """timestamp -> ten file diem LiDAR (kitti_format/training/velodyne/<idx>.bin)."""
    i2t = pickle.load(open(os.path.join(kitti_dir, 'idx2timestamp.pkl'), 'rb'))
    return {int(t): k for k, t in i2t.items()}


def load_points(kitti_dir, idx):
    """Diem LiDAR cua frame: (N, 6) x, y, z, intensity, elongation, timestamp (he toa do xe)."""
    return np.fromfile(os.path.join(kitti_dir, 'training', 'velodyne', idx + '.bin'), np.float32).reshape(-1, 6)

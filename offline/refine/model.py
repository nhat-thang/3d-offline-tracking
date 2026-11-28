"""Bước 5 của Task 1: model refine rút gọn (chỉ cần numpy + PyTorch, không cần nuscenes-devkit).

Ý tưởng giống phần track-centric refine của CTRL nhưng nhẹ hơn nhiều để chạy trên GPU 4 GB:
  1. Bộ mã hoá điểm (PointNet): mỗi box -> các điểm LiDAR quanh box (hệ box) -> MLP chung cho từng điểm -> max-pool
     -> một vector 256 chiều mô tả hình dạng vật trong box ở frame đó.
  2. Ghép vector điểm với đặc trưng box (score, kích thước, số điểm, ...) và lớp của track.
  3. Mô hình theo thời gian: các khối Conv1D giãn (dilation 1, 2, 4, 8) chạy dọc track, nhìn cả frame trước và
     frame sau (tận dụng tính offline: biết cả tương lai). Box ở frame ít điểm mượn thông tin từ frame nhiều điểm.
  4. Ba đầu ra:
       reg   (mỗi box): dx, dy, dz, dlog_w, dlog_l, dlog_h, sin(dyaw), cos(dyaw)  -> chỉnh box về gần GT
       box   (mỗi box): logit box có vật thật không
       track (mỗi track): logit track có thật không -> dùng làm score mới cho cả track
     Hai đầu box/track cộng phần chỉnh vào logit của score gốc; mọi đầu ra khởi tạo bằng 0, nên lúc bắt đầu train
     model cho ra đúng kết quả của tracker và chỉ học phần sửa thêm.
Kích thước: khoảng 0,5 triệu tham số."""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MARGIN = 1.0


class TBlock(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, m):            # x: [B, L, d], m: [B, L, 1]
        h = self.conv(F.gelu(self.norm(x)).mul(m).transpose(1, 2)).transpose(1, 2)
        x = (x + h) * m
        return (x + self.ff(x)) * m


class RefineNet(nn.Module):
    def __init__(self, nfeat=12, ncls=7, d=128):
        super().__init__()
        self.pn = nn.Sequential(nn.Conv1d(7, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.cls = nn.Embedding(ncls, 16)
        self.fuse = nn.Sequential(nn.Linear(256 + nfeat + 16, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
        self.temporal = nn.ModuleList([TBlock(d, k) for k in (1, 2, 4, 8)])
        self.reg = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 8))
        self.box = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.trk = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        for head in (self.reg, self.box, self.trk):      # đầu ra ban đầu = 0: box giữ nguyên, score giữ nguyên
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        with torch.no_grad(): self.reg[-1].bias[7] = 1.0  # cos(dyaw) = 1 -> không xoay

    def encode_points(self, pts, feat):
        """pts: [N, P, 4] (x, y, z, cường độ) hệ box; feat: [N, F]. Thêm toạ độ chuẩn hoá theo kích thước box."""
        half = torch.exp(feat[:, 2:5]) / 2 + MARGIN           # (w, l, h)/2 + lề
        xyz = pts[..., :3] / torch.stack([half[:, 1], half[:, 0], half[:, 2]], 1)[:, None, :]
        x = torch.cat([pts, xyz], -1).transpose(1, 2)          # [N, 7, P]
        return self.pn(x).max(2).values                        # [N, 256]

    def forward(self, pts, feat, tcls, mask):
        """pts [B, L, P, 4], feat [B, L, F], tcls [B], mask [B, L] (bool)."""
        B, L = mask.shape
        enc = pts.new_zeros(B, L, 256)
        enc[mask] = self.encode_points(pts[mask], feat[mask])   # chỉ mã hoá real box, bỏ phần đệm
        c = self.cls(tcls)[:, None, :].expand(B, L, -1)
        m = mask[..., None].float()
        x = self.fuse(torch.cat([enc, feat, c], -1)) * m
        for blk in self.temporal: x = blk(x, m)
        reg = self.reg(x)
        sc = reg[..., 6:8]
        reg = torch.cat([reg[..., :6], sc / sc.norm(dim=-1, keepdim=True).clamp_min(1e-6)], -1)
        mean = x.sum(1) / m.sum(1).clamp_min(1)
        mx = x.masked_fill(~mask[..., None], -1e4).max(1).values
        # điểm xuất phát là score gốc của tracker; model chỉ học phần chỉnh thêm (lúc đầu = 0 -> không làm tệ hơn bản gốc)
        s0 = feat[..., 0].clamp(1e-4, 1 - 1e-4)
        box0 = torch.log(s0 / (1 - s0))
        st = ((feat[..., 0] * m[..., 0]).sum(1) / m[..., 0].sum(1).clamp_min(1)).clamp(1e-4, 1 - 1e-4)
        trk0 = torch.log(st / (1 - st))
        return reg, box0 + self.box(x)[..., 0], trk0 + self.trk(torch.cat([mean, mx], -1))[..., 0]


# ---------------- dữ liệu ----------------
class TrackData:
    """Đọc file .npz của make_dataset.py, phục vụ theo track."""
    def __init__(self, paths):
        parts = [np.load(p) for p in paths]
        self.metas = [json.load(open(p.replace('.npz', '_meta.json'))) for p in paths]
        self.pts = np.concatenate([d['pts'] for d in parts])
        self.feat = np.concatenate([d['feat'] for d in parts]).astype(np.float32)
        self.yreg = np.concatenate([d['yreg'] for d in parts]).astype(np.float32)
        self.yreal = np.concatenate([d['yreal'] for d in parts]).astype(np.float32)
        ptr, tcls, tscene, off, soff = [], [], [], 0, 0
        for d in parts:
            ptr.append(d['ptr'][:-1] + off); off += len(d['pts'])
            tcls.append(d['tcls']); tscene.append(d['tscene'] + soff); soff += int(d['tscene'].max()) + 1 if len(d['tscene']) else 0
        self.start = np.concatenate(ptr); self.end = np.concatenate([self.start[1:], [off]]) if off else self.start
        self.tcls = np.concatenate(tcls).astype(np.int64); self.tscene = np.concatenate(tscene)
        n = self.end - self.start
        self.treal = np.array([self.yreal[s:e].mean() >= 0.5 for s, e in zip(self.start, self.end)], np.float32)
        self.n_track, self.max_len = len(self.start), int(n.max()) if len(n) else 0

    def batch(self, idx, augment=False, rng=None):
        L = int(max(self.end[i] - self.start[i] for i in idx))
        B, P = len(idx), self.pts.shape[1]
        pts = np.zeros((B, L, P, 4), np.float32); feat = np.zeros((B, L, self.feat.shape[1]), np.float32)
        yreg = np.zeros((B, L, 8), np.float32); yreal = np.zeros((B, L), np.float32); mask = np.zeros((B, L), bool)
        for b, i in enumerate(idx):
            s, e = self.start[i], self.end[i]; n = e - s
            pts[b, :n] = self.pts[s:e]; feat[b, :n] = self.feat[s:e]; yreg[b, :n] = self.yreg[s:e]
            yreal[b, :n] = self.yreal[s:e]; mask[b, :n] = True
        if augment: pts, feat, yreg = augment_batch(pts, feat, yreg, mask, rng)
        t = lambda a: torch.from_numpy(a)
        return dict(pts=t(pts), feat=t(feat), yreg=t(yreg), yreal=t(yreal), mask=t(mask),
                    tcls=t(self.tcls[idx]), treal=t(self.treal[idx]))


def augment_batch(pts, feat, yreg, mask, rng):
    """Làm biến dạng dữ liệu để model bớt phụ thuộc mật độ điểm của nuScenes:
       - giảm mật độ: mỗi box giữ ngẫu nhiên 20-100% số điểm rồi lấy lại cho đủ P điểm;
       - nhiễu toạ độ 2 cm;
       - lật gương theo trục ngang của box cho cả track (y -> -y): đổi dấu dy, sin(dyaw), vận tốc ngang."""
    B, L, P, _ = pts.shape
    keep = rng.uniform(0.2, 1.0, size=(B, L))
    for b in range(B):
        for l in range(L):
            if not mask[b, l]: continue
            k = max(1, int(P * keep[b, l]))
            sub = rng.choice(P, k, replace=False)
            pts[b, l] = pts[b, l, sub[rng.integers(0, k, P)]]
    pts[..., :3] += rng.normal(0, 0.02, size=pts[..., :3].shape).astype(np.float32)
    flip = rng.random(B) < 0.5
    pts[flip, :, :, 1] *= -1; yreg[flip, :, 1] *= -1; yreg[flip, :, 6] *= -1; feat[flip, :, 11] *= -1
    return pts, feat, yreg


def losses(out, bt, w_reg=1.0, w_box=1.0, w_trk=1.0):
    reg, box, trk = out
    m = bt['mask']; pos = m & (bt['yreal'] > 0)
    l_box = F.binary_cross_entropy_with_logits(box[m], bt['yreal'][m])
    # mỗi track nặng theo số box: evaluator tính đúng/sai trên từng box, track dài ảnh hưởng nhiều hơn track 1 box
    nb = m.sum(1).float()
    l_trk = (F.binary_cross_entropy_with_logits(trk, bt['treal'], reduction='none') * nb).sum() / nb.sum()
    if pos.any():
        l_reg = F.smooth_l1_loss(reg[pos][:, :6], bt['yreg'][pos][:, :6], beta=0.2) + \
                F.smooth_l1_loss(reg[pos][:, 6:], bt['yreg'][pos][:, 6:], beta=0.2)
    else:
        l_reg = reg.sum() * 0
    return w_reg * l_reg + w_box * l_box + w_trk * l_trk, dict(reg=l_reg.item(), box=l_box.item(), trk=l_trk.item())

"""Bước 6 của Task 1: train model refine (chạy trong Docker có PyTorch + GPU).

Dữ liệu: các file .npz do make_dataset.py tạo (tập dev). Tách theo scene: cứ 7 scene thì 1 scene làm validation,
để biết model có học thuộc lòng không. Tập test KHÔNG được dùng ở đây.
Mỗi epoch in: loss train/val và các chỉ số trên box thật của tập val, so TRƯỚC (box của tracker) và SAU (box đã chỉnh):
  - sai số tâm (m), tỉ lệ lật hướng > 90 độ, sai số kích thước (trung bình |log tỉ lệ|);
  - AUC phân biệt box thật/giả và track thật/giả: score gốc của tracker so với score model đoán;
  - AUC "box-theo-track": gán score của track cho mọi box của nó rồi so với nhãn từng box. Đây là chỉ số gần với
    AMOTA nhất, vì bộ chấm nuScenes dùng score trung bình của track cho mọi box.
Lưu model tốt nhất (loss val thấp nhất) vào offline/results/refine/ckpt/<tên>.pt.

Chạy (từ thư mục gốc):
  python offline/refine/train.py offline/results/refine/data/B0_dev.npz --name B0_s0 --seed 0
  thêm --epochs 2 --max_tracks 2000 để chạy thử nhanh."""
import os, sys, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RefineNet, TrackData, losses

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CKPT_DIR = os.path.join(ROOT, 'offline', 'results', 'refine', 'ckpt')


def auc(score, label):
    """Diện tích dưới đường ROC (xác suất một mẫu thật được chấm cao hơn một mẫu giả)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or (~label).all(): return float('nan')
    r = np.argsort(np.argsort(score)) + 1.0
    npos = label.sum(); nneg = len(label) - npos
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def evaluate(model, data, idx, dev, bs):
    model.eval()
    tot, n, R, Y, PB, YB, SB, PT, YT, ST, BT0, BT1 = 0.0, 0, [], [], [], [], [], [], [], [], [], []
    for k in range(0, len(idx), bs):
        bt = {a: v.to(dev) for a, v in data.batch(idx[k:k + bs]).items()}
        out = model(bt['pts'], bt['feat'], bt['tcls'], bt['mask'])
        l, _ = losses(out, bt); tot += float(l) * len(idx[k:k + bs]); n += len(idx[k:k + bs])
        reg, box, trk = out; m = bt['mask']; pos = m & (bt['yreal'] > 0)
        R.append(reg[pos].cpu().numpy()); Y.append(bt['yreg'][pos].cpu().numpy())
        PB.append(torch.sigmoid(box[m]).cpu().numpy()); YB.append(bt['yreal'][m].cpu().numpy()); SB.append(bt['feat'][..., 0][m].cpu().numpy())
        PT.append(torch.sigmoid(trk).cpu().numpy()); YT.append(bt['treal'].cpu().numpy())
        st = (bt['feat'][..., 0] * m).sum(1) / m.sum(1)
        ST.append(st.cpu().numpy())
        # score theo track gán cho từng box (đúng cách bộ chấm dùng): so với nhãn từng box
        BT0.append(st[:, None].expand_as(m)[m].cpu().numpy()); BT1.append(torch.sigmoid(trk)[:, None].expand_as(m)[m].cpu().numpy())
    R, Y = np.concatenate(R), np.concatenate(Y)
    yaw_t = np.arctan2(Y[:, 6], Y[:, 7]); yaw_p = np.arctan2(R[:, 6], R[:, 7])
    res = dict(loss=tot / max(n, 1), n_pos=len(Y),
               tam_truoc=float(np.hypot(Y[:, 0], Y[:, 1]).mean()), tam_sau=float(np.hypot(Y[:, 0] - R[:, 0], Y[:, 1] - R[:, 1]).mean()),
               lat_truoc=float((np.cos(yaw_t) < 0).mean()), lat_sau=float((np.cos(yaw_t - yaw_p) < 0).mean()),
               kt_truoc=float(np.abs(Y[:, 3:6]).mean()), kt_sau=float(np.abs(Y[:, 3:6] - R[:, 3:6]).mean()),
               auc_box_goc=auc(np.concatenate(SB), np.concatenate(YB)), auc_box_moi=auc(np.concatenate(PB), np.concatenate(YB)),
               auc_trk_goc=auc(np.concatenate(ST), np.concatenate(YT)), auc_trk_moi=auc(np.concatenate(PT), np.concatenate(YT)),
               auc_boxtrk_goc=auc(np.concatenate(BT0), np.concatenate(YB)), auc_boxtrk_moi=auc(np.concatenate(BT1), np.concatenate(YB)))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', nargs='+', help='các file .npz của tập dev')
    ap.add_argument('--name', default='refine')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=64, help='số track mỗi batch')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val_every', type=int, default=7, help='cứ N scene lấy 1 scene làm validation')
    ap.add_argument('--max_tracks', type=int, default=0, help='chỉ dùng N track (chạy thử)')
    ap.add_argument('--no_aug', action='store_true')
    p = ap.parse_args()
    torch.manual_seed(p.seed); rng = np.random.default_rng(p.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = TrackData(p.data)
    idx = np.arange(data.n_track)
    if p.max_tracks: idx = rng.permutation(idx)[:p.max_tracks]
    is_val = data.tscene[idx] % p.val_every == 0
    tr_idx, va_idx = idx[~is_val], idx[is_val]
    print(f'thiet bi: {dev} | track train {len(tr_idx)}, val {len(va_idx)} | box {len(data.pts)} | '
          f'ti le box that {data.yreal.mean():.3f}, track that {data.treal.mean():.3f}', flush=True)

    model = RefineNet(nfeat=data.feat.shape[1]).to(dev)
    print('so tham so:', sum(x.numel() for x in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=1e-4)
    steps = p.epochs * int(np.ceil(len(tr_idx) / p.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p.lr, total_steps=max(steps, 1), pct_start=0.1)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ck = os.path.join(CKPT_DIR, p.name + '.pt'); best = 1e9; hist = []
    t0 = time.time()
    for ep in range(p.epochs):
        perm = rng.permutation(tr_idx); run = []
        for k in range(0, len(perm), p.bs):
            bt = {a: v.to(dev) for a, v in data.batch(perm[k:k + p.bs], augment=not p.no_aug, rng=rng).items()}
            out = model(bt['pts'], bt['feat'], bt['tcls'], bt['mask'])
            l, parts = losses(out, bt)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            run.append(l.item())
        v = evaluate(model, data, va_idx, dev, p.bs * 2)
        v.update(epoch=ep + 1, train_loss=float(np.mean(run)), time_s=round(time.time() - t0))
        hist.append(v)
        print(f"ep {ep + 1:2d} | train {v['train_loss']:.3f} val {v['loss']:.3f} | tam {v['tam_truoc']:.3f}->{v['tam_sau']:.3f} m | "
              f"lat {100 * v['lat_truoc']:.1f}->{100 * v['lat_sau']:.1f}% | kt {v['kt_truoc']:.3f}->{v['kt_sau']:.3f} | "
              f"AUC box {v['auc_box_goc']:.3f}->{v['auc_box_moi']:.3f} track {v['auc_trk_goc']:.3f}->{v['auc_trk_moi']:.3f} "
              f"box-theo-track {v['auc_boxtrk_goc']:.3f}->{v['auc_boxtrk_moi']:.3f} | {v['time_s']}s", flush=True)
        if v['loss'] < best:
            best = v['loss']
            torch.save(dict(model=model.state_dict(), nfeat=data.feat.shape[1], args=vars(p), epoch=ep + 1, val=v), ck)
    json.dump(hist, open(ck.replace('.pt', '_log.json'), 'w'), indent=1)
    print(f'xong. model tot nhat (val loss {best:.3f}) luu o {ck}')


if __name__ == '__main__':
    main()

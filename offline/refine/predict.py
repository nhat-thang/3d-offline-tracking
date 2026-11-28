"""Bước 7a (phần cần PyTorch): chạy model đã train lên một file dữ liệu (.npz của make_dataset.py).
Đầu ra cùng thứ tự với file dữ liệu: reg [N, 8] (độ chỉnh từng box), pbox [N] (xác suất real box),
ptrk [T] (xác suất track thật). Sau đó dùng apply.py để ghi vào results.json.
Chạy: python offline/refine/predict.py offline/results/refine/ckpt/B0_s0.pt offline/results/refine/data/B0_test.npz"""
import os, sys, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RefineNet, TrackData

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('data')
ap.add_argument('--out', help='mặc định offline/results/refine/pred/<tên ckpt>__<tên data>.npz')
ap.add_argument('--bs', type=int, default=128)
p = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(p.ckpt, map_location=dev, weights_only=False)
model = RefineNet(nfeat=ck['nfeat']).to(dev); model.load_state_dict(ck['model']); model.eval()
data = TrackData([p.data])
reg = np.zeros((len(data.pts), 8), np.float32); pbox = np.zeros(len(data.pts), np.float32); ptrk = np.zeros(data.n_track, np.float32)
with torch.no_grad():
    for k in range(0, data.n_track, p.bs):
        idx = np.arange(k, min(k + p.bs, data.n_track))
        bt = {a: v.to(dev) for a, v in data.batch(idx).items()}
        r, b, t = model(bt['pts'], bt['feat'], bt['tcls'], bt['mask'])
        r, b = r.cpu().numpy(), torch.sigmoid(b).cpu().numpy(); ptrk[idx] = torch.sigmoid(t).cpu().numpy()
        for j, i in enumerate(idx):
            s, e = data.start[i], data.end[i]
            reg[s:e] = r[j, :e - s]; pbox[s:e] = b[j, :e - s]
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = p.out or os.path.join(root, 'offline', 'results', 'refine', 'pred',
                            os.path.splitext(os.path.basename(p.ckpt))[0] + '__' + os.path.basename(p.data))
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, reg=reg, pbox=pbox, ptrk=ptrk)
print(f'xong: {out} | {len(reg)} box, {data.n_track} track')

"""Xuat box cua cac nguon (file .bin Waymo, GT parquet v2) ra .npz de phan tich va ve chi can numpy.

Moi nguon -> <out_dir>/boxes_<ten>.npz gom cac mang cung do dai N:
  ctx (int, chi so trong danh sach context), ts (int64, micro giay), id (str, ID track/vat),
  box (N, 7: cx, cy, cz, dai, rong, cao, huong; he toa do xe), score (float; GT = 1),
  va voi GT them: npts (so diem LiDAR trong box), level (1/2, quy tac Waymo).
Chi giu lop xe (type 1). Box GT khong co diem LiDAR van duoc giu (npts = 0) de ve, phan tich tu loc.

Chay: python waymo/export_boxes_npz.py --contexts waymo/subset8.txt --gt_dir data/waymo/v2/validation/lidar_box
        --src ten1=duong_dan1.bin --src ten2=duong_dan2.bin --out offline/results/waymo/subset8/npz"""
import argparse, os
import numpy as np, pandas as pd
from waymo_open_dataset.protos import metrics_pb2

ap = argparse.ArgumentParser()
ap.add_argument('--contexts', required=True); ap.add_argument('--gt_dir', default='')
ap.add_argument('--src', action='append', default=[], help='ten=duong_dan.bin, lap lai cho nhieu nguon')
ap.add_argument('--out', required=True)
p = ap.parse_args()
ctxs = open(p.contexts).read().split(); cidx = {c: i for i, c in enumerate(ctxs)}
os.makedirs(p.out, exist_ok=True)
np.save(os.path.join(p.out, 'contexts.npy'), np.array(ctxs))

for s in p.src:
    name, path = s.split('=', 1)
    o = metrics_pb2.Objects(); o.ParseFromString(open(path, 'rb').read())
    C, T, I, B, S = [], [], [], [], []
    for x in o.objects:
        if x.context_name not in cidx or x.object.type != 1: continue
        b = x.object.box
        C.append(cidx[x.context_name]); T.append(x.frame_timestamp_micros); I.append(x.object.id)
        B.append((b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading)); S.append(x.score)
    np.savez_compressed(os.path.join(p.out, f'boxes_{name}.npz'), ctx=np.array(C, np.int16), ts=np.array(T, np.int64),
                        id=np.array(I), box=np.array(B, np.float32), score=np.array(S, np.float32))
    print(f'{name}: {len(C)} box, {len(set(I))} track')
    del o

if p.gt_dir:
    def col(df, suffix):
        m = [c for c in df.columns if c.endswith(suffix)]; assert len(m) == 1, (suffix, m); return df[m[0]].to_numpy()
    C, T, I, B, N, L = [], [], [], [], [], []
    for c in ctxs:
        df = pd.read_parquet(os.path.join(p.gt_dir, c + '.parquet'))
        m = col(df, 'LiDARBoxComponent].type').astype(int) == 1
        df = df[m]
        npts = col(df, 'LiDARBoxComponent].num_lidar_points_in_box').astype(int)
        lvl = col(df, 'LiDARBoxComponent].difficulty_level.detection').astype(int)
        box = np.stack([col(df, f'LiDARBoxComponent].box.{k}') for k in
                        ('center.x', 'center.y', 'center.z', 'size.x', 'size.y', 'size.z', 'heading')], 1)
        C += [cidx[c]] * len(df); T += list(col(df, 'key.frame_timestamp_micros').astype(np.int64))
        I += list(col(df, 'key.laser_object_id')); B.append(box); N += list(npts)
        L += [2 if (l == 2 or n <= 5) else 1 for l, n in zip(lvl, npts)]
    np.savez_compressed(os.path.join(p.out, 'boxes_gt.npz'), ctx=np.array(C, np.int16), ts=np.array(T, np.int64),
                        id=np.array(I), box=np.concatenate(B).astype(np.float32), score=np.ones(len(C), np.float32),
                        npts=np.array(N, np.int32), level=np.array(L, np.uint8))
    print(f'gt: {len(C)} box xe ({sum(n > 0 for n in N)} co diem LiDAR), {len(set(zip(C, I)))} vat')

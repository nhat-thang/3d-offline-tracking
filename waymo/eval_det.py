"""Cham diem detection 3D kieu Waymo (AP/APH, LEVEL_1/LEVEL_2) bang bo cham chinh thuc dang Python
(waymo_open_dataset.metrics, cung cau hinh voi compute_detection_metrics_main: Hungarian, IoU 0.7 vehicle / 0.5 ped, cyc, 3D).

GT doc tu thanh phan v2 `lidar_box` (parquet) cua tap validation:
  - bo box GT khong co diem LiDAR nao (giong cach tao gt.bin cua mmdet3d / CTRL),
  - do kho: LEVEL_2 neu nhan ghi LEVEL_2 hoac so diem <= 5, nguoc lai LEVEL_1 (quy tac cua Waymo).
Chi cham tren cac chuoi co trong file du doan (de cham duoc tap con).

Chay: python waymo/eval_det.py <pred.bin> <thu_muc lidar_box parquet> [--types 1] [--out ket_qua.json] [--contexts file.txt]"""
import argparse, glob, json, os, collections, gc
import numpy as np, pandas as pd, tensorflow as tf
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.metrics.python.wod_detection_evaluator import WODDetectionEvaluator

ap = argparse.ArgumentParser()
ap.add_argument('pred'); ap.add_argument('gt_dir')
ap.add_argument('--types', default='1', help='lop can cham, vi du 1 hoac 1,2,4')
ap.add_argument('--contexts', default='', help='file liet ke context_name can cham (mac dinh: moi chuoi trong file du doan)')
ap.add_argument('--keep_zero', action='store_true', help='giu ca GT 0 diem LiDAR')
ap.add_argument('--out', default='')
p = ap.parse_args()
types = [int(t) for t in p.types.split(',')]

objs = metrics_pb2.Objects(); objs.ParseFromString(open(p.pred, 'rb').read())
ctxs = set(open(p.contexts).read().split()) if p.contexts else {o.context_name for o in objs.objects}
fid = {}
def frame_id(c, t):
    return fid.setdefault((c, t), len(fid))

# doc thang vao mang numpy (it RAM hon list cua list), xong thi giai phong proto
M = len(objs.objects)
pb = np.empty((M, 7), np.float32); pt = np.empty(M, np.uint8); ps = np.empty(M, np.float32); pf = np.empty(M, np.int64)
n = 0
for o in objs.objects:
    if o.context_name not in ctxs or o.object.type not in types: continue
    b = o.object.box
    pb[n] = (b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading)
    pt[n] = o.object.type; ps[n] = o.score; pf[n] = frame_id(o.context_name, o.frame_timestamp_micros); n += 1
pb, pt, ps, pf = pb[:n].copy(), pt[:n].copy(), ps[:n].copy(), pf[:n].copy()
del objs; gc.collect()
print(f'du doan: {n} box, {len(ctxs)} chuoi')

def col(df, suffix):
    m = [c for c in df.columns if c.endswith(suffix)]
    assert len(m) == 1, (suffix, m)
    return df[m[0]].to_numpy()

gb, gt_, gd, gf = [], [], [], []
stat = collections.Counter()
files = [f for f in sorted(glob.glob(os.path.join(p.gt_dir, '*.parquet'))) if os.path.basename(f)[:-8] in ctxs]
print(f'GT: {len(files)} file parquet khop voi {len(ctxs)} chuoi')
for f in files:
    df = pd.read_parquet(f)
    typ = col(df, 'LiDARBoxComponent].type').astype(int)
    npts = col(df, 'LiDARBoxComponent].num_lidar_points_in_box').astype(int)
    lvl = col(df, 'LiDARBoxComponent].difficulty_level.detection').astype(int)
    box = np.stack([col(df, f'LiDARBoxComponent].box.{k}') for k in
                    ('center.x', 'center.y', 'center.z', 'size.x', 'size.y', 'size.z', 'heading')], 1)
    c = col(df, 'key.segment_context_name'); t = col(df, 'key.frame_timestamp_micros')
    for i in range(len(df)):
        if typ[i] not in types: continue
        if npts[i] == 0 and not p.keep_zero: stat['bo_0_diem'] += 1; continue
        d = 2 if (lvl[i] == 2 or npts[i] <= 5) else 1
        stat[f'L{d}'] += 1
        gb.append(box[i].astype(np.float32)); gt_.append(typ[i]); gd.append(d); gf.append(frame_id(c[i], int(t[i])))
print('GT:', dict(stat))

gb = np.stack(gb); gt_ = np.array(gt_, np.uint8); gd = np.array(gd, np.uint8); gf = np.array(gf, np.int64); gc.collect()
ev = WODDetectionEvaluator()
ev.update_state(
    dict(ground_truth_frame_id=tf.constant(gf, tf.int64), ground_truth_bbox=tf.constant(gb, tf.float32),
         ground_truth_type=tf.constant(gt_, tf.uint8), ground_truth_difficulty=tf.constant(gd, tf.uint8)),
    dict(prediction_frame_id=tf.constant(pf, tf.int64), prediction_bbox=tf.constant(pb, tf.float32),
         prediction_type=tf.constant(pt, tf.uint8), prediction_score=tf.constant(ps, tf.float32),
         prediction_overlap_nlz=tf.zeros(n, tf.bool)))
m = ev.result()  # tuple: (AP, APH, PR, PR_HA, breakdown)
AP_, APH_ = m[0].numpy(), m[1].numpy()
names = config_util.get_breakdown_names_from_config(ev._config)
res = {}
for i, n in enumerate(names):
    if 'OBJECT_TYPE' not in n: continue
    res[n] = dict(AP=float(AP_[i]), APH=float(APH_[i]))
for n, v in res.items():
    if any(f'TYPE_{k}' in n for k in ('VEHICLE', 'PEDESTRIAN', 'CYCLIST')):
        print(f"{n:45s} AP {100 * v['AP']:.2f}  APH {100 * v['APH']:.2f}")
if p.out:
    json.dump(dict(pred=p.pred, n_contexts=len(ctxs), gt=dict(stat), metrics=res), open(p.out, 'w'), indent=1)

"""Cham metric tracking 3D kieu Waymo (MOTA, MOTP, miss, mismatch, FP; LEVEL_1/LEVEL_2) bang op chinh thuc
waymo_open_dataset.metrics.ops.py_metrics_ops.tracking_metrics, cung cau hinh voi compute_tracking_metrics_main:
ghep Hungarian theo IoU 3D (vehicle 0.7, pedestrian/cyclist 0.5), quet nguong score 0..1 buoc 0.01, moi breakdown lay
nguong cho MOTA cao nhat.

  MOTA = 1 - (miss + mismatch + FP) / so GT ; MOTP = sai so trung binh cua cac cap ghep dung (1 - IoU)
  miss, mismatch (doi ID), FP: in ra o dang ti le tren so GT va so dem xap xi (ti le x so GT).
GT doc tu thanh phan v2 `lidar_box` (parquet); bo box GT khong co diem LiDAR; do kho tracking: LEVEL_2 neu nhan ghi
LEVEL_2 hoac box co <= 5 diem, nguoc lai LEVEL_1. Chi cham cac chuoi trong --contexts (hoac moi chuoi trong file du doan).

Chay: python waymo/eval_track.py <pred.bin> <thu_muc lidar_box parquet> [--types 1] [--contexts file.txt] [--out ket_qua.json]"""
import argparse, collections, glob, json, os, gc
import numpy as np, pandas as pd, tensorflow as tf
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import breakdown_pb2, metrics_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util

ap = argparse.ArgumentParser()
ap.add_argument('pred'); ap.add_argument('gt_dir')
ap.add_argument('--types', default='1'); ap.add_argument('--contexts', default='')
ap.add_argument('--out', default='')
p = ap.parse_args()
types = [int(t) for t in p.types.split(',')]


def get_config():
    c = metrics_pb2.Config()
    c.breakdown_generator_ids.append(breakdown_pb2.Breakdown.OBJECT_TYPE)
    d = c.difficulties.add(); d.levels.append(label_pb2.Label.LEVEL_1); d.levels.append(label_pb2.Label.LEVEL_2)
    c.matcher_type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN
    for t in (0.0, 0.7, 0.5, 0.5, 0.5): c.iou_thresholds.append(t)
    c.box_type = label_pb2.Label.Box.TYPE_3D
    for i in range(100): c.score_cutoffs.append(i * 0.01)
    c.score_cutoffs.append(1.0)
    return c


objs = metrics_pb2.Objects(); objs.ParseFromString(open(p.pred, 'rb').read())
ctxs = set(open(p.contexts).read().split()) if p.contexts else {o.context_name for o in objs.objects}
fid, oid = {}, {}
frame_id = lambda c, t: fid.setdefault((c, int(t)), len(fid))
obj_id = lambda src, c, k: oid.setdefault((src, c, str(k)), len(oid))

M = len(objs.objects)
pb = np.empty((M, 7), np.float32); pt = np.empty(M, np.uint8); ps = np.empty(M, np.float32)
pf = np.empty(M, np.int64); po = np.empty(M, np.int64); pseq = []
n = 0
for o in objs.objects:
    if o.context_name not in ctxs or o.object.type not in types: continue
    b = o.object.box
    pb[n] = (b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading)
    pt[n] = o.object.type; ps[n] = o.score; pf[n] = frame_id(o.context_name, o.frame_timestamp_micros)
    po[n] = obj_id('pred', o.context_name, o.object.id); pseq.append(o.context_name); n += 1
pb, pt, ps, pf, po = pb[:n].copy(), pt[:n].copy(), ps[:n].copy(), pf[:n].copy(), po[:n].copy()
del objs; gc.collect()
print(f'du doan: {n} box, {len(set(po.tolist()))} track, {len(ctxs)} chuoi')


def col(df, suffix):
    m = [c for c in df.columns if c.endswith(suffix)]; assert len(m) == 1, (suffix, m); return df[m[0]].to_numpy()


gb, gt_, gd, gf, go, gseq = [], [], [], [], [], []
stat = collections.Counter()
files = [f for f in sorted(glob.glob(os.path.join(p.gt_dir, '*.parquet'))) if os.path.basename(f)[:-8] in ctxs]
for f in files:
    df = pd.read_parquet(f)
    typ = col(df, 'LiDARBoxComponent].type').astype(int)
    npts = col(df, 'LiDARBoxComponent].num_lidar_points_in_box').astype(int)
    lvl = np.nan_to_num(col(df, 'LiDARBoxComponent].difficulty_level.tracking').astype(float)).astype(int)
    box = np.stack([col(df, f'LiDARBoxComponent].box.{k}') for k in
                    ('center.x', 'center.y', 'center.z', 'size.x', 'size.y', 'size.z', 'heading')], 1)
    c = col(df, 'key.segment_context_name'); t = col(df, 'key.frame_timestamp_micros'); k = col(df, 'key.laser_object_id')
    for i in range(len(df)):
        if typ[i] not in types: continue
        if npts[i] == 0: stat['bo_0_diem'] += 1; continue
        d = 2 if (lvl[i] == 2 or npts[i] <= 5) else 1
        stat[f'L{d}'] += 1
        gb.append(box[i].astype(np.float32)); gt_.append(typ[i]); gd.append(d)
        gf.append(frame_id(c[i], t[i])); go.append(obj_id('gt', c[i], k[i])); gseq.append(c[i])
print(f'GT: {len(files)} file parquet, {dict(stat)}')

cfg = get_config()
out = py_metrics_ops.tracking_metrics(
    prediction_bbox=tf.constant(pb, tf.float32), prediction_type=tf.constant(pt, tf.uint8),
    prediction_score=tf.constant(ps, tf.float32), prediction_frame_id=tf.constant(pf, tf.int64),
    prediction_sequence_id=tf.constant(pseq, tf.string), prediction_object_id=tf.constant(po, tf.int64),
    ground_truth_bbox=tf.constant(np.stack(gb), tf.float32), ground_truth_type=tf.constant(gt_, tf.uint8),
    ground_truth_frame_id=tf.constant(gf, tf.int64), ground_truth_sequence_id=tf.constant(gseq, tf.string),
    ground_truth_object_id=tf.constant(go, tf.int64), ground_truth_difficulty=tf.constant(gd, tf.uint8),
    config=cfg.SerializeToString())
mota, motp, miss, mismatch, fp, cutoff = [x.numpy() for x in out[:6]]
names = config_util.get_breakdown_names_from_config(cfg)
ngt = {1: stat['L1'], 2: stat['L1'] + stat['L2']}  # LEVEL_2 tinh tren ca GT LEVEL_1 va LEVEL_2 (so dem chi dung khi --types co 1 lop)
res = {}
for i, nm in enumerate(names):
    tmap = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 4}
    tn = [k for k in tmap if f'TYPE_{k}' in nm]
    if not tn or tmap[tn[0]] not in types: continue
    g = ngt[2 if nm.endswith('LEVEL_2') else 1]
    res[nm] = dict(MOTA=float(mota[i]), MOTP=float(motp[i]), miss=float(miss[i]), mismatch=float(mismatch[i]), fp=float(fp[i]),
                   nguong_score=float(np.ravel(cutoff[i])[0]) if np.size(cutoff[i]) else None, so_gt=g,
                   miss_dem=round(float(miss[i]) * g), mismatch_dem=round(float(mismatch[i]) * g), fp_dem=round(float(fp[i]) * g))
    v = res[nm]
    print(f"{nm:40s} MOTA {100 * v['MOTA']:.2f}  MOTP {v['MOTP']:.4f}  miss {100 * v['miss']:.2f}% ({v['miss_dem']})  "
          f"mismatch {100 * v['mismatch']:.3f}% ({v['mismatch_dem']})  FP {100 * v['fp']:.2f}% ({v['fp_dem']})")
if p.out:
    json.dump(dict(pred=p.pred, n_contexts=len(ctxs), gt=dict(stat), metrics=res), open(p.out, 'w'), indent=1)

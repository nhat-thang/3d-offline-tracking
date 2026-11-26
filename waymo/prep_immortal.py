"""Chuan bi du lieu cho ImmortalTracker-for-CTRL tu Waymo v2 (thay cho preparedata/waymo/*.py doc tfrecord v1):
  data/waymo/validation/ts_info/<seg>.json        : danh sach timestamp (micro giay) theo thu tu frame
  data/waymo/validation/ego_info/<seg>.npz        : {'0': 4x4 world_from_vehicle, '1': ...}
  data/waymo/validation/detection/<det>/dets/<seg>.npz : bboxes[frame] = [[cx,cy,cz,heading,l,w,h,score], ...], types[frame]
voi <seg> = 'segment-<context>_with_camera_labels' (ten ImmortalTracker dung va pred_bin.py tach lai context).
Chay: python waymo/prep_immortal.py waymo/subset8.txt data/waymo/ctrl_release/fsd_base_vehicle_val.bin fsd"""
import sys, os, json, collections
import numpy as np, pandas as pd
from waymo_open_dataset.protos import metrics_pb2

ctxs = open(sys.argv[1]).read().split(); det_bin, det_name = sys.argv[2], sys.argv[3]
V2 = 'data/waymo/v2/validation'; D = 'data/waymo/validation'
for sub in ('ts_info', 'ego_info', f'detection/{det_name}/dets'): os.makedirs(f'{D}/{sub}', exist_ok=True)
seg = lambda c: f'segment-{c}_with_camera_labels'

ts_of = {}
for c in ctxs:
    vp = pd.read_parquet(f'{V2}/vehicle_pose/{c}.parquet').sort_values('key.frame_timestamp_micros')
    ts = [int(t) for t in vp['key.frame_timestamp_micros']]
    tcol = [k for k in vp.columns if k.endswith('world_from_vehicle.transform')][0]
    json.dump(ts, open(f'{D}/ts_info/{seg(c)}.json', 'w'))
    np.savez_compressed(f'{D}/ego_info/{seg(c)}.npz', **{str(i): np.array(m, np.float64).reshape(4, 4) for i, m in enumerate(vp[tcol])})
    ts_of[c] = {t: i for i, t in enumerate(ts)}

objs = metrics_pb2.Objects(); objs.ParseFromString(open(det_bin, 'rb').read())
boxes = {c: collections.defaultdict(list) for c in ctxs}; types = {c: collections.defaultdict(list) for c in ctxs}
miss = 0
for o in objs.objects:
    c = o.context_name
    if c not in boxes: continue
    fi = ts_of[c].get(o.frame_timestamp_micros)
    if fi is None: miss += 1; continue
    b = o.object.box
    boxes[c][fi].append([b.center_x, b.center_y, b.center_z, b.heading, b.length, b.width, b.height, o.score])
    types[c][fi].append(o.object.type)
for c in ctxs:
    n = len(ts_of[c])
    bb = np.empty(n, object); tt = np.empty(n, object)
    for i in range(n):
        bb[i] = np.array(boxes[c][i], np.float64).reshape(-1, 8); tt[i] = np.array(types[c][i], np.int64)
    np.savez_compressed(f'{D}/detection/{det_name}/dets/{seg(c)}.npz', bboxes=bb, types=tt)
    print(f'{c}: {n} frame, {sum(len(v) for v in boxes[c].values())} box detection')
print('box khong khop timestamp:', miss)

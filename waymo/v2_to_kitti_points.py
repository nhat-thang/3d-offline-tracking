"""Chuyen LiDAR Waymo v2 (parquet) sang dinh dang diem ma CTRL/SST doc (giong bo chuyen doi waymo cua mmdet3d 0.15):
  data/waymo/kitti_format/training/velodyne/<idx>.bin : float32 [N, 6] = x, y, z, intensity, elongation, timestamp (he toa do xe)
  data/waymo/kitti_format/idx2timestamp.pkl, idx2contextname.pkl : anh xa idx -> timestamp (micro giay), idx -> ten chuoi
Ca 5 LiDAR, ca 2 return; LiDAR TOP dung pose tung pixel (lidar_pose) nhu mmdet3d. Tap validation nam trong thu muc 'training'
nhu quy uoc cua mmdet3d; idx = '1' + so thu tu chuoi (3 chu so) + so thu tu frame (3 chu so).
Chay: python waymo/v2_to_kitti_points.py waymo/subset8.txt"""
import sys, os, pickle, glob
import numpy as np, pandas as pd
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils

V2 = 'data/waymo/v2/validation'; OUT = 'data/waymo/kitti_format'
os.makedirs(f'{OUT}/training/velodyne', exist_ok=True)
all_ctx = sorted(os.path.basename(f)[:-8] for f in glob.glob(f'{V2}/lidar_box/*.parquet'))   # 202 chuoi, co dinh thu tu
todo = open(sys.argv[1]).read().split()

def load_pkl(name):
    p = f'{OUT}/{name}'
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
idx2ts, idx2ctx = load_pkl('idx2timestamp.pkl'), load_pkl('idx2contextname.pkl')

for ctx in todo:
    fi = all_ctx.index(ctx)
    rd = lambda c: pd.read_parquet(f'{V2}/{c}/{ctx}.parquet')
    lid, cal, lpo, vpo = rd('lidar'), rd('lidar_calibration'), rd('lidar_pose'), rd('vehicle_pose')
    cal_m = {int(r['key.laser_name']): v2.LiDARCalibrationComponent.from_dict(r) for _, r in cal.iterrows()}
    lpo_m = {int(r['key.frame_timestamp_micros']): v2.LiDARPoseComponent.from_dict(r) for _, r in lpo.iterrows()}
    vpo_m = {int(r['key.frame_timestamp_micros']): v2.VehiclePoseComponent.from_dict(r) for _, r in vpo.iterrows()}
    tss = sorted(int(t) for t in lid['key.frame_timestamp_micros'].unique())
    npts = []
    for k, ts in enumerate(tss):
        idx = f'1{fi:03d}{k:03d}'
        out = f'{OUT}/training/velodyne/{idx}.bin'
        idx2ts[idx], idx2ctx[idx] = ts, ctx
        if os.path.exists(out): continue
        parts = []
        for _, r in lid[lid['key.frame_timestamp_micros'] == ts].iterrows():
            l = v2.LiDARComponent.from_dict(r); ln = int(r['key.laser_name']); top = ln == 1
            for ri in (l.range_image_return1, l.range_image_return2):
                p = lidar_utils.convert_range_image_to_point_cloud(
                    ri, cal_m[ln], pixel_pose=lpo_m[ts].range_image_return1 if top else None,
                    frame_pose=vpo_m[ts] if top else None, keep_polar_features=True).numpy()
                parts.append(p[:, [3, 4, 5, 1, 2]])   # x, y, z, intensity, elongation
        pts = np.concatenate(parts).astype(np.float32)
        pts = np.hstack([pts, np.full((len(pts), 1), ts, np.float32)])
        pts.tofile(out); npts.append(len(pts))
    print(f'{ctx}: {len(tss)} frame, trung binh {np.mean(npts) if npts else 0:.0f} diem/frame', flush=True)
    pickle.dump(idx2ts, open(f'{OUT}/idx2timestamp.pkl', 'wb')); pickle.dump(idx2ctx, open(f'{OUT}/idx2contextname.pkl', 'wb'))
print('xong:', len(idx2ts), 'frame')

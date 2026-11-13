"""Thử ý tưởng 'kiểm tra box đoán bằng điểm LiDAR' trên 2 scene val có sẵn LiDAR (scene-0103, scene-0916, lấy từ nuscenes_mini).
Sinh box kéo dài thả lỏng (như R2_all), đếm số điểm LiDAR trong mỗi box, so tỉ lệ trùng GT theo số điểm."""
import sys, os, collections, argparse
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0, os.path.dirname(__file__)); import common, pipeline
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingBox
SC = ['scene-0103', 'scene-0916']
MINI = os.path.join(common.ROOT, 'data', 'nuscenes_mini')
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
tracks = {sc: dict(v) for sc, v in common.load_tracks(common.B0, where, set(SC)).items()}
p = argparse.Namespace(k=3, gap=3.0, d0=1.0, d1=1.0, d0p=0.6, d1p=0.6, min_len=2, min_score=0.1, ext=2.0, ext_tail=2.0, ext_decay=1.0)
pipeline.reassociate(tracks, p)
ego = {}; pc_global = {}
for sc in SC:
    for tok in scenes[sc]:
        sd = nusc.get('sample_data', nusc.get('sample', tok)['data']['LIDAR_TOP'])
        ep = nusc.get('ego_pose', sd['ego_pose_token']); cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ego[tok] = np.array(ep['translation'][:2])
        pc = LidarPointCloud.from_file(os.path.join(MINI, sd['filename']))
        pc.rotate(Quaternion(cs['rotation']).rotation_matrix); pc.translate(np.array(cs['translation']))
        pc.rotate(Quaternion(ep['rotation']).rotation_matrix); pc.translate(np.array(ep['translation']))
        pc_global[tok] = pc.points[:3]
pipeline.ADDED.clear()
pipeline.extend(tracks, p, scenes, where, ego)
cfg = config_factory('tracking_nips_2019')
gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
# box đầy đủ (kích thước, hướng) của box kéo dài: lấy từ track
full = {}
for sc in SC:
    for tid, seq in tracks[sc].items():
        for f, t, b in seq: full[(scenes[sc][f], tuple(np.round(b['translation'][:2], 4)))] = b
def npts(tok, b, margin=0.0):
    bx = Box(b['translation'], np.array(b['size']) + margin, Quaternion(b['rotation']))
    m = points_in_box(bx, pc_global[tok])
    zb = b['translation'][2] - b['size'][2] / 2  # đáy box; bỏ điểm mặt đất (thấp hơn đáy + 0,3 m)
    return int((m & (pc_global[tok][2] > zb + 0.3)).sum())
rows = []
for a in pipeline.ADDED:
    b = full[(a['tok'], tuple(np.round(a['x'], 4)))]
    g = [np.array(x.translation[:2]) for x in gt.boxes.get(a['tok'], []) if x.tracking_name == a['cls']]
    hit = bool(g) and min(np.linalg.norm(a['x'] - y) for y in g) <= 2.0
    rows.append((npts(a['tok'], b), hit, a['cls']))
# đối chứng: box thật (có detection) của track tin cậy
real = []
for sc in SC:
    for tid, seq in tracks[sc].items():
        for f, t, b in seq:
            if b['tracking_score'] >= 0.3:
                tok = scenes[sc][f]
                g = [np.array(x.translation[:2]) for x in gt.boxes.get(tok, []) if x.tracking_name == b['tracking_name']]
                real.append((npts(tok, b), bool(g) and min(np.linalg.norm(np.array(b['translation'][:2]) - y) for y in g) <= 2.0))
print('box keo dai:', len(rows), ' dung (khong loc):', round(np.mean([r[1] for r in rows]), 3))
for lo, hi in [(0, 0), (1, 2), (3, 9), (10, 49), (50, 10**9)]:
    s = [r[1] for r in rows if lo <= r[0] <= hi]
    print(f'  {lo}-{hi} diem: n={len(s)} dung={np.mean(s) if s else float("nan"):.2f}')
for th in [1, 3, 5, 10, 20]:
    s = [r[1] for r in rows if r[0] >= th]
    print(f'  giu box co >= {th} diem: con {len(s)} box, dung={np.mean(s) if s else float("nan"):.2f}, so box dung giu lai={sum(s)}/{sum(r[1] for r in rows)}')
print('doi chung box that score>=0.3:', len(real), 'dung', round(np.mean([r[1] for r in real]), 3), ' trung vi so diem', np.median([r[0] for r in real]))

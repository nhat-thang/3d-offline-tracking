"""Kiểm tra hướng và kích thước box so với GT (bộ chấm nuScenes không đo 2 thứ này).
Mỗi box thật (score >= 0.05) ghép với GT cùng lớp gần nhất trong 2 m; đo tỉ lệ lệch hướng > 90° và sai số kích thước (1 - IoU thể tích căn giữa).
Chạy: python offline/scripts/check_yaw_size.py <results.json> dev"""
import sys, os, collections
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0, os.path.dirname(__file__)); import common
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingBox
res, part = sys.argv[1], sys.argv[2]
cfg = config_factory('tracking_nips_2019')
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
scenes, where = common.scene_index(nusc)
keep = set(common.load_split()[part])
gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
tr = common.load_tracks(res, where, keep)
S = collections.defaultdict(list)
for sc in tr:
    for tid, seq in tr[sc].items():
        for f, t, b in seq:
            if b['tracking_score'] < 0.3: continue
            g = [x for x in gt.boxes.get(scenes[sc][f], []) if x.tracking_name == b['tracking_name']]
            if not g: continue
            d = [np.linalg.norm(np.array(x.translation[:2]) - b['translation'][:2]) for x in g]
            j = int(np.argmin(d))
            if d[j] > 2: continue
            x = g[j]
            dy = abs((Quaternion(b['rotation']).yaw_pitch_roll[0] - Quaternion(x.rotation).yaw_pitch_roll[0] + np.pi) % (2 * np.pi) - np.pi)
            inter = np.prod(np.minimum(b['size'], x.size)); iou = inter / (np.prod(b['size']) + np.prod(x.size) - inter)
            S[b['tracking_name']].append((dy > np.pi / 2, 1 - iou))
allv = [v for vs in S.values() for v in vs]
print(os.path.basename(os.path.dirname(res)), 'box ghep', len(allv), 'lech huong>90: %.2f%%' % (100 * np.mean([a for a, _ in allv])), 'sai so kich thuoc (1-IoU): %.3f' % np.mean([b for _, b in allv]))
for c in sorted(S): print('  ', c, len(S[c]), '%.2f%%' % (100 * np.mean([a for a, _ in S[c]])), '%.3f' % np.mean([b for _, b in S[c]]))

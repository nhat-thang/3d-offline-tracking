"""Phân tích độ chính xác của box kéo dài: box nào có GT (đã lọc như bộ chấm nuScenes) trong vòng 2 m thì tính là đúng.
Chạy: python offline/scripts/analyze_ext.py <tag> dev"""
import sys, os, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import common
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingBox
tag, part = sys.argv[1], sys.argv[2]
A = pickle.load(open(os.path.expanduser(f'~/scr/added_{tag}_{part}.pkl'), 'rb'))
cfg = config_factory('tracking_nips_2019')
nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
for a in A:
    g = [np.array(b.translation[:2]) for b in gt.boxes.get(a['tok'], []) if b.tracking_name == a['cls']]
    a['hit'] = bool(g) and min(np.linalg.norm(a['x'] - y) for y in g) <= 2.0
pickle.dump(A, open(os.path.expanduser(f'~/scr/added_{tag}_{part}_hit.pkl'), 'wb'))
def show(name, key):
    d = collections.defaultdict(list)
    for a in A: d[key(a)].append(a['hit'])
    print(name); [print(f'   {k}: n={len(v)} dung={np.mean(v):.2f}') for k, v in sorted(d.items())]
print('tong', len(A), 'dung', np.mean([a['hit'] for a in A]).round(3))
show('phia', lambda a: a['side'])
show('buoc', lambda a: (a['side'], a['steps']))
show('lop', lambda a: a['cls'])
show('score tb track', lambda a: min(int(a['score'] * 10) / 10, 0.8))
show('so box that', lambda a: min(a['nreal'], 20) // 5 * 5)
show('khoang cach toi xe (10m)', lambda a: int(a['dego'] // 10) * 10)
show('toc do (m/s)', lambda a: min(int(a['speed'] // 2) * 2, 10))

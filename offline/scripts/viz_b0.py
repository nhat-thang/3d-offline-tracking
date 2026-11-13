"""Tạo trang HTML xem kết quả tracking theo góc nhìn từ trên xuống (BEV) cho một scene.
Chạy (từ thư mục immortal_repro):  python offline/scripts/viz_b0.py [tên scene, vd scene-0103] [results.json]
Mặc định: B0, scene có nhiều vật bị đổi ID nhất trong b0_analysis.json.
Đầu ra: offline/results/viz/<tên>_<scene>.html (tự chứa, mở bằng trình duyệt)."""
import sys, os, json, collections, math
import numpy as np, ijson
sys.path.insert(0, os.path.dirname(__file__))
from analyze_b0 import ROOT, B0, DATAROOT, CLASSES, best_thresholds
from nuscenes import NuScenes
from nuscenes.eval.tracking.utils import category_to_tracking_name
from pyquaternion import Quaternion

res_path = sys.argv[2] if len(sys.argv) > 2 else B0
tag = os.path.basename(os.path.dirname(os.path.abspath(res_path)))
nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=False)
if len(sys.argv) > 1 and sys.argv[1] != '-':
    scene = next(s for s in nusc.scene if s['name'] == sys.argv[1])
else:
    ex = json.load(open(os.path.join(ROOT, 'offline/results/b0_analysis/b0_analysis.json')))['multi_id_examples']
    cnt = collections.Counter(e['scene'] for c in ex for e in ex[c])
    scene = nusc.get('scene', cnt.most_common(1)[0][0])
yaw = lambda q: float(Quaternion(q).yaw_pitch_roll[0])
toks, t = [], scene['first_sample_token']
while t:
    toks.append(t); t = nusc.get('sample', t)['next']
tokset = set(toks)
pred = {}
with open(res_path, 'rb') as f:
    for tok, boxes in ijson.kvitems(f, 'results', use_float=True):
        if tok in tokset:
            pred[tok] = boxes
            if len(pred) == len(tokset): break
thr = best_thresholds()
sc = collections.defaultdict(list)
for tok in toks:
    for b in pred.get(tok, []): sc[b['tracking_id']].append(b['tracking_score'])
mean = {k: float(np.mean(v)) for k, v in sc.items()}
ids, insts = {}, {}
r2 = lambda v: round(float(v), 2)
frames = []
for tok in toks:
    s = nusc.get('sample', tok)
    sd = nusc.get('sample_data', s['data']['LIDAR_TOP']); ep = nusc.get('ego_pose', sd['ego_pose_token'])
    gt = []
    for a in s['anns']:
        ann = nusc.get('sample_annotation', a)
        c = category_to_tracking_name(ann['category_name'])
        if c is None or ann['num_lidar_pts'] + ann['num_radar_pts'] == 0: continue
        gt.append([r2(ann['translation'][0]), r2(ann['translation'][1]), r2(ann['size'][0]), r2(ann['size'][1]), r2(yaw(ann['rotation'])),
                   CLASSES.index(c), insts.setdefault(ann['instance_token'], len(insts))])
    tr = []
    for b in pred.get(tok, []):
        c = b['tracking_name']; tid = b['tracking_id']
        tr.append([r2(b['translation'][0]), r2(b['translation'][1]), r2(b['size'][0]), r2(b['size'][1]), r2(yaw(b['rotation'])),
                   CLASSES.index(c), ids.setdefault(tid, len(ids)), round(mean[tid], 3),
                   1 if b['tracking_score'] < 0.01 else 0, 1 if mean[tid] >= thr[c] else 0])
    frames.append({'ego': [r2(ep['translation'][0]), r2(ep['translation'][1]), r2(yaw(ep['rotation']))], 'gt': gt, 'tr': tr})
data = {'name': scene['name'], 'desc': scene['description'], 'tag': tag, 'cls': CLASSES, 'thr': {k: round(v, 3) for k, v in thr.items()}, 'frames': frames}
html = open(os.path.join(os.path.dirname(__file__), 'viz_template.html'), encoding='utf-8').read().replace('__DATA__', json.dumps(data, separators=(',', ':')))
out = os.path.join(ROOT, 'offline', 'results', 'viz'); os.makedirs(out, exist_ok=True)
p = os.path.join(out, f"{tag}_{scene['name']}.html"); open(p, 'w', encoding='utf-8').write(html)
print(p, len(html) // 1024, 'KB', scene['name'], len(frames), 'frame', len(ids), 'track', len(insts), 'vat GT')

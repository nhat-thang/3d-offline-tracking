"""Task 2: thí nghiệm cách bộ chấm nuScenes phạt từng kiểu lỗi.
Lấy chính box GT (đã qua bộ lọc của bộ chấm) làm kết quả "hoàn hảo", rồi làm hỏng theo một kiểu duy nhất mỗi lần,
ghi results.json và chấm bằng devkit trên cùng tập scene. Mỗi thí nghiệm chỉ khác bản hoàn hảo đúng một thứ.
Chạy:  python offline/scripts/metric_experiments.py make <scene_meta.json>     # tạo các file results
       python offline/scripts/eval_split.py offline/results/metric_exp/<ten>/results_dev.json <scene_meta.json> offline/results/metric_exp/eval_<ten>"""
import sys, os, json, copy
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0, os.path.dirname(__file__)); import common
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingBox

OUT = os.path.join(common.ROOT, 'offline', 'results', 'metric_exp')
EXPS = ['hoan_hao', 'lech_0.5m', 'lech_1.5m', 'lech_1.99m', 'lech_2.5m', 'nguoc_huong_sai_kich_thuoc', 'doi_id_giua_track',
        'bo_1_trong_2_frame', 'keo_dai_2_frame', 'track_gia_score_thap', 'track_gia_score_cao', 'mat_40pt_track']


def main(meta):
    keep = set(json.load(open(meta))['scenes'])
    nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
    scenes, where = common.scene_index(nusc)
    cfg = config_factory('tracking_nips_2019')   # phải tạo cấu hình trước: nó khai báo danh sách lớp cho TrackingBox
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
    tracks = {}   # (scene, gt_id) -> [(frame, box_dict)]
    for tok in gt.sample_tokens:
        sc, fi, _ = where[tok]
        if sc not in keep: continue
        for g in gt.boxes[tok]:
            b = dict(sample_token=tok, translation=list(g.translation), size=list(g.size), rotation=list(g.rotation),
                     velocity=list(g.velocity), tracking_id=g.tracking_id, tracking_name=g.tracking_name, tracking_score=0.9)
            tracks.setdefault((sc, g.tracking_id), []).append((fi, b))
    for k in tracks: tracks[k].sort(key=lambda x: x[0])
    rng = np.random.default_rng(0)
    for name in EXPS:
        out = []
        for (sc, tid), seq in tracks.items():
            seq = [(f, copy.deepcopy(b)) for f, b in seq]
            if name.startswith('lech_'):
                d = float(name[5:-1])
                for f, b in seq:
                    a = rng.uniform(0, 2 * np.pi); b['translation'][0] += d * np.cos(a); b['translation'][1] += d * np.sin(a)
            elif name == 'nguoc_huong_sai_kich_thuoc':
                for f, b in seq:
                    b['rotation'] = list((Quaternion(b['rotation']) * Quaternion(axis=[0, 0, 1], angle=np.pi)).q)
                    b['size'] = [s * 1.5 for s in b['size']]
            elif name == 'doi_id_giua_track' and len(seq) >= 4:
                for f, b in seq[len(seq) // 2:]: b['tracking_id'] = tid + '_b'
            elif name == 'bo_1_trong_2_frame':
                seq = [x for i, x in enumerate(seq) if i % 2 == 0 or i == len(seq) - 1]
            elif name == 'keo_dai_2_frame' and len(seq) >= 2:
                toks = scenes[sc]
                (f0, b0), (f1, b1) = seq[0], seq[1]
                v = np.array(b1['translation']) - np.array(b0['translation'])
                ext = []
                for k in (1, 2):
                    if f0 - k >= 0:
                        e = copy.deepcopy(b0); e['translation'] = list(np.array(b0['translation']) - k * v); e['sample_token'] = toks[f0 - k]; ext.append((f0 - k, e))
                (fa, ba), (fb, bb) = seq[-2], seq[-1]
                v = np.array(bb['translation']) - np.array(ba['translation'])
                for k in (1, 2):
                    if fb + k < len(toks):
                        e = copy.deepcopy(bb); e['translation'] = list(np.array(bb['translation']) + k * v); e['sample_token'] = toks[fb + k]; ext.append((fb + k, e))
                seq = sorted(seq + ext, key=lambda x: x[0])
            elif name.startswith('track_gia'):
                fake = []
                for f, b in seq:
                    e = copy.deepcopy(b); e['translation'][0] += 5.0; e['tracking_id'] = tid + '_gia'
                    e['tracking_score'] = 0.3 if name.endswith('thap') else 0.95
                    fake.append((f, e))
                seq = seq + fake
            elif name == 'mat_40pt_track':
                if rng.random() < 0.4: seq = []
            out += [b for f, b in seq]
        res = {}
        for sc in keep:
            for tok in scenes[sc]: res[tok] = []
        for b in out: res[b['sample_token']].append(b)
        d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
        json.dump({'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False},
                   'results': res}, open(os.path.join(d, 'results_dev.json'), 'w'))
        print(name, len(out), 'box')


if __name__ == '__main__':
    main(sys.argv[2])

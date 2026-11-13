"""Hàm dùng chung cho module offline: đọc/ghi results.json theo track, chia tập dev/test.

Lưu ý về đầu ra của ImmortalTracker gốc trên nuScenes (evaluation/nuscenes/result_creation.py, Validity.agein2):
chỉ xuất box ở frame track được ghép với detection, cộng thêm ĐÚNG 1 frame dự đoán ngay sau khi mất detection
(score = score cũ x 0.01). Khi bị che lâu hơn, track vẫn sống bên trong tracker nhưng KHÔNG xuất box,
nên trong results.json track có khoảng trống. Detection CenterPoint có score >= 0.1, vì vậy box có score < 0.05
là box dự đoán (không có detection)."""
import os, json, collections
import numpy as np, ijson

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
B0 = os.path.join(ROOT, 'baseline_B0', 'results.json')
DATAROOT = os.path.join(ROOT, 'data', 'nuscenes_valslim')
SPLIT = os.path.join(ROOT, 'offline', 'split_dev_test.json')
CLASSES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
PRED_SCORE = 0.05  # box có score dưới mức này là box dự đoán, không có detection


def scene_index(nusc=None):
    """{scene_name: [sample_token theo thứ tự]}, {sample_token: (scene_name, frame_idx, timestamp_s)}"""
    if nusc is None:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=False)
    scenes, where = {}, {}
    for s in nusc.scene:
        toks, t = [], s['first_sample_token']
        while t:
            smp = nusc.get('sample', t)
            where[t] = (s['name'], len(toks), smp['timestamp'] / 1e6)
            toks.append(t); t = smp['next']
        scenes[s['name']] = toks
    return scenes, where


def load_tracks(path, where, keep_scenes=None):
    """{scene: {tracking_id: [(frame_idx, t, box_dict), ...] sắp theo frame}}"""
    tracks = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path, 'rb') as f:
        for tok, boxes in ijson.kvitems(f, 'results', use_float=True):
            sc, fi, ts = where[tok]
            if keep_scenes is not None and sc not in keep_scenes: continue
            for b in boxes:
                tracks[sc][b['tracking_id']].append((fi, ts, b))
    for sc in tracks:
        for tid in tracks[sc]:
            tracks[sc][tid].sort(key=lambda x: x[0])
    return tracks


def save_tracks(tracks, scenes, path, keep_scenes=None):
    res = {}
    for sc, toks in scenes.items():
        if keep_scenes is not None and sc not in keep_scenes: continue
        for tok in toks: res[tok] = []
        for tid, seq in tracks.get(sc, {}).items():
            for fi, ts, b in seq:
                res[toks[fi]].append(b)
    for tok in res:  # luật nuScenes: tối đa 500 box mỗi sample -> giữ 500 box score cao nhất
        if len(res[tok]) > 500:
            res[tok] = sorted(res[tok], key=lambda b: -b['tracking_score'])[:500]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False},
                   'results': res}, f)
    return sum(len(v) for v in res.values())


def load_split():
    return json.load(open(SPLIT))


def make_split(scenes):
    """Chia 150 scene val xen kẽ theo thứ tự tên: chẵn -> dev (dò tham số), lẻ -> test (báo cáo)."""
    names = sorted(scenes)
    sp = {'dev': names[0::2], 'test': names[1::2],
          'note': 'Chia xen ke theo ten scene. dev: thu nghiem, do tham so. test: chi dung de bao cao ket qua cuoi.'}
    json.dump(sp, open(SPLIT, 'w'), indent=1)
    return sp

"""Tao ban metadata nuScenes trainval rut gon chi cho 150 scene val, de may RAM nho load duoc.
Giu nguyen moi ban ghi cua val (sample, sample_annotation, keyframe sample_data, ego_pose cua chung),
nen ket qua prepare/eval tren val khong doi. Chay tu thu muc goc, theo tung buoc: python repro/make_val_slim.py <step>."""
import ijson, json, os, sys, shutil
from nuscenes.utils.splits import create_splits_scenes
SRC = sys.argv[2] if len(sys.argv) > 2 else 'data/nuscenes/v1.0-trainval'
DST = sys.argv[3] if len(sys.argv) > 3 else 'data/nuscenes_valslim/v1.0-trainval'
os.makedirs(DST, exist_ok=True)
step = sys.argv[1]
def stream(name):
    with open(os.path.join(SRC, name + '.json'), 'rb') as f:
        yield from ijson.items(f, 'item', use_float=True)
def dump(name, rows):
    json.dump(rows, open(os.path.join(DST, name + '.json'), 'w'))
    print(name, len(rows))
def load_dst(name): return json.load(open(os.path.join(DST, name + '.json')))
if step == 'small':
    for n in ['attribute', 'calibrated_sensor', 'category', 'log', 'map', 'sensor', 'visibility']:
        shutil.copy(os.path.join(SRC, n + '.json'), DST)
    val = set(create_splits_scenes()['val'])
    scenes = [s for s in json.load(open(os.path.join(SRC, 'scene.json'))) if s['name'] in val]
    dump('scene', scenes)
    st = {s['token'] for s in scenes}
    dump('sample', [s for s in json.load(open(os.path.join(SRC, 'sample.json'))) if s['scene_token'] in st])
elif step == 'sample_data':
    samp = {s['token'] for s in load_dst('sample')}
    dump('sample_data', [r for r in stream('sample_data') if r['is_key_frame'] and r['sample_token'] in samp])
elif step == 'ego_pose':
    need = {r['ego_pose_token'] for r in load_dst('sample_data')}
    dump('ego_pose', [r for r in stream('ego_pose') if r['token'] in need])
elif step == 'sample_annotation':
    samp = {s['token'] for s in load_dst('sample')}
    dump('sample_annotation', [r for r in stream('sample_annotation') if r['sample_token'] in samp])
elif step == 'instance':
    need = {r['instance_token'] for r in load_dst('sample_annotation')}
    dump('instance', [r for r in json.load(open(os.path.join(SRC, 'instance.json'))) if r['token'] in need])

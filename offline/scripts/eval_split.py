"""Chấm điểm nuScenes tracking (bộ chấm chính thức) trên một phần của tập val.
Chạy (từ thư mục immortal_repro):  python offline/scripts/eval_split.py <results.json> <dev|test|val> <thư_mục_ra>
Cách làm: lọc results.json chỉ giữ sample của các scene thuộc phần đã chọn, đổi danh sách scene 'val' của devkit
thành đúng phần đó, rồi gọi TrackingEval như bình thường. Với 'val' kết quả giống hệt chấm đầy đủ 150 scene."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import common, ijson
res_path, part, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
scenes, where = common.scene_index()
keep = set(scenes) if part == 'val' else set(common.load_split()[part])
import nuscenes.utils.splits as S
_orig = S.create_splits_scenes
def patched(verbose=False):
    d = _orig(verbose); d['val'] = [s for s in d['val'] if s in keep]; return d
S.create_splits_scenes = patched
import nuscenes.eval.common.loaders as CL, nuscenes.eval.tracking.loaders as TL
CL.create_splits_scenes = patched; TL.create_splits_scenes = patched
filtered = os.path.join(out, 'results_' + part + '.json')
res, dropped = {}, 0
with open(res_path, 'rb') as f:
    for tok, boxes in ijson.kvitems(f, 'results', use_float=True):
        if where[tok][0] in keep:
            if len(boxes) > 500:  # luật nuScenes: tối đa 500 box mỗi sample -> giữ 500 box score cao nhất
                dropped += len(boxes) - 500
                boxes = sorted(boxes, key=lambda b: -b['tracking_score'])[:500]
            res[tok] = boxes
json.dump({'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False}, 'results': res}, open(filtered, 'w'))
print('bo bot', dropped, 'box do vuot 500 box/sample')
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
ev = TrackingEval(config=config_factory('tracking_nips_2019'), result_path=filtered, eval_set='val', output_dir=out,
                  nusc_version='v1.0-trainval', nusc_dataroot=common.DATAROOT, verbose=False)
m = ev.main(render_curves=False)
os.remove(filtered)
keys = ['amota', 'amotp', 'mota', 'recall', 'ids', 'frag', 'fp', 'fn', 'tid', 'lgd']
line = ' | '.join(f'{m[k]:.3f}' if k in ('amota', 'amotp', 'mota', 'recall', 'tid', 'lgd') else f'{m[k]:.0f}' for k in keys)
open(os.path.join(out, 'summary_row.md'), 'w').write(f'| {os.path.basename(os.path.dirname(os.path.abspath(res_path)))} ({part}) | {line} |\n')
print('| Phien ban | ' + ' | '.join(k.upper() for k in keys) + ' |'); print(open(os.path.join(out, 'summary_row.md')).read())

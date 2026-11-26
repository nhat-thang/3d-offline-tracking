# Chay tools/test.py cua SST voi cau hinh ctrl_veh_24e.py, sua cau hinh trong bo nho (khong dump ra file:
# mmcv 1.3.9 + yapf moi bi loi FormatCode(verify=...)). Dung: python ctrl_test.py <tta 0|1> <batch> <checkpoint> <pklfile_prefix>
import os, sys, runpy
from mmcv import Config
tta, bs, ckpt, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
_orig = Config.fromfile
def fromfile(*a, **k):
    c = _orig(*a, **k)
    if tta:  # Track TTA dung nhu cac dong tac gia de comment trong config
        c.data.test.pipeline = c.tta_pipeline
        c.model.test_cfg.tta = dict(merge='weighted')
    if os.environ.get('PROPOSALS'): c.data.test.tracklet_proposals_file = os.environ['PROPOSALS']  # track dau vao khac (vd. ban de xuat)
    c.data.test.samples_per_gpu = bs; c.data.samples_per_gpu = bs; c.data.workers_per_gpu = 2
    print('TTA:', bool(tta), '| batch:', bs, '| proposals:', c.data.test.tracklet_proposals_file)
    return c
Config.fromfile = staticmethod(fromfile)
sys.argv = ['tools/test.py', 'configs/ctrl/ctrl_veh_24e.py', ckpt, '--out', prefix + '.pkl', '--eval', 'waymo', '--eval-options', f'pklfile_prefix={prefix}']
sys.path.insert(0, 'tools')
runpy.run_path('tools/test.py', run_name='__main__')

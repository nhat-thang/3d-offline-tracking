"""Chay main_nuscenes.py cua ImmortalTracker theo tung dot co gioi han thoi gian (may chi giu tien trinh <180s).
Logic giong het main() goc: cung config, cung NuScenesLoader/sequence_mot, ID = '<file_index>_<id>' voi
file_index la vi tri trong danh sach day du 150 scene (khong bi lech nhu co --skip). Bo qua scene da co file.
Dung: python run_chunk.py <name> <obj_type> <budget_s> <nproc>   (chay trong thu muc ImmortalTracker)"""
import sys, os, time, multiprocessing as mp
name, obj_type, budget, nproc = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
sys.argv = ['main_nuscenes.py', '--name', name, '--det_name', 'cp', '--config_path', 'configs/nu_configs/immortal.yaml']
sys.path.insert(0, '.')
import main_nuscenes as M
data_folder = './data/nuscenes/validation_2hz'
det_folder = os.path.join('./data/nuscenes/validation_2hz/detection', 'cp')
summary = os.path.join('./mot_results/nuscenes/validation_2hz', name, 'summary', obj_type)
os.makedirs(summary, exist_ok=True)
for t in 'car,bus,trailer,truck,pedestrian,bicycle,motorcycle'.split(','):
    os.makedirs(os.path.join('./mot_results/nuscenes/validation_2hz', name, 'summary', t), exist_ok=True)
files = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
todo = [(i, f) for i, f in enumerate(files) if not os.path.exists(os.path.join(summary, f))]
configs = M.yaml.load(open('configs/nu_configs/immortal.yaml', 'r'))
T0 = time.time()
def work(item):
    file_index, file_name = item
    if time.time() - T0 > budget:
        return 0
    seg = file_name.split('.')[0]
    loader = M.NuScenesLoader(configs, [obj_type], seg, data_folder, det_folder, 0)
    ids, bboxes, states, types = M.sequence_mot(configs, loader, obj_type, file_index, None, None, False)
    for fi in range(len(ids)):
        for i in range(len(ids[fi])):
            ids[fi][i] = '{:}_{:}'.format(file_index, ids[fi][i])
    M.np.savez_compressed(os.path.join(summary, '{}.npz'.format(seg)), ids=ids, bboxes=bboxes, states=states, types=types)
    return 1
if __name__ == '__main__':
    with mp.Pool(nproc) as p:
        done = sum(p.imap_unordered(work, todo))
    left = len(todo) - done
    print(f'{obj_type}: +{done} scene, con lai {left}')

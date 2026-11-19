"""Gộp các phần dữ liệu do make_dataset.py --chunk tạo ra thành một file.
Chạy: python offline/refine/merge_dataset.py B0_dev 3   -> đọc B0_dev_c0..c2, ghi B0_dev.npz + B0_dev_meta.json"""
import os, sys, json
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'refine', 'data')
name, k = sys.argv[1], int(sys.argv[2])
parts = [np.load(os.path.join(D, f'{name}_c{i}.npz')) for i in range(k)]
metas = [json.load(open(os.path.join(D, f'{name}_c{i}_meta.json'))) for i in range(k)]
out = {key: np.concatenate([d[key] for d in parts]) for key in ['pts', 'feat', 'yreg', 'yreal', 'gt_inst', 'frame', 'tcls']}
ptr, tsc, boff, soff = [np.zeros(1, np.int64)], [], 0, 0
for d, m in zip(parts, metas):
    ptr.append(d['ptr'][1:] + boff); tsc.append(d['tscene'] + soff)
    boff += len(d['pts']); soff += len(m['scenes'])
out['ptr'] = np.concatenate(ptr); out['tscene'] = np.concatenate(tsc)
# gt_inst là chỉ số cục bộ của từng phần: cộng lệch để không trùng giữa các phần
off, gi = 0, []
for d in parts:
    g = d['gt_inst'].copy(); g[g >= 0] += off; gi.append(g); off += int(d['gt_inst'].max()) + 1 if len(g) else 0
out['gt_inst'] = np.concatenate(gi)
np.savez(os.path.join(D, name + '.npz'), **out)
meta = dict(metas[0]); meta['scenes'] = sum([m['scenes'] for m in metas], []); meta['track_ids'] = sum([m['track_ids'] for m in metas], [])
meta['n_box'] = int(len(out['pts'])); meta['n_track'] = len(meta['track_ids']); meta['real_ratio'] = float(out['yreal'].mean())
json.dump(meta, open(os.path.join(D, name + '_meta.json'), 'w'))
print(f"xong {name}: {len(meta['scenes'])} scene, {meta['n_box']} box, {meta['n_track']} track, ti le box that {meta['real_ratio']:.3f}")

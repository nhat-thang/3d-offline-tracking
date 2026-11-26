# Tao data/waymo/kitti_format/context2timestamp.pkl cho tools/ctrl/extend_tracks.py:
# {context_name: [timestamp (micro giay) cua tung frame, tang dan]}, lay tu idx2timestamp.pkl + idx2contextname.pkl (v2_to_kitti_points.py).
import pickle, sys
from collections import defaultdict
d = sys.argv[1] if len(sys.argv) > 1 else 'data/waymo/kitti_format'
i2t = pickle.load(open(f'{d}/idx2timestamp.pkl', 'rb')); i2c = pickle.load(open(f'{d}/idx2contextname.pkl', 'rb'))
out = defaultdict(list)
for idx, ts in i2t.items():
    out[i2c[idx]].append(int(ts))
out = {k: sorted(set(v)) for k, v in out.items()}
pickle.dump(out, open(f'{d}/context2timestamp.pkl', 'wb'))
print('context2timestamp:', len(out), 'chuoi,', sum(map(len, out.values())), 'frame; vd', next(iter(out)), out[next(iter(out))][:2])

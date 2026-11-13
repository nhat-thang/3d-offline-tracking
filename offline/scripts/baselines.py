"""Hai baseline offline, mỗi cái áp riêng lên B0 (không cộng dồn):
  B1: nội suy tuyến tính các khoảng trống bên trong mỗi track (vị trí, kích thước, góc, score).
  B2: kéo dài hai chiều kiểu CTRL (arXiv 2304.12315, mục 3.2 + phụ lục D), không có bước refine.
      CTRL dùng 10 Hz: track > 100 frame kéo tới hết sequence, track ngắn hơn kéo thêm 20 frame.
      Đổi sang thời gian cho nuScenes 2 Hz: track dài > 10 s kéo tới đầu/cuối scene, còn lại kéo thêm 2 s (4 keyframe),
      cả về quá khứ và tương lai, theo vận tốc không đổi ước lượng từ các box có detection ở đầu/cuối track.
      Score box kéo dài = score detection gần nhất của track (CTRL để mạng refine chấm lại score; ở đây không có refine).
Chạy (từ thư mục immortal_repro):  python offline/scripts/baselines.py b1|b2 [val|dev|test]
Đầu ra: offline/results/B1/results.json (val, đủ 150 scene) hoặc results_<dev|test>.json (chỉ các scene của phần đó)."""
import sys, os, copy
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0, os.path.dirname(__file__))
import common


def yaw_of(b): return Quaternion(b['rotation']).yaw_pitch_roll[0]
def quat(yaw): return list(Quaternion(axis=[0, 0, 1], angle=yaw).q)


def interp_box(a, b, r):
    """Box ở tỉ lệ r giữa box a (r=0) và b (r=1)."""
    n = copy.deepcopy(a)
    n['translation'] = list((1 - r) * np.array(a['translation']) + r * np.array(b['translation']))
    n['size'] = list((1 - r) * np.array(a['size']) + r * np.array(b['size']))
    n['rotation'] = list(Quaternion.slerp(Quaternion(a['rotation']), Quaternion(b['rotation']), r).q)
    n['tracking_score'] = (1 - r) * a['tracking_score'] + r * b['tracking_score']
    return n


def b1(tracks, scenes, where):
    added = 0
    for sc in tracks:
        for tid, seq in tracks[sc].items():
            out = [seq[0]]
            for (f0, t0, a), (f1, t1, b) in zip(seq, seq[1:]):
                for f in range(f0 + 1, f1):
                    tok = scenes[sc][f]; t = where[tok][2]
                    out.append((f, t, interp_box(a, b, (t - t0) / (t1 - t0)))); added += 1
                out.append((f1, t1, b))
            tracks[sc][tid] = out
    return added


def velocity(pts):
    """pts: [(t, xyz)] -> vận tốc (m/s) theo bình phương tối thiểu; 1 điểm thì 0."""
    if len(pts) < 2: return np.zeros(3)
    t = np.array([p[0] for p in pts]); x = np.array([p[1] for p in pts])
    t = t - t.mean(); den = (t ** 2).sum()
    return (t[:, None] * (x - x.mean(0))).sum(0) / den if den > 0 else np.zeros(3)


def b2(tracks, scenes, where, long_s=10.0, ext_s=2.0, k=3):
    added = 0
    for sc in tracks:
        toks = scenes[sc]; T = [where[t][2] for t in toks]
        for tid, seq in tracks[sc].items():
            real = [(f, t, b) for f, t, b in seq if b['tracking_score'] >= common.PRED_SCORE] or seq
            dur = seq[-1][1] - seq[0][1]
            new = []
            for side in ('head', 'tail'):
                ref = real[:k] if side == 'head' else real[-k:]
                v = velocity([(t, np.array(b['translation'])) for _, t, b in ref])
                v[2] = 0.0
                end_f, end_t, end_b = seq[0] if side == 'head' else seq[-1]
                anchor_t = ref[0][1] if side == 'head' else ref[-1][1]
                anchor = np.array((ref[0] if side == 'head' else ref[-1])[2]['translation'])
                score = (ref[0] if side == 'head' else ref[-1])[2]['tracking_score']
                frames = range(end_f - 1, -1, -1) if side == 'head' else range(end_f + 1, len(toks))
                for f in frames:
                    if dur <= long_s and abs(T[f] - end_t) > ext_s + 1e-6: break
                    n = dict(end_b)
                    n['translation'] = list(anchor + v * (T[f] - anchor_t))
                    n['tracking_score'] = score
                    n['sample_token'] = toks[f]
                    new.append((f, T[f], n))
            if new:
                tracks[sc][tid] = sorted(seq + new, key=lambda x: x[0]); added += len(new)
    return added


if __name__ == '__main__':
    name = sys.argv[1]; part = sys.argv[2] if len(sys.argv) > 2 else 'val'
    scenes, where = common.scene_index()
    keep = None if part == 'val' else set(common.load_split()[part])
    tracks = common.load_tracks(common.B0, where, keep)
    added = {'b1': b1, 'b2': b2}[name](tracks, scenes, where)
    for sc in tracks:
        for tid, seq in tracks[sc].items():
            for f, t, b in seq: b['sample_token'] = scenes[sc][f]
    out = os.path.join(common.ROOT, 'offline', 'results', name.upper(), 'results.json' if part == 'val' else f'results_{part}.json')
    n = common.save_tracks(tracks, scenes, out, keep)
    print(name.upper(), 'them', added, 'box; tong', n, 'box ->', out)

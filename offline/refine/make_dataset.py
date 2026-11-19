"""Bước 4 của Task 1: tạo dữ liệu train cho model refine từ track + point cloud LiDAR keyframe.

Mỗi box của tracker thành một mẫu:
  - Đầu vào: các điểm LiDAR quanh box, đưa về hệ tọa độ của box (tâm box là gốc, trục x dọc theo chiều dài xe),
    lấy ngẫu nhiên NPTS điểm; kèm vài đặc trưng của box (score, kích thước, số điểm, khoảng cách tới xe ...).
  - Nhãn: ghép box với box GT cùng lớp trong cùng frame (Hungarian theo khoảng cách tâm, tối đa 2 m, giống ngưỡng
    ghép của bộ chấm nuScenes). Box ghép được: real = 1 và độ lệch tới GT (trong hệ box):
        dx, dy, dz, log(w_gt/w), log(l_gt/l), log(h_gt/h), sin(dyaw), cos(dyaw)
    Box không ghép được: real = 0 (box giả hoặc box lệch quá xa).
  - GT dùng đúng bộ lọc của bộ chấm (load_gt + filter_eval_boxes): bỏ box GT ngoài tầm chấm và box GT không có điểm,
    nên nhãn khớp với cách bộ chấm tính đúng/sai.
Các box được gom theo track (sắp theo thời gian) để model nhìn cả track.
Box nằm ngoài tầm chấm của lớp (40/50 m) bị bỏ qua: bộ chấm không xét chúng, gắn nhãn 0 sẽ dạy sai model.
Chỉ dùng scene có đủ file LiDAR của mọi keyframe.

Chạy (từ thư mục gốc, môi trường có nuscenes-devkit, không cần GPU):
  python offline/refine/make_dataset.py B0 dev                     # track của B0
  python offline/refine/make_dataset.py offline/results/R3b/results_dev.json dev --tag R3b
  thêm --max_scenes 3 để chạy thử nhanh.
  Máy chậm hoặc bị giới hạn thời gian mỗi lệnh: chạy từng phần --chunk 0..k-1 --nchunks k, rồi
  python offline/refine/merge_dataset.py <tag>_<part> k
Đầu ra: offline/results/refine/data/<tag>_<part>.npz và <tag>_<part>_meta.json"""
import sys, os, json, argparse, pickle, time
import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import common

NPTS = 256      # số điểm lấy cho mỗi box
MARGIN = 1.0    # nới box thêm 1 m mỗi phía khi cắt điểm (box đoán có thể lệch so với vật thật)
MATCH = 2.0     # ngưỡng ghép box với GT (m), giống bộ chấm nuScenes
RANGE = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40}
CLS_ID = {c: i for i, c in enumerate(common.CLASSES)}
BOX_FEATS = ['score', 'is_pred', 'log_w', 'log_l', 'log_h', 'z_rel_ego', 'dist_ego/50', 'log1p_n_in', 'log1p_n_crop',
             't_in_track/20', 'v_long', 'v_lat']
OUT_DIR = os.path.join(common.ROOT, 'offline', 'results', 'refine', 'data')


def yaw_of(q):
    return Quaternion(q).yaw_pitch_roll[0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_points(nusc, tok):
    """Point cloud keyframe ở hệ toàn cục: (N, 4) = x, y, z, cường độ/255; và vị trí xe (x, y, z)."""
    sd = nusc.get('sample_data', nusc.get('sample', tok)['data']['LIDAR_TOP'])
    path = os.path.join(common.DATAROOT, sd['filename'])
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])   # cảm biến -> xe
    ep = nusc.get('ego_pose', sd['ego_pose_token'])                     # xe -> toàn cục (đã kiểm ở bước 3)
    xyz = pts[:, :3] @ Quaternion(cs['rotation']).rotation_matrix.T + np.array(cs['translation'])
    xyz = xyz @ Quaternion(ep['rotation']).rotation_matrix.T + np.array(ep['translation'])
    return np.hstack([xyz, pts[:, 3:4] / 255.0]).astype(np.float32), np.array(ep['translation'])


def crop(pts, c, size, yaw, rng):
    """Cắt điểm quanh box, đổi sang hệ box. Trả về (NPTS, 4), số điểm trong vùng cắt, số điểm trong box gốc."""
    w, l, h = size
    r = 0.5 * np.hypot(w, l) + MARGIN
    p = pts[(np.abs(pts[:, 0] - c[0]) < r) & (np.abs(pts[:, 1] - c[1]) < r)]   # lọc thô cho nhanh
    d = p[:, :3] - c
    cy, sy = np.cos(yaw), np.sin(yaw)
    x = d[:, 0] * cy + d[:, 1] * sy
    y = -d[:, 0] * sy + d[:, 1] * cy
    z = d[:, 2]
    k = (np.abs(x) <= l / 2 + MARGIN) & (np.abs(y) <= w / 2 + MARGIN) & (np.abs(z) <= h / 2 + MARGIN)
    n_in = int(((np.abs(x) <= l / 2) & (np.abs(y) <= w / 2) & (np.abs(z) <= h / 2)).sum())
    loc = np.stack([x[k], y[k], z[k], p[k, 3]], 1)
    n = len(loc)
    out = np.zeros((NPTS, 4), np.float32)
    if n:
        out[:] = loc[rng.choice(n, NPTS, replace=n < NPTS)]
    return out, n, n_in


def match_frame(boxes, gts):
    """boxes: [(idx, box_dict)], gts: [TrackingBox] cùng frame. Trả về {idx: gt} ghép theo từng lớp."""
    res = {}
    for cls in common.CLASSES:
        P = [(i, b) for i, b in boxes if b['tracking_name'] == cls]
        G = [g for g in gts if g.tracking_name == cls]
        if not P or not G: continue
        D = np.array([[np.linalg.norm(np.array(b['translation'][:2]) - np.array(g.translation[:2])) for g in G] for _, b in P])
        C = np.where(D <= MATCH, D, 1e6)
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < 1e6: res[P[i][0]] = G[j]
    return res


def load_input_tracks(src, part, where, keep):
    if src == 'B0':   # dùng chung bộ nhớ tạm với pipeline.py (đọc results.json của B0 rất chậm)
        cache = os.path.join(os.path.expanduser('~'), 'scr', f'b0_tracks_{part}.pkl')
        if os.path.exists(cache): return pickle.load(open(cache, 'rb'))
        tracks = {sc: dict(v) for sc, v in common.load_tracks(common.B0, where, keep).items()}
        os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(tracks, open(cache, 'wb'))
        return tracks
    # đọc results.json bằng ijson khá chậm -> lưu tạm theo đường dẫn + thời điểm sửa file
    key = os.path.abspath(src).replace(os.sep, '_').replace(':', '') + f'_{int(os.path.getmtime(src))}'
    cache = os.path.join(os.path.expanduser('~'), 'scr', 'tracks_' + key[-120:] + '.pkl')
    if os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    tracks = {sc: dict(v) for sc, v in common.load_tracks(src, where, keep).items()}
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(tracks, open(cache, 'wb'))
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help="'B0' hoặc đường dẫn results_<part>.json của một phiên bản")
    ap.add_argument('part', choices=['dev', 'test'])
    ap.add_argument('--tag', help='tên đầu ra (mặc định B0 hoặc tên thư mục của file results)')
    ap.add_argument('--max_scenes', type=int, default=0, help='chỉ lấy N scene đầu (chạy thử)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=-1, help='chỉ làm phần thứ i (0..nchunks-1) của danh sách scene, rồi gộp bằng merge_dataset.py')
    ap.add_argument('--nchunks', type=int, default=1)
    p = ap.parse_args()
    tag = p.tag or ('B0' if p.src == 'B0' else os.path.basename(os.path.dirname(os.path.abspath(p.src))))
    rng = np.random.default_rng(p.seed)
    t0 = time.time()

    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
    from nuscenes.eval.tracking.data_classes import TrackingBox
    nusc = NuScenes(version='v1.0-trainval', dataroot=common.DATAROOT, verbose=False)
    scenes, where = common.scene_index(nusc)
    keep = set(common.load_split()[p.part])
    tracks = load_input_tracks(p.src, p.part, where, keep)
    cfg = config_factory('tracking_nips_2019')
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, 'val', TrackingBox, verbose=False)), cfg.class_range, verbose=False)
    print(f'doc du lieu xong: {time.time() - t0:.0f}s', flush=True)

    # scene có đủ LiDAR
    lidar_dir = os.path.join(common.DATAROOT, 'samples', 'LIDAR_TOP')
    have = set(os.listdir(lidar_dir)) if os.path.isdir(lidar_dir) else set()
    def complete(sc):
        return all(os.path.basename(nusc.get('sample_data', nusc.get('sample', t)['data']['LIDAR_TOP'])['filename']) in have
                   for t in scenes[sc])
    use = [sc for sc in sorted(keep) if sc in tracks and complete(sc)]
    if p.max_scenes: use = use[:p.max_scenes]
    if p.chunk >= 0:
        n = len(use); use = use[p.chunk * n // p.nchunks:(p.chunk + 1) * n // p.nchunks]
    print(f'{p.part}: dung {len(use)} / {len(keep)} scene (cac scene con lai chua du file LiDAR)', flush=True)

    PTS, FEAT, YREG, YREAL, GTI, FRAME, PTR, TCLS, TSC, TID = [], [], [], [], [], [], [0], [], [], []
    inst_id = {}
    for si, sc in enumerate(use):
        toks = scenes[sc]
        per_frame = {}    # frame -> [((tid, k), box)]
        for tid, seq in tracks[sc].items():
            for k, (f, t, b) in enumerate(seq):
                per_frame.setdefault(f, []).append(((tid, k), b))
        sample = {}       # (tid, k) -> (pts, feat, yreg, yreal, gt_inst)
        for f, items in per_frame.items():
            pts, ego = load_points(nusc, toks[f])
            items = [(key, b) for key, b in items
                     if np.linalg.norm(np.array(b['translation'][:2]) - ego[:2]) <= RANGE[b['tracking_name']]]
            m = match_frame(items, gt.boxes.get(toks[f], []))
            for key, b in items:
                c = np.array(b['translation']); size = np.array(b['size']); yaw = yaw_of(b['rotation'])
                cp, n, n_in = crop(pts, c, size, yaw, rng)
                v = np.array(b.get('velocity', [0, 0])[:2], float)
                cy, sy = np.cos(yaw), np.sin(yaw)
                feat = [b['tracking_score'], float(b['tracking_score'] < common.PRED_SCORE), *np.log(np.maximum(size, 1e-2)),
                        c[2] - ego[2], np.linalg.norm(c[:2] - ego[:2]) / 50, np.log1p(n_in), np.log1p(n), 0.0,
                        v[0] * cy + v[1] * sy, -v[0] * sy + v[1] * cy]
                yreg = np.zeros(8, np.float32); yreal = 0; gi = -1
                g = m.get(key)
                if g is not None:
                    d = np.array(g.translation) - c
                    dyaw = wrap(yaw_of(g.rotation) - yaw)
                    yreg[:] = [d[0] * cy + d[1] * sy, -d[0] * sy + d[1] * cy, d[2],
                               *np.log(np.maximum(np.array(g.size), 1e-2) / np.maximum(size, 1e-2)), np.sin(dyaw), np.cos(dyaw)]
                    yreal = 1; gi = inst_id.setdefault(g.tracking_id, len(inst_id))
                sample[key] = (cp, np.array(feat, np.float32), yreg, yreal, gi)
        for tid, seq in tracks[sc].items():
            ks = [k for k in range(len(seq)) if (tid, k) in sample]
            if not ks: continue
            t_first = seq[ks[0]][1]
            for k in ks:
                cp, feat, yreg, yreal, gi = sample[(tid, k)]
                feat[9] = (seq[k][1] - t_first) / 20.0
                PTS.append(cp.astype(np.float16)); FEAT.append(feat); YREG.append(yreg); YREAL.append(yreal); GTI.append(gi)
                FRAME.append(seq[k][0])
            PTR.append(len(PTS)); TSC.append(si); TID.append(str(tid))
            names = [seq[k][2]['tracking_name'] for k in ks]
            TCLS.append(CLS_ID[max(set(names), key=names.count)])
        print(f'  [{si + 1}/{len(use)}] {sc}: {len(PTS)} box, {len(TID)} track ({time.time() - t0:.0f}s)', flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{tag}_{p.part}' + (f'_c{p.chunk}' if p.chunk >= 0 else ''))
    np.savez(out + '.npz', pts=np.stack(PTS) if PTS else np.zeros((0, NPTS, 4), np.float16),
             feat=np.stack(FEAT), yreg=np.stack(YREG), yreal=np.array(YREAL, np.int8), gt_inst=np.array(GTI, np.int32),
             frame=np.array(FRAME, np.int16), ptr=np.array(PTR, np.int64), tcls=np.array(TCLS, np.int8),
             tscene=np.array(TSC, np.int32))
    yreal = np.array(YREAL)
    meta = dict(src=p.src, part=p.part, scenes=use, track_ids=TID, box_feats=BOX_FEATS, classes=common.CLASSES,
                npts=NPTS, margin=MARGIN, match=MATCH, n_box=len(PTS), n_track=len(TID), real_ratio=float(yreal.mean()))
    json.dump(meta, open(out + '_meta.json', 'w'))
    print(f'xong: {out}.npz | {len(PTS)} box, {len(TID)} track, ti le box that {yreal.mean():.3f} | {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

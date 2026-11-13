"""Phân tích quy mô vấn đề trên kết quả mốc B0, trước khi viết module offline.

Đo, cho từng class, tại đúng ngưỡng score mà bộ chấm chọn (ngưỡng cho MOTA cao nhất):
  1. Bao nhiêu vật thật (GT instance) bị bám bởi >= 2 ID khác nhau  -> việc của bước "nối track".
  2. Các khoảng bị mất giữa track (vật thật có mặt nhưng không được ghép), độ dài theo frame
     -> việc của bước "lấp đoạn bị che".
  3. Bỏ sót ở đầu / cuối đời vật (trước lần ghép đầu tiên, sau lần ghép cuối)
     -> việc của bước "kéo dài track".
  4. Vật bị bỏ sót hoàn toàn (không frame nào được ghép).
Ghép box giống bộ chấm: lọc GT/pred như devkit, ghép theo khoảng cách tâm <= 2 m (Hungarian từng frame).

Lưu ý: bộ chấm thay score từng box bằng score TRUNG BÌNH của cả track trước khi lọc theo ngưỡng,
nên ở ngưỡng tốt nhất cả track được giữ hoặc bỏ cùng lúc. Script làm y như vậy (dùng create_tracks của devkit).

Chạy (từ thư mục immortal_repro):
  python offline/scripts/analyze_b0.py analyze   # ghi offline/results/b0_analysis/
"""
import sys, os, json, collections
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
B0 = os.path.join(ROOT, 'baseline_B0', 'results.json')
DETAILS = os.path.join(ROOT, 'ImmortalTracker', 'mot_results', 'nuscenes', 'validation_2hz', 'immortal_noflip', 'results', 'metrics_details.json')
OUT = os.path.join(ROOT, 'offline', 'results', 'b0_analysis')
FILTERED = os.path.join(OUT, 'pred_at_best_threshold.json')
DATAROOT = os.path.join(ROOT, 'data', 'nuscenes_valslim')
CLASSES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
os.makedirs(OUT, exist_ok=True)


def best_thresholds():
    d = json.load(open(DETAILS))
    return {c: float(d[c]['confidence'][int(np.nanargmax(np.array(d[c]['mota'], dtype=float)))]) for c in CLASSES}


def step_filter():
    import ijson
    thr = best_thresholds()
    res, n_in, n_out = {}, 0, 0
    with open(B0, 'rb') as f:
        for tok, boxes in ijson.kvitems(f, 'results', use_float=True):
            keep = []
            for b in boxes:
                n_in += 1
                if b['tracking_name'] in thr and b['tracking_score'] >= thr[b['tracking_name']]:
                    keep.append(b)
            n_out += len(keep)
            res[tok] = keep
    json.dump({'results': res, 'meta': {'use_lidar': True, 'use_camera': False, 'use_radar': False, 'use_map': False, 'use_external': False},
               'thresholds': thr}, open(FILTERED, 'w'))
    print('nguong theo class:', {k: round(v, 3) for k, v in thr.items()})
    print(f'box: {n_in} -> {n_out} sau loc')


def step_analyze():
    from scipy.optimize import linear_sum_assignment
    from nuscenes import NuScenes
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from nuscenes.eval.tracking.loaders import create_tracks
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.data_classes import TrackingBox
    from nuscenes.eval.common.loaders import load_prediction, load_gt

    cfg = config_factory('tracking_nips_2019')  # phải gọi trước load_gt (đặt TRACKING_NAMES)
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=False)
    import ijson
    from nuscenes.eval.common.data_classes import EvalBoxes
    pred = EvalBoxes()
    with open(B0, 'rb') as f:
        for tok, boxes in ijson.kvitems(f, 'results', use_float=True):
            pred.add_boxes(tok, [TrackingBox.deserialize(b) for b in boxes])
    thr = best_thresholds()
    gt = load_gt(nusc, 'val', TrackingBox, verbose=False)
    pred, gt = add_center_dist(nusc, pred), add_center_dist(nusc, gt)
    pred = filter_eval_boxes(nusc, pred, cfg.class_range, verbose=False)
    gt = filter_eval_boxes(nusc, gt, cfg.class_range, verbose=False)
    # create_tracks gom theo scene + timestamp và nội suy tuyến tính khoảng trống, giống bộ chấm
    raw_scores = collections.defaultdict(list)
    for tok in pred.sample_tokens:
        sc = nusc.get('sample', tok)['scene_token']
        for b in pred.boxes[tok]:
            raw_scores[(sc, b.tracking_id)].append(b.tracking_score)
    gt_tracks = create_tracks(gt, nusc, 'val', gt=True)
    pred_tracks = create_tracks(pred, nusc, 'val', gt=False)

    stats = {c: collections.Counter() for c in CLASSES}
    gap_len = {c: collections.Counter() for c in CLASSES}
    multi_id_examples = {c: [] for c in CLASSES}
    for scene in gt_tracks:
        ts_all = sorted(gt_tracks[scene].keys())
        # theo từng class: chuỗi ghép của mỗi GT instance theo thời gian
        seq = {c: collections.defaultdict(list) for c in CLASSES}  # inst -> [(frame_idx, pred_id or None)]
        last_match = {c: {} for c in CLASSES}  # inst -> pred_id ghép gần nhất
        cand_seq = {c: collections.defaultdict(list) for c in CLASSES}  # ghép KHÔNG ngưỡng
        for fi, ts in enumerate(ts_all):
            g_all = gt_tracks[scene][ts]
            p_all = pred_tracks[scene].get(ts, [])
            for c in CLASSES:
                g = [b for b in g_all if b.tracking_name == c]
                p = [b for b in p_all if b.tracking_name == c and b.tracking_score >= thr[c]]
                matched = {}
                if g and p:
                    D = np.array([[np.linalg.norm(np.array(a.translation[:2]) - np.array(b.translation[:2])) for b in p] for a in g])
                    # giống CLEAR-MOT/motmetrics: giữ cặp ghép của frame trước nếu vẫn trong 2 m
                    pid_idx = {b.tracking_id: j for j, b in enumerate(p)}
                    used_g, used_p = set(), set()
                    for i, b in enumerate(g):
                        prev = last_match[c].get(b.tracking_id)
                        j = pid_idx.get(prev)
                        if j is not None and j not in used_p and D[i, j] <= 2.0:
                            matched[i] = prev; used_g.add(i); used_p.add(j)
                    rg = [i for i in range(len(g)) if i not in used_g]
                    rp = [j for j in range(len(p)) if j not in used_p]
                    if rg and rp:
                        C = np.where(D[np.ix_(rg, rp)] <= 2.0, D[np.ix_(rg, rp)], 1e6)
                        r, k = linear_sum_assignment(C)
                        for i, j in zip(r, k):
                            if C[i, j] < 1e6:
                                matched[rg[i]] = p[rp[j]].tracking_id
                for i, pid in matched.items():
                    last_match[c][g[i].tracking_id] = pid
                for i, b in enumerate(g):
                    seq[c][b.tracking_id].append((fi, matched.get(i)))
                # ghép không ngưỡng (đơn giản: Hungarian từng frame) để biết track có tồn tại không
                pa = [b for b in p_all if b.tracking_name == c]
                if g and pa:
                    D2 = np.array([[np.linalg.norm(np.array(a.translation[:2]) - np.array(b.translation[:2])) for b in pa] for a in g])
                    C2 = np.where(D2 <= 2.0, D2, 1e6)
                    r2, k2 = linear_sum_assignment(C2)
                    m2 = {i: pa[j].tracking_id for i, j in zip(r2, k2) if C2[i, j] < 1e6}
                else:
                    m2 = {}
                for i, b in enumerate(g):
                    cand_seq[c][b.tracking_id].append((fi, m2.get(i)))
        for c in CLASSES:
            for inst, s in seq[c].items():
                st = stats[c]
                st['gt_instances'] += 1
                st['gt_boxes'] += len(s)
                hit = [k for k, (_, pid) in enumerate(s) if pid is not None]
                if not hit:
                    st['inst_never_tracked'] += 1
                    st['fn_never_tracked'] += len(s)
                    cand = cand_seq[c].get(inst, [])
                    cids = [pid for _, pid in cand if pid is not None]
                    if cids:
                        tid = collections.Counter(cids).most_common(1)[0][0]
                        raw = np.array(raw_scores[(scene, tid)])
                        det = raw[raw >= 0.01]  # box có detection; box dự đoán khi bị che có score x0.01
                        st['never_but_track_exists'] += 1
                        st['never_fn_recoverable'] += sum(1 for _, pid in cand if pid is not None)
                        if len(det) and det.mean() >= thr[c]:
                            st['never_det_mean_above_thr'] += 1
                        if len(raw) and (raw < 0.01).mean() > 0.5:
                            st['never_track_mostly_coasting'] += 1
                    continue
                ids = [pid for _, pid in s if pid is not None]
                n_ids = len(set(ids))
                if n_ids >= 2:
                    st['inst_multi_id'] += 1
                    st['id_changes'] += sum(1 for a, b in zip(ids, ids[1:]) if a != b)
                    if len(multi_id_examples[c]) < 20:
                        multi_id_examples[c].append({'scene': scene, 'instance': inst, 'ids_in_order': [i for k, i in enumerate(ids) if k == 0 or ids[k - 1] != i]})
                st['fn_head'] += hit[0]                 # bỏ sót trước lần ghép đầu tiên
                st['fn_tail'] += len(s) - 1 - hit[-1]   # bỏ sót sau lần ghép cuối
                # các khoảng trống ở giữa
                for a, b in zip(hit, hit[1:]):
                    if b - a > 1:
                        L = b - a - 1
                        st['mid_gaps'] += 1
                        st['fn_mid'] += L
                        gap_len[c][L if L < 10 else 10] += 1
                        if s[a][1] != s[b][1]:
                            st['mid_gaps_id_changed'] += 1
    # tổng hợp
    total = collections.Counter()
    for c in CLASSES:
        total.update(stats[c])
    out = {'thresholds': thr, 'per_class': {c: dict(stats[c]) for c in CLASSES},
           'total': dict(total), 'gap_length_hist_frames(10=10+)': {c: dict(sorted(gap_len[c].items())) for c in CLASSES},
           'multi_id_examples': multi_id_examples}
    json.dump(out, open(os.path.join(OUT, 'b0_analysis.json'), 'w'), indent=1)
    keys = ['gt_instances', 'gt_boxes', 'inst_never_tracked', 'never_but_track_exists', 'never_fn_recoverable', 'never_det_mean_above_thr', 'never_track_mostly_coasting', 'inst_multi_id', 'id_changes', 'fn_head', 'fn_mid', 'fn_tail', 'fn_never_tracked', 'mid_gaps', 'mid_gaps_id_changed']
    lines = ['| class | ' + ' | '.join(keys) + ' |', '|' + '---|' * (len(keys) + 1)]
    for c in CLASSES + ['TOTAL']:
        st = total if c == 'TOTAL' else stats[c]
        lines.append(f'| {c} | ' + ' | '.join(str(st.get(k, 0)) for k in keys) + ' |')
    open(os.path.join(OUT, 'b0_analysis.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'filter': step_filter, 'analyze': step_analyze}[sys.argv[1]]()

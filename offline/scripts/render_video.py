"""Xuất video kết quả tracking: 6 camera + BEV LiDAR, box tô màu theo ID, kèm bảng metric tracking.
Dữ liệu ảnh/LiDAR: data/nuscenes_mini (2 scene thuộc tập val: scene-0103, scene-0916).
Chạy (từ thư mục immortal_repro):
  python offline/scripts/render_video.py frames <scene> [results.json] [--all] [--resume]   # vẽ khung hình (--resume: bỏ qua khung đã có)
  python offline/scripts/render_video.py video  <scene> [results.json]           # ghép MP4 bằng ffmpeg
  python offline/scripts/render_video.py table  [metrics_summary.json]           # ảnh bảng metric theo class
Mặc định chỉ vẽ track mà bộ chấm giữ lại (score trung bình >= ngưỡng tốt nhất của class); --all vẽ mọi track.
Đầu ra: offline/results/video/"""
import sys, os, json, time, subprocess, collections
import numpy as np, cv2, ijson
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.eval.tracking.utils import category_to_tracking_name

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MINI = os.path.join(ROOT, 'data', 'nuscenes_mini')
OUT = os.path.join(ROOT, 'offline', 'results', 'video')
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
CW, CH, BEV, RNG = 480, 270, 540, 50.0
DETAILS = os.path.join(ROOT, 'ImmortalTracker', 'mot_results', 'nuscenes', 'validation_2hz', 'immortal_noflip', 'results', 'metrics_details.json')
CLASSES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']


def color(tid):
    h = (hash(tid) * 2654435761) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


def thresholds():
    d = json.load(open(DETAILS))
    return {c: float(d[c]['confidence'][int(np.nanargmax(np.array(d[c]['mota'], dtype=float)))]) for c in CLASSES}


def load(scene_name, res_path, keep_all):
    nusc = NuScenes(version='v1.0-mini', dataroot=MINI, verbose=False)
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    toks, t = [], scene['first_sample_token']
    while t:
        toks.append(t); t = nusc.get('sample', t)['next']
    want, pred = set(toks), {}
    with open(res_path, 'rb') as f:
        for tok, boxes in ijson.kvitems(f, 'results', use_float=True):
            if tok in want:
                pred[tok] = boxes
                if len(pred) == len(want): break
    sc = collections.defaultdict(list)
    for tok in toks:
        for b in pred.get(tok, []): sc[b['tracking_id']].append(b['tracking_score'])
    mean = {k: np.mean(v) for k, v in sc.items()}
    thr = thresholds()
    if not keep_all:
        pred = {k: [b for b in v if mean[b['tracking_id']] >= thr[b['tracking_name']]] for k, v in pred.items()}
    return nusc, scene, toks, pred


def to_sensor(nusc, box, sd_token):
    sd = nusc.get('sample_data', sd_token)
    ep = nusc.get('ego_pose', sd['ego_pose_token']); cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    box = box.copy()
    box.translate(-np.array(ep['translation'])); box.rotate(Quaternion(ep['rotation']).inverse)
    box.translate(-np.array(cs['translation'])); box.rotate(Quaternion(cs['rotation']).inverse)
    return box, cs


def draw_cam(nusc, sample, cam, boxes, gt):
    sd_tok = sample['data'][cam]; sd = nusc.get('sample_data', sd_tok)
    img = cv2.imread(os.path.join(MINI, sd['filename']))
    H, W = img.shape[:2]
    for b, col, txt, dashed in gt + boxes:
        bb, cs = to_sensor(nusc, b, sd_tok)
        K = np.array(cs['camera_intrinsic'])
        if not box_in_image(bb, K, (W, H), vis_level=BoxVisibility.ANY): continue
        if bb.center[2] < 1: continue
        c = view_points(bb.corners(), K, normalize=True)[:2].T.astype(int)
        lw = 2 if dashed else 4
        for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            cv2.line(img, tuple(c[i]), tuple(c[j]), col, lw, cv2.LINE_AA)
        if txt:
            p = tuple(c[:4].min(0)); cv2.putText(img, txt, (p[0], max(p[1] - 8, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
    img = cv2.resize(img, (CW, CH), interpolation=cv2.INTER_AREA)
    cv2.putText(img, cam.replace('CAM_', ''), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_bev(nusc, sample, boxes, gt):
    sd_tok = sample['data']['LIDAR_TOP']; sd = nusc.get('sample_data', sd_tok)
    pc = LidarPointCloud.from_file(os.path.join(MINI, sd['filename'])).points[:3]
    img = np.zeros((BEV, BEV, 3), np.uint8); s = BEV / (2 * RNG)
    P = lambda x, y: (int(BEV / 2 - y * s), int(BEV / 2 - x * s))  # lidar x hướng lên trên ảnh
    m = (np.abs(pc[0]) < RNG) & (np.abs(pc[1]) < RNG)
    u = (BEV / 2 - pc[1, m] * s).astype(int); v = (BEV / 2 - pc[0, m] * s).astype(int)
    ok = (u >= 0) & (u < BEV) & (v >= 0) & (v < BEV); img[v[ok], u[ok]] = (200, 200, 200)
    for r in range(10, int(RNG) + 1, 10): cv2.circle(img, (BEV // 2, BEV // 2), int(r * s), (60, 60, 60), 1)
    for b, col, txt, dashed in gt + boxes:
        bb, _ = to_sensor(nusc, b, sd_tok)
        c = bb.bottom_corners()[:2].T
        pts = np.array([P(x, y) for x, y in c], np.int32)
        cv2.polylines(img, [pts], True, col, 1 if dashed else 2, cv2.LINE_AA)
        if not dashed:
            f = P(*bb.center[:2]); h = P(*((c[0] + c[1]) / 2)); cv2.line(img, f, h, col, 1, cv2.LINE_AA)
    cv2.rectangle(img, P(2.3, 1.0), P(-2.3, -1.0), (0, 0, 255), -1)
    cv2.putText(img, 'LIDAR BEV (50 m)', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def frames(scene_name, res_path, keep_all):
    nusc, scene, toks, pred = load(scene_name, res_path, keep_all)
    tag = os.path.basename(os.path.dirname(os.path.abspath(res_path))) + ('_all' if keep_all else '')
    d = os.path.join(OUT, f'{tag}_{scene_name}_frames'); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    for fi, tok in enumerate(toks):
        p = os.path.join(d, f'{fi:03d}.jpg')
        if os.path.exists(p) and '--resume' in sys.argv: continue
        if time.time() - t0 > 150: print('tam dung, chay lai voi --resume de tiep tuc'); return
        sample = nusc.get('sample', tok)
        boxes = []
        for b in pred.get(tok, []):
            bx = Box(b['translation'], b['size'], Quaternion(b['rotation']))
            pred_only = b['tracking_score'] < 0.01
            boxes.append((bx, color(b['tracking_id']), b['tracking_id'].split('_')[-1], pred_only))
        gt = []
        for a in sample['anns']:
            ann = nusc.get('sample_annotation', a)
            if category_to_tracking_name(ann['category_name']) is None: continue
            gt.append((Box(ann['translation'], ann['size'], Quaternion(ann['rotation'])), (150, 150, 150), None, True))
        cams = [draw_cam(nusc, sample, c, boxes, gt) for c in CAMS]
        grid = np.vstack([np.hstack(cams[:3]), np.hstack(cams[3:])])
        frame = np.hstack([grid, draw_bev(nusc, sample, boxes, gt)])
        bar = np.full((40, frame.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, f'ImmortalTracker ({tag})  |  {scene_name}  |  frame {fi + 1}/{len(toks)}  |  mau = ID track, xam = GT, net manh = box du doan (khong co detection)',
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(p, np.vstack([bar, frame]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('xong', len(toks), 'frame ->', d)


def video(scene_name, res_path, keep_all):
    tag = os.path.basename(os.path.dirname(os.path.abspath(res_path))) + ('_all' if keep_all else '')
    d = os.path.join(OUT, f'{tag}_{scene_name}_frames'); mp4 = os.path.join(OUT, f'{tag}_{scene_name}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '2', '-i', os.path.join(d, '%03d.jpg'),
                    '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', mp4], check=True)
    print(mp4)


def table(path):
    m = json.load(open(path)); L = m['label_metrics']
    cols = [('AMOTA', 'amota', '{:.3f}'), ('AMOTP', 'amotp', '{:.3f}'), ('MOTA', 'mota', '{:.3f}'), ('Recall', 'recall', '{:.3f}'),
            ('IDS', 'ids', '{:.0f}'), ('FRAG', 'frag', '{:.0f}'), ('FP', 'fp', '{:.0f}'), ('FN', 'fn', '{:.0f}')]
    rows = [['Class'] + [c[0] for c in cols]] + [[c] + [f.format(L[k][c]) for _, k, f in cols] for c in CLASSES]
    rows.append(['TONG'] + [f.format(m[k]) for _, k, f in cols])
    W, H = 110, 34; img = np.full((H * len(rows) + 60, 130 + W * len(cols), 3), 255, np.uint8)
    cv2.putText(img, 'Metric tracking - ImmortalTracker goc (B0), nuScenes val 150 scene', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    for r, row in enumerate(rows):
        y = 60 + r * H
        if r == 0 or r == len(rows) - 1: cv2.rectangle(img, (0, y), (img.shape[1], y + H), (235, 235, 235), -1)
        for c, v in enumerate(row):
            x = 10 if c == 0 else 130 + (c - 1) * W
            cv2.putText(img, v, (x, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2 if r in (0, len(rows) - 1) else 1, cv2.LINE_AA)
    p = os.path.join(OUT, 'metrics_table_B0.png'); cv2.imwrite(p, img)
    md = '| ' + ' | '.join(rows[0]) + ' |\n|' + '---|' * len(rows[0]) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows[1:])
    open(os.path.join(OUT, 'metrics_table_B0.md'), 'w').write(md + '\n')
    print(p); print(md)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    a = [x for x in sys.argv[1:] if not x.startswith('--')]; keep_all = '--all' in sys.argv
    if a[0] == 'table':
        table(a[1] if len(a) > 1 else os.path.join(ROOT, 'baseline_B0', 'metrics_summary.json'))
    else:
        res = a[2] if len(a) > 2 else os.path.join(ROOT, 'baseline_B0', 'results.json')
        {'frames': frames, 'video': video}[a[0]](a[1], res, keep_all)

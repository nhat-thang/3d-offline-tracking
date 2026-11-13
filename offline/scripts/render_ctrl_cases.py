"""Video 6 camera + BEV LiDAR cho các trường hợp tracking kiểu CTRL (không refine), 2 scene val trong nuscenes_mini.
Màu: box có detection = màu theo ID (nét đậm); box đoán của ImmortalTracker khi bị che = màu theo ID (nét mảnh);
box kéo dài: xanh lá = đúng vật của track, cam = trùng vật khác, tím = vật không có điểm LiDAR, đỏ = chỗ không có vật.
GT: xám nhạt; GT không có điểm LiDAR: xám đậm.
Cần chạy viz_ctrl_cases.py trước (tạo offline/results/viz/ctrl_cases_boxes.json).
Chạy (từ thư mục immortal_repro):
  python offline/scripts/render_ctrl_cases.py frames <scene> [--resume]
  python offline/scripts/render_ctrl_cases.py video <scene>
Đầu ra: offline/results/video/ctrl_cases_<scene>.mp4"""
import sys, os, json, time, subprocess
import numpy as np, cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.tracking.utils import category_to_tracking_name
sys.path.insert(0, os.path.dirname(__file__))
import render_video as rv

KIND = {'same': (91, 158, 46), 'other': (30, 138, 224), 'zeropt': (199, 92, 138), 'empty': (69, 69, 214)}
def tcolor(tid):
    # màu track trong dải xanh lơ - xanh dương (khác hẳn 4 màu của box kéo dài)
    h = 85 + (tid * 7) % 35
    c = cv2.cvtColor(np.uint8([[[h, 200, 255 - (tid * 37) % 90]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


BOXES = os.path.join(rv.ROOT, 'offline', 'results', 'viz', 'ctrl_cases_boxes.json')


def legend(w):
    bar = np.full((70, w, 3), 30, np.uint8)
    items = [((180, 180, 180), 'GT'), ((90, 90, 90), 'GT khong co diem LiDAR'), ((230, 160, 60), 'box co detection (xanh, theo ID)'),
             ((230, 160, 60), 'net manh: ImmortalTracker doan khi bi che'), (KIND['same'], 'keo dai: dung vat'),
             (KIND['other'], 'keo dai: trung vat khac'), (KIND['zeropt'], 'keo dai: vat khong co diem'), (KIND['empty'], 'keo dai: cho trong')]
    x = 10
    for i, (c, t) in enumerate(items):
        y = 48 if i >= 4 else 22
        if i == 4: x = 10
        cv2.rectangle(bar, (x, y - 12), (x + 22, y), c, -1 if i != 3 else 1)
        cv2.putText(bar, t, (x + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x += 40 + 9 * len(t)
    return bar


def frames(scene_name):
    nusc = NuScenes(version='v1.0-mini', dataroot=rv.MINI, verbose=False)
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    toks, t = [], scene['first_sample_token']
    while t: toks.append(t); t = nusc.get('sample', t)['next']
    raw = json.load(open(BOXES))
    d = os.path.join(rv.OUT, f'ctrl_cases_{scene_name}_frames'); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    for fi, tok in enumerate(toks):
        p = os.path.join(d, f'{fi:03d}.jpg')
        if os.path.exists(p) and '--resume' in sys.argv: continue
        if time.time() - t0 > 150: print('tam dung, chay lai voi --resume de tiep tuc'); return
        sample = nusc.get('sample', tok)
        boxes = []
        for tr, sz, rot, kind, tid in raw.get(tok, []):
            col = KIND.get(kind) or tcolor(tid)
            boxes.append((Box(tr, sz, Quaternion(rot)), col, str(tid), kind == 'pred'))
        gt = []
        for a in sample['anns']:
            ann = nusc.get('sample_annotation', a)
            if category_to_tracking_name(ann['category_name']) is None: continue
            vis = ann['num_lidar_pts'] + ann['num_radar_pts'] > 0
            gt.append((Box(ann['translation'], ann['size'], Quaternion(ann['rotation'])), (150, 150, 150) if vis else (80, 80, 80), None, True))
        cams = [rv.draw_cam(nusc, sample, c, boxes, gt) for c in rv.CAMS]
        grid = np.vstack([np.hstack(cams[:3]), np.hstack(cams[3:])])
        frame = np.hstack([grid, rv.draw_bev(nusc, sample, boxes, gt)])
        top = np.full((36, frame.shape[1], 3), 30, np.uint8)
        cv2.putText(top, f'Tracking kieu CTRL (ImmortalTracker + keo dai 2 chieu, KHONG refine)  |  {scene_name}  |  frame {fi + 1}/{len(toks)}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(p, np.vstack([top, frame, legend(frame.shape[1])]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('xong', len(toks), 'frame ->', d)


def video(scene_name):
    d = os.path.join(rv.OUT, f'ctrl_cases_{scene_name}_frames'); mp4 = os.path.join(rv.OUT, f'ctrl_cases_{scene_name}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '2', '-i', os.path.join(d, '%03d.jpg'),
                    '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', mp4], check=True)
    print(mp4)


if __name__ == '__main__':
    os.makedirs(rv.OUT, exist_ok=True)
    {'frames': frames, 'video': video}[sys.argv[1]](sys.argv[2])

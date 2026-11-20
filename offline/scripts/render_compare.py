"""Video so sánh kết quả tracking trên cùng một scene: 6 camera + BEV LiDAR, mỗi cấu hình một video.
Chỉ có 2 loại box cho dễ nhìn: box của tracker (màu theo ID track, cùng ID thì cùng màu) và GT (xám mảnh).
Chỉ vẽ track mà bộ chấm giữ lại: score trung bình của track >= ngưỡng cho MOTA tốt nhất của lớp đó
(lấy từ metrics_details.json của chính cấu hình), nên video khớp với con số trong bảng kết quả.
Ảnh và LiDAR lấy từ data/nuscenes_mini (scene-0103 thuộc nửa test, scene-0916 thuộc nửa dev).
Chạy (từ thư mục gốc):
  python offline/scripts/render_compare.py frames scene-0103 <results.json> <metrics_details.json> <tag> "<tiêu đề>" [--resume]
  python offline/scripts/render_compare.py video  scene-0103 <tag>
Đầu ra: offline/results/video/compare_<tag>_<scene>.mp4"""
import sys, os, json, time, zlib, subprocess
import numpy as np, cv2
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.eval.tracking.utils import category_to_tracking_name
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
import render_video as rv

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def color(tid):   # màu ổn định theo ID (hash() của Python đổi mỗi lần chạy)
    h = zlib.crc32(str(tid).encode()) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


def text_bar(w, title, sub):
    img = Image.new('RGB', (w, 64), (30, 30, 30)); d = ImageDraw.Draw(img)
    try: f1, f2 = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 16)
    except OSError: f1 = f2 = ImageFont.load_default()
    d.text((12, 6), title, font=f1, fill=(255, 255, 255)); d.text((12, 38), sub, font=f2, fill=(190, 190, 190))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def frames(scene, res, details, tag, title):
    det = json.load(open(details))
    rv.thresholds = lambda: {c: float(det[c]['confidence'][int(np.nanargmax(np.array(det[c]['mota'], dtype=float)))])
                             if c in det and not np.all(np.isnan(np.array(det[c]['mota'], dtype=float))) else 1.1 for c in rv.CLASSES}
    nusc, _, toks, pred = rv.load(scene, res, False)
    d = os.path.join(rv.OUT, f'compare_{tag}_{scene}_frames'); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    for fi, tok in enumerate(toks):
        p = os.path.join(d, f'{fi:03d}.jpg')
        if os.path.exists(p) and '--resume' in sys.argv: continue
        if time.time() - t0 > 150: print('tam dung, chay lai voi --resume'); return
        sample = nusc.get('sample', tok)
        boxes = [(Box(b['translation'], b['size'], Quaternion(b['rotation'])), color(b['tracking_id']),
                  str(b['tracking_id']).split('_')[-1][-3:], False) for b in pred.get(tok, [])]
        gt = []
        for a in sample['anns']:
            ann = nusc.get('sample_annotation', a)
            if category_to_tracking_name(ann['category_name']) is None: continue
            gt.append((Box(ann['translation'], ann['size'], Quaternion(ann['rotation'])), (150, 150, 150), None, True))
        cams = [rv.draw_cam(nusc, sample, c, boxes, gt) for c in rv.CAMS]
        frame = np.hstack([np.vstack([np.hstack(cams[:3]), np.hstack(cams[3:])]), rv.draw_bev(nusc, sample, boxes, gt)])
        bar = text_bar(frame.shape[1], title, f'{scene}  ·  frame {fi + 1}/{len(toks)}  ·  màu = ID track (cùng màu = cùng một vật)  ·  xám mảnh = GT')
        cv2.imwrite(p, np.vstack([bar, frame]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('xong', len(toks), 'frame ->', d)


def video(scene, tag):
    d = os.path.join(rv.OUT, f'compare_{tag}_{scene}_frames'); mp4 = os.path.join(rv.OUT, f'compare_{tag}_{scene}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '2', '-i', os.path.join(d, '%03d.jpg'), '-c:v', 'libx264',
                    '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', mp4], check=True)
    print(mp4, os.path.getsize(mp4) // 1024, 'KB')


if __name__ == '__main__':
    if sys.argv[1] == 'frames': frames(*sys.argv[2:7])
    else: video(sys.argv[2], sys.argv[3])

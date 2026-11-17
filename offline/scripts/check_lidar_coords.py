"""Bước 3 của Task 1: kiểm tra phép đổi tọa độ LiDAR trước khi dùng point cloud cho refine.

Ý tưởng: nuScenes ghi sẵn trong mỗi annotation số điểm LiDAR nằm trong box GT (`num_lidar_pts`).
Nếu tự đọc point cloud, đổi tọa độ cảm biến -> xe -> toàn cục, rồi đếm số điểm trong box GT mà ra
đúng con số đó, thì chuỗi đổi tọa độ là đúng.

Chuỗi đổi tọa độ:
  1. Point cloud trong file .pcd.bin ở hệ tọa độ của cảm biến LIDAR_TOP.
  2. Cảm biến -> xe:       xoay theo calibrated_sensor.rotation, rồi cộng calibrated_sensor.translation.
  3. Xe -> toàn cục:       xoay theo ego_pose.rotation, rồi cộng ego_pose.translation.
  4. Box GT (sample_annotation) đã ở hệ toàn cục, nên so trực tiếp được.

Chạy (từ thư mục gốc):  python offline/scripts/check_lidar_coords.py [số sample tối đa, mặc định 300]
Chỉ dùng các sample đã có file LiDAR trong data/nuscenes_valslim/samples/LIDAR_TOP."""
import sys, os, time
import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import points_in_box

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATAROOT = os.path.join(ROOT, 'data', 'nuscenes_valslim')
max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 300

nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=False)


def lidar_points_global(sd):
    """Đọc point cloud của một sample_data LIDAR_TOP và trả về toạ độ toàn cục (3 x N)."""
    pc = LidarPointCloud.from_file(os.path.join(DATAROOT, sd['filename']))
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])   # cảm biến -> xe
    ep = nusc.get('ego_pose', sd['ego_pose_token'])                     # xe -> toàn cục
    pc.rotate(Quaternion(cs['rotation']).rotation_matrix); pc.translate(np.array(cs['translation']))
    pc.rotate(Quaternion(ep['rotation']).rotation_matrix); pc.translate(np.array(ep['translation']))
    return pc.points[:3]


rows, t0, n_samples = [], time.time(), 0
for sample in nusc.sample:
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    if not os.path.exists(os.path.join(DATAROOT, sd['filename'])): continue
    try:
        pts = lidar_points_global(sd)
    except Exception as e:                       # file đang được ghi dở trong lúc tải
        print('bo qua file loi:', sd['filename'], e); continue
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        # lọc thô các điểm gần box cho nhanh, rồi đếm chính xác
        near = np.abs(pts[:2] - np.array(ann['translation'][:2])[:, None]).max(0) < max(ann['size']) + 1
        n_mine = int(points_in_box(box, pts[:, near]).sum())
        rows.append((ann['num_lidar_pts'], n_mine, ann['category_name']))
    n_samples += 1
    if n_samples >= max_samples or time.time() - t0 > 150: break

a = np.array([(r[0], r[1]) for r in rows])
gt, mine = a[:, 0], a[:, 1]
diff = mine - gt
print(f'sample: {n_samples}, box GT: {len(rows)}')
print(f'khop chinh xac: {np.mean(diff == 0) * 100:.1f}%  |  lech <= 2 diem: {np.mean(np.abs(diff) <= 2) * 100:.1f}%')
print(f'tong diem nuScenes ghi: {gt.sum()}  |  tong diem tu dem: {mine.sum()}  |  ti le: {mine.sum() / max(gt.sum(), 1):.4f}')
big = np.argsort(-np.abs(diff))[:5]
print('5 box lech nhieu nhat (nuScenes, tu dem, class):')
for i in big: print('  ', rows[i])

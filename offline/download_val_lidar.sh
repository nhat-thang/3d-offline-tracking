#!/bin/bash
# Tải point cloud LiDAR keyframe (LIDAR_TOP) của 150 scene val nuScenes, chỉ giữ 6.019 file cần dùng (~4 GB).
# Nguồn: bản công khai của nuScenes trên AWS Open Data (s3://motional-nuscenes, không cần tài khoản AWS).
# Mỗi gói v1.0-trainvalXX_keyframes.tgz (~4-6 GB, tổng ~45 GB) được tải dạng luồng và giải nén chỉ các file trong
# offline/val_lidar_files.txt, nên không cần ổ trống 45 GB. Dữ liệu nuScenes: chỉ dùng nội bộ, không chia sẻ (license).
# Chạy (trong Docker, từ /repro):  bash offline/download_val_lidar.sh
# Chạy lại được: gói nào đã xong (có file đánh dấu) sẽ bỏ qua.
cd "$(dirname "$0")/.."
OUT=data/nuscenes_valslim; LIST=offline/val_lidar_files.txt; LOG=offline/results/download_lidar.log
URL=https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0
mkdir -p $OUT/.done
for i in 01 02 03 04 05 06 07 08 09 10; do
  [ -f $OUT/.done/$i ] && { echo "goi $i: da xong, bo qua"; continue; }
  echo "goi $i: dang tai va giai nen ($(date +%H:%M))"
  # tar báo "Not found in archive" cho file thuộc gói khác: bình thường, ghi vào log
  if curl -sfL --retry 5 "$URL/v1.0-trainval${i}_keyframes.tgz" | tar -xz -C $OUT -T $LIST 2>>$LOG; [ ${PIPESTATUS[0]} -eq 0 ]; then
    touch $OUT/.done/$i
  else
    echo "goi $i: LOI tai (xem $LOG), chay lai script de thu lai"
  fi
  echo "  da co $(ls $OUT/samples/LIDAR_TOP 2>/dev/null | wc -l) / 6019 file"
done

#!/bin/bash
# Tai anh camera (camera_image) va thong so camera (camera_calibration, gom intrinsic/extrinsic) cua Waymo v2.0.1
# cho cac chuoi co trong video so sanh (offline/results/waymo/subset8/video/plan.json -> waymo/cases_video.txt).
# Chay trong google/cloud-sdk (tools/waymo_15_tai_camera.bat).
cd "$(dirname "$0")/.."
B=gs://waymo_open_dataset_v_2_0_1/validation
echo "== Dung luong camera_image cua cac chuoi can tai =="
for s in $(cat waymo/cases_video.txt); do gsutil du -h $B/camera_image/$s.parquet; done
for c in camera_calibration camera_image; do
  mkdir -p data/waymo/v2/validation/$c
  for s in $(cat waymo/cases_video.txt); do
    [ -f data/waymo/v2/validation/$c/$s.parquet ] && continue
    gsutil -q cp $B/$c/$s.parquet data/waymo/v2/validation/$c/ || echo "LOI: $c/$s"
  done
  echo "$c: $(ls data/waymo/v2/validation/$c | wc -l) file"
done
echo "== Da tai =="; du -sh data/waymo/v2/validation/camera_*

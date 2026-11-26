#!/bin/bash
# Buoc B.1: do dung luong cac thanh phan LiDAR cua tap validation (Waymo v2.0.1), roi tai cho cac chuoi trong waymo/subset8.txt.
# Chay trong google/cloud-sdk (tools/waymo_5_tai_lidar.bat).
cd "$(dirname "$0")/.."
B=gs://waymo_open_dataset_v_2_0_1/validation
for c in lidar lidar_calibration lidar_pose vehicle_pose; do
  mkdir -p data/waymo/v2/validation/$c
  for s in $(cat waymo/subset8.txt); do
    [ -f data/waymo/v2/validation/$c/$s.parquet ] && continue
    gsutil -q cp $B/$c/$s.parquet data/waymo/v2/validation/$c/ || echo "LOI: $c/$s"
  done
  echo "$c: $(ls data/waymo/v2/validation/$c | wc -l) file"
done
echo "== Da tai =="; du -sh data/waymo/v2/validation/*

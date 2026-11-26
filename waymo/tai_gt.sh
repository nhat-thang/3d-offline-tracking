#!/bin/bash
# Tai nhan box 3D (thanh phan lidar_box, Waymo v2.0.1) cua tap validation de cham diem; ghi danh sach thu muc ra log.
# Chay trong google/cloud-sdk sau khi da dang nhap (tools/waymo_2_dang_nhap_gcloud.bat).
cd "$(dirname "$0")/.."
B=gs://waymo_open_dataset_v_2_0_1
mkdir -p data/waymo/v2/validation/lidar_box logs
gsutil ls $B/ > logs/waymo_bucket_v2.txt 2>&1; gsutil ls $B/validation/ >> logs/waymo_bucket_v2.txt 2>&1
cat logs/waymo_bucket_v2.txt
gsutil -m cp -n "$B/validation/lidar_box/*.parquet" data/waymo/v2/validation/lidar_box/ 2>&1 | tail -3
echo "so file lidar_box: $(ls data/waymo/v2/validation/lidar_box | wc -l)"; du -sh data/waymo/v2/validation/lidar_box

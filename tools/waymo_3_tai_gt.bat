@echo off
REM Tai nhan box 3D tap validation cua Waymo (thanh phan lidar_box, v2.0.1). Log: tools\waymo_3.log
cd /d %~dp0..
docker run --rm -v gcloud_cfg:/root/.config/gcloud -v "%cd%":/repro google/cloud-sdk:slim bash /repro/waymo/tai_gt.sh > tools\waymo_3.log 2>&1
type tools\waymo_3.log
pause

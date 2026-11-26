@echo off
REM Buoc B.1: do dung luong LiDAR validation va tai 8 chuoi dau (waymo\subset8.txt). Log: tools\waymo_5.log
cd /d %~dp0..
docker run --rm -v gcloud_cfg:/root/.config/gcloud -v "%cd%":/repro google/cloud-sdk:slim bash /repro/waymo/tai_lidar_subset.sh > tools\waymo_5.log 2>&1
type tools\waymo_5.log
pause

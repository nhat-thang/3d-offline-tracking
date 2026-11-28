@echo off
REM Tai anh camera va thong so camera (intrinsic/extrinsic) cho 4 chuoi trong video so sanh. Log: tools\waymo_15.log
cd /d %~dp0..
docker run --rm -v gcloud_cfg:/root/.config/gcloud -v "%cd%":/repro google/cloud-sdk:slim bash /repro/waymo/tai_camera_cases.sh > tools\waymo_15.log 2>&1
type tools\waymo_15.log
pause

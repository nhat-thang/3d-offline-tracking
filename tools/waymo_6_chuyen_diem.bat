@echo off
REM Buoc B.2: chuyen LiDAR v2 cua cac chuoi trong waymo\subset8.txt sang dinh dang diem cua CTRL. Log: tools\waymo_6.log
cd /d %~dp0..
docker run --rm -v "%cd%":/repro waymo_eval python waymo/v2_to_kitti_points.py waymo/subset8.txt > tools\waymo_6.log 2>&1
findstr /v /c:"tensorflow" tools\waymo_6.log
pause

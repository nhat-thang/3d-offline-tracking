@echo off
REM Buoc B.4: tao ts_info, ego_info va detection FSD cho ImmortalTracker (8 chuoi). Log: tools\waymo_8.log
cd /d %~dp0..
docker run --rm -v "%cd%":/repro waymo_eval python waymo/prep_immortal.py waymo/subset8.txt data/waymo/ctrl_release/fsd_base_vehicle_val.bin fsd > tools\waymo_8.log 2>&1
findstr /v /c:"tensorflow" tools\waymo_8.log
pause

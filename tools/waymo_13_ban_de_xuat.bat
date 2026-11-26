@echo off
REM Buoc C.1: dua ket qua ban de xuat (ours.bin) qua keo dai + cat diem + refine CTRL (GPU), roi cham ca ban de xuat va CTRL o tung giai doan tren 8 chuoi.
REM Log: tools\waymo_13.log
cd /d %~dp0..
set D=%cd%\data\waymo
echo Bat dau: %date% %time% > tools\waymo_13.log
docker run --rm --gpus all --shm-size 4g -v "%D%":/opt/SST/data/waymo -v "%cd%\waymo":/repro_waymo ctrl_env2 bash /repro_waymo/run_ours_refine.sh ours >> tools\waymo_13.log 2>&1
docker run --rm -v "%cd%":/repro -w /repro waymo_eval bash waymo/eval_ours.sh >> tools\waymo_13.log 2>&1
echo Ket thuc: %date% %time% >> tools\waymo_13.log
findstr /b /c:"==" /c:"##" /c:"OBJECT_TYPE_TYPE_VEHICLE" /c:"Bat dau" /c:"Ket thuc" /c:"Traceback" /c:"Error" tools\waymo_13.log
pause

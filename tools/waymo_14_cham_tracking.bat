@echo off
REM Buoc C.2: cham metric tracking kieu Waymo (MOTA, doi ID...) cho CTRL va ban de xuat tren 8 chuoi. Log: tools\waymo_14.log
cd /d %~dp0..
docker run --rm -v "%cd%":/repro -w /repro waymo_eval bash waymo/eval_track.sh > tools\waymo_14.log 2>&1
findstr /c:"##" /c:"VEHICLE_LEVEL" /c:"Error" /c:"error" tools\waymo_14.log
pause

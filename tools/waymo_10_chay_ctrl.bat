@echo off
REM Buoc B.6: chay lai CTRL (tracker, keo dai, cat diem, refine co va khong TTA, xoa box rong) tren 8 chuoi. Dung GPU.
REM Mo tools\xem_gpu.bat song song de theo doi GPU. Log: tools\waymo_10.log
cd /d %~dp0..
set D=%cd%\data\waymo
echo Bat dau: %date% %time% > tools\waymo_10.log
docker run --rm --gpus all --shm-size 4g -v "%D%":/opt/SST/data/waymo -v "%D%":/opt/ImmortalTracker-for-CTRL/data/waymo -v "%D%\ctrl_work\mot_results":/opt/ImmortalTracker-for-CTRL/mot_results -v "%cd%\waymo":/repro_waymo ctrl_env2 bash /repro_waymo/run_ctrl.sh %1 >> tools\waymo_10.log 2>&1
echo Ket thuc: %date% %time% >> tools\waymo_10.log
findstr /b /c:"==" /c:"Bat dau" /c:"Ket thuc" /c:"Traceback" /c:"Error" tools\waymo_10.log
pause

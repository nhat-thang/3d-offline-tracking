@echo off
REM Buoc B.3: tao Docker image cho CTRL (SST) + ImmortalTracker-for-CTRL. Lau (30-60 phut, build CUDA). Log: tools\waymo_7.log
cd /d %~dp0..
echo Bat dau: %date% %time% > tools\waymo_7.log
docker build -t ctrl_env -f waymo/Dockerfile.ctrl waymo >> tools\waymo_7.log 2>&1
echo Ket thuc: %date% %time% >> tools\waymo_7.log
powershell -command "Get-Content tools\waymo_7.log -Tail 15"
pause

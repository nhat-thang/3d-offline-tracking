@echo off
REM Tao Docker image cham diem Waymo (chi can chay 1 lan, vai phut). Can Docker Desktop dang chay.
cd /d %~dp0..
docker build -t waymo_eval -f waymo/Dockerfile.eval waymo > tools\waymo_0.log 2>&1
type tools\waymo_0.log | findstr /i "error successfully naming"
echo Xong. Log: tools\waymo_0.log
pause

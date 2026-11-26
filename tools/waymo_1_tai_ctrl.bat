@echo off
REM Tai ket qua FSD, ket qua CTRL chinh thuc, poses va checkpoint CTRL (khoang 1,1 GB). Chay lai duoc neu bi ngat.
cd /d %~dp0..
docker run --rm -v "%cd%":/repro waymo_eval bash waymo/tai_ctrl.sh > tools\waymo_1.log 2>&1
type tools\waymo_1.log
pause

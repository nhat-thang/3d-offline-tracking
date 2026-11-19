@echo off
cd /d %~dp0..
echo Bat dau tai LiDAR keyframe tap val: %date% %time% > tools\download_lidar.log
echo (Co the dong cua so nay bat cu luc nao; chay lai file nay se tiep tuc tu goi dang do.) >> tools\download_lidar.log
docker run --rm -v "%cd%":/repro -w /repro python:3.10 bash offline/download_val_lidar.sh >> tools\download_lidar.log 2>&1
echo Ket thuc: %date% %time% >> tools\download_lidar.log
type tools\download_lidar.log
pause

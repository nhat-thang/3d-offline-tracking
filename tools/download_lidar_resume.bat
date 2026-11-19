@echo off
cd /d %~dp0..
echo Dung cac lan tai cu con chay ngam (neu co)...
docker rm -f lidar_dl >nul 2>&1
for /f %%c in ('docker ps -q --filter ancestor=python:3.10') do docker rm -f %%c >nul
echo Bat dau (ban tai tiep duoc): %date% %time% > tools\download_lidar.log
echo Dang tai. Muon dung thi chay tools\dung_tai_lidar.bat (dong cua so nay KHONG dung viec tai).
docker run --rm --name lidar_dl -v "%cd%":/repro -w /repro python:3.10 bash offline/download_val_lidar_resume.sh >> tools\download_lidar.log 2>&1
echo Ket thuc: %date% %time% >> tools\download_lidar.log
type tools\download_lidar.log
pause

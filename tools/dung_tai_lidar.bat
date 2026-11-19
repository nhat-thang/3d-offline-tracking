@echo off
echo Dung viec tai LiDAR...
docker rm -f lidar_dl
for /f %%c in ('docker ps -q --filter ancestor=python:3.10') do docker rm -f %%c
echo Da dung. Phan da tai van giu, chay lai download_lidar_resume.bat de tai tiep.
pause

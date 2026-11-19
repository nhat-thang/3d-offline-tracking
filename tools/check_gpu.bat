@echo off
cd /d %~dp0
echo === Docker version === > check_gpu.log
docker version >> check_gpu.log 2>&1
echo. >> check_gpu.log
echo === GPU trong Docker (nvidia-smi) === >> check_gpu.log
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >> check_gpu.log 2>&1
echo. >> check_gpu.log
echo === XONG === >> check_gpu.log
type check_gpu.log
pause

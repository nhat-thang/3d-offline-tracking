@echo off
REM Train model refine tren GPU (Docker PyTorch), roi du doan tren tap test. Log: tools\train_refine.log
REM Dong cua so nay KHONG dung viec train. Muon dung: docker rm -f refine_train
cd /d %~dp0..
set NAME=B0_s0
set SEED=0
set DATA=B0
echo Bat dau train %NAME%: %date% %time% > tools\train_refine.log
docker rm -f refine_train >nul 2>&1
docker run --rm --gpus all --name refine_train --shm-size 2g -e PYTHONUNBUFFERED=1 -v "%cd%":/repro -w /repro pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash -c "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv && python offline/refine/train.py offline/results/refine/data/%DATA%_dev.npz --name %NAME% --seed %SEED% --epochs 20 --bs 32 && python offline/refine/predict.py offline/results/refine/ckpt/%NAME%.pt offline/results/refine/data/%DATA%_test.npz" >> tools\train_refine.log 2>&1
echo Ket thuc: %date% %time% >> tools\train_refine.log
type tools\train_refine.log
pause

@echo off
REM Train refine cho 2 cau hinh con lai (Ours, CTRLext), moi cau hinh roi du doan tren tap test. Chay SAU khi train_refine.bat (B0) xong.
REM Log: tools\train_refine2.log. Muon dung: docker rm -f refine_train
cd /d %~dp0..
echo Bat dau: %date% %time% > tools\train_refine2.log
docker rm -f refine_train >nul 2>&1
docker run --rm --gpus all --name refine_train --shm-size 2g -e PYTHONUNBUFFERED=1 -v "%cd%":/repro -w /repro pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash -c "for C in Ours CTRLext; do python offline/refine/train.py offline/results/refine/data/${C}_dev.npz --name ${C}_s0 --seed 0 --epochs 12 --bs 32 && python offline/refine/predict.py offline/results/refine/ckpt/${C}_s0.pt offline/results/refine/data/${C}_test.npz; done" >> tools\train_refine2.log 2>&1
echo Ket thuc: %date% %time% >> tools\train_refine2.log
type tools\train_refine2.log
pause

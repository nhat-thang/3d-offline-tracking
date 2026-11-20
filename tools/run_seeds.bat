@echo off
REM Chay them seed 1 va 2 cho 3 cau hinh co refine (B0, Ours, CTRLext): train + du doan tren GPU, roi cham diem.
REM Chay qua dem (khoang 4 gio). Log: tools\seeds.log. Dong cua so KHONG dung viec chay; muon dung: docker rm -f refine_train refine_eval
cd /d %~dp0..
echo Bat dau: %date% %time% > tools\seeds.log
docker rm -f refine_train refine_eval >nul 2>&1
docker run --rm --gpus all --name refine_train --shm-size 2g -e PYTHONUNBUFFERED=1 -v "%cd%":/repro -w /repro pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash -c "for S in 1 2; do for C in B0 Ours CTRLext; do [ -f offline/results/refine/pred/${C}_s${S}__${C}_test.npz ] && continue; python offline/refine/train.py offline/results/refine/data/${C}_dev.npz --name ${C}_s${S} --seed ${S} --epochs 12 --bs 32 && python offline/refine/predict.py offline/results/refine/ckpt/${C}_s${S}.pt offline/results/refine/data/${C}_test.npz; done; done" >> tools\seeds.log 2>&1
for %%S in (s1 s2) do docker run --rm --name refine_eval -e PYTHONUNBUFFERED=1 -e SEED=%%S -e ONLY_FULL=1 -e ENV_DIR=/opt/imm_env -v imm_env:/opt/imm_env -v "%cd%":/repro -w /repro python:3.10 bash offline/refine/run_eval_all.sh >> tools\seeds.log 2>&1
echo Ket thuc: %date% %time% >> tools\seeds.log
type tools\seeds.log
pause

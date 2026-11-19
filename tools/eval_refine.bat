@echo off
REM Cham diem tat ca cau hinh tren 75 scene test (ca B0, Ours, CTRLext, co va khong co refine). Log: tools\eval_refine.log
cd /d %~dp0..
echo Bat dau cham: %date% %time% > tools\eval_refine.log
docker rm -f refine_eval >nul 2>&1
docker run --rm --name refine_eval -e PYTHONUNBUFFERED=1 -v "%cd%":/repro -w /repro python:3.10 bash offline/refine/run_eval_all.sh >> tools\eval_refine.log 2>&1
echo Ket thuc: %date% %time% >> tools\eval_refine.log
type tools\eval_refine.log
pause

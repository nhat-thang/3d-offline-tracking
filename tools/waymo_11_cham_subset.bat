@echo off
REM Buoc B.7: cham cac ket qua CTRL tren dung 8 chuoi (subset8.txt), so voi file chinh thuc cua tac gia. Log: tools\waymo_11.log
cd /d %~dp0..
docker run --rm -v "%cd%":/repro -w /repro waymo_eval bash waymo/eval_subset.sh > tools\waymo_11.log 2>&1
findstr /c:"##" /c:"AP " /c:"GT:" tools\waymo_11.log
pause

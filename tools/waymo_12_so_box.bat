@echo off
REM Buoc B.8: so tung box giua CTRL tu chay va file chinh thuc cua tac gia tren 8 chuoi (khong can GT). Log: tools\waymo_12.log
cd /d %~dp0..
set K=data/waymo/ctrl_work
set R=data/waymo/ctrl_release
set O=offline/results/waymo/subset8
docker run --rm -v "%cd%":/repro -w /repro waymo_eval bash -c "mkdir -p %O%; echo == CTRL TTA + xoa box rong; python waymo/compare_boxes.py %K%/ctrl_TTA_wo_empty_right.bin %R%/best_val_in_paper.bin --contexts waymo/subset8.txt --out %O%/so_box_TTA.json; echo == CTRL khong TTA + xoa box rong; python waymo/compare_boxes.py %K%/ctrl_noTTA_wo_empty_right.bin %R%/best_val_in_paper.bin --contexts waymo/subset8.txt --out %O%/so_box_noTTA.json" > tools\waymo_12.log 2>&1
type tools\waymo_12.log
pause

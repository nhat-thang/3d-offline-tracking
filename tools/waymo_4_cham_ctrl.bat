@echo off
REM Buoc A: cham ket qua CTRL chinh thuc (best_val_in_paper.bin) va ket qua FSD tren tap validation. Log: tools\waymo_4.log
cd /d %~dp0..
set R=data/waymo/ctrl_release
set G=data/waymo/v2/validation/lidar_box
docker run --rm -v "%cd%":/repro waymo_eval bash -c "python waymo/inspect_bin.py %R%/best_val_in_paper.bin; python waymo/inspect_bin.py %R%/fsd_base_vehicle_val.bin; mkdir -p offline/results/waymo; python waymo/eval_det.py %R%/best_val_in_paper.bin %G% --types 1 --out offline/results/waymo/ctrl_official_val_vehicle.json; python waymo/eval_det.py %R%/fsd_base_vehicle_val.bin %G% --types 1 --out offline/results/waymo/fsd_base_vehicle_val.json" > tools\waymo_4.log 2>&1
findstr /c:"chuoi" /c:"AP " /c:"GT:" tools\waymo_4.log
pause

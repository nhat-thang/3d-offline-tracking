#!/bin/bash
# Cham tat ca ket qua (vehicle, LEVEL_1/2 AP/APH) tren dung cac chuoi trong subset8.txt. Chay trong image waymo_eval, thu muc /repro.
G=data/waymo/v2/validation/lidar_box; C=waymo/subset8.txt; O=offline/results/waymo/subset8; mkdir -p $O
R=data/waymo/ctrl_release; K=data/waymo/ctrl_work; T=$K/mot_results/waymo/validation/immortal_fsd/bin
for x in "fsd_tac_gia:$R/fsd_base_vehicle_val.bin" "ctrl_tac_gia:$R/best_val_in_paper.bin" \
         "tracker:$T/pred_merged.bin" "tracker_keo_dai:$T/pred_merged_extend.bin" \
         "ctrl_noTTA:$K/ctrl_noTTA.bin" "ctrl_noTTA_xoa_rong:$K/ctrl_noTTA_wo_empty_right.bin" \
         "ctrl_TTA:$K/ctrl_TTA.bin" "ctrl_TTA_xoa_rong:$K/ctrl_TTA_wo_empty_right.bin"; do
  n=${x%%:*}; f=${x#*:}
  [ -f $f ] || { echo "## $n: chua co $f"; continue; }
  echo "## $n ($f)"
  python waymo/eval_det.py $f $G --types 1 --contexts $C --out $O/$n.json 2>&1 | grep -v Warning
done

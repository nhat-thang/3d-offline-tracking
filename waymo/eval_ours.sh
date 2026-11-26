#!/bin/bash
# Cham ban de xuat va CTRL o tung giai doan tren 8 chuoi (vehicle). Chay trong image waymo_eval, thu muc /repro.
G=data/waymo/v2/validation/lidar_box; C=waymo/subset8.txt; O=offline/results/waymo/subset8; mkdir -p $O
K=data/waymo/ctrl_work; T=$K/mot_results/waymo/validation/immortal_fsd/bin; U=$K/ours
for x in "ctrl_tracker:$T/pred_merged.bin" "ours:$U/ours.bin" \
         "ctrl_tracker_keo_dai:$T/pred_merged_extend.bin" "ours_keo_dai:$U/ours_extend.bin" \
         "ctrl_refine_noTTA:$K/ctrl_noTTA.bin" "ours_refine_noTTA:$U/ours_noTTA.bin" \
         "ctrl_refine_noTTA_xoa_rong:$K/ctrl_noTTA_wo_empty_right.bin" "ours_refine_noTTA_xoa_rong:$U/ours_noTTA_wo_empty_right.bin"; do
  n=${x%%:*}; f=${x#*:}
  [ -f $f ] || { echo "## $n: chua co $f"; continue; }
  echo "## $n"
  python waymo/eval_det.py $f $G --types 1 --contexts $C --out $O/$n.json 2>&1 | grep "VEHICLE_LEVEL\|du doan"
done

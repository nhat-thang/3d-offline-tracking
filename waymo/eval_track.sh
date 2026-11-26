#!/bin/bash
# Cham metric tracking kieu Waymo (MOTA, mismatch...) cho CTRL va ban de xuat o tung giai doan tren 8 chuoi (vehicle).
# Chay trong image waymo_eval, thu muc /repro.
G=data/waymo/v2/validation/lidar_box; C=waymo/subset8.txt; O=offline/results/waymo/subset8/tracking; mkdir -p $O
R=data/waymo/ctrl_release; K=data/waymo/ctrl_work; T=$K/mot_results/waymo/validation/immortal_fsd/bin; U=$K/ours
for x in "ctrl_tracker:$T/pred_merged.bin" "ours:$U/ours.bin" \
         "ctrl_tracker_keo_dai:$T/pred_merged_extend.bin" "ours_keo_dai:$U/ours_extend.bin" \
         "ctrl_refine_noTTA_xoa_rong:$K/ctrl_noTTA_wo_empty_right.bin" "ours_refine_noTTA_xoa_rong:$U/ours_noTTA_wo_empty_right.bin" \
         "ctrl_refine_TTA_xoa_rong:$K/ctrl_TTA_wo_empty_right.bin" "ctrl_tac_gia:$R/best_val_in_paper.bin"; do
  n=${x%%:*}; f=${x#*:}
  [ -f $f ] || { echo "## $n: chua co $f"; continue; }
  echo "## $n"
  python waymo/eval_track.py $f $G --types 1 --contexts $C --out $O/$n.json 2>&1 | grep "VEHICLE_LEVEL\|du doan\|Error\|error"
done

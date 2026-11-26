#!/bin/bash
# Chay lai CTRL bang checkpoint cua tac gia (khong train) tren cac chuoi da chuyen diem, trong image ctrl_env2.
# D:\Projects\immortal_repro\data\waymo duoc gan vao /opt/SST/data/waymo va /opt/ImmortalTracker-for-CTRL/data/waymo.
# Chay tung buoc: bash run_ctrl.sh <buoc> ; buoc = track | extend | input | infer | infer_tta | empty | all
set -o pipefail
W=/opt/SST/data/waymo; WK=$W/ctrl_work; mkdir -p $WK
TRK=/opt/ImmortalTracker-for-CTRL; CFG_T=configs/waymo_configs/immortal_for_ctrl_keep10.yaml
BIN_T=$TRK/mot_results/waymo/validation/immortal_fsd/bin/pred_merged.bin   # ket qua tracker
BIN_E=${BIN_T%.bin}_extend.bin                                              # sau khi keo dai ve qua khu
[ -f $W/kitti_format/poses.pkl ] || cp $W/ctrl_release/poses.pkl $W/kitti_format/poses.pkl
python /repro_waymo/make_ctx2ts.py $W/kitti_format   # context2timestamp.pkl cho extend_tracks.py
step=${1:-all}   # co the ghep nhieu buoc: infer_tta,empty
has() { [[ $step == all || ",$step," == *",$1,"* ]]; }
log() { echo "== $1 ($(date +%H:%M:%S))"; }

if has track; then
  log "1. ImmortalTracker-for-CTRL tren FSD (vehicle)"; cd $TRK
  # cau hinh dung gpu: True -> moi tien trinh goi cuda.set_device(token); may chi co 1 GPU nen chay 1 tien trinh (8 chuoi ~2-3 phut)
  python main_waymo.py --name immortal --det_name fsd --config_path $CFG_T --process 1 --obj_type vehicle --split validation || exit 1
  python evaluation/waymo/pred_bin.py --name immortal --det_name fsd --config_path $CFG_T --split validation --obj_types vehicle --no-eval || exit 1
  ls -la $BIN_T
fi
if has extend; then
  log "2. Keo dai track ve qua khu (extend_tracks.py, cau hinh goc cua tac gia)"; cd /opt/SST
  sed "s#^bin_path:.*#bin_path: $BIN_T#" tools/ctrl/data_configs/extend.yaml > $WK/extend.yaml
  python tools/ctrl/extend_tracks.py $WK/extend.yaml 2>&1 | grep -v compute_detection_metrics_main
  ls -la $BIN_E || exit 1
fi
if has input; then
  log "3. Cat diem theo track (generate_track_input.py)"; cd /opt/SST
  sed -e "s#^val_bin_path:.*#val_bin_path: $BIN_E#" -e "s#^exist_ok:.*#exist_ok: True#" tools/ctrl/data_configs/fsd_base_vehicle.yaml > $WK/fsd_base_vehicle.yaml
  python tools/ctrl/generate_track_input.py $WK/fsd_base_vehicle.yaml --process 1 || exit 1
  ls -la data/waymo/tracklet_data/
fi
infer() {  # infer <ten> <co_tta 0|1> <batch>
  cd /opt/SST
  python /repro_waymo/ctrl_test.py $2 $3 $W/ctrl_release/ctrl_veh_epoch_24.pth $WK/$1 2>&1 | grep -v "compute_detection_metrics_main\|CalledProcessError"
  ls -la $WK/$1.bin
}
if has infer; then log "4. Refine CTRL, khong TTA"; infer ctrl_noTTA 0 2; fi
if has infer_tta; then log "4b. Refine CTRL, co Track TTA"; infer ctrl_TTA 1 1; fi
if has empty; then
  log "5. Xoa box rong (remove_empty.py)"; cd /opt/SST
  # 1 GPU: nhieu tien trinh se goi cuda.set_device(1..3) va loi am tham; nhanh 1 tien trinh cua tac gia bi dao 2 tham so -> sua tren ban sao
  sed 's/out_list, input_list, args.split, ts2idx, 0, 1)/out_list, input_list, ts2idx, args.split, 0, 1)/' tools/ctrl/remove_empty.py > tools/ctrl/remove_empty_1gpu.py
  for b in $WK/ctrl_noTTA.bin $WK/ctrl_TTA.bin; do
    [ -f $b ] && python tools/ctrl/remove_empty_1gpu.py --bin-path $b --process 1 --split training --type vehicle 2>&1 | grep -v compute_detection_metrics_main
  done
  ls -la $WK/*.bin
fi
log "xong buoc: $step"

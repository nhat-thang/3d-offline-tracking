#!/bin/bash
# Dua ket qua cua phuong phap de xuat (ours.bin, tu ours_waymo.py) qua DUNG cac buoc sau tracker cua CTRL:
# keo dai ve qua khu -> cat diem theo track -> refine bang checkpoint CTRL (khong TTA) -> xoa box rong.
# Chay trong image ctrl_env2 (giong run_ctrl.sh). Dung: bash run_ours_refine.sh [ten=ours]
set -o pipefail
N=${1:-ours}
W=/opt/SST/data/waymo; WK=$W/ctrl_work; OD=$WK/$N; BIN=$OD/$N.bin; BIN_E=$OD/${N}_extend.bin
log() { echo "== $1 ($(date +%H:%M:%S))"; }
[ -f $BIN ] || { echo "thieu $BIN"; exit 1; }
python /repro_waymo/make_ctx2ts.py $W/kitti_format
cd /opt/SST
log "2. Keo dai ve qua khu ($N)"
sed "s#^bin_path:.*#bin_path: $BIN#" tools/ctrl/data_configs/extend.yaml > $OD/extend.yaml
python tools/ctrl/extend_tracks.py $OD/extend.yaml 2>&1 | grep -v "compute_detection_metrics_main\|CalledProcessError\|Traceback\|File \|raise\|ret_bytes\|return run"
ls -la $BIN_E || exit 1
log "3. Cat diem theo track ($N)"
sed -e "s#^val_bin_path:.*#val_bin_path: $BIN_E#" -e "s#^exist_ok:.*#exist_ok: True#" tools/ctrl/data_configs/fsd_base_vehicle.yaml > $OD/${N}_vehicle.yaml
python tools/ctrl/generate_track_input.py $OD/${N}_vehicle.yaml --process 1 || exit 1
log "4. Refine CTRL khong TTA ($N)"
PROPOSALS=./data/waymo/tracklet_data/${N}_vehicle_val.pkl python /repro_waymo/ctrl_test.py 0 2 $W/ctrl_release/ctrl_veh_epoch_24.pth $OD/${N}_noTTA 2>&1 | grep -v "compute_detection_metrics_main\|CalledProcessError"
ls -la $OD/${N}_noTTA.bin || exit 1
log "5. Xoa box rong ($N)"
sed 's/out_list, input_list, args.split, ts2idx, 0, 1)/out_list, input_list, ts2idx, args.split, 0, 1)/' tools/ctrl/remove_empty.py > tools/ctrl/remove_empty_1gpu.py
python tools/ctrl/remove_empty_1gpu.py --bin-path $OD/${N}_noTTA.bin --process 1 --split training --type vehicle 2>&1 | grep -v compute_detection_metrics_main
ls -la $OD/*.bin
log "xong $N"

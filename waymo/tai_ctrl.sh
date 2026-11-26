#!/bin/bash
# Tai file ket qua va checkpoint CTRL tu thu muc Google Drive trong docs/CTRL_instructions.md (tusen-ai/SST).
# Can dang ky Waymo Open Dataset truoc (giay phep phi thuong mai). Bo qua file da tai du dung luong.
cd "$(dirname "$0")/.."
D=data/waymo/ctrl_release; mkdir -p $D
get() {  # get <ten> <id> <so byte>
  if [ -f $D/$1 ] && [ $(stat -c %s $D/$1) -eq $3 ]; then echo "$1: da co"; return; fi
  echo "$1: dang tai ($(date +%H:%M))"; gdown --quiet -O $D/$1 "https://drive.google.com/uc?id=$2" || echo "$1: LOI"
  [ -f $D/$1 ] && [ $(stat -c %s $D/$1) -eq $3 ] && echo "$1: xong" || echo "$1: SAI DUNG LUONG"
}
get poses.pkl 1Y2XM2vadTnksEXIA3RNLUqXX0Ar2BGfW 38698684
get fsd_base_vehicle_val.bin 1hinxg_I8_Irq39u93cRUdGHfs7AgYhYZ 195359327
get best_val_in_paper.bin 1Z518w8RhN4tSJhGuMg1-CPQy5zLETs5v 609176745
get ctrl_veh_epoch_24.pth 1Q5Gu72KzZz23M6X_iGzvR6HfPI4h4F2s 246116998
ls -la $D

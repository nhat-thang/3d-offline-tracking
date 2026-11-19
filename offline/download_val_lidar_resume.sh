#!/bin/bash
# Tải point cloud LiDAR keyframe (LIDAR_TOP) của 150 scene val nuScenes, bản TẢI TIẾP ĐƯỢC.
# Mỗi gói được lưu xuống ổ (data/tmp_tgz/) trước; mất mạng hay tắt máy thì lần chạy sau tải tiếp từ chỗ dừng.
# Sau khi tải, so kích thước với máy chủ và kiểm tra gzip; gói hỏng thì xoá và tải lại ngay (tối đa 3 lần).
# Chỉ giải nén 6.019 file trong offline/val_lidar_files.txt rồi xoá file gói. Cần trống khoảng 6 GB cho một gói tạm.
# Chạy (trong Docker, từ /repro):  bash offline/download_val_lidar_resume.sh
cd "$(dirname "$0")/.."
OUT=data/nuscenes_valslim; LIST=offline/val_lidar_files.txt; LOG=offline/results/download_lidar.log
TMP=data/tmp_tgz; URL=https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0
mkdir -p $OUT/.done $TMP offline/results
fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }
for i in 01 02 03 04 05 06 07 08 09 10; do
  [ -f $OUT/.done/$i ] && { echo "goi $i: da xong, bo qua"; continue; }
  F=$TMP/trainval${i}_keyframes.tgz; U="$URL/v1.0-trainval${i}_keyframes.tgz"
  REMOTE=$(curl -sIL --connect-timeout 30 "$U" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
  ok=0
  for round in 1 2 3; do
    for try in $(seq 1 20); do
      have=$(fsize $F)
      [ "$REMOTE" -gt 0 ] && [ "$have" -ge "$REMOTE" ] && break
      echo "goi $i: tai lan $try, da co $((have/1048576)) / $((REMOTE/1048576)) MB ($(date +%H:%M))"
      curl -sfL -C - --connect-timeout 30 --speed-limit 1000 --speed-time 120 -o $F "$U" && break
      echo "goi $i: mat ket noi, cho 1 phut roi tai tiep tu cho dung"; sleep 60
    done
    have=$(fsize $F)
    if [ "$REMOTE" -gt 0 ] && [ "$have" -ne "$REMOTE" ]; then echo "goi $i: kich thuoc sai ($have / $REMOTE byte), xoa va tai lai"; rm -f $F; continue; fi
    echo "goi $i: kiem tra file goi ($(date +%H:%M))"
    if gzip -t $F 2>/dev/null; then ok=1; break; fi
    echo "goi $i: file goi hong, xoa va tai lai"; rm -f $F
  done
  [ $ok -eq 1 ] || { echo "goi $i: khong tai duoc sau 3 vong, chay lai file .bat sau"; continue; }
  echo "goi $i: giai nen ($(date +%H:%M))"
  tar -xzf $F -C $OUT -T $LIST 2>>$LOG; rc=$?
  if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then touch $OUT/.done/$i; rm -f $F; fi   # mã 2: có file không thuộc gói này, bình thường
  echo "  da co $(ls $OUT/samples/LIDAR_TOP 2>/dev/null | wc -l) / 6019 file"
done

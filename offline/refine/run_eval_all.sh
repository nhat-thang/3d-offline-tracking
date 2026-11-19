#!/bin/bash
# Bước 7: áp kết quả refine lên track rồi chấm điểm devkit cho mọi cấu hình trên 75 scene test, gom thành một bảng.
# Chạy trong Docker python:3.10 (tools/eval_refine.bat). Môi trường imm_env được tạo lại mỗi lần (giống README).
# Cấu hình nào chưa có file dự đoán thì bỏ qua; cấu hình đã chấm thì không chấm lại.
cd "$(dirname "$0")/../.."
S=${SEED:-s0}
if [ ! -x /tmp/imm_env/bin/python ]; then
  echo "tao moi truong (vai phut)..."; ENV=/tmp/imm_env bash repro/setup_env.sh > /tmp/setup.log 2>&1 && /tmp/imm_env/bin/pip install -q ijson >> /tmp/setup.log 2>&1 || { tail /tmp/setup.log; exit 1; }
fi
. /tmp/imm_env/bin/activate
mkdir -p offline/results/eval
ev() {   # ev <ten> <results.json>
  local o=offline/results/eval/${1}_test
  if [ -f $o/metrics_summary.json ]; then echo "$1: da cham truoc do"; return; fi
  [ -f "$2" ] || { echo "$1: chua co $2, bo qua"; return; }
  echo "$1: dang cham ($(date +%H:%M))"; python offline/scripts/eval_split.py "$2" test $o > $o.log 2>&1 || tail -3 $o.log
  python offline/scripts/check_yaw_size.py "$2" test > $o.box.txt 2>&1
}
ref() {  # ref <cau hinh> <src cho apply> <ten ra> [--no_box]
  local pred=offline/results/refine/pred/${1}_${S}__${1}_test.npz
  [ -f $pred ] || { echo "$3: chua co du doan $pred, bo qua"; return; }
  [ -f offline/results/$3/results_test.json ] || python offline/refine/apply.py $2 test --data offline/results/refine/data/${1}_test.npz --pred $pred --out_tag $3 $4 | tail -1
  ev $3 offline/results/$3/results_test.json
}
ev B0 baseline_B0/results.json
ref B0 B0 B0_refscore_$S --no_box
ref B0 B0 B0_ref_$S
ev Ours offline/results/Ours/results_test.json
ref Ours offline/results/Ours/results_test.json Ours_ref_$S
ev CTRLext offline/results/CTRLext/results_test.json
ref CTRLext offline/results/CTRLext/results_test.json CTRLext_ref_$S
python - <<'PY'
import json, os, glob
rows = ['B0', 'B0_refscore_' + os.environ.get('SEED', 's0'), 'B0_ref_' + os.environ.get('SEED', 's0'), 'Ours', 'Ours_ref_' + os.environ.get('SEED', 's0'),
        'CTRLext', 'CTRLext_ref_' + os.environ.get('SEED', 's0')]
out = ['| Cau hinh (75 scene test) | AMOTA | AMOTP | MOTA | Recall | IDS | FRAG | FP | FN | Hop box (lat huong, sai so kich thuoc) |', '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    f = f'offline/results/eval/{r}_test/metrics_summary.json'
    if not os.path.exists(f): continue
    m = json.load(open(f))
    bt = ''
    bf = f'offline/results/eval/{r}_test.box.txt'
    if os.path.exists(bf):
        l = [x for x in open(bf) if 'box ghep' in x]; bt = l[0].split('box ghep')[1].strip() if l else ''
    out.append(f"| {r} | {m['amota']:.3f} | {m['amotp']:.3f} | {m['mota']:.3f} | {m['recall']:.3f} | {m['ids']:.0f} | {m['frag']:.0f} | {m['fp']:.0f} | {m['fn']:.0f} | {bt} |")
open('offline/results/refine/summary_test.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
PY

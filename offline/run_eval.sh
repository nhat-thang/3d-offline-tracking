#!/bin/bash
# Chấm điểm nuScenes (bộ chấm chính thức) cho MỘT phiên bản trên một phần dữ liệu. Chạy từng bản một (tránh hết RAM).
# Dùng trong Docker:  cd /repro && source ~/imm_env/bin/activate && bash offline/run_eval.sh R1 dev
set -e
cd "$(dirname "$0")/.."
V=$1; P=${2:-dev}
F=offline/results/$V/results_$P.json; [ "$P" = val ] && F=offline/results/$V/results.json
python offline/scripts/eval_split.py "$F" "$P" offline/results/eval/${V}_$P 2>&1 | tee offline/results/eval/${V}_$P.log | tail -3
echo "| Phien ban | AMOTA | AMOTP | MOTA | RECALL | IDS | FRAG | FP | FN | TID | LGD |"
echo "|---|---|---|---|---|---|---|---|---|---|---|"
cat offline/results/eval/B0_dev/summary_row.md offline/results/eval/B1_dev/summary_row.md offline/results/eval/B2_dev/summary_row.md offline/results/eval/${V}_$P/summary_row.md 2>/dev/null

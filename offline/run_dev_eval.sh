#!/bin/bash
# Chấm điểm nuScenes (bộ chấm chính thức) cho B0, B1, B2 trên 75 scene dev, chạy song song 3 tiến trình.
# Dùng trong Docker:  cd /repro && source ~/imm_env/bin/activate && bash offline/run_dev_eval.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p offline/results/eval
python offline/scripts/eval_split.py baseline_B0/results.json dev offline/results/eval/B0_dev > offline/results/eval/B0_dev.log 2>&1 &
python offline/scripts/eval_split.py offline/results/B1/results.json dev offline/results/eval/B1_dev > offline/results/eval/B1_dev.log 2>&1 &
python offline/scripts/eval_split.py offline/results/B2/results_dev.json dev offline/results/eval/B2_dev > offline/results/eval/B2_dev.log 2>&1 &
wait
echo "| Phien ban | AMOTA | AMOTP | MOTA | RECALL | IDS | FRAG | FP | FN | TID | LGD |"
echo "|---|---|---|---|---|---|---|---|---|---|---|"
cat offline/results/eval/B0_dev/summary_row.md offline/results/eval/B1_dev/summary_row.md offline/results/eval/B2_dev/summary_row.md

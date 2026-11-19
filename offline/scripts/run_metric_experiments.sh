#!/bin/bash
# Chấm lần lượt các thí nghiệm của metric_experiments.py (bỏ qua thí nghiệm đã chấm). Dùng: bash ... <scene_meta.json>
cd "$(dirname "$0")/../.."
PY=${PY:-python}
for d in offline/results/metric_exp/*/; do
  n=$(basename $d); [[ $n == eval_* ]] && continue
  o=offline/results/metric_exp/eval_$n
  [ -f $o/metrics_summary.json ] && continue
  $PY offline/scripts/eval_split.py $d/results_dev.json $1 $o > /dev/null 2>&1 && echo "xong $n"
done

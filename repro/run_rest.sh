#!/bin/bash
# Chay tiep tracking (bo qua scene da xong) + eval. Chay tu thu muc goc: bash repro/run_rest.sh immortal_noflip
NAME=${1:-immortal_flip}; P=${PROC:-$(nproc)}
REPRO=$(cd "$(dirname "$0")/.." && pwd); . ${ENV:-$HOME/imm_env}/bin/activate; cd $REPRO/ImmortalTracker; mkdir -p $REPRO/logs
for t in car bus trailer truck pedestrian bicycle motorcycle; do python $REPRO/repro/run_chunk.py $NAME $t 999999 $P 2>&1 | grep "con lai"; done
bash evaluation/nuscenes/pipeline.sh $NAME 2>&1 | tail -2
OUT=./mot_results/nuscenes/validation_2hz/$NAME/results
python -m nuscenes.eval.tracking.evaluate $OUT/results.json --output_dir $OUT --eval_set val --dataroot $REPRO/data/nuscenes_valslim --version v1.0-trainval 2>&1 | tail -40 > $REPRO/logs/5_eval_$NAME.log
python $REPRO/repro/compare_paper.py $OUT/metrics_summary.json | tee $REPRO/RESULT_${NAME}_vs_paper.md

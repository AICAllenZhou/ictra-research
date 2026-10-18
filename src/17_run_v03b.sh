#!/bin/bash
# v03b_thr_sens_floor (Allen 2026-10-17; objective written in ITERATION_LOG before the run): v03 (inner_cv) with tau =
# max inner F1 subject to inner sensitivity >= S_ref - 0.02 (fallback: max sensitivity at FA <= FA_ref), S_ref/FA_ref
# from v01 per (set, model). Stage-1 runs, compared with v01 under section 1. Starts after the v02 queue is done
# (a full run costs ~2.5 h > 1 h). Workers 3, or 2 while the CPU reads > 85 C at launch.
cd "$(dirname "$0")/.." || exit 1
LOG=results/lab/logs/v03b.log
until grep -q "QUEUE DONE" results/lab/logs/queue_v02.log 2>/dev/null; do sleep 60; done
W=2
OUT=results/dev/v03b_thr_sens_floor; mkdir -p $OUT
echo "$(date +%H:%M) start v03b with $W workers (CPU $(python src/thermal.py) C)" > $LOG
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > $OUT/gpu_log.csv &
GPID=$!
python -u src/04_train_ablation.py --version v03b_thr_sens_floor --runs seizure --thr-mode inner_cv \
  --thr-objective f1_sens_floor --thr-ref v01_baseline \
  --note "v03 + tau = max inner F1 s.t. inner sensitivity >= v01 sensitivity - 0.02 (one change vs v03)" \
  --models xgb lgbm --rules --seeds 0 1 2 --workers $W >> $LOG 2> results/lab/logs/v03b.err.log
echo "exit $?" >> $LOG
kill $GPID
python -u src/04_train_ablation.py --compare v01_baseline v03b_thr_sens_floor > results/lab/logs/v03b_compare.log 2>&1
python -u src/05_operating_points.py --version v03b_thr_sens_floor --workers $W > results/lab/logs/v03b_sweep.log 2>&1
python -u src/15_version_report.py --version v03b_thr_sens_floor > /dev/null 2>&1
echo "V03B DONE $(grep -c 'Falling back' results/lab/logs/v03b.err.log) fallback warnings" >> $LOG

#!/bin/bash
# v02g_grid_fix (defect correction 2026-10-19, EVALUATION_PROTOCOL amendment): v02_full_hours' exact configuration with
# the corrected tau candidate grid (--thr-grid tail_log). Becomes the reference. v02 vs v02g is descriptive (no section-1
# decision): compare table, fixed-FA budgets 1/2/5/10 and v02's C xgb FA. 2 workers (CPU > 85 C under load).
# Launch inside keep_awake:  python src/keep_awake.py -- bash src/21_run_v02g.sh
cd "$(dirname "$0")/.." || exit 1
V=v02g_grid_fix; LOG=results/lab/logs/$V.log; OUT=results/dev/$V; mkdir -p $OUT
W=2
echo "$(date '+%m-%d %H:%M') start $V with $W workers (CPU $(python src/thermal.py) C)" >> $LOG
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 >> $OUT/gpu_log.csv &
GPID=$!
python -u src/04_train_ablation.py --version $V --runs all --thr-mode inner_split --thr-grid tail_log \
  --note "v02_full_hours re-run with the corrected tau candidate grid (defect correction, not a version)" \
  --bg-draws 5 --bg-draws-cells C_eeg_ecg_emg_acc:xgb --models xgb lgbm --rules --seeds 0 1 2 --workers $W >> $LOG 2> results/lab/logs/$V.err.log
echo "exit $?" >> $LOG
kill $GPID
python -u src/04_train_ablation.py --compare v02_full_hours $V > results/lab/logs/${V}_compare.log 2>&1
python -u src/05_operating_points.py --version $V --workers $W > results/lab/logs/${V}_sweep.log 2>&1
python -u src/15_version_report.py --version $V > /dev/null 2>&1
FA02=$(python -c "import pandas as pd;T=pd.read_csv('results/dev/v02_full_hours/summary.csv');print(round(float(T[(T.set=='C_eeg_ecg_emg_acc')&(T.model=='xgb')].fa_per_24h.iloc[0]),3))")
python -u src/18_matched_fa.py --ref v02_full_hours --new $V --fa 1 2 5 10 $FA02 > results/lab/logs/${V}_matched_fa.log 2>&1
echo "V02G DONE $(date '+%m-%d %H:%M') $(grep -c 'Falling back' results/lab/logs/$V.err.log) fallback warnings" >> $LOG

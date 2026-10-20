#!/bin/bash
# v11_causal (2026-10-20, Task 8 priority 2): v02g_grid_fix's configuration with ONE change, the causal pipeline
# (ictra_lib causal mode: causal filters, causal dropout repair, causal ACC calibration, causal R-peak HR, alarms
# timestamped at the end of the window that raised them). Waits for V02G DONE, rebuilds features into
# data/features_causal (03 --causal), trains with --causal, compares with v02g under section 1 and at fixed FA budgets.
# Launch inside keep_awake:  python src/keep_awake.py -- bash src/23_run_v11.sh
cd "$(dirname "$0")/.." || exit 1
V=v11_causal; LOG=results/lab/logs/$V.log; OUT=results/dev/$V; mkdir -p $OUT
until grep -q "V02G DONE" results/lab/logs/v02g_grid_fix.log 2>/dev/null; do sleep 120; done
W=2
echo "$(date '+%m-%d %H:%M') causal feature rebuild with $W workers (CPU $(python src/thermal.py) C)" >> $LOG
python -u src/03_features.py --causal --workers $W > results/lab/logs/${V}_features.log 2>&1
echo "$(date '+%m-%d %H:%M') features exit $? ; $(ls data/features_causal/*.parquet | wc -l) files; failed: $(grep -c ' run-' data/features_causal/_failed.txt 2>/dev/null || echo 0)" >> $LOG
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 >> $OUT/gpu_log.csv &
GPID=$!
python -u src/04_train_ablation.py --version $V --causal --runs all --thr-mode inner_split --thr-grid tail_log \
  --note "v02g + causal pipeline (one change: ictra_lib causal mode, causal features and alarm timestamps)" \
  --bg-draws 5 --bg-draws-cells C_eeg_ecg_emg_acc:xgb --models xgb lgbm --rules --seeds 0 1 2 --workers $W >> $LOG 2> results/lab/logs/$V.err.log
echo "exit $?" >> $LOG
kill $GPID
python -u src/04_train_ablation.py --compare v02g_grid_fix $V > results/lab/logs/${V}_compare.log 2>&1
python -u src/05_operating_points.py --version $V --workers $W > results/lab/logs/${V}_sweep.log 2>&1
python -u src/15_version_report.py --version $V > /dev/null 2>&1
FA=$(python -c "import pandas as pd;T=pd.read_csv('results/dev/v02g_grid_fix/summary.csv');print(round(float(T[(T.set=='C_eeg_ecg_emg_acc')&(T.model=='xgb')].fa_per_24h.iloc[0]),3))")
python -u src/18_matched_fa.py --ref v02g_grid_fix --new $V --fa 1 2 5 10 $FA > results/lab/logs/${V}_matched_fa.log 2>&1
echo "V11 DONE $(date '+%m-%d %H:%M') $(grep -c 'Falling back' results/lab/logs/$V.err.log) fallback warnings" >> $LOG

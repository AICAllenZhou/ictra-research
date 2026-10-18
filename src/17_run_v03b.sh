#!/bin/bash
# After the v02 queue: (1) v01 sweep re-run with precision/F1 columns and the v01 vs v02 fixed-FA table; (2) v03b.
# v03b_thr_sens_floor (protocol amendment 2026-10-17 evening, logged before the run): v02's setup (all runs, C xgb 3 x 5
# draws) with one change, the tau procedure: inner leave-patients-out CV, tau = max inner F1 s.t. inner sensitivity >=
# v02's matched-FA sensitivity - 0.02 (fallback: max sensitivity at FA <= v02's FA). Compared with v02 under section 1,
# plus fixed-FA budgets 1/2/5/10 and v02's own C xgb FA. 2 workers (CPU > 85 C under load tonight).
cd "$(dirname "$0")/.." || exit 1
LOG=results/lab/logs/v03b.log
until grep -q "QUEUE DONE" results/lab/logs/queue_v02.log 2>/dev/null; do sleep 60; done
W=2
echo "$(date +%H:%M) v01 sweep re-run (precision/F1 columns) and v01 vs v02 fixed-FA table" > $LOG
python -u src/05_operating_points.py --version v01_baseline --workers $W > results/lab/logs/v01_sweep_rerun.log 2>&1
python -u src/18_matched_fa.py --ref v01_baseline --new v02_full_hours --fa 1 2 5 10 > results/lab/logs/v02_vs_v01_fixed_fa.log 2>&1
echo "$(date +%H:%M) fixed-FA table written (results/dev/v02_full_hours/matched_fa_vs_v01_baseline.md)" >> $LOG
OUT=results/dev/v03b_thr_sens_floor; mkdir -p $OUT
echo "$(date +%H:%M) start v03b with $W workers (CPU $(python src/thermal.py) C)" >> $LOG
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > $OUT/gpu_log.csv &
GPID=$!
python -u src/04_train_ablation.py --version v03b_thr_sens_floor --runs all --thr-mode inner_cv \
  --thr-objective f1_sens_floor --thr-ref v02_full_hours --thr-ref-matched \
  --bg-draws 5 --bg-draws-cells C_eeg_ecg_emg_acc:xgb \
  --note "v02 + tau from inner leave-patients-out CV, max F1 s.t. sensitivity >= v02 matched-FA sensitivity - 0.02 (one change: tau procedure)" \
  --models xgb lgbm --rules --seeds 0 1 2 --workers $W >> $LOG 2> results/lab/logs/v03b.err.log
echo "exit $?" >> $LOG
kill $GPID
python -u src/04_train_ablation.py --compare v02_full_hours v03b_thr_sens_floor > results/lab/logs/v03b_compare.log 2>&1
python -u src/05_operating_points.py --version v03b_thr_sens_floor --workers $W > results/lab/logs/v03b_sweep.log 2>&1
python -u src/15_version_report.py --version v03b_thr_sens_floor > /dev/null 2>&1
FA02=$(python -c "import pandas as pd;T=pd.read_csv('results/dev/v02_full_hours/summary.csv');print(round(float(T[(T.set=='C_eeg_ecg_emg_acc')&(T.model=='xgb')].fa_per_24h.iloc[0]),3))")
python -u src/18_matched_fa.py --ref v02_full_hours --new v03b_thr_sens_floor --fa 1 2 5 10 $FA02 > results/lab/logs/v03b_matched_fa.log 2>&1
echo "V03B DONE $(grep -c 'Falling back' results/lab/logs/v03b.err.log) fallback warnings; matched FA incl. v02 C xgb FA $FA02" >> $LOG

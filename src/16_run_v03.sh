#!/bin/bash
# v03_threshold_stability (Allen 2026-10-17): v01 + --thr-mode inner_cv only (same features, models, seeds, folds, runs).
# Starts when Exp P (src/13_preictal_study.py) has finished, then compares with v01 under EVALUATION_PROTOCOL section 1.
# Workers: 3, or 2 while the CPU reads > 85 C at launch (src/thermal.py, Allen 2026-10-17).
cd "$(dirname "$0")/.." || exit 1
LOG=results/lab/logs/v03.log
while python -c "import psutil,sys;sys.exit(0 if any('13_preictal_study' in ' '.join(p.info['cmdline'] or []) and 'psutil' not in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])) else 1)"; do sleep 30; done
W=$(python src/thermal.py --workers 3); T=$(python src/thermal.py)
OUT=results/dev/v03_threshold_stability; mkdir -p $OUT
echo "$(date +%H:%M) start v03 with $W workers (CPU $T C)" > $LOG
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > $OUT/gpu_log.csv &
GPID=$!
python -u src/04_train_ablation.py --version v03_threshold_stability --runs seizure --thr-mode inner_cv \
  --note "v01 + threshold from pooled out-of-fold inner leave-patients-out CV (one change)" \
  --models xgb lgbm --rules --seeds 0 1 2 --workers $W >> $LOG 2> results/lab/logs/v03.err.log
echo "exit $?" >> $LOG
kill $GPID
python -u src/04_train_ablation.py --compare v01_baseline v03_threshold_stability > results/lab/logs/v03_compare.log 2>&1
python -u src/05_operating_points.py --version v03_threshold_stability --workers $W > results/lab/logs/v03_sweep.log 2>&1
python -u src/15_version_report.py --version v03_threshold_stability > /dev/null 2>&1
echo "V03 DONE $(grep -c 'Falling back' results/lab/logs/v03.err.log) fallback warnings" >> $LOG

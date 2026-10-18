#!/bin/bash
# Queue for v02_full_hours (Task 7; Allen 2026-10-17: start automatically, do not wait for him).
# Starts v02 only when ALL of these hold, checked every 2 min:
#   1. the nonseizure dry run reports 0 files still to download (stage 2 complete);
#   2. every dev run (2,072 in the catalog) has a feature file or a line in data/features/_failed.txt, and 03 is not running;
#   3. v03 has been decided (results/dev/v03_threshold_stability/compare_vs_v01_baseline.csv exists) and no trainer runs.
# v02 = the accepted chain on all runs: --thr-mode inner_cv if v03 was ACCEPTED, else inner_split (v01's rule).
# C xgb (the section-1 reference cell) gets the protocol 3 seeds x 5 background draws; all other cells 3 x 1 (Allen
# 2026-10-17). Workers 3, or 2 while the CPU reads > 85 C at launch (src/thermal.py).
# Then: paired bootstrap v02 vs the accepted version (C xgb) and the operating-point sweep. Log/notes are written by hand.
cd "$(dirname "$0")/.." || exit 1
LOG=results/lab/logs/queue_v02.log
say() { echo "$(date +%H:%M) $*" | tee -a $LOG; }
CMP=results/dev/v03_threshold_stability/compare_vs_v01_baseline.csv
say "queue started"
while true; do
  left=$(python src/01_download_seizeit2.py --runs nonseizure --split train 2>&1 | head -1 | sed 's/.*still to download: \([0-9]*\) files.*/\1/')
  feat=$(python - <<'EOF'
import re, pandas as pd
from pathlib import Path
R = pd.read_csv("data/catalog/seizeit2_runs.csv"); R = R[R.split == "train"]
have = {p.name for p in Path("data/features").glob("*.parquet")}
f = Path("data/features/_failed.txt")
failed = set(re.findall(r"^(sub-\d+) run-(\d+):", f.read_text(), re.M)) if f.exists() else set()
miss = [1 for s, r in zip(R["sub"], R.run) if f"{s}_run-{int(r):02d}.parquet" not in have and (s, f"{int(r):02d}") not in failed]
print(len(miss))
EOF
)
  busy=$(python -c "import psutil;print(sum(1 for p in psutil.process_iter(['cmdline']) if 'psutil' not in ' '.join(p.info['cmdline'] or []) and any(('03_features' in c or '04_train_ablation' in c) and c.endswith('.py') for c in (p.info['cmdline'] or []))))")
  if [ "$left" = "0" ] && [ "$feat" = "0" ] && [ "$busy" = "0" ] && [ -f "$CMP" ]; then break; fi
  say "waiting: nonseizure left $left, runs without features $feat, 03/04 processes $busy, v03 decided $([ -f $CMP ] && echo yes || echo no)"
  sleep 120
done
DEC=$(python -c "import pandas as pd;print(pd.read_csv('$CMP').decision.iloc[0])")
if [ "$DEC" = "ACCEPT" ]; then MODE=inner_cv; REF=v03_threshold_stability; else MODE=inner_split; REF=v01_baseline; fi
say "stage 2 complete and featured; v03 $DEC -> v02 with --thr-mode $MODE, compared against $REF"
OUT=results/dev/v02_full_hours; mkdir -p $OUT
W=2; say "workers $W (CPU $(python src/thermal.py) C)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > $OUT/gpu_log.csv &
GPID=$!
python -u src/04_train_ablation.py --version v02_full_hours --runs all --thr-mode $MODE \
  --note "accepted chain ($REF) on all dev runs incl. seizure-free hours" \
  --bg-draws 5 --bg-draws-cells C_eeg_ecg_emg_acc:xgb   --models xgb lgbm --rules --seeds 0 1 2 --workers $W > results/lab/logs/v02.log 2> results/lab/logs/v02.err.log
say "v02 trainer exit $? ; Falling back warnings: $(grep -c 'Falling back' results/lab/logs/v02.err.log)"
kill $GPID
python -u src/04_train_ablation.py --compare $REF v02_full_hours > results/lab/logs/v02_compare.log 2>&1
say "compare v02 vs $REF done"
python -u src/05_operating_points.py --version v02_full_hours --workers $W > results/lab/logs/v02_sweep.log 2>&1
python -u src/15_version_report.py --version v02_full_hours > /dev/null 2>&1
say "v02 sweep exit $? ; QUEUE DONE"

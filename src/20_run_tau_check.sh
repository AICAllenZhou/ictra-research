#!/bin/bash
# After v03b: tau quantile diagnostic (Allen 2026-10-18), 2 workers (CPU > 85 C under load).
cd "$(dirname "$0")/.." || exit 1
until grep -q "V03B DONE\|^exit [1-9]" results/lab/logs/v03b.log 2>/dev/null; do sleep 60; done
python -u src/19_tau_quantile_check.py --version v03b_thr_sens_floor --ref v02_full_hours --workers 2 > results/lab/logs/tau_check.log 2>&1
echo "TAU CHECK DONE exit $?" >> results/lab/logs/tau_check.log

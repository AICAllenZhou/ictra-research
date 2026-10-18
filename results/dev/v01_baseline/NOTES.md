# v01_baseline: notes (Task 5)

**Run.** `04_train_ablation.py --version v01_baseline --runs seizure --models xgb lgbm --rules --seeds 0 1 2 --workers 3`
(2026-10-17, 15:02–16:32, lib-2026-10-17e, xgboost on CUDA, lgbm on CPU).
- Data: dev folds, **seizure-containing runs only**: 305 runs, 2,288 scored hours, 702 seizures, 96 patients.
- Offline pipeline.
- Official table (every cell seed-mean / seed-median, FBTC first, sweep at FA/24h ≤ 1/2/5, montage): `REPORT.md`.
- Figures: `fig_ablation.png`, `fig_operating_curves.png`.

> **FA caveat (Allen, 2026-10-17):** v01's FA/24h was measured on seizure-containing runs only (2,288 h) and understates real-world false alarms. **v02_full_hours (all 8,715 dev hours, incl. seizure-free runs) is the realistic FA baseline from here on.** Compare versions across that boundary only at fixed FA/24h budgets (`matched_fa_vs_*.md`), not by the single-τ row.

## Answers

- **Does C beat A? Yes.** Paired patient bootstrap within v01 (`compare_C_eeg_ecg_emg_acc_xgb_vs_A_eeg_xgb.csv`):
  - xgb: ΔF1 +0.064 [+0.023, +0.095], P(ΔF1 > 0) = 0.998, Δsens +0.076 [+0.003, +0.145], ΔFA/24h −1.4
    [−3.7, +1.1]. The gain holds in all 5 folds (ΔF1 +0.013 … +0.113).
  - lgbm: ΔF1 +0.038 [+0.002, +0.068], P = 0.977, ΔFA +0.50 (right at the +0.5 limit).
  - These are ablation contrasts inside one version, not version decisions.
- **B vs A** (fold by fold, as asked, not B's first fold vs A's mean):
  - xgb: ΔF1 +0.043 [−0.001, +0.076], P = 0.973. Better in 4 of 5 folds (fold 2: −0.027).
  - lgbm: ΔF1 +0.015 [−0.023, +0.050], P = 0.785, worse in folds 1–2. **Not robust.**
  - The first-fold B gain seen during training (F1 0.48) was fold 0 only.
- **Best family:** xgb and lgbm are tied within the CIs (C: F1 0.325 vs 0.320). xgb stays the primary family.
- **Strongest rule:**
  - At its F1-tuned threshold, `rule_eeg` has the best F1 (0.13), at 32.7 FA/24h.
  - `rule_hr` has the highest sensitivity (0.39) but 57.7 FA/24h.
  - At a fixed budget (sweep) every rule is weak overall (≤ 0.11 at FA/24h ≤ 5, `rule_emg`).
  - On FBTC at ≤ 5 FA/24h: `rule_eeg` 0.67, `rule_emg` 0.61.
  - Every learned model beats every rule by a wide margin (F1 0.26–0.33 vs 0.09–0.13). This is the dev-set preview of H4.
- **FBTC check (< 0.80 = pipeline problem): passed.**
  - Every learned model detects 0.97–1.00 of the 33 FBTC seizures (patient-bootstrap CI lower bound ≥ 0.89).
  - The rules are lower: `rule_acc` 0.67, `rule_hr` 0.73, `rule_emg` 0.76, `rule_eeg` 0.97. That is expected for single
    signals, not a pipeline problem.

## Data findings (to know before v03)

1. **Three patients with very short seizures dominate the misses.**
   - sub-039, sub-047 and sub-035 hold 194 of the 702 seizures (28%). Median durations are 10–13 s.
   - A detects 2–5% of them; C detects 37% / 12% / 2%.
   - Without them, right-montage sensitivity (A xgb) is 0.45, against 0.51 for left. The apparent "right montage problem"
     (0.17 vs 0.51) is these three patients, not the montage.
   - Mechanism (C xgb, seed 0):
     - The median max probability during their seizures is 0.83–0.95, against thresholds of about 0.96.
     - 3-of-5 then removes about half of those that do cross: a 10-s seizure gives only 2–3 windows.
   - Pooled event sensitivity weights patients by seizure count, so these three patients pull every pooled number down.
     Per-patient medians are less affected.
   - Candidate later versions, one change each: k-of-n for short events (e.g. 2-of-3); 2-s windows. **Not changed in v01.**
2. **Missed seizures vs Exp R "no response on any signal"** (`missed_vs_expR.csv`; a seizure counts as missed if it is
   detected in fewer than half the seeds). For C xgb:

   | Exp R class | seizures | missed by C xgb | missed by A xgb |
   |---|---|---|---|
   | no change on any recorded signal | 74 | 67 (91%) | 71 (96%) |
   | changes on ≥ 1 signal | 621 | 337 (54%) | 388 (62%) |
   | not in Exp R (< 5 s) | 7 | 7 (100%) | 7 (100%) |

   - **The "invisible" seizures are almost always missed, but they are a minority of the misses.**
     - Of C's 411 misses, 67 (16%) are invisible.
     - 337 (82%) changed on at least one signal.
   - What the 337 visible misses look like:
     - 137 (41%) come from the three short-seizure patients above;
     - 185 last < 20 s;
     - their median duration is 17 s, against 80 s for detected seizures.
   - Most misses are short seizures that the signals do register, not seizures with nothing to detect.
   - By group, C's invisible misses are focal non-motor 43, focal motor 13, unknown motor 11, FBTC 0.
3. **The threshold depends on the seed.**
   - The per-seed τ spread within a fold reaches SD 0.052 (A xgb fold 1: τ = 0.965 / 0.879 / 0.971).
   - Consequences on that fold:
     - seed 1 there gives 50.9 FA/24h, against 3.9–8.2 for the other seeds;
     - **seed-mean FA/24h (11.1) therefore lies outside the seed-0 bootstrap CI [5.2, 10.6]**.
   - `REPORT.md` shows mean and median (A xgb FA/24h 11.1 / 10.9; C lgbm 9.7 / 11.5).
   - Per-fold SD and IQR of τ are in `compare_*_by_fold.csv`. This is what v03 (`--thr-mode inner_cv`) is meant to shrink.
4. **False alarms are high (about 10 / 24 h) for every learned model.**
   - v01 counts FA only in seizure-containing runs (`--runs seizure`): fewer and harder hours than the full monitoring.
   - v02 adds the seizure-free runs.
   - At a fixed budget, C xgb reaches sensitivity 0.23 / 0.29 / 0.38 at FA/24h ≤ 1 / 2 / 5 (sweep; held-out threshold,
     not deployable).

## Deviations from `EVALUATION_PROTOCOL.md` §1 in this run (stated, not hidden)
- **Repeats.** §1 says "3 seeds × 5 background-sampling draws". v01 has 3 seeds, each with one background draw (the
  sampling is seeded by the model seed), so 3 draws in total, not 15.
- **Score.** §1 says "mean over folds". v01 pools the per-patient counts over all folds, then takes the mean over
  seeds. Per-fold numbers are in `compare_*_by_fold.csv` and `scores_by_fold.csv`.
- **Peak RSS.** It is the worker process's cumulative peak working set. Workers are reused, so jobs late in the queue
  (the rules) inherit earlier peaks. True maximum: 1.59 GB per job; 3 workers ≈ 4.8 GB, under the 50% cap (16.7 GB).

## GPU check (Allen, 2026-10-17)
- nvidia-smi during the first xgb jobs (15:02–15:05): max 70% utilisation, 1,213 / 12,227 MiB (baseline 566 MiB).
  Sample line: `2026/10/17 15:02:29.207, 70 %, 1213 MiB, 12227 MiB`.
- Booster device `cuda:0`; 0 "Falling back … mismatched devices" warnings in stderr. The trainer predicts on the CPU
  after the GPU fit (`predict_on_cpu`).
- LightGBM on CPU: no GPU build installed.

## Leakage checklist (`EVALUATION_PROTOCOL.md` §6), checked on the v01 outputs
- [x] **No patient on both sides of any split.**
  - The 96 dev patients form 5 disjoint folds (0 pairwise overlaps; `splits.json` hash matches `splits.sha256`).
  - Inner threshold split = patients of the training folds only.
- [x] **Threshold and post-processing fixed on training folds only.**
  - τ comes from the inner 80/20 patient split of the 4 training folds.
  - 3-of-5 and the SzCORE constants are fixed in `ictra_lib.CFG`.
  - The held-out fold is used only for scoring. (The sweep in `operating_curve.csv` uses held-out data and is labelled
    an operating curve, not a deployable number.)
- [x] **Natural class ratio.** Validation scores every window of the held-out runs. Only training undersamples the
  background (5:1).
- [x] **`impd` masked.** impd and flat-EEG windows are masked from training, from alarms and from the FA hours
  (`score_recording(mask=...)`).
- [x] **Model version chosen before any locked or external set was opened.**
  - No file of sub-097..125 exists in `data/raw` or `data/features` (checked).
  - No external labels were read (`labels_allowed()` guard).
- [ ] **Billeci–Siena overlap.** Not checked yet; it is only relevant at the external test. Their channel lists are
  identical (DATA_NOTES §8), so overlap is likely and must be checked before both are counted.
- **"Too good" check.** FBTC sensitivity is 0.97–1.00, but at about 10 FA/24h and on large, long seizures (median 80 s
  detected), consistent with the literature on wearable FBTC detection. Focal sensitivity is 0.24–0.50. Nothing looks
  like leakage: no patient is in two folds, and the threshold never sees the held-out fold.

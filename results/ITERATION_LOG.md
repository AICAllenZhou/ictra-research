# ICTRA iteration log

Every version: one change, dev folds only, accept/reject by the rule in docs/EVALUATION_PROTOCOL.md section 1.
Synthetic smoke tests are never logged here.

## 2026-10-17 — pipeline fixes found on the first real files (before v01; not a model version)

Found in Tasks 1–3 and fixed before any feature table was kept (details and evidence: `docs/DATA_NOTES.md` §1–§7).
Features are rebuilt from scratch under `ictra_lib.LIB_VERSION = lib-2026-10-17c` (stamped in `data/features/_lib_version.txt`).
- No `events.tsv` on this machine (the git metadata clone is absent): `01_download_seizeit2.py --sidecars` fetches dev sidecars from S3.
- SensorDot sample dropouts (1–2 samples pulled toward zero every 1–4 s; up to 0.6 % of EEG/ECG/EMG and 2.4 % of ACC samples in
  affected runs) became 50–200 uV spikes after filtering: `ictra_lib.repair_dropouts`, first step of `harmonise`.
- The ACC scale differs between recordings (resting |a| 0.43–0.52 "g" in most runs, 0.167 in sub-019 run-07): `ictra_lib.acc_to_g`
  calibrates every run to 1 g at rest. The neck SensorDot is now picked by name.
- `heart_rate_series` would have needed ~42 GB on an 18-h run; `eeg_features` ~2 GB per worker: both rewritten (same output).
- `03_features.py` could save a half-downloaded run without ECG and skip it forever: it now needs every catalogued file complete.
- `04_train_ablation.py`: the column list came from the first training file (a run without ECG would have removed ECG from Model B);
  reference events were rebuilt from window overlaps instead of the annotations; FA/24h hours included masked windows.
  All three are fixed. xgboost runs on the GPU (`--device cuda`; same model).
- `02_relationship_study.py` had its own filter copies (it now uses `ictra_lib`), `np.resize` wrapped short segments
  cyclically (now NaN-padded), and it counted a missing signal as "did not respond" (now shares are over seizures with that signal).
- Kept from the teammate update (Carina's ECG audit): `flat_eeg/ecg/emg` columns, flat windows blanked and masked like `impd`.

## 2026-10-17 — H3 revised after Exp R (no model change)

H3 as first written (PLAN §5: "EEG changes before HR in most seizures"): **not supported on the dev set**. The EEG change is
strictly first vs HR in 41–56% of seizures by group (FBTC 54%, focal motor 45%, focal non-motor 41%, focal unknown motor
56%; z > 3 for 3 s, `results/expR/onsets.csv`). It is replaced in `docs/PREREGISTRATION_DRAFT.md` by H3a (FBTC: EEG change
precedes HR, ACC and EMG; median lag > 0) and H3b (focal: HR change within ±10 s of the EEG change in ≥ 70%). On dev, H3a
holds (median lags +3.5 / +8.5 / +12 s) and H3b does not (57%); both are tested on the locked test.
Update (Allen, same day): H3b's threshold form is replaced by a model-level prediction (B's median delay on non-FBTC focal
seizures not shorter than A's by more than 2 s, while B's sensitivity is higher), and H4 is added (each learned model beats
its single-signal rule(s) in event F1, P(improve) ≥ 0.90). H1–H4 are defined only in `docs/PREREGISTRATION_DRAFT.md`.

## v01_baseline (2026-10-17 16:32) default features, seizure runs only

Dev folds, pooled over patients, patient-bootstrap 95% CI.

| set               | model    |   seizures |   hours |   sensitivity | sens_ci      |   fa_per_24h | fa_ci          |    f1 | f1_ci        |   median_delay_s |
|:------------------|:---------|-----------:|--------:|--------------:|:-------------|-------------:|:---------------|------:|:-------------|-----------------:|
| A_eeg             | lgbm     |        702 | 2288.18 |         0.359 | [0.22, 0.47] |        9.178 | [4.92, 10.25]  | 0.278 | [0.21, 0.38] |           21.583 |
| A_eeg             | xgb      |        702 | 2288.18 |         0.366 | [0.21, 0.47] |       11.052 | [5.16, 10.64]  | 0.259 | [0.20, 0.37] |           21.917 |
| B_eeg_ecg         | xgb      |        702 | 2288.18 |         0.404 | [0.22, 0.49] |        9.552 | [3.99, 8.23]   | 0.3   | [0.22, 0.43] |           22     |
| B_eeg_ecg         | lgbm     |        702 | 2288.18 |         0.399 | [0.21, 0.49] |        9.789 | [3.77, 7.86]   | 0.295 | [0.22, 0.44] |           21.167 |
| C_eeg_ecg_emg_acc | xgb      |        702 | 2288.18 |         0.442 | [0.28, 0.55] |        9.667 | [4.17, 7.26]   | 0.325 | [0.27, 0.47] |           21     |
| C_eeg_ecg_emg_acc | lgbm     |        702 | 2288.18 |         0.435 | [0.27, 0.54] |        9.678 | [4.16, 7.56]   | 0.32  | [0.27, 0.47] |           20.417 |
| rule              | rule_eeg |        702 | 2288.18 |         0.372 | [0.23, 0.52] |       32.662 | [26.16, 40.66] | 0.129 | [0.09, 0.16] |           24     |
| rule              | rule_acc |        702 | 2288.18 |         0.298 | [0.20, 0.40] |       35.578 | [27.95, 44.36] | 0.098 | [0.06, 0.14] |           25     |
| rule              | rule_emg |        702 | 2288.18 |         0.296 | [0.17, 0.42] |       38.252 | [28.60, 48.17] | 0.093 | [0.06, 0.13] |           24     |
| rule              | rule_hr  |        702 | 2288.18 |         0.393 | [0.24, 0.55] |       57.74  | [42.37, 75.75] | 0.089 | [0.06, 0.12] |           11.5   |

Offline pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17).
Peak RSS per job (GB; max over folds and seeds of the worker's peak working set): A_eeg/lgbm 1.35, A_eeg/xgb 1.12, B_eeg_ecg/lgbm 1.35, B_eeg_ecg/xgb 1.35, C_eeg_ecg_emg_acc/lgbm 1.59, C_eeg_ecg_emg_acc/xgb 1.35, rule/rule_acc 1.59, rule/rule_eeg 1.59, rule/rule_emg 1.59, rule/rule_hr 1.59. Overall max 1.59 GB.

**Decision: v01 = BASELINE (the first accepted version; reference for v03).** C xgb F1 0.325 [0.27, 0.47], sensitivity
0.44, 9.7 FA/24h on seizure-containing runs only. C beats A in paired bootstrap (xgb ΔF1 +0.064, P = 0.998); every learned
model beats every rule; FBTC check passed (0.97–1.00 for all learned models). Details: `results/dev/v01_baseline/NOTES.md`,
official table `REPORT.md` (seed-mean / seed-median).
- **GPU:** nvidia-smi during the first xgb jobs, `2026/10/17 15:02:29.207, 70 %, 1213 MiB, 12227 MiB` (baseline 566 MiB);
  booster device cuda:0; 0 "Falling back" warnings; LightGBM on CPU.
- **Peak RSS:** the numbers above are the cumulative peak of reused workers, true maximum 1.59 GB per job; 3 workers stay
  under the 50%-of-RAM cap.
- **Leakage checklist (§6):**
  - [x] no patient on both sides (5 disjoint folds, splits hash matches; inner split inside the training folds);
  - [x] threshold and post-processing from training folds only;
  - [x] natural class ratio in validation;
  - [x] impd and flat EEG masked from training, alarms and FA hours;
  - [x] no locked or external data opened (no sub-097..125 file on disk);
  - [ ] Billeci–Siena overlap: open, needed only before the external tests.
- **Found:**
  - three short-seizure patients (sub-039/047/035, 28% of seizures, median 10–13 s) are mostly missed;
  - 82% of C's misses changed on ≥ 1 signal in Exp R (only 16% are "invisible");
  - the per-seed threshold spread reaches SD 0.052 within a fold, which is why v03 targets it.

## 2026-10-17 — plan after the v01 review (Allen; order binding, no model change)

1. v03_threshold_stability (stage 1, `--thr-mode inner_cv`) vs v01.
2. v02_full_hours: the accepted chain on all runs, C xgb with 3 seeds × 5 background draws, other cells 3 × 1. It becomes
   the reference.
3. v07: 2-s window / 1-s hop, via a runtime CFG override and its own feature directory; ictra_lib.py is not edited.
4. v08: k-of-n grid 2/3, 3/5, 4/6, tuned with τ on the inner split. **v08's report must show the < 15 s row for every
   model and for `rule_hr` / `rule_acc` side by side, before and after the change. That row is the test of whether v08
   did its job.**
5. v11 causal pipeline (`--causal` in ictra_lib, version bump, feature rebuild), compared against the accepted offline
   version.
6. FA audit of v02 (`FA_AUDIT.md`); then v09.
7. v04 relationship features.
8. `src/14_export_model.py` (frozen xgb → JSON/ONNX + numpy inference to 1e-6).
9. Task 9 headers.

Every table reports sensitivity by seizure length (< 15 s, 15–60 s, > 60 s). Use `--workers 2` while the CPU is above
85 °C (`src/thermal.py`).

## 2026-10-17 — Exp P pre-ictal drift (study, not a version; exploratory)

`src/13_preictal_study.py` covers 538 eligible dev seizures (93 patients) against 5,350 same-patient pseudo-onsets. It
looks for drift in heart rate, HRV (`ecg_rmssd`), EMG log RMS and movement over −120..−30 s compared with −300..−120 s.
**No signal shows pre-ictal drift in any group: 0 of 20 cells, and 0 of 20 with −60..−30 vs −300..−60 s.** The only CI
that excludes 0 is FBTC HR, and it points the other way (fewer seizures than controls with z > 2). The curves separate
only in the last ~20 s before the annotated onset, which is excluded by design and matches Exp R's early changes. On this
evidence a minutes-ahead warning is not plausible from these signals. Details: `results/expP/FINDINGS.md`.

## 2026-10-17 — v03b objective (written BEFORE running; Allen)

v03b = v03 (`--thr-mode inner_cv`, same features, models, seeds, folds and runs as v01) with one change: the rule that
picks τ.
- **Candidates:** the same 40 τ candidates as v01/v03, applied to the pooled out-of-fold inner predictions of all
  training patients.
- **Objective:** maximise pooled inner event F1 **subject to** pooled inner sensitivity ≥ S_ref − 0.02.
- **Fallback:** if no candidate meets the floor, maximise inner sensitivity subject to inner FA/24h ≤ FA_ref. If none
  meets that either, take the lowest-FA candidate.
- **Ties:** fewer false alarms.
- **References per (set, model):** S_ref and FA_ref are v01's held-out pooled values (`results/dev/v01_baseline/summary.csv`,
  seed mean). Values:

  | | A xgb | A lgbm | B xgb | B lgbm | C xgb | C lgbm | rule_eeg | rule_hr | rule_emg | rule_acc |
  |---|---|---|---|---|---|---|---|---|---|---|
  | S_ref | 0.3656 | 0.3590 | 0.4041 | 0.3993 | 0.4421 | 0.4349 | 0.3718 | 0.3932 | 0.2963 | 0.2977 |
  | FA_ref | 11.05 | 9.18 | 9.55 | 9.79 | 9.67 | 9.68 | 32.66 | 57.74 | 38.25 | 35.58 |

- **Note:** the floor is checked on inner (training-patient) predictions, not on the held-out fold; the held-out sensitivity
  can still differ.
- **Evaluation:** compared with v01 under §1 on C xgb. Runs right after v02, because a full run costs about 2.5 h (> 1 h).

## v03_threshold_stability (2026-10-17 20:10) v01 + threshold from pooled out-of-fold inner leave-patients-out CV (one change)

Dev folds, pooled over patients, patient-bootstrap 95% CI.

| set               | model    |   seizures |   hours |   sensitivity | sens_ci      |   fa_per_24h | fa_ci          |    f1 | f1_ci        |   median_delay_s |
|:------------------|:---------|-----------:|--------:|--------------:|:-------------|-------------:|:---------------|------:|:-------------|-----------------:|
| A_eeg             | lgbm     |        702 | 2288.18 |         0.302 | [0.21, 0.44] |        5.493 | [3.94, 8.41]   | 0.295 | [0.22, 0.39] |           27.917 |
| A_eeg             | xgb      |        702 | 2288.18 |         0.301 | [0.20, 0.44] |        6.174 | [5.11, 10.15]  | 0.282 | [0.19, 0.35] |           26.25  |
| B_eeg_ecg         | lgbm     |        702 | 2288.18 |         0.349 | [0.23, 0.51] |        4.587 | [3.52, 7.12]   | 0.353 | [0.25, 0.46] |           25.333 |
| B_eeg_ecg         | xgb      |        702 | 2288.18 |         0.343 | [0.22, 0.48] |        4.419 | [3.23, 6.72]   | 0.352 | [0.24, 0.46] |           24.542 |
| C_eeg_ecg_emg_acc | xgb      |        702 | 2288.18 |         0.379 | [0.27, 0.55] |        4.479 | [3.25, 5.52]   | 0.381 | [0.29, 0.51] |           22.333 |
| C_eeg_ecg_emg_acc | lgbm     |        702 | 2288.18 |         0.376 | [0.26, 0.54] |        4.709 | [3.50, 6.06]   | 0.371 | [0.28, 0.48] |           25.333 |
| rule              | rule_eeg |        702 | 2288.18 |         0.353 | [0.24, 0.49] |       30.323 | [23.37, 38.17] | 0.128 | [0.09, 0.17] |           21.875 |
| rule              | rule_emg |        702 | 2288.18 |         0.158 | [0.09, 0.23] |       11.191 | [7.71, 15.34]  | 0.119 | [0.07, 0.17] |           35     |
| rule              | rule_acc |        702 | 2288.18 |         0.252 | [0.17, 0.33] |       23.925 | [20.46, 27.81] | 0.112 | [0.07, 0.16] |           26.75  |
| rule              | rule_hr  |        702 | 2288.18 |         0.4   | [0.26, 0.54] |       48.615 | [34.59, 64.08] | 0.103 | [0.07, 0.14] |           21     |

Offline pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17).
Peak RSS per job (GB; max over folds and seeds of the worker's peak working set): A_eeg/lgbm 1.55, A_eeg/xgb 1.30, B_eeg_ecg/lgbm 1.59, B_eeg_ecg/xgb 1.55, C_eeg_ecg_emg_acc/lgbm 2.08, C_eeg_ecg_emg_acc/xgb 1.66, rule/rule_acc 2.08, rule/rule_eeg 2.08, rule/rule_emg 2.08, rule/rule_hr 2.08. Overall max 2.08 GB.

**Decision v03_threshold_stability vs v01_baseline (§1, C xgb, paired patient bootstrap): REJECT.**
- ΔF1 +0.061 [+0.015, +0.104], P(ΔF1 > 0) = 0.992;
- ΔFA/24h −5.19 [−6.81, −3.73] (9.67 → 4.48);
- **Δsensitivity −0.063 [−0.111, −0.024] (0.442 → 0.379): fails the −0.02 sensitivity cap**, the only failed condition.

The stability goal was met: the per-seed τ SD per fold fell from 0.005–0.031 (v01) to 0.0002–0.0023 (v03). The loss is
an operating-point shift (a slightly higher τ moves along the same curve), not a model change. v02 therefore runs with
v01's threshold method (inner_split), and v03b (sensitivity-floor objective, logged above) runs after v02. Official
table: `results/dev/v03_threshold_stability/REPORT.md`.
Rejected on sensitivity only; τ stability goal achieved — carried into v03b with a sensitivity-constrained τ.

## 2026-10-17 — protocol amendment: v03b compared with v02 on full hours (Allen; before v03b started)

This supersedes the v03b reference values logged above (v01-based S_ref / FA_ref).
- v03b runs on all dev hours and is compared under §1 with v02_full_hours.
- Sensitivity floor per cell = v02's matched-FA sensitivity (operating curve at FA/24h ≤ v02's deployable FA) − 0.02;
  fallback: max sensitivity at FA ≤ v02's FA.
- The only change relative to v02 is the τ-selection procedure (inner leave-patients-out CV + constrained objective).
- Full text: `docs/EVALUATION_PROTOCOL.md`, amendment "2026-10-17 (evening)".
- Fair comparisons from here on: v01 vs v02 (and v02 vs v03b) at fixed FA/24h budgets 1 / 2 / 5 / 10, with sensitivity
  and F1, next to the single-τ table.

## v02_full_hours (2026-10-18 03:54) accepted chain (v01_baseline) on all dev runs incl. seizure-free hours

Dev folds, pooled over patients, patient-bootstrap 95% CI.

| set               | model    |   seizures |   hours |   sensitivity | sens_ci      |   fa_per_24h | fa_ci          |    f1 | f1_ci        |   median_delay_s |
|:------------------|:---------|-----------:|--------:|--------------:|:-------------|-------------:|:---------------|------:|:-------------|-----------------:|
| A_eeg             | lgbm     |        702 | 8641.56 |         0.174 | [0.09, 0.21] |        1.989 | [0.57, 2.27]   | 0.159 | [0.10, 0.21] |           29.458 |
| A_eeg             | xgb      |        702 | 8641.56 |         0.171 | [0.05, 0.12] |        2.501 | [0.14, 0.38]   | 0.144 | [0.09, 0.19] |           29.417 |
| B_eeg_ecg         | lgbm     |        702 | 8641.56 |         0.223 | [0.05, 0.26] |        2.019 | [0.10, 0.27]   | 0.202 | [0.09, 0.38] |           31     |
| B_eeg_ecg         | xgb      |        702 | 8641.56 |         0.229 | [0.08, 0.28] |        2.203 | [0.46, 1.25]   | 0.196 | [0.11, 0.34] |           31.25  |
| C_eeg_ecg_emg_acc | lgbm     |        702 | 8641.56 |         0.232 | [0.09, 0.31] |        1.563 | [0.40, 1.24]   | 0.229 | [0.12, 0.37] |           30.417 |
| C_eeg_ecg_emg_acc | xgb      |        702 | 8641.56 |         0.256 | [0.09, 0.29] |        3.001 | [0.40, 1.24]   | 0.191 | [0.13, 0.34] |           30.242 |
| rule              | rule_eeg |        702 | 8641.56 |         0.217 | [0.13, 0.34] |       24.054 | [17.28, 32.48] | 0.033 | [0.02, 0.05] |           29.5   |
| rule              | rule_emg |        702 | 8641.56 |         0.198 | [0.11, 0.29] |       24.037 | [17.73, 30.90] | 0.029 | [0.02, 0.05] |           32.75  |
| rule              | rule_acc |        702 | 8641.56 |         0.283 | [0.18, 0.38] |       36.449 | [30.47, 42.97] | 0.029 | [0.02, 0.05] |           25     |
| rule              | rule_hr  |        702 | 8641.56 |         0.366 | [0.24, 0.50] |       60.539 | [46.63, 76.43] | 0.024 | [0.01, 0.04] |           13     |

Offline pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17).
Peak RSS per job (GB; max over folds and seeds of the worker's peak working set): A_eeg/lgbm 2.32, A_eeg/xgb 2.32, B_eeg_ecg/lgbm 2.75, B_eeg_ecg/xgb 2.75, C_eeg_ecg_emg_acc/lgbm 3.16, C_eeg_ecg_emg_acc/xgb 3.16, rule/rule_acc 3.16, rule/rule_eeg 3.16, rule/rule_emg 3.16, rule/rule_hr 3.16. Overall max 3.16 GB.

**Decision: v02_full_hours becomes the reference (evaluation-population change, not a §1 method decision).**
- The §1 compare against v01 (`compare_vs_v01_baseline.csv`) reads REJECT: ΔF1 −0.137, Δsens −0.186, ΔFA/24h −6.67
  (C xgb). It is not a valid §1 test: v01 scored 2,288 h of seizure-containing runs, v02 scores 8,642 h of all dev runs,
  so FA/24h and precision have different denominators. Allen's plan (2026-10-17) set v02 as the reference before the run.
- Fixed-FA comparison (`matched_fa_vs_v01_baseline.md`), C xgb, seed-mean, v01 → v02: sensitivity at FA/24h ≤ 1 / 2 / 5
  / 10 = 0.235 → 0.268, 0.293 → 0.324, 0.377 → 0.414, 0.448 → 0.498. Every learned model is +0.01 to +0.06 at every
  budget. F1 at the same budget is lower in v02 (e.g. 0.346 → 0.307 at ≤ 1) because the same FA rate over 3.8× the hours
  is 3.8× the false-alarm count. Part of the sensitivity gain may be easier seizure-free hours rather than a better model;
  that is not separated here.
- Why the single-τ row drops: the inner split now contains the seizure-free hours, and the max-F1 objective picks a
  higher τ (C xgb median τ per fold 0.987–0.992), trading sensitivity for precision: 0.256 at 3.0 FA/24h, against 0.364
  on the curve at the same FA. This is the gap v03b's sensitivity floor targets.
- FBTC check passes for every learned model (0.83–0.92 at the deployable τ; 1.00 for C at FA/24h ≤ 1). τ is still
  repeat-dependent: C xgb per-fold τ range reaches 0.944–0.994 over the 15 repeats, with FA/24h 0.0–12.7.

## v03b_thr_sens_floor (2026-10-18 14:03) v02 + tau from inner leave-patients-out CV, max F1 s.t. sensitivity >= v02 matched-FA sensitivity - 0.02 (one change: tau procedure)

Dev folds, pooled over patients, patient-bootstrap 95% CI.

| set               | model    |   seizures |   hours |   sensitivity | sens_ci      |   fa_per_24h | fa_ci          |    f1 | f1_ci        |   median_delay_s |
|:------------------|:---------|-----------:|--------:|--------------:|:-------------|-------------:|:---------------|------:|:-------------|-----------------:|
| A_eeg             | lgbm     |        702 | 8641.56 |         0.351 | [0.24, 0.49] |        6.734 | [5.32, 9.32]   | 0.146 | [0.10, 0.19] |           23.167 |
| A_eeg             | xgb      |        702 | 8641.56 |         0.342 | [0.23, 0.49] |        6.522 | [5.00, 8.70]   | 0.146 | [0.10, 0.20] |           25.833 |
| B_eeg_ecg         | xgb      |        702 | 8641.56 |         0.435 | [0.30, 0.58] |        7.274 | [6.24, 9.28]   | 0.168 | [0.11, 0.22] |           20.542 |
| B_eeg_ecg         | lgbm     |        702 | 8641.56 |         0.444 | [0.32, 0.60] |        7.685 | [6.50, 9.83]   | 0.164 | [0.11, 0.22] |           20.5   |
| C_eeg_ecg_emg_acc | lgbm     |        702 | 8641.56 |         0.48  | [0.34, 0.62] |        7.862 | [6.42, 9.50]   | 0.173 | [0.12, 0.23] |           20.667 |
| C_eeg_ecg_emg_acc | xgb      |        702 | 8641.56 |         0.473 | [0.35, 0.63] |        7.82  | [6.56, 9.46]   | 0.172 | [0.12, 0.23] |           20.558 |
| rule              | rule_eeg |        702 | 8641.56 |         0.321 | [0.21, 0.45] |       28.628 | [23.08, 34.48] | 0.04  | [0.02, 0.06] |           21     |
| rule              | rule_emg |        702 | 8641.56 |         0.246 | [0.17, 0.32] |       27.001 | [22.32, 32.01] | 0.034 | [0.02, 0.05] |           35     |
| rule              | rule_acc |        702 | 8641.56 |         0.285 | [0.19, 0.38] |       34.891 | [31.21, 39.24] | 0.03  | [0.02, 0.05] |           26     |
| rule              | rule_hr  |        702 | 8641.56 |         0.46  | [0.31, 0.59] |       60.225 | [47.70, 74.29] | 0.029 | [0.02, 0.05] |           18.25  |

Offline pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17).
Peak RSS per job (GB; max over folds and seeds of the worker's peak working set): A_eeg/lgbm 2.95, A_eeg/xgb 2.83, B_eeg_ecg/lgbm 3.52, B_eeg_ecg/xgb 3.38, C_eeg_ecg_emg_acc/lgbm 3.97, C_eeg_ecg_emg_acc/xgb 3.97, rule/rule_acc 3.97, rule/rule_eeg 3.97, rule/rule_emg 3.97, rule/rule_hr 3.97. Overall max 3.97 GB.

**Decision: REJECT (§1, v02_full_hours → v03b, C xgb, paired patient bootstrap, 96 patients).**
- ΔF1 −0.011 [−0.044, +0.026], P(ΔF1 > 0) = 0.289; Δsens +0.217 [+0.156, +0.274] (0.256 → 0.473); ΔFA/24h +4.82
  [+3.90, +5.82] (3.00 → 7.82). Fails the F1 gain, P(improve) and FA cap; passes the sensitivity cap
  (`compare_vs_v02_full_hours.csv`). Second rejection under the rule written before the run.
- **Same model, threshold chosen in the wrong place — not a worse model.** At fixed FA/24h budgets 1 / 2 / 5 / 10 / 3.00
  every learned model sits on v02's curve: |Δsens| ≤ 0.013, A models exactly 0; C xgb 0.268 → 0.271 at ≤ 1 and
  0.364 → 0.365 at ≤ 3.00 (`matched_fa_vs_v02_full_hours.md`).
- **τ stability goal achieved, recorded although the version is rejected:** C xgb per-fold τ SD 0.013–0.019 (v02) →
  0.0012–0.0016; fold 0's 15 repeats 0.9496–0.9550 (v02: 0.944–0.994) (`compare_vs_v02_full_hours_by_fold.csv`).
- **τ quantile check** (`src/19_tau_quantile_check.py`, Allen 2026-10-18; repeat 0 of every learned model × 5 folds +
  4 rules × 5 folds; `tau_quantile_check.csv/.md`):
  - Reproduction: τ re-selected from the re-created inner OOF predictions equals the logged τ in 50 / 50 jobs.
  - Scale drift not supported: P_final / P_inner median 0.97–1.32 per model, P_final > P_inner in only 2–3 of 5 folds;
    pooled inner and held-out operating points agree (C xgb median inner sens 0.456 vs held-out 0.493, FA/24h 7.66 vs
    8.51). The fold-to-fold spread of P_final (0.0011–0.0043) is the same size as the rules', which have no fitted scale.
  - The floor was not the active constraint at the chosen point: inner sensitivity 0.28–0.46 against floors 0.17–0.34.
  - **What sets the operating point is the τ candidate grid.** `tune_threshold` sweeps 40 quantiles of the pooled
    probabilities, `np.linspace(0.90, 0.9999, 40)`. All 30 learned-model jobs picked candidate 38 (the 0.99734 quantile,
    P ≈ 0.0027); the next candidate is the 0.9999 quantile (P ≈ 0.0001), a 27× lower alarm rate with nothing between.
    Rules picked candidates 24–36. Whether candidate 39 was excluded by the floor or lost on F1 was not recorded.
  - The same grid plausibly explains the seed-dependent τ since v01: v02 C xgb fold 0 held-out exceedance at τ splits
    into 0.00007–0.00045 (high-τ repeats) and 0.0030–0.0049 (low-τ repeats), one value between (0.00137), i.e. jumps
    between the top grid points. Hypothesis, not tested: a version with a denser top grid (one change) would test it.

## 2026-10-19 — defect correction: τ candidate grid (Allen; logged before the run)

The τ search in `tune_threshold` was too coarse at the top (linear quantiles 0.90–0.9999, 40 points; last step a 27×
jump in alarm rate), which pinned τ in v01–v03b (v03b quantile check above). Corrected to `q = 1 − logspace(−1, −5, 81)`
(log-spaced tail probability). Not a §1 version: v02's configuration is re-run unchanged except for the grid as
**v02g_grid_fix**, which becomes the reference for all later versions. Full text: `docs/EVALUATION_PROTOCOL.md`,
amendment 2026-10-19. One-job smoke test before the run (A xgb fold 0 seed 0, same inner split): τ 0.9877 → 0.9727,
inner F1 0.156 → 0.328, held-out sens / FA/24h / F1 0.108 / 0.13 / 0.186 → 0.303 / 2.43 / 0.269 (one job; not a result).

## 2026-10-20 — v11_causal: what changes (logged before the run)

One change relative to the reference v02g_grid_fix: the causal pipeline (`ictra_lib` lib-2026-10-20f, `CFG["causal"]`;
features rebuilt into `data/features_causal/` by `03_features.py --causal`; `04 --causal`). Every step uses only samples
up to the end of the window it describes:
- band-pass and mains notch: forward-only `sosfilt` / `lfilter` (same Butterworth / notch designs) instead of zero-phase;
- no whole-chunk median removal (filters start in steady state at the first sample);
- dropout detection judged with the previous 1-s block's level and a running median of the past 60 blocks; repaired by
  holding the last good sample instead of interpolating to the next;
- ACC calibration: running median |a| of the past 10 min instead of the whole-recording median;
- heart rate: causal Pan-Tompkins-style R-peak detector (neurokit's uses a centred smoother and a whole-recording QRS
  threshold); HR held at the last final beat instead of interpolated;
- alarms timestamped at the END of the window that raised them (offline: the window start), so delays are not one
  window early.
Bounded look-ahead left: `resample_poly` (≈ 40 ms; none on a device sampling at 256 Hz) and the dropout check's right
neighbour (1–2 samples). Checks (`src/22_causality_check.py`, `results/data_checks/causality_check.md`):
- offline path unchanged: sub-001 run-01 recomputed, 32,658 windows × 61 columns, max |diff| 0.0, same NaN pattern;
- truncation test (sub-001 run-01, cut at 30 min): causal mode 0 differing windows on EEG, ECG, EMG, ACC (898 each);
  offline mode differs (EEG 4, EMG 1, ACC 898 windows), so the test detects look-ahead;
- causal vs neurokit HR on 12 dev runs: median |diff| 0.85–5.0 bpm, share within 5 bpm 0.50–0.995; four noisy runs
  disagree strongly (p90 40–80 bpm). The cost to detection is what v11 measures.
Decision by §1 against v02g; the offline-vs-causal delta is reported whether or not v11 is accepted (TASKS v11).

## v02g_grid_fix (2026-10-21 10:19) v02_full_hours re-run with the corrected tau candidate grid (defect correction, not a version)

Dev folds, pooled over patients, patient-bootstrap 95% CI.

| set               | model    |   seizures |   hours |   sensitivity | sens_ci      |   fa_per_24h | fa_ci          |    f1 | f1_ci        |   median_delay_s |
|:------------------|:---------|-----------:|--------:|--------------:|:-------------|-------------:|:---------------|------:|:-------------|-----------------:|
| A_eeg             | lgbm     |        702 | 8641.56 |         0.151 | [0.10, 0.28] |        0.867 | [0.75, 1.71]   | 0.188 | [0.12, 0.28] |           31.5   |
| A_eeg             | xgb      |        702 | 8641.56 |         0.141 | [0.08, 0.24] |        0.901 | [0.78, 1.79]   | 0.175 | [0.10, 0.25] |           31.75  |
| B_eeg_ecg         | lgbm     |        702 | 8641.56 |         0.23  | [0.11, 0.39] |        1.158 | [0.59, 1.20]   | 0.254 | [0.14, 0.40] |           30.583 |
| B_eeg_ecg         | xgb      |        702 | 8641.56 |         0.236 | [0.11, 0.39] |        1.32  | [0.37, 0.80]   | 0.254 | [0.17, 0.44] |           32.833 |
| C_eeg_ecg_emg_acc | xgb      |        702 | 8641.56 |         0.268 | [0.11, 0.40] |        1.312 | [0.26, 0.63]   | 0.28  | [0.17, 0.48] |           30.9   |
| C_eeg_ecg_emg_acc | lgbm     |        702 | 8641.56 |         0.236 | [0.11, 0.40] |        1.008 | [0.35, 0.88]   | 0.276 | [0.16, 0.45] |           33.833 |
| rule              | rule_emg |        702 | 8641.56 |         0.078 | [0.04, 0.12] |        5.499 | [3.84, 7.20]   | 0.041 | [0.02, 0.06] |           41     |
| rule              | rule_eeg |        702 | 8641.56 |         0.085 | [0.05, 0.14] |        9.568 | [6.84, 12.85]  | 0.029 | [0.02, 0.04] |           33.25  |
| rule              | rule_acc |        702 | 8641.56 |         0.162 | [0.08, 0.27] |       21.327 | [14.80, 29.02] | 0.028 | [0.01, 0.05] |           25     |
| rule              | rule_hr  |        702 | 8641.56 |         0.162 | [0.08, 0.26] |       25.109 | [16.90, 34.69] | 0.024 | [0.01, 0.05] |           15.75  |

Offline pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17).
Peak RSS per job (GB; max over folds and seeds of the worker's peak working set): A_eeg/lgbm 2.30, A_eeg/xgb 2.30, B_eeg_ecg/lgbm 2.61, B_eeg_ecg/xgb 2.61, C_eeg_ecg_emg_acc/lgbm 3.15, C_eeg_ecg_emg_acc/xgb 3.15, rule/rule_acc 3.15, rule/rule_eeg 3.15, rule/rule_emg 3.15, rule/rule_hr 3.15. Overall max 3.15 GB.

**Outcome: v02g_grid_fix is the reference from here on (defect correction; descriptive comparison, not a §1 decision).**
- Same models, better deployable point: C xgb v02 → v02g sensitivity 0.256 → 0.268, FA/24h 3.00 → 1.31, F1 0.183 →
  0.276 (paired bootstrap ΔF1 +0.093 [+0.049, +0.140], ΔFA −1.69 [−1.99, −1.41], Δsens +0.012 [−0.010, +0.036];
  `compare_vs_v02_full_hours.csv` prints ACCEPT, but this is not a version decision). F1 rises in every learned cell
  (A xgb 0.144 → 0.175, B xgb 0.196 → 0.254, C lgbm 0.229 → 0.276) at lower FA/24h (0.87–1.32). FBTC 0.89–0.96.
- Same curve, as expected: at fixed FA/24h 1 / 2 / 5 / 10 / 3.00 every model's sensitivity and F1 are identical to
  v02's (`matched_fa_vs_v02_full_hours.md`), so the whole gain is the operating point the corrected search finds.
- τ is steadier but still repeat-dependent: C xgb per-fold τ SD 0.013–0.019 → 0.0045–0.0085; per-fold τ range over the
  15 repeats still 0.961–0.994, held-out FA/24h 0.01–8.95 (`report_by_fold_seed.csv`).
- The rules move to much higher thresholds under max inner F1 (rule_hr 60.5 → 25.1 FA/24h, sensitivity 0.366 → 0.162;
  rule_eeg 24.1 → 9.6, 0.217 → 0.085); their F1 stays 0.024–0.041 and their curves are unchanged. Rule FBTC at the
  deployable point is now below 0.80 (0.36–0.67), a property of single signals at a high threshold.
- Short seizures (< 15 s) at the deployable point: C xgb 0.055 → 0.035 (fewer alarms); the curve is unchanged.

## 2026-10-21 — how v11 is treated (Allen; logged before v11's results)

v11_causal is a correctness fix, not a §1 candidate: its §1 comparison with v02g is recorded as a descriptive offline−causal
difference, not as ACCEPT/REJECT (same as v02g; `docs/EVALUATION_PROTOCOL.md` amendment 2026-10-21). v02g stays the
offline reference (information content, comparison with published work); v11 is the causal reference (real-time alarm,
delay, deployability; demo and website). Both are frozen and tested in the single locked-set opening.

## v11_causal (2026-10-21 23:59) v02g + causal pipeline (one change: ictra_lib causal mode, causal features and alarm timestamps)

Dev folds, pooled over patients, patient-bootstrap 95% CI.

| set               | model    |   seizures |   hours |   sensitivity | sens_ci      |   fa_per_24h | fa_ci          |    f1 | f1_ci        |   median_delay_s |
|:------------------|:---------|-----------:|--------:|--------------:|:-------------|-------------:|:---------------|------:|:-------------|-----------------:|
| A_eeg             | lgbm     |        702 | 8641.55 |         0.154 | [0.10, 0.29] |        1.134 | [0.84, 1.91]   | 0.177 | [0.11, 0.28] |           37     |
| A_eeg             | xgb      |        702 | 8641.55 |         0.149 | [0.10, 0.28] |        1.131 | [0.88, 2.09]   | 0.171 | [0.11, 0.26] |           37.333 |
| B_eeg_ecg         | lgbm     |        702 | 8641.55 |         0.223 | [0.11, 0.35] |        1.032 | [0.47, 0.84]   | 0.258 | [0.15, 0.41] |           33.167 |
| B_eeg_ecg         | xgb      |        702 | 8641.55 |         0.193 | [0.10, 0.32] |        1.041 | [0.39, 0.80]   | 0.231 | [0.15, 0.39] |           34.167 |
| C_eeg_ecg_emg_acc | lgbm     |        702 | 8641.55 |         0.213 | [0.09, 0.35] |        0.669 | [0.25, 0.54]   | 0.275 | [0.15, 0.43] |           36.75  |
| C_eeg_ecg_emg_acc | xgb      |        702 | 8641.55 |         0.258 | [0.12, 0.39] |        1.345 | [0.40, 0.94]   | 0.268 | [0.17, 0.43] |           34.733 |
| rule              | rule_emg |        702 | 8641.55 |         0.077 | [0.04, 0.12] |        5.891 | [3.95, 7.89]   | 0.038 | [0.02, 0.06] |           38.375 |
| rule              | rule_eeg |        702 | 8641.55 |         0.09  | [0.05, 0.15] |       10.651 | [7.67, 14.15]  | 0.027 | [0.02, 0.04] |           34.5   |
| rule              | rule_acc |        702 | 8641.55 |         0.288 | [0.14, 0.45] |       41.098 | [30.23, 52.47] | 0.027 | [0.01, 0.05] |           17.625 |
| rule              | rule_hr  |        702 | 8641.55 |         0.204 | [0.11, 0.32] |       40.504 | [27.79, 54.85] | 0.019 | [0.01, 0.03] |           14     |

Offline pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17).
Peak RSS per job (GB; max over folds and seeds of the worker's peak working set): A_eeg/lgbm 2.10, A_eeg/xgb 2.10, B_eeg_ecg/lgbm 2.76, B_eeg_ecg/xgb 2.60, C_eeg_ecg_emg_acc/lgbm 3.19, C_eeg_ecg_emg_acc/xgb 3.19, rule/rule_acc 3.19, rule/rule_eeg 3.19, rule/rule_emg 3.19, rule/rule_hr 3.19. Overall max 3.19 GB.

**v11_causal outcome (descriptive, not a §1 decision; amendment 2026-10-21). v11 is the causal reference.**
- Cost of making the pipeline real-time, C xgb vs v02g (paired patient bootstrap, 96 patients): ΔF1 −0.011
  [−0.027, +0.007], Δsens −0.009 [−0.031, +0.012], ΔFA/24h +0.03 [−0.08, +0.14] (0.276 / 0.268 / 1.31 → 0.265 / 0.258 /
  1.34). `compare_vs_v02g_grid_fix.csv` prints REJECT because §1 asks for ΔF1 ≥ +0.01; per the amendment this is not a
  decision. Per fold ΔF1 −0.031 … +0.018. Other cells: A xgb −0.003, A lgbm −0.011, B lgbm +0.004, B xgb −0.023,
  C lgbm −0.000 (F1, single τ). At fixed FA/24h ≤ 1 / 2 / 5 / 10: C xgb sensitivity −0.018 / −0.013 / −0.016 / −0.010.
- Where the loss is (`delta_vs_v02g_grid_fix.md`, Δ sensitivity [95% CI], repeat SD 0.008–0.022 overall):
  - By group, focal non-motor shows no loss in any model with ECG (B lgbm 0.000, B xgb −0.017 [−0.044, +0.017], C xgb
    −0.001, C lgbm −0.005), so the loss does not concentrate where heart rate is the main signal; Allen's prediction
    for a heart-rate-detector loss is not supported.
  - By duration, the only CIs that exclude 0 are in models that use ECG and in seizures > 60 s: B xgb −0.089
    [−0.132, −0.022], C lgbm −0.053 [−0.083, −0.005] (C xgb −0.029 [−0.058, +0.011]); A (no ECG) > 60 s: −0.001 / +0.006.
    Long seizures are mostly motor, where the ECG carries movement artefact; a causal R-peak detector that is less
    robust to artefact than offline neurokit would act there. This is a pattern, not a test; most cells are inside
    the repeat noise.
  - Reading: making the pipeline causal costs about 0.01–0.02 F1 overall; between-model differences are mostly within
    repeat noise, except B xgb.
- Alarm delay, causal and timestamped at the window end (official metric: median over runs of the per-run median, seed
  mean): A xgb 37.3 s, B xgb 34.2 s, C xgb 34.7 s, C lgbm 36.8 s (v02g, window-start timestamps: 30.9–33.8 s).
  Paired on seizures both A and C xgb detect (v11, same seed, 119 seizures): C alarms earlier in 52 %, same in 15 %;
  median paired difference −2.0 s [−4.0, 0.0]; lgbm 0.0 s [−2.0, 0.0]. Fusion is earlier by about one 2-s hop at
  most; the evidence is weak.

## 2026-10-22 — notes from the demo videos (no version run)

- **Offline-only vs causal-only detections (C xgb, majority over the 15 repeats):** 26 seizures are detected offline
  (v02g) but not causally (v11), 16 the other way; net −10 of 702 (−1.4 %), consistent with Δsens −0.009. Concrete
  example: sub-080 run-19 (focal motor, 43 s) is detected by the v02g fold-4 model (delay 10 s) and missed by the v11
  fold-4 model (`docs/DEMO_RUNS.txt`).
- **Heart-rate artefact during convulsions (observed while building the demo, `src/27_demo_video.py`):** in the two FBTC
  demo seizures the ECG-derived heart rate jumps by up to 96–111 bpm from one second to the next (values 32–181 bpm),
  i.e. the ECG is dominated by movement artefact. This fits the v11 breakdown (loss only in models with ECG and only in
  seizures > 60 s). Hypothesis, not tested.
- **Candidate next step (not run; Allen 2026-10-22):** give the model a heart-rate quality flag (e.g. share of seconds
  with a jump > 25 bpm in the window) so it can discount heart rate when the ECG is artefact, rather than replacing the
  R-peak detector. One change, small, mechanism stated in advance.
- **"27 %" on the website/script:** 0.268 is the offline (v02g) curve point at FA/24h ≤ 1; the causal (v11) value is 0.250;
  v11's deployable point is 0.258 at 1.34 FA/24h. All are pooled over the 702 seizures of the 96 dev patients
  (detected seizures / all seizures), not per patient or per recording.

## 2026-10-26 — Task 10: freeze and pre-registration

- Frozen (`src/06_freeze.py`, commit 168dbb9): v02g (offline) and v11 (causal) lineages; A, B, C xgboost as 3-seed
  averages trained on all 96 dev patients, plus the four rules; τ from one inner split (19 inner-validation patients),
  corrected grid, max pooled inner F1, 3 of 5. C xgb τ: v02g 0.989742, v11 0.984999.
- Reproduction check before hashing: τ re-selected for four logged dev jobs (v02g / v11 × A / C xgb, fold 0, seed 0)
  with the freeze's helpers — identical to the logged τ in all four (`results/freeze/freeze_tau_check.log`).
- `docs/PREREGISTRATION.md` (commit 571901f), SHA-256 bc5048c88ae3524348215e4c62801ec66fedbe5c4e4ac616bcbb3c3944fedca6:
  per-patient aggregation (primary), H1–H4 analysis plan for both lineages, τ uncertainty, deviations (xgboost only;
  external sub-models replaced by frozen rules; §3.1 items not implemented). From here on nothing in `models/` or the
  listed code changes; the locked set is opened once (Task 11).

## 2026-10-26 12:58 -0700 — opening record: locked internal test (Task 11), written before any sub-097..125 file is downloaded

- Pre-registration: `docs/PREREGISTRATION.md`, SHA-256 bc5048c88ae3524348215e4c62801ec66fedbe5c4e4ac616bcbb3c3944fedca6
  (commit 571901f).
- Frozen metadata: `models/ictra_v02g.meta.json` SHA-256 fad9ac800cc178aac3cd7a68982d705a7b6f052b6c4a1f406e1f7fad09ee0909;
  `models/ictra_v11.meta.json` SHA-256 bf06b3a8c0ec2bb28ac470bd7940cbf28f673a1ea7060b3d7fab79bb2e309a8c (commit 168dbb9).
- Analysis script: `src/07_locked_test.py`, commit ce62feb, SHA-256 67c0cf8eaeb0fc5721781993d9d0a2e37d3b84e484cd365e590400241c986b82
  (git blob aa497faf2c144389871a7758a9cae70e639bdf0b). Dry run on the development patients ended DONE (2026-10-26 12:24–12:51
  local); its numbers are in-sample and are not results.
- Steps from here, each once: download sub-097..125 (`01_download_seizeit2.py --runs all --split locked_test --go`), check that
  only these 29 patients were added; `03_features.py --allow-locked` and `03_features.py --allow-locked --causal`; one run of
  `07_locked_test.py` → `results/locked_test/`. No threshold, feature, code or analysis changes after the opening; an error
  stops the run, is fixed and logged here, and the script is then run once.
- Once 07 has run, sub-097..125 are development data and cannot be used for any independent test.

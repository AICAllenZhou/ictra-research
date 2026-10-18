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

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

# ICTRA evaluation protocol: training and testing kept apart

Written 2026-10-17, before any signal file was opened. This is the fixed rule set, like SlopeAI's `ITERATION_LOG.md`
protocol section. Nothing here changes after the first model is trained, except through a dated amendment at the bottom.

## 0. Why three tiers

The 2025 SzCORE seizure-detection challenge let teams train on any public data and scored them on a hidden hospital
dataset. Teams' self-reported F1 of 80–90% fell to 2–43% on the hidden data. Cross-dataset tests of CHB-MIT ↔ TUSZ models
drop from AUC 0.86–0.90 to 0.62–0.64. In-dataset cross-validation is therefore not evidence. ICTRA reports three tiers,
each weaker than the next, and the headline number is always the strongest tier available.

| Tier | Data | What it shows | When it is run |
|---|---|---|---|
| 1 Development | SeizeIT2 sub-001..096, 5-fold GroupKFold by patient | choosing between versions | repeatedly |
| 2 Locked internal test | SeizeIT2 sub-097..125 | same device, new patients | **once**, after pre-registration |
| 3 External tests | Billeci (EEG+ECG), Siena (EEG+ECG), TUSZ (EEG+EKG), MSG Seer (EMG+ACC), szdb (ECG), OSDB (ACC, home) | new device, new hospital, new country, new population | **once each**, after pre-registration |

SeizeIT2's authors call sub-097..125 a "validation" set, so some published models were selected on it. For ICTRA it is a
locked test only because we never tune on it; tier 3 is still the stronger claim.

## 1. Tier 1: development on sub-001..096

- `GroupKFold(n_splits=5, groups=subject)`. Every run of a patient stays in the same fold (`sub-xxx` has up to ~60 runs).
- Nested: thresholds and post-processing are tuned on the 4 training folds only (inner split), never on the held-out fold.
- All seizure-free hours are kept in the validation folds so false alarms per 24 h are real. Training may undersample
  background windows to about 5:1; validation never does.
- Windows that straddle a seizure boundary are labelled by overlap fraction for training and excluded from window-level
  validation scoring (event scoring handles the edges).
- `impd` epochs (464 impedance-check epochs exist in the events files) are masked out of training and out of the
  false-alarm denominator.
- Repeats: 3 seeds × 5 background-sampling draws. Score = mean over folds of the event-based metrics below.
- One change per version, logged in `results/ITERATION_LOG.md`. Accept a change only if **all** hold:
  event F1 +0.01 or more; patient-bootstrap P(improvement > 0) ≥ 0.90; sensitivity does not fall by more than 0.02;
  FA/24h does not rise by more than 0.5.

## 2. Tier 2: locked internal test, sub-097..125

- Before downloading or opening any sub-097..125 file: write `docs/PREREGISTRATION.md` with the frozen model files,
  threshold, post-processing, metrics, subgroup list and hypotheses; commit; store the SHA-256 in
  `docs/PREREGISTRATION.sha256`.
- Run once. No re-runs, no second checkpoint, no "just checking".
- Report every hypothesis, including failed ones. After it is opened, the set becomes development data and the report
  says so (SlopeAI rule).
- Contents: 29 patients, 2,912 h, 181 seizures (22 FBTC from 9 patients, 54 focal motor, 75 non-motor, 30 unknown).
  15 of these seizures have no ECG/EMG file: report with and without them.

## 3. Tier 3: external tests on data never used for training

The model is trained only on SeizeIT2. External sets differ in device, electrode positions, sampling rate, hospital
and country, exactly like SlopeAI training on China and Latin America and testing on British Columbia.

### 3.1 Harmonisation (fixed in code, applied identically to train and test)
- Resample every signal to 256 Hz. Band-pass EEG 1–25 Hz (SeizeIT2 baseline choice), ECG 0.5–40 Hz, EMG 20–100 Hz
  (needs ≥ 250 Hz), 50/60 Hz notch chosen by recording country.
- Behind-the-ear proxy from 10-20 scalp EEG: bipolar **T7–P7** and **T8–P8** (older names T3–T5, T4–T6), plus a
  cross-head pair for the "left"/"right" montage models. This is an imperfect proxy: SeizeIT2's upper electrode sits near
  T7/T8 and the lower one near P9/P10. State it as a limitation.
- Per-recording robust scaling (5th–95th percentile) so amplifier gain differences cancel.
- Left/right flip augmentation during training so the model does not depend on the seizure side.
- ECG goes through R-peak detection → heart rate and HRV features, which are device-independent by construction.
- EMG goes to an RMS envelope and >20 Hz power; ACC to magnitude. These are also unit-insensitive.

### 3.2 Which model is tested on which external set

| External set | Signals present | Model tested | Headline metric |
|---|---|---|---|
| Billeci Zenodo 1415495 (15 pts, 38 sz) | EEG + ECG | **Model B** (EEG+ECG) and Model A (EEG) | event sensitivity, FA/24h |
| Siena PhysioNet (14 pts, 47 sz) | EEG + ECG | Model B, Model A | same; check patient overlap with Billeci (same hospital) |
| TUSZ v2.0.3 eval split (registration) | EEG + EKG | Model A, Model B (files with EKG) | same, by seizure type incl. TCSZ/GNSZ |
| MSG Seer Sensor Dot (5 pts) | EMG + ACC | EMG+ACC sub-model only | sensitivity on motor seizures, FA/24h |
| PhysioNet szdb (5 pts, 11 sz) | ECG | ECG sub-model / HR rule | sanity only; too small for claims |
| Mayo MSG Zenodo 17380899 (11 pts, ~337 days each) | HR 1 Hz | HR-jump rule only | FA/24h of an HR rule over months at home |
| OSDB (49 users, 146 sz + 348 false alarms, home) | wrist ACC + HR | ACC sub-model | share of real false alarms rejected |

No public dataset has all four signals, so **Model C (EEG+ECG+EMG+ACC) can only be fully tested on SeizeIT2 tier 2**.
The external tests show that each branch generalises; the fusion claim rests on tier 2. Say this plainly.

### 3.3 Missing modalities
Train with modality dropout (zero a whole signal with p ≈ 0.25 per window) and use late fusion: one sub-model per
signal, probabilities averaged over the signals that are present, weights renormalised. Then any external set can be
scored by the same frozen model. Also report the single-branch result as the headline external number.

### 3.4 Also pre-registered
Each external test gets its own section in `docs/PREREGISTRATION.md` and is opened once. Negative-control sets
(UE4W daily life, EPIStress) count false alarms only and are never used as training negatives.

## 4. Metrics (SzCORE event scoring)

- A detection counts if it overlaps a reference seizure, allowing 30 s before onset and 60 s after offset. Alarms less
  than 90 s apart are merged; events longer than 5 min are split.
- Report: event sensitivity, precision, F1, **false alarms per 24 h**, detection delay (s from onset), per patient
  then averaged, with patient-level bootstrap 95% CIs (1,000 resamples). Paired bootstrap for A vs B vs C.
- Always next to the models: the single-input rules (HR +30% for 10 s; EMG RMS threshold; EEG line length threshold;
  ACC magnitude threshold).
- Never accuracy, never specificity: seizures are under 1% of hours.
- Breakdown by seizure group: FBTC, focal motor, focal non-motor, unknown. By montage: left, right, generalized.

## 5. Statistical honesty
- FA/24h differences need hours, not seizures: at 80% power, 1.0 vs 0.5 FA/24h needs about 1,100 h per arm. Tier 2 has
  2,912 h, so only a paired comparison on the same hours (A vs C on identical recordings) can detect it.
- 22 FBTC test seizures: 22/22 gives a 95% CI of 85–100%; 18/22 gives 62–93%. FBTC results are exploratory with CI, never
  a superiority claim.
- Independent re-computation of every number before it enters the report or the website.

## 6. Leakage checklist (ticked before every table)
- [ ] No patient appears on both sides of any split (grouped by `sub-`, not by run file).
- [ ] Threshold and post-processing were fixed on training folds / pre-registration only.
- [ ] Test sets kept their natural class ratio.
- [ ] `impd` epochs masked.
- [ ] Model version was chosen before the locked or external set was opened.
- [ ] Billeci and Siena patient overlap checked before both are counted as independent.

## Amendments

### 2026-10-17 — all reported numbers are offline evaluations

Found while documenting the real-data fixes (`docs/DATA_NOTES.md` §3b). The signal path in `src/ictra_lib.py` is not
causal:
- `repair_dropouts` looks ahead 1–2 samples and up to 1 s, with a noise threshold taken over each ~1-h chunk;
- `acc_to_g` divides by the median |a| of the whole recording;
- `bandpass` / `notch` are zero-phase (`sosfiltfilt` / `filtfilt`).

The relative features, the ECG baseline and the k-of-n post-processing are causal.

Therefore **every number from tiers 1, 2 and 3 is an offline evaluation**: the same code path runs on training and test
data, so the comparisons are valid, but they do not show what a real-time device would achieve. Every results table,
the report, the website and the demo say "offline pipeline". A causal pipeline is version candidate **v11**
(`docs/TASKS.md` Task 8). It must report its delta against the accepted offline version on the same dev folds
(Δsensitivity, ΔFA/24h, ΔF1, Δdelay, with paired-bootstrap CIs). Until v11 exists, the hardware replay demo labels
itself "offline pipeline".

### 2026-10-17 (evening) — v03b reference and sensitivity floor (Allen; logged before v03b started)

- **Reference.** v03b is compared under §1 with **v02_full_hours**, on the same full-hours data (`--runs all`, 3 seeds; C
  xgb 3 × 5 draws), not with v01. v01's FA/24h was measured on seizure-containing runs only and understates real-world FA.
- **Sensitivity floor.** Per (set, model): S_ref = v02's matched-FA sensitivity, i.e. the best held-out sensitivity on
  v02's operating curve at FA/24h ≤ FA_ref, where FA_ref is v02's deployable FA/24h for that cell (seed mean).
  - τ maximises pooled inner event F1 subject to inner sensitivity ≥ S_ref − 0.02.
  - If that is infeasible: maximise inner sensitivity subject to inner FA/24h ≤ FA_ref. Otherwise: the lowest-FA
    candidate.
- **What changes relative to v02.** Only the τ-selection procedure: τ comes from pooled out-of-fold predictions of a
  5-fold inner leave-patients-out CV with the constrained objective, instead of an F1-max τ on one 80/20 inner split.
  Features, models, seeds, draws and post-processing are unchanged.
- **Reporting.** Next to the §1 decision: sensitivity and F1 at fixed FA/24h budgets of 1, 2, 5 and 10 (operating sweep),
  and the per-fold τ and per-seed sensitivity (median, range).

### 2026-10-19 — defect correction: τ candidate grid; v02g replaces v02 as the reference (Allen; logged before the run)

- **Defect.** `04_train_ablation.tune_threshold` tried 40 quantiles of the inner predictions, linear in q from 0.90 to
  0.9999. The top two candidates (q = 0.99734 and 0.9999) differ 27× in alarm rate with nothing between. The v03b
  quantile check found every learned-model job on the same candidate (q = 0.99734), and v02's held-out exceedance at τ
  splits into two groups matching the top grid points. The deployable τ of v01–v03b was therefore set by grid
  resolution, not by the objective.
- **Correction.** Candidates are log-spaced in the tail probability, which the false-alarm rate is proportional to:
  `q = 1 − logspace(−1, −5, 81)` (tail 0.1 → 1e-5, 20 per decade). The upper end keeps the old grid's 0.90 start (rules
  picked tails 0.008–0.036 in v03b). Flag `--thr-grid tail_log` (default); `legacy` reproduces v01–v03b only. Nothing else
  changes: features, models, seeds, draws, inner split, objective (max inner F1), k-of-n, scoring.
- **This is a defect correction, not a version.** It is not compared with v02 under §1: v02 is pinned by the same grid,
  so a §1 difference would be partly the grid. Instead v02's exact configuration is re-run with the corrected grid as
  **v02g_grid_fix**, which **replaces v02_full_hours as the reference** for every later version (§1 comparisons from
  here on are against v02g). v02 vs v02g is reported descriptively: the single-τ table side by side (does v02's own
  deployable point improve?), and fixed FA/24h budgets 1 / 2 / 5 / 10 (expected the same curve: same models).
- **Earlier decisions stand as made.** v03 and v03b were rejected under the grid they ran with; their log entries keep
  their numbers. Their fixed-FA comparisons (same curve) do not depend on the grid (05_operating_points.py already
  sweeps a log-spaced tail grid).

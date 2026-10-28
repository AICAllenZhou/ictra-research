# ICTRA task list for Claude Code (in order; each task has a definition of done)

Rules in `CLAUDE.md` apply to every task. Tick a task only when its "Done when" holds. If a task reveals a problem in an
earlier script, fix the script, re-run from that point, and note it in `results/ITERATION_LOG.md`.

---

## Stage 1 — data in, baselines out (target: Oct 24)

### Task 1. Download the training seizure runs
```
python src/00_seizeit2_catalog.py
python src/01_download_seizeit2.py --runs seizure --split train          # dry run, check ~25 GB and free disk
python src/01_download_seizeit2.py --runs seizure --split train --go
```
**Done when:** the script ends with `failed: 0`; `data/raw/ds005873/` holds 1,163 .edf files; a spot check of 5 random
files with `mne.io.read_raw_edf` shows 256 Hz (25 Hz for mov), the expected channel counts (EEG 2, ECG 1, EMG 1) and
durations matching `data/catalog/seizeit2_runs.csv`. Record the real channel names of each modality in
`docs/DATA_NOTES.md` (new file), including the exact accelerometer channel names, and the physical units from the EDF header.

### Task 2. Verify `ictra_lib` assumptions against real files
Open 3 real runs (one per montage: left, right, generalized). Check and write to `docs/DATA_NOTES.md`:
- EDF units: are EEG/ECG/EMG stored in µV or V (MNE returns V; `03_features.py` multiplies by 1e6)? Typical EEG std
  should be 5–50 µV, ECG R-peak amplitude 0.3–3 mV, EMG at rest < 20 µV. Fix the scale if wrong.
- ACC channel names/units (`03_features.py` picks channels containing "acc"; adjust if the names differ).
- Does `events.tsv` onset align with a visible change in the behind-the-ear EEG for an FBTC seizure? Plot 3 examples
  (±120 s) to `results/data_checks/`.
- The impedance-check (`impd`) epochs: confirm they look artefactual.
**Done when:** `docs/DATA_NOTES.md` exists with these answers and any fixes are committed and re-smoke-tested.

### Task 3. Feature extraction
```
python src/03_features.py --workers <n_cores>
```
**Done when:** `data/features/` has one parquet per downloaded run (1,163 for stage 1), `_failed.txt` is empty or every
failure is explained in `DATA_NOTES.md`, and a summary (`results/data_checks/feature_summary.md`) reports: total
windows, seizure windows, edge windows, impd windows, share of windows with each modality present, and the per-feature
NaN rate. Flag any feature with > 5% NaN outside missing-modality runs.

### Task 4. Relationship study (Exp R)
```
python src/02_relationship_study.py --plot
```
**Done when:** `results/expR/summary_by_group.csv` and the four timeline figures exist, and `results/expR/FINDINGS.md`
(new) states in plain words, per seizure group: which signals respond (share), the median lag of heart rate and muscle
after EEG onset with IQR, the share of seizures with HR rise ≥ 20 and ≥ 50 bpm, and what this implies for which signal to
trust for which seizure type. Compare with the literature numbers already in `docs/PLAN.md` (ictal tachycardia in ~56–80%
of focal seizures; EMG/ACC only for motor seizures). Also compute the lead–lag with the alternative definition
(z > 2 for 5 s) and report whether the conclusions change (robustness).

### Task 5. Baseline ablation v01
```
python src/04_train_ablation.py --version v01_baseline --note "default features, seizure runs only" --models xgb lgbm logit rf --rules --seeds 0 1 2 --workers <n_cores>
python src/04_train_ablation.py --version v01_baseline --sets ecg_only emg_only acc_only emg_acc --models xgb --seeds 0 --workers <n_cores>
```
**Done when:** `results/dev/v01_baseline/summary.csv` exists; `ITERATION_LOG.md` has the v01 table; a figure
`results/dev/v01_baseline/fig_ablation.png` shows sensitivity vs FA/24h for A, B, C and the four rules (one axis, per
seizure group as panels); and `results/dev/v01_baseline/NOTES.md` answers: does C have fewer FA/24h than A at equal or
higher sensitivity? Which model family is best? Which rule is strongest? Any sign of leakage (checklist §6)?
**Caveat to write down:** stage-1 data contains only seizure-containing runs (~3,000 h of the 8,700 dev hours), so
FA/24h is measured on a harder-than-average subset; the full-hours number comes in Task 7.

### Task 6. Add an operating-point sweep to the trainer
Extend `04_train_ablation.py` (or add `05_operating_points.py`) so that from the saved window probabilities it reports,
per (set, model): sensitivity at FA/24h budgets of 0.5, 1, 2 and 5, and FA/24h at sensitivities of 80, 90 and 95%, pooled
over folds, by seizure group. Thresholds for this sweep are chosen on the held-out fold (so label it clearly as an
operating curve, not a deployable threshold).
**Done when:** `results/dev/<version>/operating_points.csv` and a sensitivity–FA curve figure are produced for v01.

## Stage 2 — full hours, iteration (target: Nov 7)

### Task 7. Download the seizure-free dev runs and re-run
```
python src/01_download_seizeit2.py --runs all --split train --go      # adds ~70 GB; check disk first
python src/03_features.py --workers <n_cores>
python src/04_train_ablation.py --version v02_full_hours --note "all dev hours" --models xgb lgbm --rules --seeds 0 1 2
```
**Done when:** v02 table is in the log with FA/24h on all 8,715 dev hours, and NOTES.md compares v01 vs v02.
From here on v02 is the reference every later version is compared with.

### Task 8. Self-iteration (one change per version, ≥ 6 versions)

**Priority order (Allen, 2026-10-19; supersedes the order below).** Done so far: v01 baseline, v02 full hours (reference),
v03 REJECT, v03b REJECT. Then:
1. **Must — τ grid defect correction + baseline re-run** (`v02g_grid_fix`, EVALUATION_PROTOCOL amendment 2026-10-19).
   Not a version: v02's configuration with the corrected τ candidate grid; becomes the reference for everything after.
   Report v02 vs v02g side by side (does v02's own deployable point improve without changing any model?) and at fixed
   FA/24h 1 / 2 / 5 / 10. Running since 2026-10-19 16:35.
2. **Must — v11 causal pipeline** (description below). Not an improvement: the precondition for the real-time claim
   ("wakes a carer within seconds") on the website and in the demo. Current features use future samples, so the model
   as evaluated cannot run live. Same priority as 1.
3. **If time — v04 relationship features from Exp R** (EEG changes first, HR later: the lag becomes a model input). The
   only version that builds on our own Exp R finding; supports Originality. Not needed for any other conclusion.
4. **Cut / after the freeze — v07 (2-s windows), v08 (k-of-n grid).** Expected ΔF1 about 0.01–0.03 each; changes no
   conclusion. Not run before the freeze.
Freeze (Task 10) as soon as 1, 2 (and optionally 3) are done, earlier than Nov 14 if possible; the remaining time goes to
the hardware demo (Nov 10), the 6-minute video and rehearsal. Submission Nov 29.
Every long run is launched inside `python src/keep_awake.py -- bash src/NN_run_<version>.sh` (blocks idle sleep only;
not a lid close or power-button press).

Candidates, original list (order superseded above); each is a flag or a small function in `ictra_lib.py` / `04_train_ablation.py`:
- v03 relationship features from Exp R (e.g. HR rise within 20 s of an EEG line-length jump; EMG burst following EEG change)
- v04 montage-aware EEG features (left / right / generalized handled by channel ordering or a montage indicator)
- v05 longer context (rolling 20 s mean/max of key features; previous-window deltas)
- v06 late fusion: separate xgb per modality, probabilities averaged with weights fit on the inner split, missing-modality renormalisation
- v07 window length 2 s / 8 s
- v08 post-processing k-of-n grid (2/3, 3/5, 4/6, 5/8) tuned on inner split
- v09 per-patient baseline normalisation (first 10 min of each recording)
- v10 class ratio 1:3 vs 1:10 and class weights
- v11 causal pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17): causal filters (`sosfilt`, group delay reported),
  causal dropout threshold (running median of |diff| over past minutes), causal ACC calibration (first-10-min or running
  median |a|). Report the offline-vs-causal delta (Δsens, ΔFA/24h, ΔF1, Δdelay, paired bootstrap) whether or not it is
  accepted; the demo replay uses v11 once it exists and says "offline pipeline" until then.
Implement the paired patient bootstrap comparison (`compare` mode, like SlopeAI's `09_iterate.py compare`) that prints
Δsensitivity, ΔFA/24h, ΔF1 with CI and P(improve) and the ACCEPT/REJECT decision by the §1 rule.
**Done when:** each version has a log entry with the decision; the accepted chain is stated at the top of the log.

### Task 9. External-dataset adapters (no labels opened yet)
Write `src/08_external_adapters.py` with one loader per external set that returns (signals dict, fs, events) in the
same shape `03_features.py` expects, using `ictra_lib.bte_proxy_from_scalp` for scalp EEG:
- Billeci Zenodo 1415495 (EDF + per-recording .txt with seizure start/end)
- Siena PhysioNet (PNxx EDF + `Seizures-list-PNxx.txt`)
- PhysioNet szdb (WFDB ECG + `times.seize`)
- My Seizure Gauge Seer export (after registration), Open Seizure Database JSON (after access)
Only the file-format code is written and tested on ONE recording's header per set; **no seizure labels are scored**.
**Done when:** each adapter loads a header/first minute and prints channel names, fs, duration; documented in DATA_NOTES.md.

## Stage 3 — pre-registration, locked test, external tests (target: Nov 14)

### Task 10. Freeze and pre-register
Write `src/06_freeze.py`: train the accepted version on all 96 dev patients (3 seeds, averaged), save models + threshold +
post-processing to `models/ictra_v<N>_seed{0,1,2}.json` and `models/ictra_v<N>.meta.json`. Then write
`docs/PREREGISTRATION.md` (template: `../SlopeAI/results/v4_future/PREREGISTRATION.md`): frozen files, hypotheses
H1–H4 exactly as defined in `docs/PREREGISTRATION_DRAFT.md` (the one place they are defined), metrics, subgroup list, external sets and which model is
tested on each. `sha256sum docs/PREREGISTRATION.md > docs/PREREGISTRATION.sha256`. Commit.
**Done when:** the hash file exists and nothing in `models/` or the accepted version's code changes afterwards.

### Task 11. Locked internal test (once)
```
python src/01_download_seizeit2.py --runs all --split locked_test --go
python src/03_features.py --allow-locked
python src/07_locked_test.py           # new: frozen model, frozen threshold, SzCORE scoring, by group, bootstrap CIs
```
**Done when:** `results/locked_test/RESULTS.md` reports every pre-registered hypothesis (held / failed), with and without
the 15 seizures lacking ECG/EMG, and states that the set is now development data.

### Task 12. External tests (once each)
`python src/08_external_test.py --dataset billeci|siena|szdb|...` using the frozen models per `EVALUATION_PROTOCOL.md` §3.2.
Check Billeci–Siena patient overlap first (demographics / recording dates) and report it.
**Done when:** `results/external/<dataset>/RESULTS.md` per set, plus `results/external/SUMMARY.md` with the tier-3 table.

### Task 13. Independent review
Re-compute every number in `results/**/RESULTS.md` and `ITERATION_LOG.md` from the saved predictions with a separate
script (`src/09_independent_check.py`), as `../SlopeAI/src/05_review_checks.py` does. List mismatches.
**Done when:** `results/INDEPENDENT_REVIEW.md` says every number reproduced, or lists what did not and why.

## Stage 4 — hardware and demo (parallel track, hardware owner)
- Fix the stuck-at-ADC-limit problem in `biosignal-studio` (58–60% of samples at 0/1023); target < 1%.
- Multi-channel acquisition at 256 Hz; EMG filter 20–120 Hz + 60 Hz notch; add an accelerometer.
- `replay` mode: feed a SeizeIT2 EDF through `ictra_lib` features + the frozen model + alert, for the demo video. The replay screen shows "offline pipeline" until v11 (causal) exists.
- Healthy-volunteer nuisance trials (chewing, walking, stairs, typing, sleep posture), consent recorded; report FA per hour
  on the team's own board. Clearly labelled simulation only (fist clench, arm shake) for EMG/ACC response.

---

## Note on Carina's ECG audit (2026-10-17)

Carina audited the SeizeIT2 ECG files for sub-001..042 on her Mac (`dataset_audit.md`, `ecg_inventory.csv`,
`annotation_summary.csv`). Verified against `data/catalog/`: her train-side numbers match exactly (277 recordings,
2,906.22 h; 95 vs 96 seizures). Her findings that are now built into the pipeline:
- **Flat / saturated stretches** (9 files with ≥ 1 h flat, e.g. sub-021 run-08, 8.05 h at digital −128): `ictra_lib.flat_windows`
  marks them; `03_features.py` writes `flat_eeg/ecg/emg` and blanks the features; `04_train_ablation.py` masks `flat_eeg`
  like `impd` (no alarm counted, not used as background).
- **Boundary windows excluded** from training and window scoring (our `edge` column).
- **Impedance checks excluded** (`impd`).
Her proposals that are **not** adopted, with reasons:
- Split sub-001..030 / sub-031..042: the "test" side is dominated by two patients (sub-039: 75 seizures, sub-035: 55 of 187),
  has 2 FBTC, and its seizure types differ from the training side (she noticed this herself). Use `data/catalog/splits.json`;
  her 12 subjects are already spread over the 5 dev folds. The locked test stays sub-097..125, which she correctly does not have.
- 10-s non-overlapping windows labelled only when fully inside a seizure: drops the 44 seizures shorter than 10 s and most
  of every seizure's first/last 10 s. Keep 4-s windows, 2-s hop, ≥ 50 % overlap.
- Her copy is missing 9 truncated files + sub-039 run-06: 7 recordings, ~135 h and ~58 seizures (mostly sub-039). Re-download
  with `01_download_seizeit2.py` (SHA-256 verified) rather than skipping them.
Action for Carina: run `01_download_seizeit2.py --runs seizure --split train --modalities eeg ecg emg mov --subjects sub-001 … sub-042`
to add the other modalities (~6 GB), then own Task 4 (relationship study) and the `ecg_only` / `rule_hr` rows of Task 5.

---

## Task 14. "Good enough for WAICY" decision and disk cleanup (Allen signs)

Raw EDFs (95–126 GB) may be deleted only after the model is frozen and every test has run, because any later feature
change (window length, new relationship features, montage handling) needs the raw files. `src/99_cleanup_raw.py`
enforces six gates and refuses otherwise. Gate G5 is a human decision: Allen writes `docs/DONE_FOR_WAICY.md` containing
the word "signed", his name and the date, after checking this list:

- [ ] Tier 2 (locked SeizeIT2 test) run once; every pre-registered hypothesis reported, held or failed.
- [ ] Tier 3 external tests run once each on at least Billeci, Siena and szdb (Seer / OSDB if access arrived).
- [ ] The ablation shows, with patient-bootstrap CIs, how FA/24h changes A → B → C at matched sensitivity, by seizure group,
      with the single-input rules in the same table. (A negative or mixed result is still "good enough" if it is reported honestly.)
- [ ] Exp R findings written up: which signals respond, in what order, by seizure type, including the share of seizures
      with no behind-the-ear EEG change.
- [ ] `results/INDEPENDENT_REVIEW.md` reproduces every number in the report.
- [ ] The demo replay works from the kept demo runs (`docs/DEMO_RUNS.txt`: 3–5 run stems, e.g. one FBTC, one focal motor,
      one non-motor, one nuisance-rich seizure-free run) and the frozen model.
- [ ] `models/` and `results/**/preds` are backed up (they are small) somewhere other than this disk.

Then: `python src/99_cleanup_raw.py --keep-demo` (dry run) → `--keep-demo --go`.
Disk plan without deleting anything: 149 GB free − 95 GB dev − 31 GB locked test ≈ 23 GB spare, so deletion is optional
if nothing else lands on the drive; an external USB drive for `data/raw/` is the simplest alternative.

---

## Task 15. Results export for the website (`src/95_export_results.py`)

Implements `docs/DELIVERABLES_SPEC.md`. `python src/95_export_results.py --stage dev --version v01_baseline` writes
`results/export/ictra_results_<YYYYMMDD>_dev_v01/` and zips it; `--stage locked_test` / `--stage external` only work after
`docs/PREREGISTRATION.sha256` exists (same guard as 03). Every number in `metrics/` must be re-derivable from `predictions*/`
with `src/09_independent_check.py` (Task 13), which should also run on the zip before it is sent.
**Done when:** the zip opens, `README.md` lists all deviations from the checklist (spec §Deviations), per-second files have
no gaps (`t_sec` contiguous per run), and `events_*.csv` counts match `data/catalog/seizeit2_seizures.csv` for the exported split.

## Task 12b. Third-party blind evaluation (added 2026-10-18; after Task 10 freeze + hash, never before)

Two external evaluators can test ICTRA on data we have never seen. Both count as Tier 3; each is submitted ONCE.
1. **SzCORE / epilepsybenchmarks.com (EPFL)** — continuously open blind benchmark on private 19-channel scalp EEG.
   Submit Model A (EEG-only) through the T7–P7 / T8–P8 bte proxy as a Docker image in the SzCORE submission format
   (input: EDF 10-20 scalp; output: TSV events; scoring library = the one `ictra_lib.score_events` follows).
   Build the image in `src/08_external_test.py --export-szcore` together with Task 15; test it locally on one
   Siena recording before submitting.
2. **KU Leuven (SeizeIT2 authors)** — request a one-time run on their non-public SeizeIT2 hold-out (same sensors).
   Package: Docker image of the frozen Model C (bte-EEG + ECG + EMG + ACC input), `docs/PREREGISTRATION.md` + its
   SHA-256, locked-test RESULTS.md, and a one-page README. Allen sends the e-mail after Task 11; Claude drafts it.
**Done when:** the SzCORE result (or submission receipt) and the KU Leuven reply (or no reply by 25 Nov) are recorded in
`results/external/SUMMARY.md`, with the numbers exactly as returned.

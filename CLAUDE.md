# ICTRA — working rules for anyone (human or AI) changing this project

ICTRA is Allen's WAICY 2026 project: detect epileptic seizures from EEG + ECG + EMG (+ accelerometer) with an AI model,
following the same procedure as the sibling project `../SlopeAI` (relationship study → single-input baselines →
A/B/C ablation → self-iteration under a fixed rule → pre-registered locked test → external tests). Read, in this order:

1. `README.md` — folder map, reproduce commands
2. `docs/PLAN.md` — the idea, dataset decision, step-by-step plan, honest limits
3. `docs/EVALUATION_PROTOCOL.md` — the fixed rules for train/test separation, metrics, accept/reject. **Binding.**
4. `docs/TASKS.md` — what to do next, in order, with the definition of done for each task
5. `docs/TEAM_PLAN.md` — who owns which signal branch and the fusion step
6. `docs/DASHBOARD_SPEC.md` — the ICTRA Lab page: live progress, results, and the WAICY figures (one data contract, scripts write JSON, page only reads)
7. `results/ITERATION_LOG.md` — what has been tried

## Rules that are never broken

- **Training data is SeizeIT2 development patients only** (`data/catalog/splits.json` → `dev_folds`, sub-001..096).
  Never read, download, feature-extract or look at sub-097..125 (`locked_test`) or any external dataset's labels until
  `docs/PREREGISTRATION.md` exists and its SHA-256 is in `docs/PREREGISTRATION.sha256`. `03_features.py` refuses
  `--allow-locked` without that file; keep that guard, and add the same guard to every new script that touches test data.
- **Split by patient, never by file or window.** A patient's runs all go to the same fold. Thresholds and post-processing
  are tuned on training folds (inner split) only. Test sets keep their natural class ratio.
- **One signal-processing code path.** Everything that must be identical between training and any test data (resampling,
  filters, windowing, features, post-processing, scoring) lives in `src/ictra_lib.py`. Do not copy a filter or feature
  into another script; import it. Changing `ictra_lib.py` means re-running `03_features.py` for every run and bumping the
  version name.
- **No external negatives.** Never add EEGMMIDB, ds005505, Bonn or any other dataset as "healthy" training negatives;
  negatives are SeizeIT2's own seizure-free hours. (Negative-control sets may be used to COUNT false alarms, never to train.)
- **Metrics:** event-based sensitivity, precision, F1, false alarms per 24 h and detection delay (SzCORE rules in
  `ictra_lib.score_events`), per patient then pooled, with patient-bootstrap 95% CIs, broken down by seizure group
  (FBTC / focal_motor / focal_nonmotor / focal_unknown_motor). Never report accuracy or specificity. Never report FA per hour.
- **Single-input rules next to every model.** Every results table shows `rule_hr`, `rule_emg`, `rule_eeg`, `rule_acc`
  alongside the learned models, and every failed hypothesis is reported.
- **One change per version.** A version = one edit, run on dev folds via `04_train_ablation.py --version vNN_<name>`,
  accepted only by the rule in `EVALUATION_PROTOCOL.md` §1 (event F1 +0.01, P(improve) ≥ 0.90, sensitivity −0.02 max,
  FA/24h +0.5 max, compared with paired patient bootstrap against the current accepted version). Log every version,
  accepted or rejected, in `results/ITERATION_LOG.md` with the number, the decision and one sentence of why.
- **Synthetic data is for smoke tests only** and is never logged, plotted into the report, or left in `data/`.
- **Mask `impd` epochs** (impedance checks) from training and from the false-alarm denominator.
- **Independent re-computation** of every number before it goes into a document, slide or website.
- **Hardware recordings** are made on consenting healthy volunteers only, never on patients, and are used for false-alarm
  counting and demos, never for reported detection metrics. ICTRA is a research prototype, not a medical device.

## Conventions

- Scripts are numbered in run order in `src/` (`00_…`, `01_…`). A new step gets the next number; never renumber.
- Results go under `results/<stage>/<version>/`; raw downloads under `data/raw/` (git-ignored); features under
  `data/features/` (git-ignored). Catalogs under `data/catalog/` are committed.
- Python 3.10+, dependencies: `pandas numpy requests mne edfio scipy neurokit2 pyarrow scikit-learn xgboost lightgbm matplotlib tabulate`.
  No PyTorch until a GPU is available (Colab); tree models and sklearn first.
- The machine running training may have 2–4 CPU cores and 3–8 GB RAM. Stream files; never load all raw EDFs into memory.
  Every long step must be resumable (skip outputs that already exist).
- Comments and docs in plain English; numbers with their source; no superlatives.
- When a result looks too good (e.g. > 90% sensitivity with < 1 FA/24h on all focal seizures), suspect leakage first and
  write down what was checked (`EVALUATION_PROTOCOL.md` §6 checklist).

## Current state (2026-10-17)

- Done: dataset audit and inventory, SeizeIT2 catalog (883 seizures, 2,850 runs, SHA-256 per file), patient-wise splits
  (hashed), downloader, feature extractor, relationship study, parallel A/B/C trainer with rules and bootstrap.
  All smoke-tested on synthetic SeizeIT2-shaped recordings; **no real signal data has been processed yet.**
- Next: `docs/TASKS.md` task 1.

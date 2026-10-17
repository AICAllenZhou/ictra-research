# ICTRA — WAICY 2026

**Question:** around a seizure, how do brain (EEG), heart (ECG) and muscle (EMG + movement) signals relate in time? Does adding heart and muscle to the brain signal cut false alarms without missing more seizures?

The project follows the same procedure as SlopeAI:

1. **Catalog** the data.
2. **Relationship study** of the signals.
3. **Single-input baselines.**
4. **Model ablation:** A (EEG), then B (+ECG), then C (+EMG/ACC).
5. **Self-iteration** under a fixed accept/reject rule.
6. **Pre-registered locked test.**
7. **Hardware feasibility trials.**

See `docs/PLAN.md`.

## Reproduce (Python 3.10+)

One-shot for stage 1 on Windows: `run_stage1.bat` (download → features → relationship study → parallel A/B/C training). Every script is resumable.


```bash
pip install pandas numpy requests mne scipy scikit-learn xgboost shap neurokit2 matplotlib
python src/00_seizeit2_catalog.py                                        # metadata only, about 1 min
python src/01_download_seizeit2.py --runs seizure --split train          # dry run: shows the files and GB
python src/01_download_seizeit2.py --runs seizure --split train --go     # about 25 GB
```

## Folder map

| Path | What it is |
|---|---|
| `data/catalog/dataset_inventory.csv` | Every candidate dataset, checked, with a keep or drop decision and the reason |
| `data/catalog/seizeit2_runs.csv` | 2,850 recording runs: hours, EEG montage, bytes per signal, split, number of seizures |
| `data/catalog/seizeit2_seizures.csv` | 883 seizures: onset, duration, ILAE type, group (FBTC, focal motor, non-motor, unknown), split, which signals exist |
| `data/catalog/seizeit2_files.csv` | 11,009 .edf files with size and SHA-256, used by the downloader |
| `data/raw/` | Downloaded data (not in git) |
| `docs/PLAN.md` | Re-evaluation, dataset decision, step-by-step plan, limits |
| `src/` | Numbered scripts, as in SlopeAI |

## Rules (same as SlopeAI)

- **Split by patient.** sub-001–096 are for development. sub-097–125 are the locked test: they stay closed until `docs/PREREGISTRATION.md` is written and hashed, and they are used once.
- **Report the simple baselines** (heart-rate jump, EMG threshold, EEG line length) next to every model, and report every failed hypothesis.
- **Report sensitivity, false alarms per 24 h and detection delay**, broken down by seizure group. Never report accuracy alone.
- **ICTRA is a research prototype, not a medical device.** No patient recordings are made without ethics approval.

## Evaluation tiers (see `docs/EVALUATION_PROTOCOL.md`)

| Tier | Data | Opened |
|---|---|---|
| 1 Development | SeizeIT2 sub-001..096, 5 patient-wise folds (`data/catalog/splits.json`, hashed) | repeatedly |
| 2 Locked internal test | SeizeIT2 sub-097..125 | once, after pre-registration |
| 3 External tests, never trained on | Billeci EEG+ECG (Zenodo 1415495), Siena EEG+ECG, TUSZ EEG+EKG, My Seizure Gauge EMG+ACC, szdb ECG, Open Seizure Database wrist ACC (home) | once each, after pre-registration |

Only SeizeIT2 is used for training. Step 1 (`src/02_relationship_study.py`) runs on development patients only and was
smoke-tested on a synthetic recording with planted lags (EEG 0 s, heart +5 s, muscle +12 s, movement +15 s), which it recovered.

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

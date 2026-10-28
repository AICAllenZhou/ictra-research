# Results hand-off to the website (ictra_results_YYYYMMDD.zip)

Source: Allen's "ICTRA 训练输出清单" (2026-10-17). The website draws every chart itself from CSV/JSON; training only
exports numbers, plus the few things that must be images/video. Implemented by `src/95_export_results.py` (Task 15).
Column names: lowercase_snake, `_pct` = 0–100, `_s` = seconds, probabilities 0–1, missing = empty/NaN (never 0 or -1), UTF-8 CSV.

## Mapping: what the pipeline already produces → what the zip must contain

| Zip path | Built from | Notes |
|---|---|---|
| `predictions/<model>/sub-XXX.csv(.gz)` | `results/<stage>/<version>/preds/*.parquet` (2-s hop window probs) | expand to 1 row per second (hold each window prob for its 2 s); columns `subject, session, run, t_sec, prob, label, missing`; `fusion/` adds `prob_eeg, prob_ecg, prob_emg` from the per-modality sub-models (late-fusion version) |
| `predictions/events_test.csv` | `data/catalog/seizeit2_seizures.csv` filtered to the exported split | `event_id = sub-XXX_ses-01_run-NN_eNN`; `seizure_type` = ILAE string from events.tsv; `vigilance` from events.tsv if present; `modalities_available` from has_ecg_emg/has_mov + flat masks |
| `predictions_val/` | dev out-of-fold preds (sub-001..096) | same format; this is what proves τ and k/n were chosen on validation |
| `metrics/ablation.csv` | `summary.csv` | one row per model config incl. `fusion_no_kofn` (threshold only, no k-of-n) |
| `metrics/operating_curve.csv` | `05_operating_points.py` | ≥ 40 thresholds per model, `sensitivity_pct, fa_per_24h, latency_median_s` |
| `metrics/per_type.csv`, `per_patient.csv`, `latency.csv` | `scores_by_group.csv`, `scores_by_patient.csv`, hyp/ref overlap times | latency = first alarm − annotated onset (may be negative within the 30-s tolerance); also export `latency_from_bte_change_s` |
| `metrics/cross_dataset.csv` | `results/external/*` | rows for seizeit2_test, siena, billeci, szdb, chbmit (+ seer/osdb if accessed); `channels_used` e.g. `T7-P7;T8-P8` |
| `metrics/training_curves.csv` | xgb/lgbm eval logs per fold | tree models: `iteration`, `val_logloss`, `val_auprc`; add `val_event_f1` at the end of each fold only |
| `metrics/feature_importance_ecg.csv`, `_emg.csv`, `_eeg.csv` | gain importance averaged over folds | we also give `_eeg.csv` (tree model) |
| `metrics/false_alarm_context.csv`, `missed_events.csv` | fusion preds + events | `note` column filled by a human reviewer (team task), left empty by the script |
| `metrics/runtime.json`, `dataset_stats.json` | timing harness + catalog/splits | includes excluded runs and reasons, LIB_VERSION, git commits |
| `figures/example_events/example_<event_id>.csv` | raw EDF + preds | 3 cases (FBTC, focal impaired awareness, EEG+ECG-only), −60 s to +60 s, 64 Hz EEG/ECG, 16 Hz EMG envelope, 1 Hz HR/probs/alert |
| `figures/saliency/` | **not available**: no EEG CNN in the frozen model | instead: `figures/explain/shap_<event_id>.npz/.png` (per-window SHAP of the tree model over time) — tell the website author |
| `figures/device_recordings/`, `gui/`, `photos/`, `demo/` | hardware track (Allen/Carina) | not produced by the training code |
| `config/`, `README.md` | `models/*.meta.json`, `splits.json`, `pip freeze`, git hashes | README lists every deviation below |

## Deviations from the checklist, stated in README.md (do not silently "comply")

1. **Preprocessing is NOT matched to the firmware rates.** Checklist §D4 asks EEG 256 Hz / 0.5–29.5 Hz, ECG 125 Hz / 0.5–44.5 Hz,
   EMG 500 Hz / 60–100 Hz. The pipeline harmonises everything to 256 Hz with EEG 1–25 Hz, ECG 0.5–40 Hz, EMG 20–100 Hz + 50/60 Hz
   notch (`ictra_lib.harmonise`), because 60 Hz is North-American mains and the firmware is the thing that should change
   (TASKS stage 4). The sentence "the evaluated model is the model that runs on the device" holds once the firmware resamples
   to 256 Hz; until then say "same code path, firmware update pending".
2. **Normalisation**: rolling causal 5-min baseline features + per-recording robust scaling, not a z-score on the first 5 min.
   Same intent (no cross-recording leakage), different mechanics; documented in DATA_NOTES.
3. **Dropout repair / ACC calibration / zero-phase filters are offline (non-causal)** — see EVALUATION_PROTOCOL amendment; a
   causal version is candidate v11. All exported numbers are offline evaluations.
4. **Window hop is 2 s, exported at 1 s** by holding each window's probability; alarms (`alert`) come from k-of-n on windows.
5. **No EEG CNN, hence no Grad-CAM saliency**; SHAP time-courses of the tree model are exported instead.
6. **Test predictions (sub-097..125) exist only after pre-registration** (Task 10/11, ~Nov 14). Before that, the zip ships
   `predictions_val/` (dev out-of-fold) and the website Results section must label them "development (cross-validation)".
7. **Cross-dataset**: also includes Billeci, szdb (and Seer/OSDB if access arrives), not only CHB-MIT and Siena; all via the
   T7–P7 / T8–P8 behind-the-ear proxy for scalp EEG.
8. **Single-modality thresholds** are tuned on validation to maximise event F1 (same rule as fusion), and additionally
   reported at the ≤ 1 FA/24h operating point in `operating_curve.csv`, so both comparisons are available.
9. **FA counting covers all exported hours** (stage-2 seizure-free runs included from v02 on). v01 exports are labelled
   "seizure-containing runs only" in `dataset_stats.json`.

## Export schedule

| When | Zip contents |
|---|---|
| After v01 (tonight) | `predictions_val/` for A/B/C + rules, `metrics/*` from dev folds, `dataset_stats.json`, `training_curves.csv`, example_events ×3, README with "development only" banner |
| After v02 | same, full hours |
| After the accepted final version + pre-registration + locked test (Task 11) | `predictions/` for the test split, `metrics/*` on test, `cross_dataset.csv` |
| After external tests (Task 12) | final zip, `ictra_results_final.zip` |

Minimum viable version (if time is short): `predictions*/` for the four models + `events_*.csv` + `training_curves.csv` +
`dataset_stats.json`; the website computes the rest.

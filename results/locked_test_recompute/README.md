# Independent re-computation of the locked internal test

`python src/28_recompute_locked.py` (2026-10-28, 761 s), after `07_locked_test.py` had run once. No threshold, model, feature or
rule was changed; the frozen models and their frozen τ were re-scored along a different code path than 07 (seizures from each
run's `events.tsv` instead of the catalog; xgboost `Booster` instead of `XGBClassifier`; `ictra_lib.score_recording` instead of
07's own matching; own per-patient aggregation). The console line "scored 111 runs" counts distinct run numbers; 778
(subject, run) files were scored (asserted in the script).

Result of the comparison with `results/locked_test/`:

- Frozen-τ table, all seizures, both lineages, 7 models each: all 12 metrics (per-patient sensitivity, FA/24h, precision, F1,
  delay; pooled sensitivity, FA/24h, precision, F1, delay; detected; seizures) equal; largest difference 5.3e-15.
- Per patient (`per_patient_v02g.csv`, `per_patient_v11.csv`, 203 rows each): seizures, detected, false alarms and alarms
  identical in every row; hours differ by at most 1.4e-14.
- Point estimates of H2, H3b, H4 and the offline − causal Δ (C) equal to `hypotheses.csv` at the printed precision.
  Bootstrap CIs use a different generator here and are not compared. H1 and H3a were not re-computed.

`dev_same_aggregation.csv`: the development results (out-of-fold, per-fold τ, mean over seeds) in the locked test's per-patient
aggregation next to the pooled one. v02g C xgb: per-patient sensitivity 0.402, FA/24h 1.47, F1 0.292; pooled 0.268, 1.31, 0.280.
These are not the same procedure as the locked test (fold models and per-fold τ vs. one 3-seed ensemble trained on all 96
patients with one τ), so the difference mixes a generalisation gap with a different operating point.

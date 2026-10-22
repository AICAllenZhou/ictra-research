# Q&A preparation: numbers and sources (dev set only; the locked test is not opened)

Every number here is from the development cross-validation (sub-001..096, 702 seizures, 8,642 h). Sensitivity is
pooled over seizures (detected seizures / all seizures), not averaged per patient.

| Question | Answer | Source |
|---|---|---|
| Can it run in real time? | The causal model uses only data up to the end of each 4-s window (apart from ~40 ms of resampling look-ahead). Truncating the signal leaves every earlier window unchanged. | `results/data_checks/causality_check.md` |
| What does real time cost? | F1 −0.011 [−0.027, +0.007] (C xgb, causal vs offline, paired patient bootstrap); 26 seizures offline-only vs 16 causal-only. | ITERATION_LOG v11 entry, 2026-10-22 notes |
| How fast is the alarm? | Median 34.7 s after the annotated onset (causal C xgb; median over development recordings of each recording's median delay, detected seizures). | `results/dev/v11_causal/summary.csv` |
| How many seizures does it catch? | Causal C xgb: 26 % at 1.3 false alarms per day (deployable threshold); 25 % if set to 1 false alarm per day; 49 % at 10 per day. | `summary.csv`, `matched_fa_vs_v02g_grid_fix.md` |
| What does it miss? | Short and focal non-motor seizures: 12 % of focal non-motor seizures detected (causal). The demo shows one: probability near 0.9, below τ = 0.989, no alarm. | `delta_vs_v02g_grid_fix.md`, demo video |
| What would you do next? | Tell the model when heart rate is unreliable. During convulsions the ECG is swamped by movement (heart rate jumping 30–180 bpm second to second); the real-time loss sits in long, mostly motor seizures in models that use ECG. A heart-rate quality flag as an input is a small change with a stated mechanism. Not yet tested. | ITERATION_LOG 2026-10-22 notes |
| Does EEG change before heart rate? | In our relationship study, yes in the median (Exp R); in the demo seizure (sub-070) EEG +6 s, heart rate +9 s. We only annotate it where the data show it. | `results/expR/FINDINGS.md`, `onsets.csv` |
| Is the model better than one signal? | Yes on every comparison (learned models vs single-signal rules); multi-signal C beats EEG-only A (v01: ΔF1 +0.064, P = 0.998). | ITERATION_LOG v01 entry |

# ICTRA pre-registration — DRAFT (not hashed, not binding)

> Status: working draft. `docs/PREREGISTRATION.md` is written from this file at Task 10. Only that file is hashed
> (`docs/PREREGISTRATION.sha256`), after the frozen model exists. Nothing here permits opening sub-097..125 or any
> external labels.

## Hypotheses (H1–H4: the one place they are defined; Task 10 copies them into PREREGISTRATION.md)

Tiers: 1 = dev cross-validation (sub-001..096), 2 = locked internal test (sub-097..125), 3 = external tests.
**H3a and H3b were motivated by the dev-set Exp R, so they are tested only on tiers 2–3; the dev numbers below cannot
confirm them.**

- **H1:** C has fewer false alarms per 24 h than A at the same sensitivity, on FBTC and focal motor seizures. *(PLAN §5,
  unchanged)*
- **H2:** B is better than A on focal non-motor seizures. *(PLAN §5, unchanged)*
- **H3:** revised after Exp R on the dev set (Allen, 2026-10-17). The original wording and why it was dropped are in
  `results/ITERATION_LOG.md` ("H3 as first written").
  - **H3a:** in FBTC seizures, the behind-the-ear EEG change precedes the HR, ACC and EMG changes. That is, the median of
    `lag_eeg_to_X = t_X − t_eeg` is > 0 for each of X = HR, ACC, EMG.
    - Definitions as in `src/02_relationship_study.py`: change = z > 3 vs the −300..−60 s baseline for ≥ 3 s, searched
      from −30 s; denominator = seizures in which both changes are found. Also reported under z > 2 / 5 s.
    - Dev-set values (motivation only):

      | | z > 3, 3 s | z > 2, 5 s |
      |---|---|---|
      | lag EEG→HR | +3.5 s [−1.0, +19.2], n = 26 | +3.0 s [−2.0, +25.0], n = 29 |
      | lag EEG→ACC | +8.5 s [0.0, +20.0], n = 32 | +8.5 s [−0.2, +22.0], n = 32 |
      | lag EEG→EMG | +12.0 s [−0.5, +26.8], n = 26 | +15.0 s [−5.0, +27.5], n = 27 |
  - **H3b (model-level):** on non-FBTC focal seizures (focal motor, focal non-motor, focal unknown motor) in the locked
    set, two things hold together:
    - Model B's median detection delay is not shorter than Model A's by more than 2 s:
      Δdelay = median delay(B) − median delay(A) ≥ −2 s.
    - B's event sensitivity is higher than A's.
    - **Delay** = start of the first alarm overlapping the seizure (SzCORE tolerance) − annotated onset. Median over each
      model's detected seizures; also reported over the seizures both models detect.
    - **Rationale:** Exp R shows HR and EEG change near-simultaneously in focal seizures (median lag 0–2 s), so fusion
      should add coverage, not speed.
    - **Superseded:** the earlier threshold form ("HR within ±10 s of the EEG change in ≥ 70%") is replaced. On dev it
      was 57% (212 / 373), so it is recorded here and not tested.
- **H4:** each learned model beats its single-signal rule(s) in event F1 on the locked set:
  - A > `rule_eeg`;
  - B > max(`rule_eeg`, `rule_hr`);
  - C > the best of `rule_eeg`, `rule_hr`, `rule_emg`, `rule_acc`.

  Each comparison also needs a paired patient bootstrap P(ΔF1 > 0) ≥ 0.90. "Best" / "max" = the rule with the highest
  locked-set event F1. All rules and models use their frozen dev-tuned thresholds.

## Exploratory studies (not hypotheses; no accept/reject; never used to choose a model version)

- **Exp P: pre-ictal drift** (`src/13_preictal_study.py`, Allen 2026-10-17; dev patients only).
  - **Question:** in the 2 min before the annotated onset, do HR, HRV (`ecg_rmssd`), EMG log-RMS or movement drift
    compared with the 3 min before that?
  - **Design:**
    - Drift z = (mean[−120..−30 s] − mean[−300..−120 s]) / SD[−300..−120 s].
    - The control null comes from 10 same-patient pseudo-onsets per seizure, each ≥ 30 min from any seizure.
    - A signal is called "drifting" only if the seizure share of z > 2 exceeds the control share with a
      patient-bootstrap CI above 0.
    - Robustness: −60..−30 vs −300..−60 s.
  - **Status:** descriptive only. Any pre-ictal claim it suggests would need its own pre-registered hypothesis, tested on
    tiers 2–3.

## Design notes (prospective)

- **Matched-FA comparison for threshold-method changes (Allen, 2026-10-17; effective from v04, not applied
  retroactively to v03).**
  - A version that changes only how τ (or k-of-n) is chosen moves the operating point along the model's sensitivity–FA
    curve rather than changing the curve.
  - Such versions are therefore additionally compared at matched FA/24h budgets of 1, 2 and 5, using the operating-point
    sweep (`operating_points.csv`, `sens_at_fa_le_*`), next to the §1 decision.
  - The §1 rule stays the accept/reject criterion.

## Two frozen models, one opening of the locked set (Allen, 2026-10-21; before the freeze)

- The locked test (sub-097..125) evaluates **two** frozen models, both fixed and hashed before the set is opened:
  - **offline model** (v02g lineage: zero-phase filters, whole-recording calibration; the evaluated pipeline of the
    development results), used for claims about how much seizure information the signals carry and for comparison with
    published work;
  - **causal model** (v11 lineage: every window uses only samples up to its end; alarms timestamped at the window end),
    used for every claim about real-time alarms, alarm delay and deployability, including the demo and the website's
    "wakes a carer" statements.
- The set is opened **once**; both models are scored in that single opening with the same scoring code, and both are
  reported together, whatever their results. Running a second model after the opening would be a second look and is
  not allowed.
- H1–H4 are tested on each model separately and reported side by side; the offline−causal difference on the locked set is
  reported as a result (on dev: about −0.016 F1 for C xgb, v11 vs v02g, final number in ITERATION_LOG).

## Still to fill at Task 10
- frozen model files and hashes (both models), threshold, post-processing;
- metrics and subgroup list;
- the external sets and which model is tested on each.

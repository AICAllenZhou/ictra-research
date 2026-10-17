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

## Still to fill at Task 10
- frozen model files and hashes, threshold, post-processing;
- metrics and subgroup list;
- the external sets and which model is tested on each.

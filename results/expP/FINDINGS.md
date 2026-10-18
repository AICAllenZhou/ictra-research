# Exp P findings: is there a pre-ictal drift in HR, HRV, EMG or movement? (EXPLORATORY)

Source: `python src/13_preictal_study.py --workers 2` (2026-10-17, lib-2026-10-17e) → `results/expP/summary.csv`,
`drift.csv`, `eligibility.csv`, `figures/preictal_<group>.png`. Development patients only (sub-001..096).
Registered as exploratory in `docs/PREREGISTRATION_DRAFT.md`: no hypothesis, no accept/reject, not used to choose any
model version.

## What was measured

- **Seizures.** 538 of 702 dev seizures qualify (93 patients): FBTC 32, focal motor 272, focal non-motor 159, focal
  unknown motor 75.
  - To qualify, a seizure needs ≥ 600 s of recording before its onset and no other seizure in those 600 s.
  - Excluded: 122 had another seizure within 600 s, 41 started < 600 s into the recording, and 1 had no ECG, EMG or
    movement file.
- **Controls.** 10 pseudo-onsets per seizure (5,350 in total) from the same patient's seizure-containing runs, each with
  [t − 300, t] ≥ 30 min from any seizure. 3 seizures have no eligible control time.
  - Controls are **not** matched for sleep/wake. Vigilance is annotated only for seizures, and the recording clock is
    de-identified.
- **Signals** (1 s, via ictra_lib):
  - heart rate;
  - HRV = ictra_lib's `ecg_rmssd`, the RMS of successive differences of the 1-s HR in 4-s windows. This is the model's
    HRV feature, **not** the clinical R-R RMSSD;
  - EMG log RMS;
  - movement = SD of the neck |a|.
- **Drift z** = (mean[pre] − mean[ref]) / SD[ref]. Main windows: ref = −300..−120 s, pre = −120..−30 s. Robustness: ref
  = −300..−60 s, pre = −60..−30 s. The onset region (−30..0 s) is excluded by design.
- **Criterion (fixed before the run).** A signal "shows pre-ictal drift" only if the share of seizures with z > 2
  exceeds the control share, with the patient-bootstrap 95% CI of the difference above 0.

## Result: no signal shows pre-ictal drift, in any group, under either window definition (0 of 40 cells)

Main definition, share of z > 2 (seizures vs controls) and the difference [95% CI]:

| group | heart rate | HRV (ecg_rmssd) | EMG log RMS | movement |
|---|---|---|---|---|
| FBTC (32) | 0.0% vs 2.2%, −2.2 [−3.8, −0.8] | 6.9% vs 3.4%, +3.5 [−4.7, +15.0] | 6.9% vs 3.4%, +3.5 [−4.9, +15.8] | 9.4% vs 5.1%, +4.3 [−5.4, +16.1] |
| focal motor (272) | 4.1% vs 2.1%, +2.0 [−1.2, +5.0] | 5.9% vs 3.5%, +2.4 [−2.1, +7.7] | 2.6% vs 3.3%, −0.7 [−2.9, +0.9] | 6.7% vs 5.5%, +1.2 [−2.7, +4.9] |
| focal non-motor (159) | 1.9% vs 2.5%, −0.6 [−2.4, +1.7] | 1.9% vs 2.2%, −0.3 [−2.2, +2.0] | 3.8% vs 3.2%, +0.6 [−2.4, +5.1] | 9.8% vs 6.2%, +3.7 [−1.4, +9.8] |
| focal unknown motor (75) | 2.8% vs 2.2%, +0.6 [−3.2, +5.2] | 4.2% vs 2.3%, +1.9 [−2.8, +6.4] | 1.4% vs 2.8%, −1.4 [−4.7, +0.5] | 4.6% vs 4.8%, −0.2 [−5.8, +5.7] |
| all (528) | 3.0% vs 2.2%, +0.8 [−1.4, +3.4] | 4.5% vs 3.0%, +1.6 [−1.3, +5.4] | 3.0% vs 3.2%, −0.2 [−1.9, +1.6] | 7.4% vs 5.6%, +1.9 [−1.3, +5.2] |

(Differences in percentage points. n per signal can be lower than the group total where that signal was not recorded.)

- **Median drift z** (main definition) is between −0.13 and +0.08 for seizures and between −0.08 and 0.00 for
  controls, and the CIs overlap in every cell. Robust definition: −0.18 to +0.11 vs −0.17 to −0.02.
- **The only CI that excludes 0 points the wrong way.** FBTC heart rate has *fewer* seizures with z > 2 than controls
  (0 of 29). Under the robust windows, FBTC EMG and movement do the same.
- **Falling values (z < −2, secondary column)** are ≤ 0.4% for seizures and controls in every cell except FBTC heart
  rate: 1 of 29 seizures (3.4%) vs 0.4%, CI of the difference [−1.0, +10.7]. No drop in HRV, EMG or movement either.
- **Slopes** over −300..−30 s are about 0 (median |slope| ≤ 0.04 SD per minute) for seizures and controls.
- **Robustness** (−60..−30 vs −300..−60 s): same conclusion, 0 of 20 cells.
- **One weak hint, below the criterion:** FBTC HRV median z = −0.13 [−0.22, −0.05], a slight fall, against controls
  −0.02 [−0.09, +0.04]. With 29 seizures the CIs overlap, and the share criterion is not met. Not a finding.

## What this does and does not show

- **Does show:** in these four wearable signals, the 2 minutes before the annotated onset (up to −30 s) look like any
  other stretch of the same patient's recording. No group or signal separates from the same-patient control
  distribution.
- **The figures:** the seizure and control median curves overlap from −300 s to about −30 s. The only change comes in
  the last ~20 s before the annotated onset (HR, movement and EMG rising; focal motor clearest). That window is excluded
  by design, and it matches Exp R: the signals often change at or slightly before the annotated onset (annotation
  timing, early ictal activity), not as a pre-ictal state.
- **Does not show:**
  - **Not that no pre-ictal state exists.** This tests only these four signals, at 1-s resolution, over a 5-min horizon,
    with large-shift criteria: z > 2 means the 90-s mean moves by two second-to-second SDs. Slow changes over 10–60 min,
    scalp-EEG precursors and the clinical R-R HRV are not tested.
  - **Not the effect of vigilance.** Controls are not matched for sleep/wake, which could hide or create small
    differences.
  - **Not a test of the locked set.** Dev patients only.
- **Short-horizon warning:** from these signals, a warning minutes before onset is not plausible on this evidence. The
  wearable signals change at the seizure, not before it, so ICTRA stays a detection system, not a forecasting system.

# Exp R findings: timing of brain, heart and muscle signals around SeizeIT2 seizures (Task 4)

Source: `python src/02_relationship_study.py --plot --workers 3` (2026-10-17, `ictra_lib` lib-2026-10-17e) →
`results/expR/summary_by_group.csv`, `onsets.csv`, `figures/timeline_*.png`. Robustness:
`--from-curves --z-thr 2 --min-dur 5 --tag z2_5s` → `results/expR/z2_5s/`. Development patients only (sub-001..096):
695 seizures ≥ 5 s, 96 patients, 0 failures (`failed.txt` empty). Labels and SzCORE tolerances are unchanged.

## Definitions (fixed in the script header before the run)

- Per seizure and signal, a 1-s curve from −300 s to +179 s around the **annotated onset**. Baseline = −300..−60 s.
- **Change time** `t_X` = first second ≥ −30 s at which the signal exceeds baseline mean + 3 SD for ≥ 3 s (EEG: either
  direction). EEG = earlier of line length and 1–25 Hz power; EMG = earlier of RMS and high-frequency power; HR from
  R-peaks; ACC = neck |a| in g. The search starts at −30 s, so −30 is a floor; no change before +179 s means "no response".
- Every timing is given twice:
  - `t_X` = change time **relative to the annotated onset**;
  - `lag_eeg_to_X` = `t_X − t_eeg` = change time **relative to the first behind-the-ear (bte) EEG change** (positive: X later).
- "Responds" share = seizures with a change time, over seizures where that signal was recorded.
- HR rise = max HR in 0..60 s minus baseline median, raw and with a 5-s rolling median (`med5`, robust to single
  mis-detected R-peaks). Medians with [IQR], rounded to whole seconds / bpm as in `summary_by_group.csv` (Python
  rounds halves to even, so a median of 18.5 s shows as 18 here and as 18.5 on the Lab page; exact values: `onsets.csv`).

## Results by seizure group (z > 3 for 3 s)

| | FBTC | focal motor | focal non-motor | focal unknown motor |
|---|---|---|---|---|
| seizures / patients | 33 / 28 | 315 / 48 | 244 / 39 | 103 / 22 |
| responds: EEG | 100% of 33 | 83% of 313 | 58% of 238 | 80% of 97 |
| responds: HR | 87% of 30 | 75% of 312 | 58% of 235 | 76% of 91 |
| responds: EMG | 87% of 30 | 78% of 313 | 59% of 235 | 79% of 91 |
| responds: ACC | 97% of 33 | 84% of 313 | 59% of 204 | 86% of 77 |
| HR rise ≥ 20 bpm (raw / med5) | 97% / 93% | 92% / 88% | 70% / 57% | 84% / 81% |
| HR rise ≥ 50 bpm (raw / med5) | 83% / 63% | 58% / 32% | 22% / 12% | 58% / 40% |
| HR rise, bpm (med5) | 55 [42, 67] | 39 [27, 56] | 22 [13, 35] | 43 [25, 58] |
| **relative to annotated onset**, s | | | | |
| t_eeg | 7 [0, 30] | 2 [−5, 18] | 11 [0, 53] | 4 [0, 20] |
| t_hr | 16 [0, 31] | 3 [−7, 16] | 6 [−1, 28] | 7 [1, 18] |
| t_emg | 22 [10, 32] | 6 [−2, 30] | 13 [−4, 47] | 4 [0, 26] |
| t_acc | 21 [2, 34] | 0 [−4, 15] | 18 [3, 68] | 1 [−1, 16] |
| **relative to first bte-EEG change**, s | | | | |
| HR after EEG | 4 [−1, 19] n=26 | 0 [−8, 8] n=209 | −2 [−20, 8] n=101 | 1 [−4, 8] n=63 |
| EMG after EEG | 12 [0, 27] n=26 | 2 [−3, 25] n=222 | 0 [−11, 18] n=103 | 1 [−2, 13] n=68 |
| ACC after EEG | 8 [0, 20] n=32 | 0 [−5, 6] n=231 | 1 [−9, 22] n=94 | 0 [−4, 1] n=58 |
| share EEG strictly first vs HR | 54% (14/26) | 45% | 41% | 56% |
| share EEG strictly first vs EMG | 73% | 56% | 50% | 51% |

## In plain words

- **FBTC (33 seizures).** Every signal responds in nearly every seizure. The bte EEG changes first (median 7 s after the
  annotated onset). HR follows about 4 s after the EEG change, ACC about 8 s after and EMG about 12 s after (the tonic–clonic
  phase). HR rises by a median of 55 bpm (med5), ≥ 20 bpm in 93%. All four signals are usable; the motor signals come
  later but are very large (median curve peaks at z ≈ 30–35 for EMG RMS and ACC, figure `timeline_FBTC.png`).
- **Focal motor (315).** About 75–84% respond on each signal. All four change at nearly the same time: median lags to EEG
  are 0–2 s, and HR is strictly first about as often as EEG (46% vs 45%). HR rises ≥ 20 bpm in 88% (med5).
- **Focal non-motor (244).** Only 58–59% respond on any signal. The EEG change is later and more spread out (11 [0, 53] s),
  and HR is the signal most often first (51% vs 41%). The HR rise is the smallest (22 bpm med5; ≥ 20 bpm in 57%).
  **EMG and ACC also respond in 59% of these "non-motor" seizures**, mostly later (ACC 18 [3, 68] s): this looks like
  movement during or after the seizure (automatisms, arousal), not a motor onset.
- **Focal unknown motor (103).** Behaves like focal motor (responses 76–86%, lags 0–1 s).
- **What to trust for which type.** For FBTC, every signal works; EMG/ACC are the most specific, but they come about 10 s
  after the EEG. For focal motor, all signals arrive together, so fusion gains coverage (seizures one signal misses)
  rather than speed. For focal non-motor, the wearable signals miss about 40% of seizures under this definition; HR is the
  earliest when it responds, but the rise is small. Expect the lowest sensitivity in this group for every model.

## Comparison with the literature numbers in `docs/PLAN.md`

- Ictal tachycardia in ~56–80% of focal seizures: here the HR change (z > 3) appears in 58–76% of focal groups. The
  med5 ≥ 20 bpm rise appears in 57–88%. Consistent.
- "EMG/ACC only for motor seizures": partly contradicted. Motor groups respond on EMG/ACC 78–86%. But non-motor seizures
  still show EMG/ACC changes in 59%, about as often as EEG (58%), and later. These are probably movement after onset,
  not ictal motor activity. A detector will use them, but they shouldn't be read as a motor sign.

## Robustness (z > 2 for 5 s, `results/expR/z2_5s/`)

The conclusions do not change. Response shares move by ≤ 10 points (e.g. focal non-motor HR 58% → 68%, unknown-motor
EEG 80% → 88%). Median change times move by ≤ 6 s. The ordering is unchanged:
- FBTC is EEG first (EEG 9, HR 15, ACC 16, EMG 18 s);
- the focal motor groups are near simultaneous (lags 0–4 s);
- in focal non-motor, HR is first slightly more often than EEG (52% vs 42%).

The HR-rise columns do not depend on the threshold.

## Limits

- **The z-threshold is relative to each seizure's own baseline variability.** If the baseline is active (awake, moving),
  a visible change can stay below 3 SD. Example: sub-016 run-03. By eye the bte EEG and the EMG change at 0 s
  (`docs/DATA_NOTES.md` §4). But the baseline line length varies ±47% of its mean and HR is 96 ± 10 bpm, so the curve only
  reaches z > 3 at +105 s (+104 s with z > 2 / 5 s). In such cases `t_X` is an upper bound on the visible change.
- The raw HR-rise maximum is inflated by mis-detected R-peaks during movement (raw ≥ 50 bpm 83% vs med5 63% in FBTC).
  Report the med5 numbers.
- The annotations were made on full-scalp video-EEG; the bte channels see spread activity later (see `docs/DATA_NOTES.md`
  §4b for the distribution of t_eeg).
- All timings come from the offline (non-causal, zero-phase) signal path (`docs/EVALUATION_PROTOCOL.md` amendment
  2026-10-17).

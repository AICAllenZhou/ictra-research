# ICTRA: re-evaluation and plan, using the SlopeAI procedure

Written 2026-10-17. All SeizeIT2 counts below were recomputed from every `events.tsv` and sidecar file
(`src/00_seizeit2_catalog.py` → `data/catalog/`). No signal file has been opened yet.

## 1. The idea in one line

SlopeAI asks how three signal families relate on a slope: **rain (trigger) → soil water (state) → ground motion
(confirmation)**. It then tests whether adding each family gives earlier warnings and fewer false alarms
(Model A → B → C).

ICTRA asks the same question of the body:

| | SlopeAI | ICTRA |
|---|---|---|
| Signal 1, the source | Rainfall: the trigger | **EEG**: the seizure starts in the brain |
| Signal 2, the system's response | Soil moisture: water state, with a delay (wetting front) | **ECG**: the autonomic response (ictal heart-rate rise, usually at or just after EEG onset) |
| Signal 3, the visible outcome | BNO085 tilt/motion: the slope actually moves | **EMG + ACC**: the muscles actually stiffen or jerk (motor and tonic-clonic seizures only) |
| Feature describing the relationship | `wetting_front_delay`, moisture gradient | `eeg_to_hr_lag`, `hr_rise_pct`, `emg_to_acc_lag`, EMG burst rhythm |
| Nuisance that must not alarm | Tapping the box, wind, footsteps | Chewing, walking, exercise, scratching, sleep movement |
| Model A / B / C | rain / + soil / + motion | **EEG / + ECG / + EMG+ACC** |

**Research question:**
- Around a seizure, do brain, heart and muscle signals change in a consistent order? Does that order depend on seizure type?
- Does adding heart and muscle to the brain signal cut false alarms per day, without missing more seizures?

## 2. Dataset decision

Full table: `data/catalog/dataset_inventory.csv`.

**Primary dataset: SeizeIT2 (OpenNeuro ds005873 v1.1.0), CC0.** It is the only open dataset where EEG, ECG and EMG
are recorded at the same time on the same patient.

| | Value (recomputed) |
|---|---|
| Patients / recording runs / hours | 125 / 2,850 / 11,626 h |
| Signals | Behind-the-ear EEG: 2 channels, 256 Hz. ECG: 1 channel, chest. EMG: 1 channel, left deltoid, 256 Hz. Movement (ACC+GYR): 12 channels, 25 Hz |
| Seizures | 883 (every patient has ≥1; median 3 per patient, maximum 75) |
| Seizure groups (all / train / locked test) | Focal-to-bilateral tonic-clonic (FBTC) 55 / 33 / 22. Focal motor 371 / 317 / 54. Focal non-motor 324 / 249 / 75. Motor status unknown 133 / 103 / 30 |
| Seizure duration | Median 33 s. FBTC median 108 s. 87 seizures are under 10 s |
| Missing ECG/EMG | 27 seizures (12 train, 15 test) have no ECG/EMG file. Movement data is missing for 74 |
| Size | 126 GB total. **Runs that contain seizures: 33 GB (3,062 h)**. Train-only seizure runs: 25 GB |
| Split | sub-001–096: train/dev (8,715 h, 702 seizures). **sub-097–125: locked test** (2,912 h, 181 seizures) |

**Secondary relationship dataset: Zenodo 3885755** (Arbune et al. 2020), 794 MB.
- Contents: EMG and EEG of convulsive seizures in 42 patients, labelled with how long the brain stays "silent" afterwards (postictal generalized EEG suppression, PGES, a marker of SUDEP risk).
- Use: a second, clean relationship test. Can the muscle signal during a seizure predict what the brain does after it?

**External test: Siena Scalp EEG** (EEG + ECG, 14 adults, 47 seizures). This is the equivalent of SlopeAI's
"never trained on Canada" test.

**Dropped:**
- **EEGMMIDB and ds005505 as "healthy" negatives.** Different device and montage, so the model would learn which dataset a recording came from. ds005505 is also not a healthy cohort.
- **Bonn/UCI.** Its accuracy is inflated by leakage.
- **Mendeley.** 6 patients.
- **SeizeIT1.** No longer shared; its ethics approval has expired.
- **ds006525.** Could not be identified.

Negatives come from the 8,000+ seizure-free hours of the same SeizeIT2 patients, exactly as SlopeAI uses ordinary
days at the same place.

**The team doc's "ECG epilepsy dataset" link (Nature s41597-025-05580-x) is the SeizeIT2 paper itself.** It is not a
separate dataset.

## 3. What changes from the current team plan

| Current plan | Problem | Change |
|---|---|---|
| Healthy-person datasets as negatives | Dataset identity leaks | Use same-patient seizure-free hours |
| Random window split | The same patient ends up in both train and test | Split by patient; locked test sub-097–125 |
| Report accuracy | Seizures are under 1% of hours, so accuracy is meaningless | Event-based sensitivity, precision, F1 and **false alarms per 24 h** (SzCORE scoring), plus detection delay |
| Pre-ictal prediction | Not supported by these data | Drop it, or keep it as future work only |
| 2D CNN on spectrograms first | Overfits; hard to explain | Feature + XGBoost first, as in SlopeAI. A small 1D CNN later as one accept/reject version |
| "More accurate than products" | No evidence | "Fusion cuts false alarms on public data, at X% of the cost" |
| One target for every seizure | FBTC behaves very differently from non-motor focal seizures | Report every result by seizure group. The headline claim is about convulsive and motor seizures |

## 4. Procedure (the SlopeAI steps, in order)

### Step 0: catalog (done)
- `00_seizeit2_catalog.py` → runs, seizures and files with SHA-256.
- `01_download_seizeit2.py` → resumable, checksum-verified download.

### Step 1: relationship study (Exp R), before any classifier
For every train seizure that has all three signals, compute one time series per signal relative to the annotated
onset, from −120 s to +180 s:

- EEG: line length and band power from each behind-the-ear channel.
- ECG: heart rate from R-peaks, and its change versus the 5-minute baseline.
- EMG: RMS envelope and high-frequency (>20 Hz) power.
- ACC: magnitude.

Then:
1. **Per-signal change time:** the first time each signal leaves its baseline band (for example z > 3 for at least 3 s).
2. **Lead–lag between signals:** `eeg_to_hr_lag`, `eeg_to_emg_lag`, `emg_to_acc_lag`, both per seizure and pooled. Show them as a timeline figure for each seizure group.
3. **Which signals respond at all, by seizure group:** the share of seizures with a heart-rate rise above 10 / 20 / 50 bpm, with an EMG burst, and with ACC above threshold.
4. **Zenodo:** correlate ictal EMG features (tonic and clonic phase durations, silent periods) with PGES duration.

This is the "wetting-front delay" of ICTRA. It is a finding that holds even if the detector is modest, and it tells
the team which signal to trust for which seizure type.

### Step 2: single-input baselines (SlopeAI's "7-day rain index alone")
Each baseline is a simple rule with its threshold tuned on dev folds:

- Heart-rate jump: HR at least 30% above the 5-minute baseline for at least 10 s.
- EMG RMS above threshold (SPEAC-style).
- EEG line length above threshold.
- ACC magnitude above threshold.

These are always reported next to every model.

### Step 3: ablation, Model A / B / C
- **A:** EEG features only.
- **B:** A + ECG/HRV features.
- **C:** B + EMG + ACC features.
- All three use XGBoost on 4-s windows with 2-s steps, plus the post-processing below.
- Post-processing: an alarm needs at least 3 of 5 positive windows; alarms are merged within 90 s.
- Also try late fusion, one model per signal with probabilities averaged, as one version.

### Step 4: self-iteration with a fixed rule (SlopeAI v3 → v18)
- **Dev folds:** 5-fold GroupKFold by patient on sub-001–096.
- **Repeats:** 3 seeds × 5 negative draws.
- **One change per version**, logged in `results/ITERATION_LOG.md`.
- **Accept a change only if all of these hold:**
  - event F1 improves by at least 0.01;
  - patient-bootstrap P(improvement > 0) is at least 0.90;
  - sensitivity does not drop by more than 0.02;
  - false alarms per 24 h do not rise by more than 0.5.
- **Candidate versions:**
  - 1D CNN on raw EEG
  - channel choice by montage (left / right / generalized)
  - relationship features from Step 1
  - per-patient baseline normalisation
  - window length
  - late versus early fusion

### Step 5: pre-registration and locked test
- **Before opening sub-097–125**, write `docs/PREREGISTRATION.md` with:
  - the frozen model and threshold;
  - the hypotheses.
- **Example hypotheses** (first wording; the definitive H1–H4 are in `docs/PREREGISTRATION_DRAFT.md`, H3 revised after Exp R):
  - **H1:** C has fewer false alarms per 24 h than A at the same sensitivity, on FBTC and focal motor seizures.
  - **H2:** B is better than A on focal non-motor seizures.
  - **H3:** EEG changes before HR in most seizures.
- Hash the file. Open the test once and report every hypothesis, including the ones that fail.
- **Then the external test on Siena:** Model B only, because Siena has no EMG.

### Step 6: honesty layer
- Use SHAP to show which signal drives each alarm.
- Report results per seizure group, with patient-bootstrap 95% confidence intervals.
- Run an independent re-computation of every number in the report.
- Keep an evolution log for the website, as in the SlopeAI Explorer.

### Step 7: hardware, as in SlopeAI's trial protocol
- Healthy volunteers only, with consent.
- Run nuisance trials (chewing, walking, stairs, typing, sleep posture) to measure false alarms per hour on the team's own board.
- Use deliberate muscle clenching and arm shaking to show that EMG and ACC respond. Label these clearly as simulation.
- Replay SeizeIT2 seizures through the same live pipeline and website.
- No patient recordings without ethics approval.

## 5. Honest limits to state up front
- **Hospital patients only.** Everyone in SeizeIT2 had focal epilepsy and was recorded in a hospital monitoring unit. There are no infants with infantile spasms, so infantile spasms stay "future work".
- **Few convulsive seizures in the test set.** There are only 22 FBTC seizures from 9 test patients, so confidence intervals for convulsive seizures will be wide.
- **Labels may not be visible on the wearable channels.** Onsets were annotated on full-scalp video-EEG, so some seizures may not show up on behind-the-ear channels.
- **Recording centre is unknown.** `participants.tsv` gives only sex, so a leave-one-centre-out test (SlopeAI's leave-one-region-out) is not possible unless the paper's supplement lists centres.
- **The team's 10-bit Arduino board is far noisier than the clinical recorder.** The board currently shows 58–60% of samples stuck at the ADC limits. Model results come from SeizeIT2. The board demonstrates feasibility; it is not validation.

## 6. Download plan
- Run downloads on the team laptop. The Cowork sandbox cannot reach openneuro.org, zenodo.org or physionet.org.
- **Step 1:** `python src/01_download_seizeit2.py --runs seizure --split train --go` (about 25 GB). This is enough for Exp R, the baselines and the first version of A/B/C.
- **Step 2:** `--runs all --split train` (about 95 GB in total). Needed for realistic false alarms per 24 h.
- **Step 3:** `--split locked_test`, only after the pre-registration is written.
- **Zenodo 3885755:** 794 MB, download via the record page or its API.

Sources: SeizeIT2 paper https://www.nature.com/articles/s41597-025-05580-x · fusion on SeizeIT2 FBTC
https://arxiv.org/pdf/2403.13066 · ECG on SeizeIT2 https://pmc.ncbi.nlm.nih.gov/articles/PMC12736484/ · SzCORE
https://arxiv.org/html/2402.13005v3 · Zenodo EMG/PGES https://zenodo.org/records/3885755 · SeizeIT1 status
https://doi.org/10.48804/P5Q0OJ

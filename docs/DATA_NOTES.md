# ICTRA data notes: what the real SeizeIT2 files contain (Tasks 1–3)

Written 2026-10-17 from the first real files (dev subjects sub-001..021 only). Every number below comes from
`src/10_data_checks.py` and the CSV/PNG it writes to `results/data_checks/` (named next to each section).

## 1. File format (all files checked: EDF+C)

Source: `python src/10_data_checks.py headers sub-001:3 sub-015:4 sub-016:3` → `results/data_checks/edf_headers.csv`
(and `edf_headers_extra.csv` for sub-019 run-07 and sub-014 run-01).

- Every file is **EDF+C** (continuous; not EDF+D, even though the JSON sidecars say `RecordingType: discontinuous`),
  with 1-s data records and an `EDF Annotations` channel that MNE drops. 16-bit samples (digital −32768..32767).
- The four files of one run (eeg/ecg/emg/mov) have the **same number of records**, and the duration equals
  `hours` in `data/catalog/seizeit2_runs.csv` to 3 decimals in all 7 runs checked.
- Start dates are anonymised (`01.01.0000` for eeg/ecg/emg, a fake 2516 date for mov), so they cannot be used to align
  files; alignment is by sample index, which is what `03_features.py` does.

### Channel names, units and sampling rates

| Modality | Channels (exact EDF labels) | Unit in header | fs |
|---|---|---|---|
| EEG, montage `left` | `BTEleft SD`, `CROSStop SD` | uV | 256 Hz |
| EEG, montage `right` | `BTEright SD`, `CROSStop SD` | uV | 256 Hz |
| EEG, montage `generalized` | `BTEleft SD`, `BTEright SD` | uV | 256 Hz |
| ECG | `ECG SD` | uV | 256 Hz |
| EMG | `EMG SD` | uV | 256 Hz |
| MOV (usual, 12 ch) | `EEG SD ACC X/Y/Z`, `EEG SD GYR A/B/C` (SensorDot on the back of the neck), `ECGEMG SD ACC X/Y/Z`, `ECGEMG SD GYR A/B/C` (SensorDot on the left chest) | g (also written for GYR, which is wrong for a gyroscope) | 25 Hz |
| MOV, sub-019 run-07 (9 ch) | neck ACC only (no neck GYR) + chest ACC + chest GYR | g | 25 Hz |
| MOV, sub-014 run-01 (6 ch) | chest device only: `ECGEMG SD ACC X/Y/Z`, `ECGEMG SD GYR A/B/C` | g | 25 Hz |

Physical ranges are set per file (e.g. EEG −1528..1601 uV in sub-001 run-03, −1926..1909 uV in sub-016 run-03).

**Consequence for the features (not fixed in v01, logged as candidate v04):** `eeg_*_0` is always a behind-the-ear
channel (left BTE for `left` and `generalized`, right BTE for `right`), but `eeg_*_1` is the cross-head channel in
`left`/`right` runs and the *right BTE* channel in `generalized` runs. `eeg_asym_ll` therefore means "BTE minus cross" in
1,739 dev runs and "left minus right" in 333. Of the 2,072 dev runs, 1,058 are left, 681 right, 333 generalized.

Task 1 spot check (`python src/10_data_checks.py spot --n 5 --seed 0` → `spot_check.csv`): 5 random downloaded files
(eeg ×3, emg, mov) had fs 256 / 256 / 25 Hz, 2 / 1 / 12 channels and durations equal to the catalog.

## 2. Units and amplitudes (Task 2)

Source: `python src/10_data_checks.py amplitudes sub-001:3 sub-015:4 sub-016:3 sub-002:3` → `amplitudes.csv`
(30 min from the middle of each run, after `ictra_lib.harmonise`).

The EDF unit is uV. MNE converts to V, and `03_features.py` multiplies by 1e6, so features are in **uV**. The scale is correct:

| Check (TASKS.md expectation) | sub-001 r3 | sub-015 r4 | sub-016 r3 | sub-002 r3 |
|---|---|---|---|---|
| EEG std per 4-s window, median, ch0 / ch1 (5–50 uV) | 5.9 / 10.4 | 8.6 / 14.9 | 6.7 / 10.4 | 7.4 / 13.7 |
| ECG R-peak amplitude, median (0.3–3 mV) | 0.34 mV | 0.30 mV | **0.08 mV** | 0.35 mV |
| Heart rate (bpm) | 72 | 82 | 56 | 70 |
| EMG RMS at rest, 10th pct / median (< 20 uV) | 4.8 / 5.3 | 9.1 / 9.9 | 5.3 / 6.0 | 18.9 / 22.2 |

ECG amplitudes are at the low end of the expected range: a single chest lead with close electrodes, measured after the
0.5–40 Hz filter. sub-016's 0.08 mV is low but R-peaks are clearly detected (see `onset_FBTC_sub-016_run-03_3076.png`).
Heart rate comes from R-peak timing, so amplitude does not enter the features.

### Accelerometer scale is not consistent: calibrated to gravity (fix)

At rest, |a| should be 1 g. The stored |a| has a median of **0.43–0.52** in sub-001, sub-002, sub-015 and sub-016 (both devices)
and **0.167** in sub-019 run-07. No samples sit at the physical limits (share < 1e-6), so this is a scale problem, not
clipping. Fix: `ictra_lib.acc_to_g` divides each run's ACC by its median |a|, so the resting magnitude is 1 g by definition.
This needs no device constant and works the same way on external ACC data. `03_features.py` applies it before `harmonise`.

ACC device used for features: the **neck** SensorDot (`EEG SD ACC X/Y/Z`), picked by name. If a run has no neck ACC
(e.g. sub-014 run-01), the first three ACC channels are used, which is the chest device. Which device was used is not stored.

Exactly constant ACC: 1.34 % of sub-019 run-07's windows have zero ACC variance (`acc_logsd` = log(1e-9) = −20.7).
This could be a very still sensor below one quantisation step, or missing data. It is left as is and documented here.

## 3. Sample dropouts in the SensorDot signals (fix)

Sources: `results/data_checks/dropout_repair_examples.png`, `dropout_check_sub-019_emg.png`,
`python src/10_data_checks.py dropouts <12 runs>` → `dropouts.csv`, `dropouts_sub-015_run-04.png`.

Some files contain runs of 1–2 samples that are pulled toward zero (raw ADC zero, not the signal's baseline), at regular
intervals of about 1–4 s. Examples (raw uV): EEG `64, 82, 79, 1.0, 68`; ECG `663, 300, 378, 738`; ACC X
`−0.403, −0.046, −0.357, −0.403`. Raw DC offsets reach about ±1,000 uV, so after band-pass filtering each dropout becomes a spike of
50–200 uV in ECG/EMG/EEG. That inflates line length and EMG RMS, and it creates false R-peaks (heart-rate spikes).

Prevalence (12 runs, 12 × 60 s segments each, share of samples flagged):

| | median over runs | worst run |
|---|---|---|
| EEG channels | 0.06 % | 0.56 % (sub-017 r1) |
| ECG | 0.02 % | 0.23 % (sub-011 r3) |
| EMG | 0.10 % | 0.47 % (sub-016 r3) |
| ACC axes | 0.11 % | 2.36 % (sub-015 r4) |

Three of the 12 runs (sub-015 r4, sub-016 r3, sub-017 r1) are heavily affected. sub-001 r3 and sub-002 r3 are at ≤ 0.03 %
(EEG/ECG), which is roughly the detector's false-positive floor.

Fix: `ictra_lib.dropout_mask` / `repair_dropouts`, called first inside `ictra_lib.harmonise`, so every dataset goes
through it. A run of 1–2 samples is replaced by linear interpolation only if **all** of these hold:
- every sample is closer to zero than the mean of its two neighbours, and at least one is closer by more than 25 %;
- the run is a V-shape: the deviation is more than 3 × the difference between the two neighbours;
- the deviation is more than 6 × the median absolute first difference;
- the local 1-s level is more than 3 × the local MAD. Where the level is near zero a dropout is invisible, and real
  signal must not be touched.

Checks:
- Glitch spikes vanish in ECG/EMG/EEG/ACC, and R-peaks are unchanged (sub-015 r4, sub-016 r3).
- During sub-017's FBTC, EMG (raw level near 0) has 6 of 8,192 samples flagged.
- During sub-019's FBTC the raw EMG swings rail to rail (±3,000–4,000 uV) and about 1.4 % of samples are flagged. That
  segment is saturated anyway and its EMG features stay extreme. This is an accepted, documented trade-off.

## 4. Onset alignment, FBTC (Task 2)

Source: `python src/10_data_checks.py onsets --group FBTC --n 4` → `results/data_checks/onset_FBTC_*.png`
(±120 s; EEG, ECG and EMG after `ictra_lib.harmonise`, HR from `ictra_lib.heart_rate_series`, neck ACC |a| calibrated).

| Seizure | Behind-the-ear EEG change relative to the annotated onset | Other signals |
|---|---|---|
| sub-015 run-04, 52,473 s, 30 s | at 0 s: high-amplitude rhythmic activity on both channels | EMG burst at 0 s, larger from +17 s; ACC from +2 s; ECG lost in artefact from 0 s |
| sub-016 run-03, 3,076 s, 301 s | at 0 s: amplitude change on both BTE channels | EMG burst exactly at 0 s; HR about 85 → 108 bpm by +10 s |
| sub-017 run-01, 21,268 s, 134 s | **none visible until about +35 s**; large from +50 s | EMG, ACC and ECG artefact from +35–50 s |
| sub-019 run-07, 33,765 s, 154 s | slow build-up from +40 s; large from +100 s | EMG bursts from +8 s; HR rises from +5 s |

In 2 of 4 FBTC seizures the annotated onset coincides with a visible change on the wearable channels. In the other 2,
the change on the behind-the-ear channels comes 35–100 s later, presumably when the seizure spreads (the onset was marked
on full-scalp video-EEG; `docs/PLAN.md` §5). Consequence: with the SzCORE tolerance (30 s before / 60 s after the
*annotated* event), a detector that fires on the visible change still counts as a hit, because these seizures last
134–154 s. But detection delays measured from the annotated onset include this spread time. Exp R (Task 4) measures the
lag directly.

## 5. Impedance checks (`impd`)

Source: `python src/10_data_checks.py impd sub-001:3 sub-001:5 sub-003:6 sub-007:3 sub-011:2 sub-014:1 sub-015:6 sub-016:3`
→ `impd_check.csv`, `impd_sub-001_run-03_23059.png`.

- 28 impd epochs, 5–346 s long.
- EEG std inside the epoch versus the 60 s on either side: median ratio 1.18 (channel 0) and 1.52 (channel 1).
- 68 % of epochs have a ratio > 1.5 in at least one channel. Peak-to-peak inside is often 200–4,000 uV.
- The example (sub-001 run-03, 23,059 s) shows bursts in EEG (cross channel ±200 uV), ECG (±1,000 uV) and EMG
  (±500 uV) at once, which is a stimulation or contact artefact, not physiology.
- Epochs with a ratio near 1 sit inside stretches that are already artefactual (e.g. sub-011 run-02 at 513–3,812 s:
  surrounding EEG std ~300–400 uV).

Conclusion: the impd epochs are artefact, and masking them (training and FA denominator) is correct.

## 6. Problems found in the code and fixed before any feature was kept

1. **No `events.tsv` on this machine.** `01_download_seizeit2.py` copies sidecars only from the git metadata clone
   `data/raw/ds005873_meta`, which is absent here, and `03_features.py` needs `events.tsv` for labels and impd. Added
   `01_download_seizeit2.py --sidecars`, which fetches events.tsv/json sidecars from OpenNeuro S3 for the selected
   runs only. It refuses any split other than `train` until `docs/PREREGISTRATION.sha256` exists, so no locked-test
   path is requested.
2. **`heart_rate_series` quality flag** built a seconds × beats distance matrix (65,000 × 80,000 ≈ 42 GB for an
   18-h run). Replaced by a binary search with the same result and O(n) memory.
3. **`eeg_features`** held a whole run's FFT in memory (~2 GB per worker). Now computed in blocks of 2,048 windows;
   the output is identical.
4. **`03_features.py` could save a half-downloaded run**: a run whose EEG had arrived but whose ECG was still a `.part`
   would be saved without ECG and then skipped forever. It now processes a run only when every catalogued file has its
   catalogued size and `events.tsv` exists.
5. Dropout repair (§3) and ACC calibration (§2), both in `ictra_lib`.

After the fixes, one 18-h run (sub-001 run-03) takes 28 s and peaks at 1.3 GB RAM; a 22-h run (sub-019 run-07)
takes 78 s and 1.5 GB. Feature extraction uses 6 workers rather than 11 (12 physical cores minus one): the machine had
about 6 GB of RAM free while the download ran, so 11 workers would swap.

No features existed before these fixes, so there is no earlier feature version. v01 is the first.

## 7. Feature extraction failures

None so far (see `data/features/_failed.txt`). Updated at the end of Task 3.

## 8. External datasets: file formats only (Task 9)

Source: `python src/08_external_adapters.py probe --dataset <name>`. Each probe fetches only the EDF header and the first
60 s (HTTP range requests; files in `data/raw/external/<name>/_probe/`, git-ignored). No label file was opened:
`Seizures-list-PNxx.txt`, the Billeci `.txt` members, szdb header comments and annotation files, OSDB event types. Any
label reading in `08_external_adapters.py` must go through `labels_allowed()`, which refuses until
`docs/PREREGISTRATION.sha256` exists.

| Set | Probed recording | Format | Channels used | fs | Duration (header) | Status |
|---|---|---|---|---|---|---|
| Siena (PhysioNet siena-scalp-eeg 1.0.0) | PN00/PN00-1.edf | EDF, 10-20 names `EEG T3` etc. | EEG proxy T7–P7 (`T3`–`T5`), T8–P8 (`T4`–`T6`); ECG `EKG EKG` | 512 Hz | 0.73 h | loads; harmonised to 256 Hz (proxy EEG std 5.8 uV, ECG std 125 uV) |
| Billeci (Zenodo 1415495) | POne_pat_1-3.zip → PATIENT01/Patient01.edf | 5 zips of 2.3–3.4 GB; per patient folder with `.edf` and `.txt` | same as Siena; ECG `EKG EKG` | 256 Hz | 4.84 h | loads (member streamed from the zip); proxy EEG std 30.7 uV, ECG std 114 uV |
| szdb (PhysioNet szdb 1.0.0) | sz01 | WFDB, 1 channel `ECG` | ECG (converted mV → uV) | 200 Hz | 1.50 h | loads; ECG std 217 uV after harmonisation |
| My Seizure Gauge (Seer) | — | Seer export (per-channel segments) | EMG + 3-axis ACC | 250 / 25 Hz (published) | — | **untested: registration not obtained** |
| Open Seizure Database | — | JSON events with 5-s `rawData` blocks | wrist ACC magnitude (+ HR) | 25 Hz | — | **untested: data-sharing agreement not obtained** |

Observations to carry into Tasks 10–12:
- **Siena and Billeci have identical channel lists** (29 EEG including `F9`/`F10`, `EKG EKG`, `SPO2`, `HR`, `MK`; Siena
  adds `1`, `2`). Both come from the same Siena hospital, so a patient overlap is likely. EVALUATION_PROTOCOL.md §6 requires
  checking it (demographics, recording dates) before both count as independent.
- Siena is at 512 Hz and szdb at 200 Hz. `ictra_lib.harmonise` resamples both to 256 Hz with `resample_poly`.
  szdb's 200 Hz limits the ECG band to < 100 Hz, which is fine for the 0.5–40 Hz ECG filter.
- Neither scalp set has EMG or ACC, so only Models A and B (and the ECG rule) apply there, as planned in §3.2.

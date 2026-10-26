# ICTRA pre-registration: locked internal test (sub-097..125) and external tests

Written 2026-10-26, **before** any file of sub-097..125 was downloaded, feature-extracted or looked at, and
before any external dataset's labels were opened. Code at commit `168dbb9c2a` plus the frozen files below. Nothing in
`models/` or in the code files listed below changes after this file is hashed (`docs/PREREGISTRATION.sha256`).

## Why

All results so far are 5-fold cross-validation on the 96 development patients (sub-001..096), and the versions were
chosen by looking at those results. The 29 locked patients (sub-097..125) have never been opened. This test measures the
frozen models on patients that no version has seen, once.

## Data

- **Tier 2, locked internal test:** SeizeIT2 sub-097..125 (`data/catalog/splits.json` → `locked_test`): 29 patients,
  2,912 h, 181 seizures (22 FBTC from 9 patients, 54 focal motor, 75 focal non-motor, 30 focal unknown motor;
  `docs/EVALUATION_PROTOCOL.md` §2). 15 of these seizures have no ECG/EMG file: every result is reported with and
  without them. All runs are used (seizure-free runs included), natural class ratio.
- Downloaded with `src/01_download_seizeit2.py --split locked_test` and featurised with `src/03_features.py --allow-locked`
  (offline features) and `--allow-locked --causal` (causal features), the same code as for the development data.
- **After opening, the set becomes development data** and every later report says so.

## Models (no new choices)

Two frozen lineages, scored in the **same single opening** with the same scoring code and reported together whatever
the results (`docs/EVALUATION_PROTOCOL.md` amendment 2026-10-21):

1. **Offline model, v02g lineage** (`v02g_grid_fix`): zero-phase filters and whole-recording calibration, offline
   features (`data/features`), alarms timestamped at the window start. Used for claims about how much seizure information
   the signals carry and for comparison with published work.
2. **Causal model, v11 lineage** (`v11_causal`): every window uses only samples up to its end (`ictra_lib` causal mode,
   `lib-2026-10-20f`), causal features (`data/features_causal`), alarms timestamped at the window end. Used for every
   claim about real-time alarms, alarm delay and deployability.

Frozen per lineage (`src/06_freeze.py`): models A (`A_eeg`), B (`B_eeg_ecg`) and C (`C_eeg_ecg_emg_acc`), **xgboost
only**, and the four single-signal rules (`rule_eeg`, `rule_hr`, `rule_emg`, `rule_acc`). Each learned model is the
**mean probability of three seed models** (seeds 0, 1, 2) trained on all 96 development patients with the evaluated
configuration (5:1 background sampling, modality dropout 0.25, same hyper-parameters).

### How τ was chosen, and its uncertainty

- τ was chosen on development data only, from **one inner patient split** of the 96 development patients
  (numpy `default_rng(0)` shuffle of the sorted patient list; the first 20 % = 19 patients are the inner-validation set).
  Three inner models (seeds 0–2) were trained on the other 77 patients; their mean probability on the 19 inner-validation
  patients was swept over the corrected τ grid (1 - logspace(-1, -5, 81) quantiles of the inner-validation scores (04_train_ablation.thr_quantiles, tail_log)) and τ = the maximum pooled event F1 point,
  alarm rule 3 of 5 windows. The rules' τ were chosen the same way on the same split. Same procedure as the evaluated
  versions (`04_train_ablation.py`, inner_split, tail_log grid), except that the three seed models are averaged.
- Inner-validation patients (the same for both lineages): sub-006, sub-009, sub-010, sub-011, sub-012, sub-014, sub-017, sub-021, sub-028, sub-035, sub-037, sub-051, sub-053, sub-067, sub-083, sub-088, sub-091, sub-094, sub-095.
  SHA-256 of this comma-separated list: `7c2e15b128bc3b62dbb10d1953847833294506184823b9e1706faa352601a325`.
- **This τ carries split-to-split randomness.** In development, the single-split τ of C xgb varied across the 15 repeats
  of one fold between 0.961 and 0.994 (v02g; 0.950–0.992 for v11; `report_by_fold_seed.csv`), with held-out FA/24h
  between about 0.01 and 9 per repeat. Averaging three seed models reduces this variability but does not remove it.
  The locked-set operating curve (all τ) is reported next to the frozen-τ result so that the effect of τ is visible.

### Reproduction check before hashing

`src/06_freeze.py --check-tau` re-ran the τ selection of logged development jobs with the same helper functions the freeze
uses (training-data sampling, model, modality dropout, τ grid, τ objective) and compared with the τ in the fold files:

- 10-26 09:56 v02g_grid_fix A_eeg/xgb fold 0 seed 0: tau re-selected 0.972702253 vs logged 0.972702253 (IDENTICAL); inner F1 0.328358 vs logged 0.328358
- 10-26 09:57 v11_causal A_eeg/xgb fold 0 seed 0: tau re-selected 0.968679650 vs logged 0.968679650 (IDENTICAL); inner F1 0.254777 vs logged 0.254777
- 10-26 10:00 v02g_grid_fix C_eeg_ecg_emg_acc/xgb fold 0 seed 0: tau re-selected 0.978864778 vs logged 0.978864778 (IDENTICAL); inner F1 0.375301 vs logged 0.375301
- 10-26 10:02 v11_causal C_eeg_ecg_emg_acc/xgb fold 0 seed 0: tau re-selected 0.977149802 vs logged 0.977149802 (IDENTICAL); inner F1 0.344371 vs logged 0.344371

### Frozen thresholds and post-processing

Alarm = at least k of the last n windows (4-s windows, 2-s hop) with score > τ; impd and flat-EEG windows never alarm and
are removed from the false-alarm hours; alarms closer than 90 s are merged and events longer than 300 s are split.

| lineage | model | τ | alarm rule | inner F1 | inner FA/24h |
|---|---|---|---|---|---|
| v02g | A_eeg/xgb | 0.980945 | 3 of 5 | 0.143 | 1.91 |
| v02g | B_eeg_ecg/xgb | 0.988125 | 3 of 5 | 0.265 | 0.32 |
| v02g | C_eeg_ecg_emg_acc/xgb | 0.989742 | 3 of 5 | 0.286 | 0.14 |
| v02g | rule_hr (`ecg_hr_rise_pct`) | 129.7596 | 3 of 5 | 0.028 | 8.26 |
| v02g | rule_emg (`emg_logrms_rel`) | 4.3567 | 3 of 5 | 0.114 | 2.32 |
| v02g | rule_eeg (`eeg_ll_max_rel`) | 2.3019 | 3 of 5 | 0.050 | 9.40 |
| v02g | rule_acc (`acc_logsd_rel`) | 9.8147 | 3 of 5 | 0.026 | 3.65 |
| v11 | A_eeg/xgb | 0.975529 | 3 of 5 | 0.122 | 3.40 |
| v11 | B_eeg_ecg/xgb | 0.990460 | 3 of 5 | 0.224 | 0.20 |
| v11 | C_eeg_ecg_emg_acc/xgb | 0.984999 | 3 of 5 | 0.273 | 0.35 |
| v11 | rule_hr (`ecg_hr_rise_pct`) | 225.8671 | 3 of 5 | 0.038 | 2.84 |
| v11 | rule_emg (`emg_logrms_rel`) | 4.1149 | 3 of 5 | 0.107 | 4.20 |
| v11 | rule_eeg (`eeg_ll_max_rel`) | 2.4132 | 3 of 5 | 0.048 | 8.24 |
| v11 | rule_acc (`acc_logsd_rel`) | 10.0429 | 3 of 5 | 0.030 | 2.91 |

### Frozen files (SHA-256)

SHA-256 of the file bytes in the working tree at the freeze commit; the git blob ID (`git rev-parse HEAD:<path>`) does
not depend on line-ending settings and identifies the committed content.

| lineage | model | file | SHA-256 | git blob |
|---|---|---|---|---|
| v02g (offline pipeline) | A_eeg/xgb | `models/ictra_v02g_A_eeg_xgb_seed0.json` | `51ca6fb24bfc731b69ac39c42c0b32a33add670c587c8a9e4b5d6b77228cd95d` | `d35cfdec49cfa1a3dea2a558213fc703ce1d1d93` |
| v02g (offline pipeline) | A_eeg/xgb | `models/ictra_v02g_A_eeg_xgb_seed1.json` | `d51e6a21585b02e48ccc63f25f9e9025152a2a0ae72fd91f932ed082eaa093b6` | `e985bd7d8d55ec5c7ba30074d0ffe598f47ed31e` |
| v02g (offline pipeline) | A_eeg/xgb | `models/ictra_v02g_A_eeg_xgb_seed2.json` | `e812d6f52b5874dc0ef4d81231e35730b27df842b2e6075a64702357d9e61a64` | `fd83fd287ca5c46096a93030a9abf15d21da052f` |
| v02g (offline pipeline) | B_eeg_ecg/xgb | `models/ictra_v02g_B_eeg_ecg_xgb_seed0.json` | `a054019a528f2872caf0d54ed9834ebdd3837c5bbdec402c29a2851e478e6cf0` | `558928dbe19eaa1625a04fd5b7166b4c93a57a65` |
| v02g (offline pipeline) | B_eeg_ecg/xgb | `models/ictra_v02g_B_eeg_ecg_xgb_seed1.json` | `682d90401d491787ea4ddb2c1bf0fe6dd5aa60a7378ffc1ca823f7850f332b64` | `d02f2279acc1ed20803d452203b098e8d370fc84` |
| v02g (offline pipeline) | B_eeg_ecg/xgb | `models/ictra_v02g_B_eeg_ecg_xgb_seed2.json` | `95156c110160fb26e7e762518136da714ed88d448d44c4c2b0acfac01b667411` | `6b75a3040f8abd90c2c4b009cc495d7f7b917053` |
| v02g (offline pipeline) | C_eeg_ecg_emg_acc/xgb | `models/ictra_v02g_C_eeg_ecg_emg_acc_xgb_seed0.json` | `f02f74234fe0cefdb41df1aadc5fe0032f59546e35cb3d8978bab2674904c7cb` | `25d38dcab1726ef2fc357e891fe0a207c0438070` |
| v02g (offline pipeline) | C_eeg_ecg_emg_acc/xgb | `models/ictra_v02g_C_eeg_ecg_emg_acc_xgb_seed1.json` | `c0e1ba7e672106fca2ca59f1a8e444259eef280d007d0765733623bb50f819fd` | `c2fa4ed915620d781c6fbe80c993e9e82c246126` |
| v02g (offline pipeline) | C_eeg_ecg_emg_acc/xgb | `models/ictra_v02g_C_eeg_ecg_emg_acc_xgb_seed2.json` | `3c46080f9e2d97498283b036a48b7e81dee621f7dc533af056804b28e973b4dd` | `fc1ae82caf2533896420b9a49ccab788dfd5cee0` |
| v02g (offline pipeline) | metadata (τ, k-of-n, columns, subjects) | `models/ictra_v02g.meta.json` | `fad9ac800cc178aac3cd7a68982d705a7b6f052b6c4a1f406e1f7fad09ee0909` | `2872ed3617bef2200cb646e808201e3125b9aed4` |
| v11 (causal pipeline) | A_eeg/xgb | `models/ictra_v11_A_eeg_xgb_seed0.json` | `0815afbd543713ce15b12b784cf7b01ec9429881323928905a62e06427a6ca3f` | `64041939232a46887abf0a0c331bb89c30871932` |
| v11 (causal pipeline) | A_eeg/xgb | `models/ictra_v11_A_eeg_xgb_seed1.json` | `4ba40fad3f8b2d2f2e70d891213a964463bf6b0a54dd627e14aea91ce3f55122` | `9fe2e9397e246eb6147034f697d590be42059978` |
| v11 (causal pipeline) | A_eeg/xgb | `models/ictra_v11_A_eeg_xgb_seed2.json` | `360b9d793bd40209ef9f9703cfcda9c5ce1bb5a3bf2c9eb413be93e4477eeeef` | `9114d22770369a075201a6da1b7bdbf8ec378b96` |
| v11 (causal pipeline) | B_eeg_ecg/xgb | `models/ictra_v11_B_eeg_ecg_xgb_seed0.json` | `c5354a152d1bcfa87bf3ead1e6b5d0e428d3708fad45eaea055b5c5a799b2089` | `aef720966b972c474e344ea1f4a58665f196994f` |
| v11 (causal pipeline) | B_eeg_ecg/xgb | `models/ictra_v11_B_eeg_ecg_xgb_seed1.json` | `9c16c80ccd101f247c86e44975d598f9cc7dbd63db49a49c59cd8f818c96054b` | `98fd72f362f5b6fd805f8d40574786f64b0d65df` |
| v11 (causal pipeline) | B_eeg_ecg/xgb | `models/ictra_v11_B_eeg_ecg_xgb_seed2.json` | `80c1d1abf58abb079bffb8bff6d4b351848e1b19075e36b513e9f9bc07c7dce0` | `1a1fdc97922922bf5c864b880d922c758cf273f0` |
| v11 (causal pipeline) | C_eeg_ecg_emg_acc/xgb | `models/ictra_v11_C_eeg_ecg_emg_acc_xgb_seed0.json` | `0dc6dedf78247b3fdb377ed537e4e6440b95ebfc79a844bdef5e85c5c1a13b9c` | `7cbf2e9be1bc76549ee804a90e28e8c2650dbbea` |
| v11 (causal pipeline) | C_eeg_ecg_emg_acc/xgb | `models/ictra_v11_C_eeg_ecg_emg_acc_xgb_seed1.json` | `9d5e0a5aa1bad9547b8e0599d2047bd86f8f71242c6c48508e7e91b266892c7c` | `895333dfe5595577829daa2d74b5b7969854e1f4` |
| v11 (causal pipeline) | C_eeg_ecg_emg_acc/xgb | `models/ictra_v11_C_eeg_ecg_emg_acc_xgb_seed2.json` | `400f80a5b2480a158240969e76e6f2113d95fc00bce591713ea24bb1b281a16c` | `3e4c587eaff734617533965e3990af88283960f2` |
| v11 (causal pipeline) | metadata (τ, k-of-n, columns, subjects) | `models/ictra_v11.meta.json` | `bf06b3a8c0ec2bb28ac470bd7940cbf28f673a1ea7060b3d7fab79bb2e309a8c` | `ab4b9d93d6dfd690e5b5bba8afb73fcf763bd83e` |

### Code that must not change (SHA-256 at freeze)

| file | SHA-256 | git blob |
|---|---|---|
| `src/ictra_lib.py` | `9051cf852c2a1eb83d99104f4e5fb20b71cfb6ea192b399ff202391db93467e9` | `c78f77be847230e745c366928c35f2b98694cd70` |
| `src/03_features.py` | `b4be31b42fc0abf791072b259d6102707c6742076212019aad009282b4a94bfe` | `b0c849919e3ab678705316e48aa94c6bd37ba08e` |
| `src/04_train_ablation.py` | `8943cca169065738e20319d06fe9b0985609008ed817dcb03f8cf81038f514c7` | `b51a20f689782e19c36a7d6fd4e93afebb87cb76` |
| `src/05_operating_points.py` | `1245ff88ca32adada77c227b2565da3d17155e7522b7862156c2f4469f255b75` | `58aeb8d9383118001e94591af2bc935c34e32ef2` |
| `src/06_freeze.py` | `5a2ec3e71f95edff7a989adba88b061de1c757eed3f3440f530bb5d118afc4d8` | `da8948c89351d9e2c089cf934af566734f6d3484` |
| `src/02_relationship_study.py` | `3f7ffa3352d63e66e1e62592169f63f7123286775b54ed9af3cab92cf04722d0` | `4550628e3cb272c78666e115901fb06af2261056` |

## Metrics (SzCORE event scoring, `ictra_lib.score_events`)

- A detection counts if an alarm overlaps a reference seizure, allowing 30 s before onset and 60 s after offset. Never
  accuracy or specificity.
- **Primary aggregation: per patient, then averaged** (`CLAUDE.md`; `docs/EVALUATION_PROTOCOL.md` §4). Each metric is
  computed for every locked-test patient and the patient values are averaged:
  - sensitivity: detected / annotated seizures, over patients with at least one seizure in the group considered;
  - FA/24h: false alarms × 24 / unmasked hours, over all 29 patients (seizure-free patients included);
  - precision: true alarms / alarms, over patients with at least one alarm;
  - F1: 2 · precision · sensitivity / (precision + sensitivity) per patient with at least one seizure (0 if the patient
    has no alarm or no detection), averaged over those patients;
  - delay (start of the first overlapping alarm − annotated onset): median over each patient's detected seizures, then
    the median over patients with at least one detected seizure.
- 95% intervals by patient bootstrap (1,000 resamples of patients; all patient-level metrics recomputed in each
  resample). Every model comparison uses a **paired** patient bootstrap (the same resampled patients for both models).
- Secondary, for comparison with the development tables (which pooled counts over patients): the same metrics with
  counts summed over all locked-test patients. The development per-patient-average values are recomputed from
  `scores_by_patient.csv` so that both aggregations can be compared with development.
- Operating points: the frozen τ (headline) and the operating curve (τ swept on the locked set; a curve, not a
  deployable number), with per-patient-average sensitivity at per-patient-average FA/24h ≤ 1, 2, 5, 10.
- Subgroups (all reported, none chosen afterwards): seizure group (FBTC, focal motor, focal non-motor, focal unknown
  motor); seizure duration (< 15 s, 15–60 s, > 60 s); EEG montage (left, right, generalized); with and without the 15
  seizures that lack ECG/EMG. FBTC results (22 seizures, 9 patients) are reported with CIs and are exploratory, never a
  superiority claim (`docs/EVALUATION_PROTOCOL.md` §5).
- The offline − causal difference for C (ΔF1, Δsensitivity, ΔFA/24h, Δdelay; paired patient bootstrap) is reported as a
  result (development value, pooled: ΔF1 −0.011 [−0.027, +0.007]).

## Hypotheses (copied verbatim from docs/PREREGISTRATION_DRAFT.md)

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

## Analysis plan for H1–H4 (how each hypothesis is decided; fixed before opening)

- **Both lineages, separately.** Every hypothesis is tested on the offline (v02g) and on the causal (v11) frozen models
  separately, with the same procedure, and both results are reported side by side. H3a is a property of the signals and
  is computed once with the Exp R definitions.
- **Metrics in the tests** are the per-patient averages defined above; where a hypothesis defines its own statistic
  (H3a: median lag over seizures; H3b: median delay over each model's detected seizures), that definition is used as
  written. No correction for multiple comparisons; every result is reported, held or failed.
- **H1** (C vs A on FBTC + focal motor seizures, in each lineage):
  1. Target sensitivity = A's per-patient-average sensitivity on FBTC + focal motor seizures at A's frozen τ.
  2. C's operating curve: C's τ is swept over the same grid as the development operating curves
     (`05_operating_points.py`); at each τ, C's per-patient-average sensitivity on FBTC + focal motor seizures and
     per-patient-average FA/24h are computed. The matched point is the τ with the lowest FA/24h among those whose
     sensitivity is ≥ the target.
  3. ΔFA = C's FA/24h at the matched point − A's FA/24h at its frozen τ.
  4. Paired patient bootstrap: in each resample, steps 1–3 are recomputed on the resampled patients.
  5. **Held if ΔFA < 0 and P(ΔFA < 0) ≥ 0.90.**
- **H2** (B vs A on focal non-motor seizures, in each lineage): Δ = B − A per-patient-average sensitivity on focal
  non-motor seizures, each at its frozen τ, over patients with at least one focal non-motor seizure. **Held if Δ > 0 and
  paired patient bootstrap P(Δ > 0) ≥ 0.90.** Also reported (not decisive): the same comparison at matched
  per-patient-average FA/24h = 1 on the operating curves.
- **H3a:** computed with `src/02_relationship_study.py` (same definitions and baseline) on the locked-test FBTC seizures.
  Held if the median of lag_eeg_to_X is > 0 for each of X = HR, ACC, EMG; the medians, their patient-bootstrap CIs and
  the n of each are reported, also under z > 2 / 5 s.
- **H3b:** as written above, on the locked non-FBTC focal seizures, alarms at each model's frozen τ, in each lineage.
- **H4:** as written above, at the frozen thresholds, per-patient-average F1, paired patient bootstrap
  P(ΔF1 > 0) ≥ 0.90 for each comparison, in each lineage; "best"/"max" rule = highest locked-set per-patient-average F1.

## External tests (tier 3; each opened once, each with its own results file)

No public dataset has all four signals, so model C is tested only on tier 2. External EEG is the behind-the-ear proxy
from scalp EEG (`ictra_lib.bte_proxy_from_scalp`: T7−P7 and T8−P8) with the EEG montage indicator set to
"generalized" (`ictra_lib.montage_features`). Signals are harmonised by the same `ictra_lib` code (resampling, filters,
mains notch by recording country). Both lineages are scored. Metrics as above (per patient, then averaged).

| external set (if access is obtained) | signals | frozen models scored | headline |
|---|---|---|---|
| Billeci, Zenodo 1415495 | EEG + ECG | **A and B** (as planned); `rule_hr`, `rule_eeg` | sensitivity, FA/24h |
| Siena, PhysioNet | EEG + ECG | **A and B** (as planned); `rule_hr`, `rule_eeg` | same; Billeci–Siena patient overlap checked and reported first |
| TUSZ v2.0.3 eval split | EEG + EKG | **A**; **B** on files with EKG (as planned); rules | same, by seizure type |
| My Seizure Gauge Seer, Sensor Dot | EMG + ACC | `rule_emg`, `rule_acc` (deviation 3) | sensitivity on motor seizures, FA/24h |
| PhysioNet szdb | ECG | `rule_hr` (deviation 3) | sanity check only (5 patients, 11 seizures) |
| Mayo MSG, Zenodo 17380899 | HR (1 Hz) | `rule_hr` (as planned) | FA/24h over months at home |
| OSDB | wrist ACC + HR | `rule_acc`, `rule_hr` (deviation 3) | share of real false alarms not raised |

## Deviations from the evaluation plan and the draft, stated before the test

1. **Only xgboost is frozen; lightgbm does not enter the locked test.** `docs/PREREGISTRATION_DRAFT.md` (two-model
   section) says H1–H4 are tested on each model. This pre-registration departs from that: only the xgboost models A, B
   and C (and the four rules) are frozen and tested, in both lineages; no lightgbm model is frozen, and no hypothesis is
   tested on lightgbm. Reason: xgboost has been the primary family since v01 (`results/dev/v01_baseline/NOTES.md`). The
   lightgbm development results are still reported in the development tables as before.
2. **τ from one inner split of an averaged ensemble** (see "How τ was chosen"): the development results are per-seed
   models with per-fold τ, averaged as metrics; the frozen models are 3-seed averages with one τ.
3. **External sub-models not built.** `docs/EVALUATION_PROTOCOL.md` §3.2 names learned single-branch models (EMG+ACC,
   ECG-only, ACC-only) and §3.3 a late-fusion design; none was built or evaluated on development data, so none is
   frozen. The external sets that would have used them are scored with the frozen single-signal rules instead:
   - **My Seizure Gauge Seer** (EMG + ACC): planned EMG+ACC sub-model → `rule_emg` and `rule_acc`;
   - **PhysioNet szdb** (ECG): planned ECG sub-model / HR rule → `rule_hr` only;
   - **Mayo MSG** (HR 1 Hz): planned HR-jump rule → `rule_hr` (unchanged in kind; the frozen `rule_hr` feature is the
     % HR rise over the causal 5-min baseline);
   - **OSDB** (wrist ACC + HR): planned ACC sub-model → `rule_acc` and `rule_hr`.
   Billeci, Siena and TUSZ keep the frozen models A and B as planned.
4. **§3.1 items not implemented:** per-recording 5th–95th percentile scaling and left/right flip augmentation. Features
   are normalised against each recording's causal 5-min rolling baseline (`*_rel` features) instead.
5. **Primary aggregation per patient**, as the protocol requires, whereas the development tables pooled counts over
   patients; both are reported (see Metrics).

All results are reported whether or not they support these hypotheses. No model, threshold or code listed here changes
after the locked set is opened; the set becomes development data from then on.

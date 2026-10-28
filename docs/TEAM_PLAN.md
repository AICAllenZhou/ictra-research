# ICTRA team plan: who does what for the fusion model

Written 2026-10-17. Deadline: WAICY submission 29 Nov 2026. Everyone reads `CLAUDE.md` and `docs/EVALUATION_PROTOCOL.md`
first. One dataset trains everything (SeizeIT2 dev patients); each person owns one signal branch, then the branches meet
in the fusion step. Task numbers refer to `docs/TASKS.md`.

## The model, in one picture

```
 EEG (2 ch) ──► EEG features ──► Model A ─┐
 ECG (1 ch) ──► HR / HRV features ────────┼─► Model B = A + ECG ─┐
 EMG (1 ch) ──► envelope / burst features ┤                      ├─► Model C = B + EMG + ACC  ──► k-of-n rule ──► alarm
 ACC (3 ax) ──► movement features ────────┘                      │
                                        late fusion: one sub-model per signal, probabilities averaged, missing signal re-weighted
```

Two fusion styles are compared on the dev folds and the better one is frozen:
- **Early fusion** (default in `04_train_ablation.py`): one XGBoost over the concatenated feature blocks.
- **Late fusion** (version candidate v06): one sub-model per signal, probabilities averaged with weights fit on the inner
  split; a missing or flat signal is dropped and the weights renormalised. Degrades gracefully when an electrode falls off.

## Owners

| Person | Branch | Owns | Deliverables (file) | Due |
|---|---|---|---|---|
| **Helena** | **EEG branch + website** | Model A; EEG features in `ictra_lib.eeg_features`; montage handling (left / right / generalized); EEG external tests (CHB-MIT, Siena via T7–P7 / T8–P8 proxy); the replay mode and alert view of the website | `results/dev/*/summary.csv` rows `A_eeg`, `rule_eeg`; `docs/DATA_NOTES.md` EEG section; `src/08_external_adapters.py` CHB-MIT + Siena loaders; website replay demo | Tasks 2, 3, 5 (A rows) by Oct 24; Task 9 adapters by Nov 7 |
| **Carina** | **ECG branch + relationship study** | Model B's ECG block; HR / HRV features; `rule_hr`; Exp R (when does the heart respond relative to EEG onset, by seizure type); ECG external tests (Billeci, Siena ECG, szdb); flat / saturation audit (done) | `results/expR/FINDINGS.md` + timeline figures; `summary.csv` rows `B_eeg_ecg`, `ecg_only`, `rule_hr`; Billeci/szdb adapters | Task 4 by Oct 28; Task 5 (B rows) by Oct 31 |
| **Tiffany** | **EMG + ACC branch** | Model C's EMG and ACC blocks; `emg_features`, `acc_features`; `rule_emg`, `rule_acc`; the second relationship study on Arbune 2020 (ictal EMG → post-ictal EEG suppression); EMG+ACC external test (My Seizure Gauge Sensor Dot, after registration); electrode-placement write-up for the deltoid site | `summary.csv` rows `emg_only`, `acc_only`, `emg_acc`, `rule_emg`, `rule_acc`; `results/expR_emg/FINDINGS.md` (Arbune); Seer adapter | Task 5 (EMG/ACC rows) by Oct 31; Arbune study by Nov 7 |
| **Constance** | **Fusion, evaluation and honesty layer** | Model C runs; late-fusion version (v06); operating-point sweep (Task 6); the ablation figure (sensitivity vs FA/24h, A → B → C and the rules, per seizure group); `ITERATION_LOG.md` upkeep; pre-registration draft; independent re-computation (Task 13); ethics + limitations sections of the report | `results/dev/*/fig_ablation.png`, `operating_points.csv`; `docs/PREREGISTRATION.md` draft; `results/INDEPENDENT_REVIEW.md` | Task 6 by Oct 31; iteration log weekly; pre-registration draft Nov 10 |
| **Allen (coach)** | **Infrastructure, hardware, review** | Downloads and compute; code review of every PR; `biosignal-studio` fixes (ADC saturation, multi-channel 256 Hz, accelerometer); volunteer nuisance trials; final freeze, locked test and external tests run once (Tasks 10–12); approves every ITERATION_LOG entry | frozen `models/`; `results/locked_test/RESULTS.md`; `results/external/SUMMARY.md` | Nov 14 |

Everyone: can explain, in 30 seconds, why their signal helps, what it misses, and what the A → B → C table shows.

## Shared rules for the branches

- Every branch's features go into `src/ictra_lib.py` behind one function; nobody keeps a private copy. A change there means
  re-running `03_features.py` for every run and a new version name.
- Every branch reports the same four numbers on the same dev folds: event sensitivity, FA/24h, F1, median delay, with
  patient-bootstrap CIs, by seizure group (`04_train_ablation.py` does this; do not write another evaluator).
- A branch is "done" when its single-signal rule, its single-signal model and its contribution inside Model C are all in
  one `summary.csv`, with a one-paragraph NOTES.md: what the signal catches, what it misses, where its false alarms come from.
- Nobody opens sub-097..125 or any external dataset's labels before `docs/PREREGISTRATION.sha256` exists.
- Datasets: one copy of SeizeIT2, downloaded with `01_download_seizeit2.py` (SHA-256 verified). Kaggle re-uploads of
  SeizeIT2 are not used (see below). No external "healthy" negatives.

## Weekly rhythm

- Monday: each owner posts their `summary.csv` rows and NOTES.md in the team doc; Constance updates the ablation figure.
- Wednesday: Allen reviews PRs; one accepted/rejected version per branch per week is the pace.
- Friday: 15-minute Q&A rehearsal, one question per person.

## EMG dataset decision (for Tiffany, 2026-10-17)

The three Kaggle links were checked:
- `furqanfareed/emg-epilleptic-seizure-dataset-of-upper-arm`: a 125.85 GB verbatim re-upload of OpenNeuro ds005873 (SeizeIT2),
  no attribution, licence "unknown", and it accidentally includes someone's Kaggle API credential file. **Duplicate. Not used.**
- `talhaarshad1/processed-emg-seizure-batch-001`: the EMG channel of the first ~5 SeizeIT2 subjects dumped to `.npz`, no README,
  no sampling rate, no labels. **Duplicate, undocumented. Not used.**
- `jeannepomuscene/myoclonic-seizure-ecg-and-emg`: a notebook that generates random numbers (`np.random.normal`, random labels;
  models score 50%). **Synthetic. Not used.**

A second sweep (Zenodo, Figshare, Mendeley, Dryad, Harvard Dataverse, OSF, IEEE DataPort, PhysioNet, OpenNeuro, Chinese sources)
found no other public surface-EMG seizure dataset. The complete open sEMG-seizure corpus is:
1. **SeizeIT2** deltoid EMG, 256 Hz, 856 seizures with an EMG file (55 FBTC) — the training data; the EMG branch is trained here.
2. **Arbune et al. 2020** (Zenodo 3885755 = Dryad 10.5061/dryad.14004q5), deltoid + biceps EMG of convulsive seizures in 42
   patients with post-ictal EEG-suppression (PGES) labels, 794 MB, open — the second relationship study and an external
   check of the EMG burst features on convulsive seizures.
3. **My Seizure Gauge** (Epilepsy Ecosystem / Seer), Byteflies Sensor Dot EMG + ACC, 5 patients, registration required — the
   EMG+ACC external test.
Everything else (Brain Sentinel SPEAC, Beniczky/Conradsen, NightWatch, Böttcher RADAR-CNS, Chinese wristband sEMG) was never released.

## Who presents what (added 2026-10-18, after Exp R, Exp P and v01)

Each member owns one branch of the evidence, presents it in 30 s and takes the questions on it. The same split is on
the website's Team page (`ictra-academic-site/team.html#parts`) and as owner tags on the Evidence exhibits.

| Person | Owns on the evidence page | 30-second part | Questions to be ready for | Next task |
|---|---|---|---|---|
| **Helena** | Model A rows of v01 (0.98 FBTC, 0.37 overall); the montage finding (right 0.17 → 0.45 without sub-039/047/035); EEG feature design | Two ear electrodes see rhythmic discharges within seconds; alone they catch nearly every convulsive seizure but a third of focal ones — the floor the other signals must lift | Why line length? Why 1–25 Hz? What does bte-EEG miss? Why did the right montage look worse? | v04 montage-aware features; CHB-MIT/Siena loaders (T7–P7 proxy); website replay demo |
| **Carina** | Exp R (Exhibit 1); Exp P (null result); the flat-ECG audit; B vs A (+0.043, not robust) | In 695 seizures the heart follows the brain by 4 s in convulsive seizures and moves with it in focal ones — heart rate adds coverage, not speed; and in the two minutes before a seizure there is nothing to see | Why was H3 rewritten? What is the control in Exp P? Why does rule_hr beat the models on seizures < 15 s? | v04 relationship features (HR rise within 20 s of EEG jump); Billeci + szdb adapters |
| **Tiffany** | C vs A (+11 points at 1 FA/day, F1 +0.064, P 0.998, all 5 folds); EMG/ACC timing in Exp R (z ≈ 30–40 in FBTC; late response in 59 % of non-motor); the Kaggle duplicate decision | Muscle and movement are the most specific signals we have; adding them to EEG and heart catches 11 points more seizures at the same false-alarm rate — the biggest single gain in the project | Why the deltoid? Why not a wrist watch? What does ACC see that EMG does not? | Arbune 2020 study; My Seizure Gauge external test |
| **Constance** | Three tiers; Dead ends section; the accept rule and iteration log; ethics and limits (40 % invisible non-motor) | First version: 97 % of convulsive seizures at 1 FA/day, fails on short focal ones, and we can show which patients and why; every number is from a patient we never trained on, under a rule written before we looked | Why not accuracy? What is a patient bootstrap? What if the locked test disagrees with dev? | v06 late fusion + modality dropout; PREREGISTRATION.md; independent re-computation script |
| **Allen** | Pipeline, hardware, demo | — | — | Hardware line: wearable prototype + real-time causal inference video by Nov 10 |

Attribution is by responsibility: the first sprint (17–18 Oct) was built by the coach with an AI coding assistant; each
member takes over, verifies and extends their branch and presents only what they can explain.

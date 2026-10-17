# ICTRA Lab: one page for progress, results and the WAICY figures

Modelled on the SlopeAI Explorer (`../SlopeAI/results/explorer/`, built by `src/28_build_explorer.py` from
`evolution.json` and friends). Two faces of the same page:

- **Live mode** (tonight, on Allen's PC): `python src/90_lab_server.py` serves `results/lab/index.html` at
  127.0.0.1:8766 and refreshes every 30 s from JSON status files. Anyone on the team can watch downloads, feature
  extraction, training jobs and new results appear.
- **Static build** (for sharing and for the WAICY submission): `python src/91_build_lab.py` writes a single
  self-contained `results/lab/ictra_lab.html` (inline CSS/JS/data, PNG figures as data URIs) that Allen publishes as a
  Claude artifact and links from the project website. Same page, frozen data.

## Data contract: every script writes JSON, the page only reads

| File (under `results/lab/data/`) | Written by | Content |
|---|---|---|
| `status.json` | `01_download_seizeit2.py`, `03_features.py`, `04_train_ablation.py` (append on every job) | per-stage progress: files/GB downloaded vs total, runs featured vs downloaded, jobs done/running/queued with set, model, fold, seed, start time, minutes; disk free GB; last update time |
| `catalog.json` | `00_seizeit2_catalog.py` | patients, hours, seizures by group and split, montage counts, modality availability |
| `expR.json` | `02_relationship_study.py` | median z-curves per signal per seizure group (−120..+180 s, 1 Hz), lag tables, response shares, both onset definitions |
| `dev_<version>.json` | `04_train_ablation.py` → `summarise()` | summary table rows (set, model, sens, CI, FA/24h, CI, F1, delay, per-group sens), per-patient points for the dot plot, operating-point curves from `05_operating_points.py` |
| `evolution.json` | hand-edited + `compare` mode of 04 | one entry per version: id, label, change, result, decision, metrics, lesson (SlopeAI rule: never rewrite old entries) |
| `locked_test.json`, `external_<dataset>.json` | 07 / 08 | same shape as dev_, plus hypothesis table (held/failed) |
| `hardware.json` | biosignal-studio logs | volunteer nuisance-trial FA/hour, replay demo status |

Rule: the page never computes a metric; it only draws what the scripts wrote. That keeps the website numbers identical
to the report numbers (independent review checks the JSON, not the page).

## Tabs

1. **Now** (live): download bars (seizure runs / seizure-free runs / locked test, grey until pre-registration), feature
   extraction counter, training job grid (one cell per set×model×fold×seed: queued / running / done with F1), disk gauge,
   log tail. Turns into a static "how it was built" timeline in the frozen build.
2. **Data**: SeizeIT2 at a glance (patients, hours, seizures by group, stacked by split), montage pie, modality coverage,
   the dataset inventory table (keep / test / drop, with reasons), the three-tier split diagram.
3. **Signals** (Exp R): the timeline figure per seizure group with the IQR band; a lag dot-plot (EEG→HR, EEG→EMG, EMG→ACC)
   per group; a "which signal responds" heat table (group × signal, % of seizures). Toggle: lag relative to annotation vs
   relative to first bte-EEG change.
4. **Ablation**: the headline chart, sensitivity vs FA/24h (log x) with A, B, C and the four rules as points with CI bars,
   one panel per seizure group; the summary table; per-patient dot plot (sensitivity per patient, A vs C) to show the
   spread; operating-point table (sens at 0.5 / 1 / 2 / 5 FA/24h).
5. **How it evolved**: the version chain (accepted = filled, rejected = hollow, like SlopeAI), each with change, result,
   lesson; the accept/reject rule printed at the top.
6. **Tests**: locked test (hypotheses held/failed, by group, with and without missing-ECG seizures) and external tests
   (one card per dataset: what it has, which model was tested, sens / FA/24h / CI, the gap vs dev). Greyed with a lock
   icon until each is opened; shows the pre-registration hash once it exists.
7. **Honesty**: leakage checklist (ticked), limits, what we did not test, independent-review status, flat-signal and
   dropout-repair stats, the late-bte-EEG-onset distribution.
8. **Figures for WAICY**: every figure below rendered at 1920×1080 and 1080×1080, light and dark, downloadable as
   PNG/SVG, with its caption and source line baked in. Same palette as the project website (EEG blue, ECG orange,
   EMG aqua, ACC grey-green, accent deep teal, alert red only for alarms).

## The WAICY figure list (slide → figure)

| ID | Figure | Slide | Source JSON |
|---|---|---|---|
| F1 | Seizures missed: Hoppe 2007 + witnesses 2025 (already on website) | 1–2 | static |
| F2 | Three signals, one seizure: the Exp R timeline for FBTC (median z, IQR), annotated with the lags | 3 | expR |
| F3 | Lag dot-plot by seizure group (EEG → heart → muscle → movement) | 3 | expR |
| F4 | Architecture diagram (already on website) | 4 | static |
| F5 | Data: 125 patients / 11,626 h / 883 seizures by group, the three-tier split | 5 | catalog |
| F6 | **The headline**: sensitivity vs FA/24h, A → B → C and the four rules, per seizure group, with CIs | 6 | dev_v<final> |
| F7 | Per-patient dot plot: A vs C sensitivity per patient | 6 | dev_v<final> |
| F8 | How it evolved: version chain with accept/reject | 6 | evolution |
| F9 | Tests: locked + external cards, dev vs test gap | 6 | locked_test, external_* |
| F10 | Honesty: share of seizures with no bte-EEG change, by group; flat-signal hours; leakage checklist | 8 | expR, status |
| F11 | Hardware: volunteer nuisance trials FA/hour on the team's board; replay screenshot | 7 | hardware |

Each figure is produced by one function in `src/92_figures.py` that reads JSON and writes PNG+SVG to
`results/lab/figures/F<n>_<slug>_{light,dark}_{16x9,1x1}.{png,svg}`; the Lab page embeds the same files. No figure is
drawn twice in two places.

## Tonight's minimum (before v01 finishes)

- `status.json` written by 01/03/04 (append-only, cheap), `90_lab_server.py` serving a first page with tab 1 and tab 2.
- Tab 3 and tab 4 appear automatically when `expR.json` and `dev_v01_baseline.json` land.
Everything else is built over the following days; the static build and F-figures by Nov 14 (Task 10 freeze).

## Implementation notes for Claude Code

- Plain HTML + inline JS (Chart.js or D3 inlined), no build step, no framework; dark/light via `prefers-color-scheme`
  with a toggle; phone-responsive. The static build must be one file under 16 MB.
- `04_train_ablation.py` already writes per-job CSVs; add a tiny `lab_status(update: dict)` helper in `ictra_lib.py`
  that merges into `results/lab/data/status.json` with a file lock, and call it from 01/03/04.
- Keep numbers formatted once, in JS, from raw JSON (no pre-rounded strings), so the independent review can diff JSON.
- Follow `docs/EVALUATION_PROTOCOL.md`: never show accuracy; always FA/24h; always CIs; always rules next to models;
  locked/external tabs stay greyed until the pre-registration hash exists.

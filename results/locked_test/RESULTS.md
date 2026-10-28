# Locked internal test — results

**These are the results of the locked internal test (sub-097..125), run once as pre-registered in `docs/PREREGISTRATION.md`. From now on this set is development data and cannot be used for an independent test.**

Run 2026-10-28 10:20:59. pre-registration SHA-256 bc5048c88ae3524348215e4c62801ec66fedbe5c4e4ac616bcbb3c3944fedca6 OK; 26 frozen model/metadata/code files match their recorded SHA-256.

## Data

- 29 patients; 2,906 unmasked hours; 181 seizures (FBTC 22, focal_motor 54, focal_nonmotor 75, focal_unknown_motor 30); 15 without ECG/EMG.

## Hypotheses (decided as in the pre-registration's analysis plan)

Decisions use all seizures of the pre-registered set and per-patient averages (the pre-registered primary aggregation). The same hypotheses are also shown without the seizures that lack ECG/EMG.

### All seizures (pre-registered set)

| hypothesis | lineage | decision | estimate | 95% CI | bootstrap |
|---|---|---|---|---|---|
| H1 | v02g | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -0.82 (C 0.04 vs A 0.86; target sensitivity 0.381) | [-1.29, -0.36] | P(ΔFA < 0) = 1.000 |
| H2 | v02g | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = +0.083 (B 0.169, A 0.086); secondary at FA/24h ≤ 1: +0.214 | [+0.000, +0.200] | P(Δ > 0) = 0.881 |
| H3b | v02g | **held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = -1.5 s (over seizures both detect: -6.0 s); Δsensitivity (B − A) = +0.102 | Δdelay [-16.0, +30.0] s; Δsens [+0.024, +0.192] | for reference only: P(Δdelay ≥ −2) = 0.680; P(Δsens > 0) = 0.996 |
| H4 (A vs rule_eeg) | v02g | **held** | ΔF1 = +0.161 (A 0.215, rule_eeg 0.054; best of rule_eeg) | [+0.063, +0.274] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v02g | **held** | ΔF1 = +0.260 (B 0.314, rule_eeg 0.054; best of rule_eeg, rule_hr) | [+0.147, +0.388] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v02g | **held** | ΔF1 = +0.282 (C 0.337, rule_emg 0.055; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.149, +0.418] | P(ΔF1 > 0) = 1.000 |
| H1 | v11 | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -1.88 (C 0.18 vs A 2.06; target sensitivity 0.452) | [-2.83, +0.71] | P(ΔFA < 0) = 0.946 |
| H2 | v11 | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = +0.081 (B 0.169, A 0.088); secondary at FA/24h ≤ 1: +0.083 | [-0.005, +0.200] | P(Δ > 0) = 0.881 |
| H3b | v11 | **held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = +8.5 s (over seizures both detect: -8.5 s); Δsensitivity (B − A) = +0.007 | Δdelay [-14.0, +32.0] s; Δsens [-0.069, +0.093] | for reference only: P(Δdelay ≥ −2) = 0.793; P(Δsens > 0) = 0.566 |
| H4 (A vs rule_eeg) | v11 | **held** | ΔF1 = +0.123 (A 0.175, rule_eeg 0.052; best of rule_eeg) | [+0.044, +0.221] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v11 | **held** | ΔF1 = +0.233 (B 0.286, rule_eeg 0.052; best of rule_eeg, rule_hr) | [+0.122, +0.358] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v11 | **held** | ΔF1 = +0.205 (C 0.264, rule_emg 0.059; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.095, +0.323] | P(ΔF1 > 0) = 1.000 |
| offline−causal Δf1 (C) | v11 − v02g | **reported** | -0.073 | [-0.135, -0.017] |  |
| offline−causal Δsensitivity (C) | v11 − v02g | **reported** | +0.017 | [-0.008, +0.051] |  |
| offline−causal Δfa_per_24h (C) | v11 − v02g | **reported** | +0.761 | [+0.176, +1.601] |  |
| offline−causal Δdelay_s (C) | v11 − v02g | **reported** | +0.750 | [-18.000, +9.012] |  |
| H3a (z > 3, 3 s) | signals (Exp R) | **not held** | EEG→HR -1.0 s [-15.0, +6.0], n = 21; EEG→ACC +3.0 s [+0.0, +14.0], n = 19; EEG→EMG +0.0 s [-7.0, +21.0], n = 21 | | decision rule: median > 0 for all three |
| H3a (z > 2, 5 s) | signals (Exp R) | **reported** | EEG→HR +0.0 s [-14.0, +6.0], n = 21; EEG→ACC +11.0 s [+1.0, +17.0], n = 19; EEG→EMG +2.0 s [-6.0, +21.0], n = 21 | | robustness, not decisive |

H3b is decided by the point estimates, as the hypothesis is worded in the pre-registration (Δdelay ≥ −2 s and B's sensitivity higher than A's; no bootstrap threshold). Its bootstrap probabilities and 95% CIs are reported for reference only. H1, H2 and H4 use P ≥ 0.90 as pre-registered.

### Without the seizures lacking ECG/EMG

| hypothesis | lineage | decision | estimate | 95% CI | bootstrap |
|---|---|---|---|---|---|
| H1 | v02g | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -0.85 (C 0.01 vs A 0.86; target sensitivity 0.396) | [-1.29, -0.37] | P(ΔFA < 0) = 1.000 |
| H2 | v02g | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = +0.091 (B 0.184, A 0.094); secondary at FA/24h ≤ 1: +0.234 | [+0.000, +0.219] | P(Δ > 0) = 0.881 |
| H3b | v02g | **not held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = -7.0 s (over seizures both detect: -6.0 s); Δsensitivity (B − A) = +0.109 | Δdelay [-17.0, +27.8] s; Δsens [+0.027, +0.207] | for reference only: P(Δdelay ≥ −2) = 0.529; P(Δsens > 0) = 0.997 |
| H4 (A vs rule_eeg) | v02g | **held** | ΔF1 = +0.163 (A 0.218, rule_eeg 0.055; best of rule_eeg) | [+0.062, +0.278] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v02g | **held** | ΔF1 = +0.271 (B 0.326, rule_eeg 0.055; best of rule_eeg, rule_hr) | [+0.153, +0.399] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v02g | **held** | ΔF1 = +0.292 (C 0.349, rule_emg 0.057; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.154, +0.433] | P(ΔF1 > 0) = 1.000 |
| H1 | v11 | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -1.88 (C 0.18 vs A 2.06; target sensitivity 0.471) | [-2.86, +0.27] | P(ΔFA < 0) = 0.962 |
| H2 | v11 | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = +0.088 (B 0.184, A 0.096); secondary at FA/24h ≤ 1: +0.091 | [-0.006, +0.215] | P(Δ > 0) = 0.881 |
| H3b | v11 | **held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = +1.0 s (over seizures both detect: -8.5 s); Δsensitivity (B − A) = +0.011 | Δdelay [-15.0, +32.0] s; Δsens [-0.068, +0.102] | for reference only: P(Δdelay ≥ −2) = 0.694; P(Δsens > 0) = 0.600 |
| H4 (A vs rule_eeg) | v11 | **held** | ΔF1 = +0.123 (A 0.177, rule_eeg 0.054; best of rule_eeg) | [+0.042, +0.225] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v11 | **held** | ΔF1 = +0.243 (B 0.298, rule_eeg 0.054; best of rule_eeg, rule_hr) | [+0.129, +0.372] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v11 | **held** | ΔF1 = +0.213 (C 0.273, rule_emg 0.061; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.098, +0.331] | P(ΔF1 > 0) = 1.000 |
| offline−causal Δf1 (C) | v11 − v02g | **reported** | -0.075 | [-0.139, -0.018] |  |
| offline−causal Δsensitivity (C) | v11 − v02g | **reported** | +0.018 | [-0.009, +0.053] |  |
| offline−causal Δfa_per_24h (C) | v11 − v02g | **reported** | +0.761 | [+0.176, +1.601] |  |
| offline−causal Δdelay_s (C) | v11 − v02g | **reported** | +0.750 | [-19.000, +10.012] |  |
| H3a (z > 3, 3 s) | signals (Exp R) | **not held** | EEG→HR -1.0 s [-15.0, +6.0], n = 21; EEG→ACC +3.5 s [+0.0, +14.0], n = 18; EEG→EMG +0.0 s [-7.0, +21.0], n = 21 | | decision rule: median > 0 for all three |
| H3a (z > 2, 5 s) | signals (Exp R) | **reported** | EEG→HR +0.0 s [-14.0, +6.0], n = 21; EEG→ACC +12.5 s [+1.0, +17.0], n = 18; EEG→EMG +2.0 s [-6.0, +21.0], n = 21 | | robustness, not decisive |

H3b is decided by the point estimates, as the hypothesis is worded in the pre-registration (Δdelay ≥ −2 s and B's sensitivity higher than A's; no bootstrap threshold). Its bootstrap probabilities and 95% CIs are reported for reference only. H1, H2 and H4 use P ≥ 0.90 as pre-registered.

## Frozen-τ results — all seizures, per-patient average (primary)

Pre-registered primary aggregation: each metric per patient, then averaged over patients; patient-bootstrap 95% CIs.

| lineage | model | sensitivity [CI] | FA/24h [CI] | precision | F1 [CI] | median delay s |
|---|---|---|---|---|---|---|
| v02g | A | 0.273 [0.145, 0.413] | 0.86 [0.48, 1.32] | 0.303 | 0.215 [0.113, 0.338] | 24.00 |
| v02g | B | 0.356 [0.221, 0.499] | 0.42 [0.23, 0.64] | 0.447 | 0.314 [0.196, 0.448] | 28.00 |
| v02g | C | 0.331 [0.197, 0.480] | 0.23 [0.06, 0.48] | 0.657 | 0.337 [0.205, 0.477] | 39.25 |
| v02g | rule_hr | 0.215 [0.113, 0.332] | 15.60 [7.11, 27.84] | 0.036 | 0.050 [0.024, 0.080] | 47.00 |
| v02g | rule_emg | 0.102 [0.027, 0.187] | 3.04 [1.27, 5.07] | 0.071 | 0.055 [0.009, 0.116] | 61.00 |
| v02g | rule_eeg | 0.331 [0.190, 0.491] | 14.15 [7.59, 23.58] | 0.039 | 0.054 [0.030, 0.086] | 31.50 |
| v02g | rule_acc | 0.043 [0.000, 0.095] | 2.53 [0.29, 5.16] | 0.005 | 0.004 [0.000, 0.009] | 1.00 |
| v11 | A | 0.312 [0.174, 0.456] | 2.06 [1.15, 3.10] | 0.203 | 0.175 [0.091, 0.284] | 27.50 |
| v11 | B | 0.308 [0.176, 0.455] | 0.30 [0.12, 0.52] | 0.560 | 0.286 [0.170, 0.414] | 31.00 |
| v11 | C | 0.348 [0.206, 0.506] | 0.99 [0.26, 2.05] | 0.443 | 0.264 [0.158, 0.389] | 40.00 |
| v11 | rule_hr | 0.074 [0.017, 0.138] | 4.58 [1.61, 8.77] | 0.026 | 0.022 [0.002, 0.061] | 86.00 |
| v11 | rule_emg | 0.136 [0.057, 0.230] | 4.39 [2.06, 7.36] | 0.049 | 0.059 [0.017, 0.115] | 73.50 |
| v11 | rule_eeg | 0.294 [0.154, 0.451] | 11.37 [6.96, 16.67] | 0.041 | 0.052 [0.027, 0.086] | 35.75 |
| v11 | rule_acc | 0.026 [0.000, 0.069] | 2.46 [0.24, 5.10] | 0.004 | 0.003 [0.000, 0.008] | -2.50 |

## Frozen-τ results — all seizures, pooled counts (secondary)

Secondary: counts summed over all patients, then the rates computed; for comparison with the development tables.

| lineage | model | seizures | detected | sensitivity | FA/24h | precision | F1 | median delay s |
|---|---|---|---|---|---|---|---|---|
| v02g | A | 181 | 28 | 0.155 | 0.93 | 0.200 | 0.174 | 26.50 |
| v02g | B | 181 | 32 | 0.177 | 0.40 | 0.400 | 0.245 | 31.00 |
| v02g | C | 181 | 29 | 0.160 | 0.15 | 0.617 | 0.254 | 36.00 |
| v02g | rule_hr | 181 | 21 | 0.116 | 16.47 | 0.010 | 0.019 | 49.00 |
| v02g | rule_emg | 181 | 11 | 0.061 | 3.57 | 0.025 | 0.035 | 61.00 |
| v02g | rule_eeg | 181 | 42 | 0.232 | 14.47 | 0.023 | 0.043 | 23.50 |
| v02g | rule_acc | 181 | 4 | 0.022 | 2.21 | 0.015 | 0.018 | 1.00 |
| v11 | A | 181 | 32 | 0.177 | 2.06 | 0.113 | 0.138 | 26.50 |
| v11 | B | 181 | 27 | 0.149 | 0.23 | 0.491 | 0.229 | 35.00 |
| v11 | C | 181 | 32 | 0.177 | 0.60 | 0.305 | 0.224 | 37.00 |
| v11 | rule_hr | 181 | 6 | 0.033 | 4.80 | 0.010 | 0.016 | 82.50 |
| v11 | rule_emg | 181 | 16 | 0.088 | 5.13 | 0.027 | 0.041 | 60.50 |
| v11 | rule_eeg | 181 | 39 | 0.215 | 12.13 | 0.027 | 0.047 | 29.00 |
| v11 | rule_acc | 181 | 3 | 0.017 | 2.11 | 0.012 | 0.014 | 0.00 |

## Frozen-τ results — without the seizures lacking ECG/EMG, per-patient average

Pre-registered primary aggregation: each metric per patient, then averaged over patients; patient-bootstrap 95% CIs.

| lineage | model | sensitivity [CI] | FA/24h [CI] | precision | F1 [CI] | median delay s |
|---|---|---|---|---|---|---|
| v02g | A | 0.279 [0.145, 0.428] | 0.86 [0.48, 1.32] | 0.303 | 0.218 [0.111, 0.342] | 25.75 |
| v02g | B | 0.373 [0.233, 0.521] | 0.42 [0.23, 0.64] | 0.447 | 0.326 [0.205, 0.464] | 28.00 |
| v02g | C | 0.342 [0.205, 0.491] | 0.23 [0.06, 0.48] | 0.657 | 0.349 [0.214, 0.490] | 39.25 |
| v02g | rule_hr | 0.226 [0.121, 0.354] | 15.60 [7.11, 27.84] | 0.036 | 0.051 [0.025, 0.083] | 47.00 |
| v02g | rule_emg | 0.108 [0.028, 0.199] | 3.04 [1.27, 5.07] | 0.071 | 0.057 [0.010, 0.120] | 61.00 |
| v02g | rule_eeg | 0.340 [0.194, 0.503] | 14.15 [7.59, 23.58] | 0.039 | 0.055 [0.030, 0.088] | 38.00 |
| v02g | rule_acc | 0.045 [0.000, 0.100] | 2.53 [0.29, 5.16] | 0.005 | 0.004 [0.000, 0.009] | 1.00 |
| v11 | A | 0.320 [0.179, 0.468] | 2.06 [1.15, 3.10] | 0.203 | 0.177 [0.090, 0.287] | 28.75 |
| v11 | B | 0.323 [0.191, 0.474] | 0.30 [0.12, 0.52] | 0.560 | 0.298 [0.176, 0.431] | 31.00 |
| v11 | C | 0.359 [0.218, 0.515] | 0.99 [0.26, 2.05] | 0.443 | 0.273 [0.159, 0.398] | 40.00 |
| v11 | rule_hr | 0.077 [0.019, 0.144] | 4.58 [1.61, 8.77] | 0.026 | 0.023 [0.002, 0.062] | 86.00 |
| v11 | rule_emg | 0.144 [0.058, 0.244] | 4.39 [2.06, 7.36] | 0.049 | 0.061 [0.017, 0.121] | 73.50 |
| v11 | rule_eeg | 0.304 [0.159, 0.457] | 11.37 [6.96, 16.67] | 0.041 | 0.054 [0.028, 0.087] | 35.75 |
| v11 | rule_acc | 0.027 [0.000, 0.074] | 2.46 [0.24, 5.10] | 0.004 | 0.003 [0.000, 0.008] | -2.50 |

## Frozen-τ results — without the seizures lacking ECG/EMG, pooled counts (secondary)

Secondary: counts summed over all patients, then the rates computed; for comparison with the development tables.

| lineage | model | seizures | detected | sensitivity | FA/24h | precision | F1 | median delay s |
|---|---|---|---|---|---|---|---|---|
| v02g | A | 166 | 26 | 0.157 | 0.93 | 0.200 | 0.176 | 26.50 |
| v02g | B | 166 | 32 | 0.193 | 0.40 | 0.400 | 0.260 | 31.00 |
| v02g | C | 166 | 28 | 0.169 | 0.15 | 0.617 | 0.265 | 36.00 |
| v02g | rule_hr | 166 | 21 | 0.127 | 16.47 | 0.010 | 0.019 | 49.00 |
| v02g | rule_emg | 166 | 11 | 0.066 | 3.57 | 0.025 | 0.036 | 61.00 |
| v02g | rule_eeg | 166 | 40 | 0.241 | 14.47 | 0.023 | 0.043 | 24.00 |
| v02g | rule_acc | 166 | 4 | 0.024 | 2.21 | 0.015 | 0.018 | 1.00 |
| v11 | A | 166 | 30 | 0.181 | 2.06 | 0.113 | 0.139 | 26.50 |
| v11 | B | 166 | 27 | 0.163 | 0.23 | 0.491 | 0.244 | 35.00 |
| v11 | C | 166 | 31 | 0.187 | 0.60 | 0.305 | 0.232 | 35.00 |
| v11 | rule_hr | 166 | 6 | 0.036 | 4.80 | 0.010 | 0.016 | 82.50 |
| v11 | rule_emg | 166 | 16 | 0.096 | 5.13 | 0.027 | 0.042 | 60.50 |
| v11 | rule_eeg | 166 | 38 | 0.229 | 12.13 | 0.027 | 0.048 | 28.00 |
| v11 | rule_acc | 166 | 3 | 0.018 | 2.11 | 0.012 | 0.014 | 0.00 |

## Subgroups at the frozen τ (models A, B, C) — all seizures

Sensitivity per patient, then averaged (primary, with patient-bootstrap 95% CI) next to the pooled sensitivity (secondary: detected / all seizures of the subgroup). The two can differ a lot when a few patients have many seizures. Montage rows also give FA/24h and F1 per patient. Rules: `subgroups_*.csv`.

| lineage | model | subgroup | seizures | patients | sensitivity [CI] (per patient, primary) | pooled sensitivity (secondary) | median delay s | FA/24h (montage) | F1 (montage) |
|---|---|---|---|---|---|---|---|---|---|
| v02g | A | group: FBTC | 22 | 9 | 0.811 [0.550, 1.000] | 0.864 | 23.75 |  |  |
| v02g | A | group: focal_motor | 54 | 15 | 0.172 [0.011, 0.369] | 0.130 | 20.50 |  |  |
| v02g | A | group: focal_nonmotor | 75 | 12 | 0.086 [0.000, 0.275] | 0.027 | 21.50 |  |  |
| v02g | A | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | A | duration: < 15 s | 45 | 9 | 0.011 [0.000, 0.037] | 0.022 | 8.00 |  |  |
| v02g | A | duration: 15-60 s | 85 | 18 | 0.085 [0.000, 0.206] | 0.047 | 28.00 |  |  |
| v02g | A | duration: > 60 s | 51 | 20 | 0.405 [0.220, 0.609] | 0.451 | 27.25 |  |  |
| v02g | A | montage: left | 122 | 15 | 0.300 [0.118, 0.505] | 0.148 | 25.75 | 1.34 | 0.173 |
| v02g | A | montage: right | 37 | 10 | 0.157 [0.000, 0.389] | 0.081 | 21.00 | 0.41 | 0.178 |
| v02g | A | montage: generalized | 22 | 4 | 0.464 [0.000, 1.000] | 0.318 | 149.50 | 0.21 | 0.464 |
| v02g | B | group: FBTC | 22 | 9 | 0.793 [0.550, 0.972] | 0.818 | 34.25 |  |  |
| v02g | B | group: focal_motor | 54 | 15 | 0.356 [0.153, 0.593] | 0.167 | 19.00 |  |  |
| v02g | B | group: focal_nonmotor | 75 | 12 | 0.169 [0.003, 0.369] | 0.053 | 29.00 |  |  |
| v02g | B | group: focal_unknown_motor | 30 | 6 | 0.017 [0.000, 0.056] | 0.033 | 9.00 |  |  |
| v02g | B | duration: < 15 s | 45 | 9 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | B | duration: 15-60 s | 85 | 18 | 0.158 [0.039, 0.292] | 0.094 | 23.25 |  |  |
| v02g | B | duration: > 60 s | 51 | 20 | 0.462 [0.265, 0.670] | 0.471 | 35.00 |  |  |
| v02g | B | montage: left | 122 | 15 | 0.352 [0.154, 0.568] | 0.148 | 27.50 | 0.56 | 0.284 |
| v02g | B | montage: right | 37 | 10 | 0.323 [0.125, 0.563] | 0.189 | 27.25 | 0.24 | 0.324 |
| v02g | B | montage: generalized | 22 | 4 | 0.454 [0.025, 1.000] | 0.318 | 28.00 | 0.34 | 0.402 |
| v02g | C | group: FBTC | 22 | 9 | 0.774 [0.527, 0.952] | 0.818 | 35.25 |  |  |
| v02g | C | group: focal_motor | 54 | 15 | 0.322 [0.111, 0.560] | 0.148 | 37.50 |  |  |
| v02g | C | group: focal_nonmotor | 75 | 12 | 0.167 [0.000, 0.367] | 0.040 | 39.00 |  |  |
| v02g | C | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | C | duration: < 15 s | 45 | 9 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | C | duration: 15-60 s | 85 | 18 | 0.120 [0.000, 0.246] | 0.059 | 28.00 |  |  |
| v02g | C | duration: > 60 s | 51 | 20 | 0.453 [0.255, 0.661] | 0.471 | 45.00 |  |  |
| v02g | C | montage: left | 122 | 15 | 0.333 [0.134, 0.556] | 0.131 | 43.00 | 0.35 | 0.336 |
| v02g | C | montage: right | 37 | 10 | 0.273 [0.079, 0.519] | 0.162 | 35.00 | 0.11 | 0.299 |
| v02g | C | montage: generalized | 22 | 4 | 0.464 [0.000, 1.000] | 0.318 | 151.25 | 0.06 | 0.438 |
| v11 | A | group: FBTC | 22 | 9 | 0.811 [0.550, 1.000] | 0.864 | 26.75 |  |  |
| v11 | A | group: focal_motor | 54 | 15 | 0.272 [0.078, 0.500] | 0.167 | 34.00 |  |  |
| v11 | A | group: focal_nonmotor | 75 | 12 | 0.088 [0.000, 0.278] | 0.040 | 23.00 |  |  |
| v11 | A | group: focal_unknown_motor | 30 | 6 | 0.017 [0.000, 0.056] | 0.033 | 17.00 |  |  |
| v11 | A | duration: < 15 s | 45 | 9 | 0.011 [0.000, 0.037] | 0.022 | 10.00 |  |  |
| v11 | A | duration: 15-60 s | 85 | 18 | 0.095 [0.007, 0.218] | 0.071 | 23.50 |  |  |
| v11 | A | duration: > 60 s | 51 | 20 | 0.455 [0.261, 0.666] | 0.490 | 27.50 |  |  |
| v11 | A | montage: left | 122 | 15 | 0.368 [0.178, 0.581] | 0.172 | 30.00 | 3.42 | 0.141 |
| v11 | A | montage: right | 37 | 10 | 0.157 [0.000, 0.389] | 0.081 | 25.00 | 0.64 | 0.118 |
| v11 | A | montage: generalized | 22 | 4 | 0.489 [0.025, 1.000] | 0.364 | 26.00 | 0.49 | 0.446 |
| v11 | B | group: FBTC | 22 | 9 | 0.774 [0.533, 0.958] | 0.773 | 37.50 |  |  |
| v11 | B | group: focal_motor | 54 | 15 | 0.278 [0.094, 0.500] | 0.111 | 25.00 |  |  |
| v11 | B | group: focal_nonmotor | 75 | 12 | 0.169 [0.003, 0.369] | 0.053 | 33.00 |  |  |
| v11 | B | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | B | duration: < 15 s | 45 | 9 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | B | duration: 15-60 s | 85 | 18 | 0.094 [0.002, 0.191] | 0.059 | 28.50 |  |  |
| v11 | B | duration: > 60 s | 51 | 20 | 0.445 [0.242, 0.652] | 0.431 | 37.00 |  |  |
| v11 | B | montage: left | 122 | 15 | 0.309 [0.134, 0.517] | 0.131 | 29.25 | 0.41 | 0.265 |
| v11 | B | montage: right | 37 | 10 | 0.273 [0.079, 0.519] | 0.162 | 25.00 | 0.22 | 0.274 |
| v11 | B | montage: generalized | 22 | 4 | 0.393 [0.000, 1.000] | 0.227 | 159.00 | 0.13 | 0.393 |
| v11 | C | group: FBTC | 22 | 9 | 0.922 [0.800, 1.000] | 0.909 | 31.50 |  |  |
| v11 | C | group: focal_motor | 54 | 15 | 0.311 [0.107, 0.548] | 0.130 | 40.00 |  |  |
| v11 | C | group: focal_nonmotor | 75 | 12 | 0.211 [0.003, 0.455] | 0.067 | 39.00 |  |  |
| v11 | C | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | C | duration: < 15 s | 45 | 9 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | C | duration: 15-60 s | 85 | 18 | 0.141 [0.002, 0.286] | 0.082 | 42.00 |  |  |
| v11 | C | duration: > 60 s | 51 | 20 | 0.510 [0.308, 0.711] | 0.490 | 38.50 |  |  |
| v11 | C | montage: left | 122 | 15 | 0.344 [0.147, 0.566] | 0.148 | 40.50 | 1.54 | 0.285 |
| v11 | C | montage: right | 37 | 10 | 0.307 [0.082, 0.583] | 0.189 | 19.00 | 0.45 | 0.212 |
| v11 | C | montage: generalized | 22 | 4 | 0.464 [0.000, 1.000] | 0.318 | 148.00 | 0.24 | 0.317 |

## Subgroups at the frozen τ (models A, B, C) — without the seizures lacking ECG/EMG

Sensitivity per patient, then averaged (primary, with patient-bootstrap 95% CI) next to the pooled sensitivity (secondary: detected / all seizures of the subgroup). The two can differ a lot when a few patients have many seizures. Montage rows also give FA/24h and F1 per patient. Rules: `subgroups_*.csv`.

| lineage | model | subgroup | seizures | patients | sensitivity [CI] (per patient, primary) | pooled sensitivity (secondary) | median delay s | FA/24h (montage) | F1 (montage) |
|---|---|---|---|---|---|---|---|---|---|
| v02g | A | group: FBTC | 21 | 9 | 0.811 [0.550, 1.000] | 0.857 | 22.25 |  |  |
| v02g | A | group: focal_motor | 41 | 14 | 0.179 [0.000, 0.389] | 0.146 | 28.00 |  |  |
| v02g | A | group: focal_nonmotor | 74 | 11 | 0.094 [0.000, 0.303] | 0.027 | 21.50 |  |  |
| v02g | A | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | A | duration: < 15 s | 35 | 8 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | A | duration: 15-60 s | 82 | 17 | 0.090 [0.000, 0.222] | 0.049 | 28.00 |  |  |
| v02g | A | duration: > 60 s | 49 | 20 | 0.403 [0.220, 0.608] | 0.449 | 27.00 |  |  |
| v02g | A | montage: left | 109 | 14 | 0.316 [0.123, 0.538] | 0.156 | 28.00 | 1.34 | 0.176 |
| v02g | A | montage: right | 36 | 10 | 0.157 [0.000, 0.389] | 0.083 | 21.00 | 0.41 | 0.178 |
| v02g | A | montage: generalized | 21 | 4 | 0.458 [0.000, 1.000] | 0.286 | 147.50 | 0.21 | 0.461 |
| v02g | B | group: FBTC | 21 | 9 | 0.811 [0.550, 1.000] | 0.857 | 34.25 |  |  |
| v02g | B | group: focal_motor | 41 | 14 | 0.381 [0.166, 0.617] | 0.220 | 19.00 |  |  |
| v02g | B | group: focal_nonmotor | 74 | 11 | 0.184 [0.004, 0.409] | 0.054 | 29.00 |  |  |
| v02g | B | group: focal_unknown_motor | 30 | 6 | 0.017 [0.000, 0.056] | 0.033 | 9.00 |  |  |
| v02g | B | duration: < 15 s | 35 | 8 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | B | duration: 15-60 s | 82 | 17 | 0.167 [0.041, 0.301] | 0.098 | 23.25 |  |  |
| v02g | B | duration: > 60 s | 49 | 20 | 0.468 [0.269, 0.676] | 0.490 | 35.00 |  |  |
| v02g | B | montage: left | 109 | 14 | 0.377 [0.174, 0.605] | 0.165 | 27.50 | 0.56 | 0.305 |
| v02g | B | montage: right | 36 | 10 | 0.323 [0.125, 0.563] | 0.194 | 27.25 | 0.24 | 0.324 |
| v02g | B | montage: generalized | 21 | 4 | 0.483 [0.025, 1.000] | 0.333 | 28.00 | 0.34 | 0.408 |
| v02g | C | group: FBTC | 21 | 9 | 0.774 [0.527, 0.952] | 0.810 | 35.25 |  |  |
| v02g | C | group: focal_motor | 41 | 14 | 0.345 [0.125, 0.591] | 0.195 | 37.50 |  |  |
| v02g | C | group: focal_nonmotor | 74 | 11 | 0.182 [0.000, 0.409] | 0.041 | 39.00 |  |  |
| v02g | C | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | C | duration: < 15 s | 35 | 8 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | C | duration: 15-60 s | 82 | 17 | 0.127 [0.000, 0.260] | 0.061 | 28.00 |  |  |
| v02g | C | duration: > 60 s | 49 | 20 | 0.452 [0.255, 0.659] | 0.469 | 45.00 |  |  |
| v02g | C | montage: left | 109 | 14 | 0.357 [0.156, 0.584] | 0.147 | 43.00 | 0.35 | 0.360 |
| v02g | C | montage: right | 36 | 10 | 0.273 [0.079, 0.519] | 0.167 | 35.00 | 0.11 | 0.299 |
| v02g | C | montage: generalized | 21 | 4 | 0.458 [0.000, 1.000] | 0.286 | 149.50 | 0.06 | 0.435 |
| v11 | A | group: FBTC | 21 | 9 | 0.811 [0.550, 1.000] | 0.857 | 26.25 |  |  |
| v11 | A | group: focal_motor | 41 | 14 | 0.286 [0.077, 0.500] | 0.195 | 38.00 |  |  |
| v11 | A | group: focal_nonmotor | 74 | 11 | 0.096 [0.000, 0.306] | 0.041 | 23.00 |  |  |
| v11 | A | group: focal_unknown_motor | 30 | 6 | 0.017 [0.000, 0.056] | 0.033 | 17.00 |  |  |
| v11 | A | duration: < 15 s | 35 | 8 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | A | duration: 15-60 s | 82 | 17 | 0.100 [0.008, 0.229] | 0.073 | 23.50 |  |  |
| v11 | A | duration: > 60 s | 49 | 20 | 0.453 [0.260, 0.665] | 0.490 | 27.50 |  |  |
| v11 | A | montage: left | 109 | 14 | 0.389 [0.189, 0.618] | 0.183 | 33.50 | 3.42 | 0.143 |
| v11 | A | montage: right | 36 | 10 | 0.157 [0.000, 0.389] | 0.083 | 25.00 | 0.64 | 0.118 |
| v11 | A | montage: generalized | 21 | 4 | 0.483 [0.025, 1.000] | 0.333 | 24.00 | 0.49 | 0.444 |
| v11 | B | group: FBTC | 21 | 9 | 0.789 [0.546, 0.967] | 0.810 | 37.50 |  |  |
| v11 | B | group: focal_motor | 41 | 14 | 0.298 [0.104, 0.526] | 0.146 | 25.00 |  |  |
| v11 | B | group: focal_nonmotor | 74 | 11 | 0.184 [0.004, 0.409] | 0.054 | 33.00 |  |  |
| v11 | B | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | B | duration: < 15 s | 35 | 8 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | B | duration: 15-60 s | 82 | 17 | 0.100 [0.002, 0.204] | 0.061 | 28.50 |  |  |
| v11 | B | duration: > 60 s | 49 | 20 | 0.450 [0.250, 0.657] | 0.449 | 37.00 |  |  |
| v11 | B | montage: left | 109 | 14 | 0.331 [0.150, 0.552] | 0.147 | 29.25 | 0.41 | 0.284 |
| v11 | B | montage: right | 36 | 10 | 0.273 [0.079, 0.519] | 0.167 | 25.00 | 0.22 | 0.274 |
| v11 | B | montage: generalized | 21 | 4 | 0.417 [0.000, 1.000] | 0.238 | 159.00 | 0.13 | 0.404 |
| v11 | C | group: FBTC | 21 | 9 | 0.922 [0.800, 1.000] | 0.905 | 31.50 |  |  |
| v11 | C | group: focal_motor | 41 | 14 | 0.333 [0.114, 0.578] | 0.171 | 40.00 |  |  |
| v11 | C | group: focal_nonmotor | 74 | 11 | 0.230 [0.004, 0.500] | 0.068 | 39.00 |  |  |
| v11 | C | group: focal_unknown_motor | 30 | 6 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | C | duration: < 15 s | 35 | 8 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v11 | C | duration: 15-60 s | 82 | 17 | 0.149 [0.002, 0.300] | 0.085 | 42.00 |  |  |
| v11 | C | duration: > 60 s | 49 | 20 | 0.508 [0.307, 0.710] | 0.490 | 38.50 |  |  |
| v11 | C | montage: left | 109 | 14 | 0.369 [0.163, 0.596] | 0.165 | 40.50 | 1.54 | 0.305 |
| v11 | C | montage: right | 36 | 10 | 0.307 [0.082, 0.583] | 0.194 | 19.00 | 0.45 | 0.212 |
| v11 | C | montage: generalized | 21 | 4 | 0.458 [0.000, 1.000] | 0.286 | 147.50 | 0.24 | 0.315 |

Subgroups (seizure group, duration, montage), operating-curve points (sensitivity at FA/24h ≤ 1, 2, 5, 10; τ swept on this set — a curve, not a deployable number), per-patient tables and the H3a onsets are in the CSV files next to this file.

*This is the locked test result. From now on sub-097..125 are development data.*

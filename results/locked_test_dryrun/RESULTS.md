# DRY RUN (development patients, in-sample — NOT a result)

**Dry run of `src/07_locked_test.py` on sub-001..096.** The frozen models were trained on these patients, so every number here is in-sample. This file only shows that the script runs end to end. It is not a result.

Run 2026-10-26 12:24:34. pre-registration SHA-256 bc5048c88ae3524348215e4c62801ec66fedbe5c4e4ac616bcbb3c3944fedca6 OK; 26 frozen model/metadata/code files match their recorded SHA-256.

## Data

- 96 patients; 8,642 unmasked hours; 702 seizures (FBTC 33, focal_motor 317, focal_nonmotor 249, focal_unknown_motor 103); 12 without ECG/EMG.

## Hypotheses (decided as in the pre-registration's analysis plan)

Decisions use all seizures of the pre-registered set and per-patient averages (the pre-registered primary aggregation). The same hypotheses are also shown without the seizures that lack ECG/EMG.

### All seizures (pre-registered set)

| hypothesis | lineage | decision | estimate | 95% CI | bootstrap |
|---|---|---|---|---|---|
| H1 | v02g | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -1.10 (C 0.15 vs A 1.25; target sensitivity 0.488) | [-1.89, -0.52] | P(ΔFA < 0) = 0.999 |
| H2 | v02g | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = -0.027 (B 0.122, A 0.149); secondary at FA/24h ≤ 1: +0.049 | [-0.096, +0.020] | P(Δ > 0) = 0.201 |
| H3b | v02g | **held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = +4.0 s (over seizures both detect: +0.0 s); Δsensitivity (B − A) = +0.053 | Δdelay [-4.0, +11.0] s; Δsens [-0.005, +0.117] | for reference only: P(Δdelay ≥ −2) = 0.948; P(Δsens > 0) = 0.967 |
| H4 (A vs rule_eeg) | v02g | **held** | ΔF1 = +0.240 (A 0.280, rule_eeg 0.040; best of rule_eeg) | [+0.183, +0.299] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v02g | **held** | ΔF1 = +0.308 (B 0.347, rule_eeg 0.040; best of rule_eeg, rule_hr) | [+0.244, +0.375] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v02g | **held** | ΔF1 = +0.318 (C 0.362, rule_emg 0.044; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.243, +0.396] | P(ΔF1 > 0) = 1.000 |
| H1 | v11 | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -2.03 (C 0.60 vs A 2.63; target sensitivity 0.584) | [-3.29, -0.46] | P(ΔFA < 0) = 0.994 |
| H2 | v11 | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = -0.083 (B 0.073, A 0.156); secondary at FA/24h ≤ 1: +0.078 | [-0.156, -0.019] | P(Δ > 0) = 0.003 |
| H3b | v11 | **not held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = +9.0 s (over seizures both detect: +8.5 s); Δsensitivity (B − A) = -0.114 | Δdelay [-1.0, +14.0] s; Δsens [-0.178, -0.059] | for reference only: P(Δdelay ≥ −2) = 0.987; P(Δsens > 0) = 0.000 |
| H4 (A vs rule_eeg) | v11 | **held** | ΔF1 = +0.235 (A 0.278, rule_eeg 0.044; best of rule_eeg) | [+0.179, +0.292] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v11 | **held** | ΔF1 = +0.229 (B 0.272, rule_eeg 0.044; best of rule_eeg, rule_hr) | [+0.168, +0.298] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v11 | **held** | ΔF1 = +0.282 (C 0.331, rule_emg 0.049; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.219, +0.348] | P(ΔF1 > 0) = 1.000 |
| offline−causal Δf1 (C) | v11 − v02g | **reported** | -0.031 | [-0.080, +0.014] |  |
| offline−causal Δsensitivity (C) | v11 − v02g | **reported** | +0.031 | [+0.000, +0.064] |  |
| offline−causal Δfa_per_24h (C) | v11 − v02g | **reported** | +0.445 | [+0.269, +0.656] |  |
| offline−causal Δdelay_s (C) | v11 − v02g | **reported** | -4.500 | [-12.250, +3.750] |  |
| H3a (z > 3, 3 s) | signals (Exp R) | **held** | EEG→HR +3.5 s [-1.0, +16.5], n = 26; EEG→ACC +8.5 s [+1.5, +16.0], n = 32; EEG→EMG +12.0 s [+1.0, +26.0], n = 26 | | decision rule: median > 0 for all three |
| H3a (z > 2, 5 s) | signals (Exp R) | **reported** | EEG→HR +3.0 s [-1.0, +19.0], n = 29; EEG→ACC +8.5 s [+1.0, +18.0], n = 32; EEG→EMG +15.0 s [-3.5, +24.0], n = 27 | | robustness, not decisive |

H3b is decided by the point estimates, as the hypothesis is worded in the pre-registration (Δdelay ≥ −2 s and B's sensitivity higher than A's; no bootstrap threshold). Its bootstrap probabilities and 95% CIs are reported for reference only. H1, H2 and H4 use P ≥ 0.90 as pre-registered.

### Without the seizures lacking ECG/EMG

| hypothesis | lineage | decision | estimate | 95% CI | bootstrap |
|---|---|---|---|---|---|
| H1 | v02g | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -1.13 (C 0.12 vs A 1.25; target sensitivity 0.465) | [-1.89, -0.52] | P(ΔFA < 0) = 0.999 |
| H2 | v02g | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = -0.028 (B 0.125, A 0.153); secondary at FA/24h ≤ 1: +0.050 | [-0.098, +0.021] | P(Δ > 0) = 0.201 |
| H3b | v02g | **held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = +4.0 s (over seizures both detect: +0.0 s); Δsensitivity (B − A) = +0.054 | Δdelay [-4.0, +11.0] s; Δsens [-0.005, +0.120] | for reference only: P(Δdelay ≥ −2) = 0.948; P(Δsens > 0) = 0.967 |
| H4 (A vs rule_eeg) | v02g | **held** | ΔF1 = +0.238 (A 0.279, rule_eeg 0.041; best of rule_eeg) | [+0.180, +0.298] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v02g | **held** | ΔF1 = +0.307 (B 0.349, rule_eeg 0.041; best of rule_eeg, rule_hr) | [+0.244, +0.377] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v02g | **held** | ΔF1 = +0.314 (C 0.359, rule_emg 0.045; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.238, +0.394] | P(ΔF1 > 0) = 1.000 |
| H1 | v11 | **held** | ΔFA/24h (C at matched sensitivity − A at frozen τ) = -2.03 (C 0.60 vs A 2.63; target sensitivity 0.565) | [-3.29, -0.46] | P(ΔFA < 0) = 0.994 |
| H2 | v11 | **not held** | Δsensitivity on focal non-motor (B − A, frozen τ) = -0.085 (B 0.075, A 0.160); secondary at FA/24h ≤ 1: +0.080 | [-0.161, -0.019] | P(Δ > 0) = 0.003 |
| H3b | v11 | **not held** | Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = +9.0 s (over seizures both detect: +8.5 s); Δsensitivity (B − A) = -0.117 | Δdelay [-1.0, +14.0] s; Δsens [-0.182, -0.059] | for reference only: P(Δdelay ≥ −2) = 0.987; P(Δsens > 0) = 0.000 |
| H4 (A vs rule_eeg) | v11 | **held** | ΔF1 = +0.236 (A 0.282, rule_eeg 0.046; best of rule_eeg) | [+0.175, +0.293] | P(ΔF1 > 0) = 1.000 |
| H4 (B vs rule_eeg) | v11 | **held** | ΔF1 = +0.235 (B 0.281, rule_eeg 0.046; best of rule_eeg, rule_hr) | [+0.172, +0.304] | P(ΔF1 > 0) = 1.000 |
| H4 (C vs rule_emg) | v11 | **held** | ΔF1 = +0.281 (C 0.332, rule_emg 0.051; best of rule_eeg, rule_hr, rule_emg, rule_acc) | [+0.218, +0.350] | P(ΔF1 > 0) = 1.000 |
| offline−causal Δf1 (C) | v11 − v02g | **reported** | -0.027 | [-0.078, +0.020] |  |
| offline−causal Δsensitivity (C) | v11 − v02g | **reported** | +0.033 | [+0.001, +0.067] |  |
| offline−causal Δfa_per_24h (C) | v11 − v02g | **reported** | +0.445 | [+0.269, +0.656] |  |
| offline−causal Δdelay_s (C) | v11 − v02g | **reported** | -4.750 | [-12.500, +4.256] |  |
| H3a (z > 3, 3 s) | signals (Exp R) | **held** | EEG→HR +3.5 s [-0.5, +17.0], n = 26; EEG→ACC +5.0 s [+0.0, +16.0], n = 29; EEG→EMG +12.0 s [+1.0, +26.5], n = 26 | | decision rule: median > 0 for all three |
| H3a (z > 2, 5 s) | signals (Exp R) | **reported** | EEG→HR +3.0 s [-1.0, +19.0], n = 29; EEG→ACC +6.0 s [+1.0, +19.5], n = 29; EEG→EMG +15.0 s [-2.0, +25.5], n = 27 | | robustness, not decisive |

H3b is decided by the point estimates, as the hypothesis is worded in the pre-registration (Δdelay ≥ −2 s and B's sensitivity higher than A's; no bootstrap threshold). Its bootstrap probabilities and 95% CIs are reported for reference only. H1, H2 and H4 use P ≥ 0.90 as pre-registered.

## Frozen-τ results — all seizures, per-patient average (primary)

Pre-registered primary aggregation: each metric per patient, then averaged over patients; patient-bootstrap 95% CIs.

| lineage | model | sensitivity [CI] | FA/24h [CI] | precision | F1 [CI] | median delay s |
|---|---|---|---|---|---|---|
| v02g | A | 0.352 [0.276, 0.433] | 1.25 [0.66, 2.06] | 0.434 | 0.280 [0.222, 0.341] | 29.00 |
| v02g | B | 0.398 [0.318, 0.488] | 0.60 [0.38, 0.88] | 0.555 | 0.347 [0.283, 0.417] | 33.50 |
| v02g | C | 0.364 [0.283, 0.453] | 0.20 [0.13, 0.29] | 0.683 | 0.362 [0.286, 0.441] | 36.50 |
| v02g | rule_hr | 0.145 [0.092, 0.210] | 10.63 [7.20, 14.49] | 0.057 | 0.035 [0.021, 0.052] | 43.00 |
| v02g | rule_emg | 0.071 [0.035, 0.113] | 3.20 [2.30, 4.24] | 0.091 | 0.044 [0.021, 0.070] | 41.00 |
| v02g | rule_eeg | 0.282 [0.210, 0.370] | 15.86 [12.05, 20.32] | 0.040 | 0.040 [0.026, 0.054] | 22.00 |
| v02g | rule_acc | 0.019 [0.003, 0.044] | 3.03 [1.02, 5.43] | 0.003 | 0.002 [0.000, 0.004] | 17.25 |
| v11 | A | 0.416 [0.336, 0.499] | 2.63 [1.44, 4.12] | 0.352 | 0.278 [0.219, 0.336] | 33.00 |
| v11 | B | 0.286 [0.212, 0.371] | 0.34 [0.18, 0.54] | 0.606 | 0.272 [0.207, 0.348] | 39.00 |
| v11 | C | 0.394 [0.315, 0.482] | 0.65 [0.42, 0.91] | 0.491 | 0.331 [0.267, 0.397] | 32.00 |
| v11 | rule_hr | 0.046 [0.015, 0.090] | 2.41 [1.26, 3.74] | 0.094 | 0.024 [0.007, 0.047] | 33.00 |
| v11 | rule_emg | 0.126 [0.075, 0.182] | 5.02 [3.69, 6.50] | 0.070 | 0.049 [0.025, 0.076] | 40.50 |
| v11 | rule_eeg | 0.268 [0.197, 0.352] | 14.38 [10.67, 18.80] | 0.046 | 0.044 [0.029, 0.060] | 29.00 |
| v11 | rule_acc | 0.016 [0.000, 0.043] | 2.71 [0.91, 4.87] | 0.003 | 0.002 [0.000, 0.005] | 38.00 |

## Frozen-τ results — all seizures, pooled counts (secondary)

Secondary: counts summed over all patients, then the rates computed; for comparison with the development tables.

| lineage | model | seizures | detected | sensitivity | FA/24h | precision | F1 | median delay s |
|---|---|---|---|---|---|---|---|---|
| v02g | A | 702 | 156 | 0.222 | 1.04 | 0.292 | 0.252 | 28.50 |
| v02g | B | 702 | 184 | 0.262 | 0.47 | 0.521 | 0.349 | 32.00 |
| v02g | C | 702 | 157 | 0.224 | 0.17 | 0.724 | 0.342 | 32.00 |
| v02g | rule_hr | 702 | 68 | 0.097 | 10.54 | 0.018 | 0.030 | 32.50 |
| v02g | rule_emg | 702 | 42 | 0.060 | 3.37 | 0.034 | 0.044 | 37.50 |
| v02g | rule_eeg | 702 | 120 | 0.171 | 16.41 | 0.020 | 0.036 | 19.50 |
| v02g | rule_acc | 702 | 7 | 0.010 | 2.10 | 0.009 | 0.010 | 1.00 |
| v11 | A | 702 | 196 | 0.279 | 2.24 | 0.195 | 0.230 | 29.00 |
| v11 | B | 702 | 142 | 0.202 | 0.26 | 0.600 | 0.303 | 38.00 |
| v11 | C | 702 | 195 | 0.278 | 0.50 | 0.521 | 0.362 | 31.00 |
| v11 | rule_hr | 702 | 28 | 0.040 | 2.37 | 0.032 | 0.035 | 18.00 |
| v11 | rule_emg | 702 | 65 | 0.093 | 5.50 | 0.032 | 0.048 | 36.00 |
| v11 | rule_eeg | 702 | 119 | 0.170 | 14.96 | 0.022 | 0.039 | 24.00 |
| v11 | rule_acc | 702 | 6 | 0.009 | 1.92 | 0.009 | 0.009 | 6.00 |

## Frozen-τ results — without the seizures lacking ECG/EMG, per-patient average

Pre-registered primary aggregation: each metric per patient, then averaged over patients; patient-bootstrap 95% CIs.

| lineage | model | sensitivity [CI] | FA/24h [CI] | precision | F1 [CI] | median delay s |
|---|---|---|---|---|---|---|
| v02g | A | 0.340 [0.268, 0.421] | 1.25 [0.66, 2.06] | 0.434 | 0.279 [0.217, 0.342] | 28.50 |
| v02g | B | 0.388 [0.304, 0.474] | 0.60 [0.38, 0.88] | 0.555 | 0.349 [0.284, 0.420] | 33.00 |
| v02g | C | 0.352 [0.271, 0.441] | 0.20 [0.13, 0.29] | 0.683 | 0.359 [0.279, 0.442] | 36.25 |
| v02g | rule_hr | 0.151 [0.098, 0.217] | 10.63 [7.20, 14.49] | 0.057 | 0.037 [0.022, 0.053] | 43.00 |
| v02g | rule_emg | 0.074 [0.036, 0.117] | 3.20 [2.30, 4.24] | 0.091 | 0.045 [0.022, 0.074] | 41.00 |
| v02g | rule_eeg | 0.274 [0.200, 0.356] | 15.86 [12.05, 20.32] | 0.040 | 0.041 [0.028, 0.056] | 22.00 |
| v02g | rule_acc | 0.020 [0.003, 0.047] | 3.03 [1.02, 5.43] | 0.003 | 0.002 [0.000, 0.004] | 17.25 |
| v11 | A | 0.407 [0.324, 0.493] | 2.63 [1.44, 4.12] | 0.352 | 0.282 [0.220, 0.342] | 32.00 |
| v11 | B | 0.293 [0.216, 0.379] | 0.34 [0.18, 0.54] | 0.606 | 0.281 [0.213, 0.359] | 39.00 |
| v11 | C | 0.385 [0.306, 0.471] | 0.65 [0.42, 0.91] | 0.491 | 0.332 [0.266, 0.405] | 31.50 |
| v11 | rule_hr | 0.048 [0.016, 0.094] | 2.41 [1.26, 3.74] | 0.094 | 0.025 [0.007, 0.048] | 33.00 |
| v11 | rule_emg | 0.132 [0.079, 0.192] | 5.02 [3.69, 6.50] | 0.070 | 0.051 [0.026, 0.079] | 40.50 |
| v11 | rule_eeg | 0.265 [0.192, 0.346] | 14.38 [10.67, 18.80] | 0.046 | 0.046 [0.031, 0.063] | 29.00 |
| v11 | rule_acc | 0.017 [0.000, 0.045] | 2.71 [0.91, 4.87] | 0.003 | 0.002 [0.000, 0.005] | 38.00 |

## Frozen-τ results — without the seizures lacking ECG/EMG, pooled counts (secondary)

Secondary: counts summed over all patients, then the rates computed; for comparison with the development tables.

| lineage | model | seizures | detected | sensitivity | FA/24h | precision | F1 | median delay s |
|---|---|---|---|---|---|---|---|---|
| v02g | A | 690 | 153 | 0.222 | 1.04 | 0.292 | 0.252 | 28.00 |
| v02g | B | 690 | 181 | 0.262 | 0.47 | 0.521 | 0.349 | 32.00 |
| v02g | C | 690 | 154 | 0.223 | 0.17 | 0.724 | 0.341 | 32.00 |
| v02g | rule_hr | 690 | 68 | 0.099 | 10.54 | 0.018 | 0.030 | 32.50 |
| v02g | rule_emg | 690 | 42 | 0.061 | 3.37 | 0.034 | 0.044 | 37.50 |
| v02g | rule_eeg | 690 | 117 | 0.170 | 16.41 | 0.020 | 0.036 | 19.00 |
| v02g | rule_acc | 690 | 7 | 0.010 | 2.10 | 0.009 | 0.010 | 1.00 |
| v11 | A | 690 | 193 | 0.280 | 2.24 | 0.195 | 0.230 | 29.00 |
| v11 | B | 690 | 141 | 0.204 | 0.26 | 0.600 | 0.305 | 38.00 |
| v11 | C | 690 | 192 | 0.278 | 0.50 | 0.521 | 0.363 | 31.00 |
| v11 | rule_hr | 690 | 28 | 0.041 | 2.37 | 0.032 | 0.036 | 18.00 |
| v11 | rule_emg | 690 | 65 | 0.094 | 5.50 | 0.032 | 0.048 | 36.00 |
| v11 | rule_eeg | 690 | 117 | 0.170 | 14.96 | 0.022 | 0.039 | 24.00 |
| v11 | rule_acc | 690 | 6 | 0.009 | 1.92 | 0.009 | 0.009 | 6.00 |

## Subgroups at the frozen τ (models A, B, C) — all seizures

Sensitivity per patient, then averaged (primary, with patient-bootstrap 95% CI) next to the pooled sensitivity (secondary: detected / all seizures of the subgroup). The two can differ a lot when a few patients have many seizures. Montage rows also give FA/24h and F1 per patient. Rules: `subgroups_*.csv`.

| lineage | model | subgroup | seizures | patients | sensitivity [CI] (per patient, primary) | pooled sensitivity (secondary) | median delay s | FA/24h (montage) | F1 (montage) |
|---|---|---|---|---|---|---|---|---|---|
| v02g | A | group: FBTC | 33 | 28 | 0.929 [0.821, 1.000] | 0.939 | 31.50 |  |  |
| v02g | A | group: focal_motor | 317 | 48 | 0.235 [0.152, 0.324] | 0.259 | 34.50 |  |  |
| v02g | A | group: focal_nonmotor | 249 | 39 | 0.149 [0.072, 0.238] | 0.092 | 28.00 |  |  |
| v02g | A | group: focal_unknown_motor | 103 | 22 | 0.159 [0.034, 0.307] | 0.194 | 16.00 |  |  |
| v02g | A | duration: < 15 s | 212 | 18 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | A | duration: 15-60 s | 225 | 65 | 0.178 [0.104, 0.259] | 0.156 | 15.75 |  |  |
| v02g | A | duration: > 60 s | 265 | 70 | 0.492 [0.390, 0.596] | 0.457 | 34.75 |  |  |
| v02g | A | montage: left | 334 | 50 | 0.371 [0.274, 0.485] | 0.335 | 23.50 | 1.88 | 0.280 |
| v02g | A | montage: right | 287 | 34 | 0.373 [0.231, 0.513] | 0.108 | 29.00 | 0.61 | 0.302 |
| v02g | A | montage: generalized | 81 | 12 | 0.215 [0.052, 0.404] | 0.160 | 48.00 | 0.11 | 0.218 |
| v02g | B | group: FBTC | 33 | 28 | 0.964 [0.882, 1.000] | 0.970 | 36.00 |  |  |
| v02g | B | group: focal_motor | 317 | 48 | 0.335 [0.228, 0.457] | 0.315 | 35.25 |  |  |
| v02g | B | group: focal_nonmotor | 249 | 39 | 0.122 [0.055, 0.203] | 0.104 | 23.25 |  |  |
| v02g | B | group: focal_unknown_motor | 103 | 22 | 0.222 [0.085, 0.372] | 0.252 | 24.00 |  |  |
| v02g | B | duration: < 15 s | 212 | 18 | 0.042 [0.000, 0.109] | 0.009 | 29.50 |  |  |
| v02g | B | duration: 15-60 s | 225 | 65 | 0.211 [0.128, 0.294] | 0.147 | 14.50 |  |  |
| v02g | B | duration: > 60 s | 265 | 70 | 0.576 [0.476, 0.688] | 0.562 | 40.00 |  |  |
| v02g | B | montage: left | 334 | 50 | 0.387 [0.281, 0.512] | 0.344 | 31.00 | 0.80 | 0.321 |
| v02g | B | montage: right | 287 | 34 | 0.437 [0.287, 0.591] | 0.146 | 33.00 | 0.39 | 0.367 |
| v02g | B | montage: generalized | 81 | 12 | 0.332 [0.169, 0.519] | 0.333 | 43.00 | 0.22 | 0.399 |
| v02g | C | group: FBTC | 33 | 28 | 0.964 [0.882, 1.000] | 0.970 | 36.00 |  |  |
| v02g | C | group: focal_motor | 317 | 48 | 0.272 [0.173, 0.389] | 0.265 | 36.50 |  |  |
| v02g | C | group: focal_nonmotor | 249 | 39 | 0.090 [0.028, 0.168] | 0.076 | 36.00 |  |  |
| v02g | C | group: focal_unknown_motor | 103 | 22 | 0.172 [0.045, 0.325] | 0.214 | 22.00 |  |  |
| v02g | C | duration: < 15 s | 212 | 18 | 0.028 [0.000, 0.091] | 0.005 | 41.00 |  |  |
| v02g | C | duration: 15-60 s | 225 | 65 | 0.197 [0.116, 0.283] | 0.120 | 16.00 |  |  |
| v02g | C | duration: > 60 s | 265 | 70 | 0.527 [0.426, 0.646] | 0.487 | 44.50 |  |  |
| v02g | C | montage: left | 334 | 50 | 0.343 [0.235, 0.468] | 0.290 | 36.00 | 0.23 | 0.332 |
| v02g | C | montage: right | 287 | 34 | 0.433 [0.279, 0.588] | 0.143 | 32.00 | 0.18 | 0.426 |
| v02g | C | montage: generalized | 81 | 12 | 0.252 [0.074, 0.452] | 0.235 | 48.00 | 0.11 | 0.301 |
| v11 | A | group: FBTC | 33 | 28 | 0.964 [0.882, 1.000] | 0.970 | 32.00 |  |  |
| v11 | A | group: focal_motor | 317 | 48 | 0.374 [0.279, 0.483] | 0.360 | 36.00 |  |  |
| v11 | A | group: focal_nonmotor | 249 | 39 | 0.156 [0.071, 0.255] | 0.104 | 21.00 |  |  |
| v11 | A | group: focal_unknown_motor | 103 | 22 | 0.177 [0.052, 0.321] | 0.233 | 17.00 |  |  |
| v11 | A | duration: < 15 s | 212 | 18 | 0.005 [0.000, 0.015] | 0.009 | 19.00 |  |  |
| v11 | A | duration: 15-60 s | 225 | 65 | 0.251 [0.170, 0.345] | 0.231 | 18.75 |  |  |
| v11 | A | duration: > 60 s | 265 | 70 | 0.562 [0.453, 0.662] | 0.536 | 38.00 |  |  |
| v11 | A | montage: left | 334 | 50 | 0.435 [0.325, 0.544] | 0.410 | 25.00 | 4.06 | 0.276 |
| v11 | A | montage: right | 287 | 34 | 0.450 [0.308, 0.594] | 0.146 | 36.00 | 1.16 | 0.292 |
| v11 | A | montage: generalized | 81 | 12 | 0.237 [0.067, 0.433] | 0.210 | 42.00 | 0.17 | 0.250 |
| v11 | B | group: FBTC | 33 | 28 | 0.821 [0.667, 0.963] | 0.848 | 36.00 |  |  |
| v11 | B | group: focal_motor | 317 | 48 | 0.241 [0.143, 0.342] | 0.252 | 40.00 |  |  |
| v11 | B | group: focal_nonmotor | 249 | 39 | 0.073 [0.020, 0.150] | 0.068 | 53.50 |  |  |
| v11 | B | group: focal_unknown_motor | 103 | 22 | 0.117 [0.020, 0.237] | 0.165 | 30.00 |  |  |
| v11 | B | duration: < 15 s | 212 | 18 | 0.001 [0.000, 0.003] | 0.005 | 12.00 |  |  |
| v11 | B | duration: 15-60 s | 225 | 65 | 0.154 [0.083, 0.235] | 0.120 | 20.75 |  |  |
| v11 | B | duration: > 60 s | 265 | 70 | 0.446 [0.345, 0.565] | 0.430 | 43.00 |  |  |
| v11 | B | montage: left | 334 | 50 | 0.300 [0.200, 0.412] | 0.269 | 35.00 | 0.50 | 0.266 |
| v11 | B | montage: right | 287 | 34 | 0.310 [0.168, 0.450] | 0.111 | 38.50 | 0.13 | 0.307 |
| v11 | B | montage: generalized | 81 | 12 | 0.160 [0.040, 0.296] | 0.247 | 51.00 | 0.13 | 0.199 |
| v11 | C | group: FBTC | 33 | 28 | 0.964 [0.882, 1.000] | 0.970 | 37.00 |  |  |
| v11 | C | group: focal_motor | 317 | 48 | 0.376 [0.267, 0.495] | 0.353 | 37.50 |  |  |
| v11 | C | group: focal_nonmotor | 249 | 39 | 0.125 [0.057, 0.205] | 0.112 | 26.00 |  |  |
| v11 | C | group: focal_unknown_motor | 103 | 22 | 0.144 [0.038, 0.273] | 0.223 | 22.00 |  |  |
| v11 | C | duration: < 15 s | 212 | 18 | 0.030 [0.000, 0.095] | 0.019 | 26.00 |  |  |
| v11 | C | duration: 15-60 s | 225 | 65 | 0.228 [0.142, 0.314] | 0.187 | 18.50 |  |  |
| v11 | C | duration: > 60 s | 265 | 70 | 0.569 [0.467, 0.678] | 0.562 | 42.00 |  |  |
| v11 | C | montage: left | 334 | 50 | 0.388 [0.286, 0.510] | 0.359 | 29.00 | 0.72 | 0.316 |
| v11 | C | montage: right | 287 | 34 | 0.429 [0.280, 0.573] | 0.160 | 37.00 | 0.64 | 0.359 |
| v11 | C | montage: generalized | 81 | 12 | 0.320 [0.128, 0.533] | 0.358 | 43.50 | 0.25 | 0.313 |

## Subgroups at the frozen τ (models A, B, C) — without the seizures lacking ECG/EMG

Sensitivity per patient, then averaged (primary, with patient-bootstrap 95% CI) next to the pooled sensitivity (secondary: detected / all seizures of the subgroup). The two can differ a lot when a few patients have many seizures. Montage rows also give FA/24h and F1 per patient. Rules: `subgroups_*.csv`.

| lineage | model | subgroup | seizures | patients | sensitivity [CI] (per patient, primary) | pooled sensitivity (secondary) | median delay s | FA/24h (montage) | F1 (montage) |
|---|---|---|---|---|---|---|---|---|---|
| v02g | A | group: FBTC | 30 | 25 | 0.920 [0.792, 1.000] | 0.933 | 31.00 |  |  |
| v02g | A | group: focal_motor | 317 | 48 | 0.235 [0.152, 0.324] | 0.259 | 34.50 |  |  |
| v02g | A | group: focal_nonmotor | 246 | 38 | 0.153 [0.074, 0.245] | 0.093 | 28.00 |  |  |
| v02g | A | group: focal_unknown_motor | 97 | 20 | 0.175 [0.038, 0.338] | 0.206 | 16.00 |  |  |
| v02g | A | duration: < 15 s | 209 | 18 | 0.000 [0.000, 0.000] | 0.000 | – |  |  |
| v02g | A | duration: 15-60 s | 220 | 63 | 0.168 [0.098, 0.248] | 0.155 | 16.00 |  |  |
| v02g | A | duration: > 60 s | 261 | 67 | 0.492 [0.391, 0.600] | 0.456 | 34.75 |  |  |
| v02g | A | montage: left | 332 | 48 | 0.345 [0.249, 0.448] | 0.331 | 19.50 | 1.88 | 0.268 |
| v02g | A | montage: right | 279 | 32 | 0.365 [0.219, 0.504] | 0.108 | 32.50 | 0.61 | 0.314 |
| v02g | A | montage: generalized | 79 | 12 | 0.256 [0.052, 0.494] | 0.165 | 48.00 | 0.11 | 0.232 |
| v02g | B | group: FBTC | 30 | 25 | 0.960 [0.870, 1.000] | 0.967 | 35.00 |  |  |
| v02g | B | group: focal_motor | 317 | 48 | 0.335 [0.228, 0.457] | 0.315 | 35.25 |  |  |
| v02g | B | group: focal_nonmotor | 246 | 38 | 0.125 [0.058, 0.208] | 0.106 | 23.25 |  |  |
| v02g | B | group: focal_unknown_motor | 97 | 20 | 0.244 [0.096, 0.403] | 0.268 | 24.00 |  |  |
| v02g | B | duration: < 15 s | 209 | 18 | 0.042 [0.000, 0.109] | 0.010 | 29.50 |  |  |
| v02g | B | duration: 15-60 s | 220 | 63 | 0.202 [0.118, 0.284] | 0.145 | 15.00 |  |  |
| v02g | B | duration: > 60 s | 261 | 67 | 0.580 [0.476, 0.690] | 0.563 | 40.00 |  |  |
| v02g | B | montage: left | 332 | 48 | 0.362 [0.263, 0.480] | 0.340 | 30.00 | 0.80 | 0.307 |
| v02g | B | montage: right | 279 | 32 | 0.433 [0.276, 0.590] | 0.147 | 34.50 | 0.39 | 0.382 |
| v02g | B | montage: generalized | 79 | 12 | 0.373 [0.175, 0.592] | 0.342 | 43.00 | 0.22 | 0.427 |
| v02g | C | group: FBTC | 30 | 25 | 0.960 [0.870, 1.000] | 0.967 | 35.50 |  |  |
| v02g | C | group: focal_motor | 317 | 48 | 0.272 [0.173, 0.389] | 0.265 | 36.50 |  |  |
| v02g | C | group: focal_nonmotor | 246 | 38 | 0.093 [0.029, 0.171] | 0.077 | 36.00 |  |  |
| v02g | C | group: focal_unknown_motor | 97 | 20 | 0.190 [0.050, 0.357] | 0.227 | 22.00 |  |  |
| v02g | C | duration: < 15 s | 209 | 18 | 0.028 [0.000, 0.091] | 0.005 | 41.00 |  |  |
| v02g | C | duration: 15-60 s | 220 | 63 | 0.188 [0.109, 0.271] | 0.118 | 15.75 |  |  |
| v02g | C | duration: > 60 s | 261 | 67 | 0.528 [0.422, 0.647] | 0.487 | 39.75 |  |  |
| v02g | C | montage: left | 332 | 48 | 0.316 [0.204, 0.435] | 0.286 | 35.50 | 0.23 | 0.311 |
| v02g | C | montage: right | 279 | 32 | 0.429 [0.273, 0.588] | 0.143 | 34.25 | 0.18 | 0.442 |
| v02g | C | montage: generalized | 79 | 12 | 0.294 [0.075, 0.528] | 0.241 | 48.00 | 0.11 | 0.329 |
| v11 | A | group: FBTC | 30 | 25 | 0.960 [0.870, 1.000] | 0.967 | 31.00 |  |  |
| v11 | A | group: focal_motor | 317 | 48 | 0.374 [0.279, 0.483] | 0.360 | 36.00 |  |  |
| v11 | A | group: focal_nonmotor | 246 | 38 | 0.160 [0.074, 0.261] | 0.106 | 21.00 |  |  |
| v11 | A | group: focal_unknown_motor | 97 | 20 | 0.195 [0.056, 0.354] | 0.247 | 17.00 |  |  |
| v11 | A | duration: < 15 s | 209 | 18 | 0.005 [0.000, 0.015] | 0.010 | 19.00 |  |  |
| v11 | A | duration: 15-60 s | 220 | 63 | 0.243 [0.162, 0.340] | 0.232 | 19.00 |  |  |
| v11 | A | duration: > 60 s | 261 | 67 | 0.564 [0.454, 0.667] | 0.536 | 38.00 |  |  |
| v11 | A | montage: left | 332 | 48 | 0.412 [0.301, 0.518] | 0.407 | 23.00 | 4.06 | 0.265 |
| v11 | A | montage: right | 279 | 32 | 0.447 [0.307, 0.593] | 0.147 | 36.50 | 1.16 | 0.309 |
| v11 | A | montage: generalized | 79 | 12 | 0.279 [0.067, 0.518] | 0.215 | 42.00 | 0.17 | 0.278 |
| v11 | B | group: FBTC | 30 | 25 | 0.880 [0.724, 1.000] | 0.900 | 36.00 |  |  |
| v11 | B | group: focal_motor | 317 | 48 | 0.241 [0.143, 0.342] | 0.252 | 40.00 |  |  |
| v11 | B | group: focal_nonmotor | 246 | 38 | 0.075 [0.021, 0.154] | 0.069 | 53.50 |  |  |
| v11 | B | group: focal_unknown_motor | 97 | 20 | 0.129 [0.022, 0.261] | 0.175 | 30.00 |  |  |
| v11 | B | duration: < 15 s | 209 | 18 | 0.001 [0.000, 0.003] | 0.005 | 12.00 |  |  |
| v11 | B | duration: 15-60 s | 220 | 63 | 0.159 [0.086, 0.240] | 0.123 | 20.75 |  |  |
| v11 | B | duration: > 60 s | 261 | 67 | 0.459 [0.355, 0.578] | 0.433 | 42.00 |  |  |
| v11 | B | montage: left | 332 | 48 | 0.291 [0.190, 0.401] | 0.268 | 34.50 | 0.50 | 0.264 |
| v11 | B | montage: right | 279 | 32 | 0.330 [0.175, 0.479] | 0.115 | 38.50 | 0.13 | 0.327 |
| v11 | B | montage: generalized | 79 | 12 | 0.202 [0.041, 0.398] | 0.253 | 51.00 | 0.13 | 0.227 |
| v11 | C | group: FBTC | 30 | 25 | 0.960 [0.870, 1.000] | 0.967 | 35.00 |  |  |
| v11 | C | group: focal_motor | 317 | 48 | 0.376 [0.267, 0.495] | 0.353 | 37.50 |  |  |
| v11 | C | group: focal_nonmotor | 246 | 38 | 0.128 [0.059, 0.210] | 0.114 | 26.00 |  |  |
| v11 | C | group: focal_unknown_motor | 97 | 20 | 0.158 [0.042, 0.296] | 0.237 | 22.00 |  |  |
| v11 | C | duration: < 15 s | 209 | 18 | 0.030 [0.000, 0.095] | 0.019 | 26.00 |  |  |
| v11 | C | duration: 15-60 s | 220 | 63 | 0.221 [0.135, 0.304] | 0.186 | 18.50 |  |  |
| v11 | C | duration: > 60 s | 261 | 67 | 0.573 [0.467, 0.684] | 0.563 | 39.00 |  |  |
| v11 | C | montage: left | 332 | 48 | 0.363 [0.258, 0.479] | 0.355 | 28.00 | 0.72 | 0.298 |
| v11 | C | montage: right | 279 | 32 | 0.428 [0.279, 0.570] | 0.161 | 37.50 | 0.64 | 0.380 |
| v11 | C | montage: generalized | 79 | 12 | 0.362 [0.143, 0.598] | 0.367 | 43.50 | 0.25 | 0.341 |

Subgroups (seizure group, duration, montage), operating-curve points (sensitivity at FA/24h ≤ 1, 2, 5, 10; τ swept on this set — a curve, not a deployable number), per-patient tables and the H3a onsets are in the CSV files next to this file.

*Dry run on development patients: in-sample, not a result.*

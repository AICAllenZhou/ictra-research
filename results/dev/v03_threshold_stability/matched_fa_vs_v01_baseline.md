# Matched-FA operating point: v03_threshold_stability vs v01_baseline at FA/24h <= 9.67

Best sensitivity on the held-out operating curve at this FA budget (seed-mean / seed-median). Deployable points (summary.csv: own tuned threshold) are given for context. The section-1 decision is unchanged by this table.

| model | group | v01_baseline | v03_threshold_stability | Δ (mean) |
|---|---|---|---|---|
| A_eeg lgbm | all | 0.370 / 0.370 | 0.370 / 0.370 | +0.000 |
| A_eeg lgbm | FBTC | 0.970 / 0.970 | 0.970 / 0.970 | +0.000 |
| A_eeg lgbm | focal_motor | 0.425 / 0.426 | 0.425 / 0.426 | +0.000 |
| A_eeg lgbm | focal_nonmotor | 0.225 / 0.221 | 0.225 / 0.221 | +0.000 |
| A_eeg lgbm | focal_unknown_motor | 0.359 / 0.359 | 0.359 / 0.359 | +0.000 |
| A_eeg xgb | all | 0.364 / 0.363 | 0.364 / 0.363 | +0.000 |
| A_eeg xgb | FBTC | 0.970 / 0.970 | 0.970 / 0.970 | +0.000 |
| A_eeg xgb | focal_motor | 0.405 / 0.407 | 0.405 / 0.407 | +0.000 |
| A_eeg xgb | focal_nonmotor | 0.240 / 0.241 | 0.240 / 0.241 | +0.000 |
| A_eeg xgb | focal_unknown_motor | 0.343 / 0.350 | 0.343 / 0.350 | +0.000 |
| B_eeg_ecg lgbm | all | 0.421 / 0.420 | 0.430 / 0.429 | +0.009 |
| B_eeg_ecg lgbm | FBTC | 1.000 / 1.000 | 1.000 / 1.000 | +0.000 |
| B_eeg_ecg lgbm | focal_motor | 0.479 / 0.483 | 0.487 / 0.483 | +0.007 |
| B_eeg_ecg lgbm | focal_nonmotor | 0.284 / 0.285 | 0.285 / 0.289 | +0.001 |
| B_eeg_ecg lgbm | focal_unknown_motor | 0.388 / 0.388 | 0.421 / 0.417 | +0.032 |
| B_eeg_ecg xgb | all | 0.410 / 0.407 | 0.419 / 0.425 | +0.009 |
| B_eeg_ecg xgb | FBTC | 1.000 / 1.000 | 1.000 / 1.000 | +0.000 |
| B_eeg_ecg xgb | focal_motor | 0.457 / 0.461 | 0.468 / 0.470 | +0.011 |
| B_eeg_ecg xgb | focal_nonmotor | 0.266 / 0.261 | 0.281 / 0.293 | +0.015 |
| B_eeg_ecg xgb | focal_unknown_motor | 0.421 / 0.427 | 0.414 / 0.408 | -0.006 |
| C_eeg_ecg_emg_acc lgbm | all | 0.450 / 0.450 | 0.449 / 0.446 | -0.001 |
| C_eeg_ecg_emg_acc lgbm | FBTC | 1.000 / 1.000 | 1.000 / 1.000 | +0.000 |
| C_eeg_ecg_emg_acc lgbm | focal_motor | 0.509 / 0.508 | 0.499 / 0.502 | -0.009 |
| C_eeg_ecg_emg_acc lgbm | focal_nonmotor | 0.307 / 0.305 | 0.308 / 0.305 | +0.001 |
| C_eeg_ecg_emg_acc lgbm | focal_unknown_motor | 0.440 / 0.447 | 0.460 / 0.456 | +0.019 |
| C_eeg_ecg_emg_acc xgb | all | 0.446 / 0.446 | 0.445 / 0.444 | -0.001 |
| C_eeg_ecg_emg_acc xgb | FBTC | 1.000 / 1.000 | 1.000 / 1.000 | +0.000 |
| C_eeg_ecg_emg_acc xgb | focal_motor | 0.495 / 0.495 | 0.491 / 0.495 | -0.004 |
| C_eeg_ecg_emg_acc xgb | focal_nonmotor | 0.304 / 0.301 | 0.301 / 0.297 | -0.003 |
| C_eeg_ecg_emg_acc xgb | focal_unknown_motor | 0.463 / 0.466 | 0.472 / 0.476 | +0.010 |
| rule rule_acc | all | 0.095 / 0.095 | 0.095 / 0.095 | +0.000 |
| rule rule_acc | FBTC | 0.242 / 0.242 | 0.242 / 0.242 | +0.000 |
| rule rule_acc | focal_motor | 0.117 / 0.117 | 0.117 / 0.117 | +0.000 |
| rule rule_acc | focal_nonmotor | 0.044 / 0.044 | 0.044 / 0.044 | +0.000 |
| rule rule_acc | focal_unknown_motor | 0.107 / 0.107 | 0.107 / 0.107 | +0.000 |
| rule rule_eeg | all | 0.132 / 0.132 | 0.132 / 0.132 | +0.000 |
| rule rule_eeg | FBTC | 0.818 / 0.818 | 0.818 / 0.818 | +0.000 |
| rule rule_eeg | focal_motor | 0.126 / 0.126 | 0.126 / 0.126 | +0.000 |
| rule rule_eeg | focal_nonmotor | 0.044 / 0.044 | 0.044 / 0.044 | +0.000 |
| rule rule_eeg | focal_unknown_motor | 0.146 / 0.146 | 0.146 / 0.146 | +0.000 |
| rule rule_emg | all | 0.137 / 0.137 | 0.137 / 0.137 | +0.000 |
| rule rule_emg | FBTC | 0.606 / 0.606 | 0.606 / 0.606 | +0.000 |
| rule rule_emg | focal_motor | 0.180 / 0.180 | 0.180 / 0.180 | +0.000 |
| rule rule_emg | focal_nonmotor | 0.028 / 0.028 | 0.028 / 0.028 | +0.000 |
| rule rule_emg | focal_unknown_motor | 0.117 / 0.117 | 0.117 / 0.117 | +0.000 |
| rule rule_hr | all | 0.091 / 0.091 | 0.091 / 0.091 | +0.000 |
| rule rule_hr | FBTC | 0.333 / 0.333 | 0.333 / 0.333 | +0.000 |
| rule rule_hr | focal_motor | 0.098 / 0.098 | 0.098 / 0.098 | +0.000 |
| rule rule_hr | focal_nonmotor | 0.028 / 0.028 | 0.028 / 0.028 | +0.000 |
| rule rule_hr | focal_unknown_motor | 0.146 / 0.146 | 0.146 / 0.146 | +0.000 |

Deployable points (sensitivity / FA/24h):

- A_eeg lgbm: v01_baseline 0.359 / 9.18; v03_threshold_stability 0.302 / 5.49
- A_eeg xgb: v01_baseline 0.366 / 11.05; v03_threshold_stability 0.301 / 6.17
- B_eeg_ecg xgb: v01_baseline 0.404 / 9.55; v03_threshold_stability 0.343 / 4.42
- B_eeg_ecg lgbm: v01_baseline 0.399 / 9.79; v03_threshold_stability 0.349 / 4.59
- C_eeg_ecg_emg_acc xgb: v01_baseline 0.442 / 9.67; v03_threshold_stability 0.379 / 4.48
- C_eeg_ecg_emg_acc lgbm: v01_baseline 0.435 / 9.68; v03_threshold_stability 0.376 / 4.71
- rule rule_eeg: v01_baseline 0.372 / 32.66; v03_threshold_stability 0.353 / 30.32
- rule rule_acc: v01_baseline 0.298 / 35.58; v03_threshold_stability 0.252 / 23.92
- rule rule_emg: v01_baseline 0.296 / 38.25; v03_threshold_stability 0.158 / 11.19
- rule rule_hr: v01_baseline 0.393 / 57.74; v03_threshold_stability 0.400 / 48.62

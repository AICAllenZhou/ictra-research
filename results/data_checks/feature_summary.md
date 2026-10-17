# Feature summary (Task 3)

Computed by `python src/10_data_checks.py features` from 607 parquet files in `data/features/`.

| quantity | value |
|---|---|
| runs | 607 |
| hours (windows x 2 s) | 5,117.6 |
| seizures with >= 1 window | 702 |
| total windows | 9,211,695 |
| seizure windows (overlap >= 0.5) | 21,255 (0.231%) |
| edge windows (0 < overlap < 0.5) | 683 (0.007%) |
| impd windows | 45,206 (0.491%) |
| flat_eeg windows | 51,214 (0.556%) |
| flat_ecg windows | 54,181 (0.588%) |
| flat_emg windows | 98,952 (1.074%) |
| windows with ecg present | 98.3% |
| windows with emg present | 98.3% |
| windows with acc present | 99.0% |

## NaN rate per feature

Denominator: windows of runs where the modality exists, excluding impd and that modality's flat windows. Flag = NaN rate > 5%.

| feature | NaN rate | flag |
|---|---|---|
| acc_logjerk | 0.00% |  |
| acc_logsd | 0.00% |  |
| acc_logsd_rel | 0.06% |  |
| acc_p1_8 | 0.00% |  |
| ecg_hr | 0.07% |  |
| ecg_hr_max | 0.07% |  |
| ecg_hr_rise | 0.28% |  |
| ecg_hr_rise_pct | 0.28% |  |
| ecg_hr_slope | 0.09% |  |
| ecg_q | 0.00% |  |
| ecg_rmssd | 0.07% |  |
| ecg_sdnn | 0.07% |  |
| eeg_a_0 | 0.00% |  |
| eeg_a_1 | 0.00% |  |
| eeg_asym_ll | 0.00% |  |
| eeg_b_0 | 0.00% |  |
| eeg_b_1 | 0.00% |  |
| eeg_d_0 | 0.00% |  |
| eeg_d_1 | 0.00% |  |
| eeg_kurt_0 | 0.00% |  |
| eeg_kurt_1 | 0.00% |  |
| eeg_ll_0 | 0.00% |  |
| eeg_ll_0_rel | 0.07% |  |
| eeg_ll_1 | 0.00% |  |
| eeg_ll_1_rel | 0.07% |  |
| eeg_ll_max | 0.00% |  |
| eeg_ll_max_rel | 0.07% |  |
| eeg_logvar_0 | 0.00% |  |
| eeg_logvar_0_rel | 0.07% |  |
| eeg_logvar_1 | 0.00% |  |
| eeg_logvar_1_rel | 0.07% |  |
| eeg_mont_generalized | 0.00% |  |
| eeg_mont_left | 0.00% |  |
| eeg_mont_right | 0.00% |  |
| eeg_sef_0 | 0.00% |  |
| eeg_sef_1 | 0.00% |  |
| eeg_th_0 | 0.00% |  |
| eeg_th_1 | 0.00% |  |
| eeg_zcr_0 | 0.00% |  |
| eeg_zcr_1 | 0.00% |  |
| emg_burst_frac | 0.00% |  |
| emg_logrms | 0.00% |  |
| emg_logrms_rel | 0.06% |  |
| emg_zcr | 0.00% |  |

# Causal-mode checks (lib-2026-10-20f)

## Offline path unchanged

| sub | run | windows | columns | missing_in_new | numeric_max_abs_diff | nan_pattern_equal | non_numeric_equal |
|---|---|---|---|---|---|---|---|
| sub-001 | 1 | 32658 | 61 | [] | 0.0 | True | True |

## Truncation test (sub-001 run-01, first 40 min, cut at T = 30 min, windows ending <= T - 1 s)

| mode    | modality   |   windows_compared |   max_abs_diff |   nan_mismatch |   windows_differing |   first_differing_window_end_s |
|:--------|:-----------|-------------------:|---------------:|---------------:|--------------------:|-------------------------------:|
| causal  | eeg        |                898 |    0           |              0 |                   0 |                            nan |
| causal  | ecg        |                898 |    0           |              0 |                   0 |                            nan |
| causal  | emg        |                898 |    0           |              0 |                   0 |                            nan |
| causal  | acc        |                898 |    0           |              0 |                   0 |                            nan |
| offline | eeg        |                898 |    0.001055    |              0 |                   4 |                           1792 |
| offline | ecg        |                898 |    0           |              0 |                   0 |                            nan |
| offline | emg        |                898 |    2.14577e-06 |              0 |                   1 |                           1798 |
| offline | acc        |                898 |    0.80283     |              0 |                 898 |                              4 |

Causal mode must show max_abs_diff 0 and no differing windows; offline mode differing is expected and shows the test detects look-ahead.

## Heart rate: causal R peaks vs neurokit (first 40 min of each run)

| sub     |   run |   seconds |   both_finite |   offline_finite |   causal_finite |   median_abs_diff_bpm |   p90_abs_diff_bpm |   within_5_bpm |   mean_hr_offline |   mean_hr_causal |
|:--------|------:|----------:|--------------:|-----------------:|----------------:|----------------------:|-------------------:|---------------:|------------------:|-----------------:|
| sub-001 |     1 |      2400 |          2398 |             2398 |            2399 |                 0.981 |              2.215 |          0.995 |            68.569 |           68.587 |
| sub-002 |     1 |      2400 |          2397 |             2398 |            2398 |                 1.411 |              3.876 |          0.933 |            70.175 |           69.714 |
| sub-003 |     1 |      2400 |          2398 |             2398 |            2399 |                 1.659 |              6.877 |          0.861 |            70.729 |           70.614 |
| sub-004 |     1 |      2400 |          2397 |             2397 |            2399 |                 1.876 |             55.238 |          0.654 |            82.226 |           90.054 |
| sub-005 |     1 |      2400 |          2398 |             2398 |            2399 |                 1.709 |              5.269 |          0.889 |            89.941 |           90.288 |
| sub-006 |     1 |      2400 |          2390 |             2390 |            2399 |                 2.2   |             63.152 |          0.657 |            66.68  |           77.321 |
| sub-007 |     1 |      2400 |          2398 |             2398 |            2399 |                 1.189 |              3.436 |          0.961 |            84.68  |           84.618 |
| sub-008 |     1 |      2400 |          2397 |             2397 |            2399 |                 4.171 |             40.873 |          0.554 |            79.568 |           82.901 |
| sub-009 |     1 |      2400 |          2396 |             2396 |            2399 |                 1.549 |             10.316 |          0.863 |            65.233 |           65.801 |
| sub-010 |     1 |      2400 |          2397 |             2398 |            2398 |                 0.889 |             11.691 |          0.876 |            63.096 |           62.686 |
| sub-011 |     1 |      2400 |          2398 |             2398 |            2398 |                 4.996 |             79.737 |          0.5   |            74.247 |           91.132 |
| sub-012 |     1 |      2400 |          2397 |             2397 |            2399 |                 0.853 |             22.322 |          0.859 |            73.024 |           77.13  |


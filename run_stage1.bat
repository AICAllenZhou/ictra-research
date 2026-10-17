@echo off
REM ICTRA stage 1: download train seizure runs -> features -> relationship study -> parallel A/B/C training (dev folds only).
REM Run from a normal Windows terminal (Anaconda Prompt / PowerShell), NOT inside Cowork. Every step is resumable: re-run after any interruption.
cd /d %~dp0
pip install -q pandas numpy requests mne edfio scipy neurokit2 pyarrow scikit-learn xgboost lightgbm matplotlib tabulate
python src\00_seizeit2_catalog.py
python src\01_download_seizeit2.py --runs seizure --split train --go
python src\03_features.py --workers 4
python src\02_relationship_study.py --plot
python src\04_train_ablation.py --workers 4 --version v01_baseline --note "default features, seizure runs only" --models xgb lgbm logit rf --rules --seeds 0 1 2
python src\04_train_ablation.py --workers 4 --version v01_baseline --sets ecg_only emg_only acc_only emg_acc --models xgb --seeds 0
echo.
echo Stage 1 done. Results in results\dev\v01_baseline\summary.csv and results\ITERATION_LOG.md
pause

"""Checks for the v11 causal mode of ictra_lib (dev runs only; nothing is written to data/).

    python src/22_causality_check.py --regression      # offline path unchanged: recompute one run, compare to its parquet
    python src/22_causality_check.py --causality       # causal path: truncating the signal at T leaves earlier windows unchanged
    python src/22_causality_check.py --hr              # causal R-peak HR vs neurokit HR on the same ECG

--causality: for each modality, the raw signal of one dev run (first 40 min) is processed in causal mode twice, whole and
cut at T = 30 min. Every window that ends at least --margin seconds before T must have identical features (max |diff|
reported). In offline mode the same test is expected to FAIL (zero-phase filters, whole-signal medians), which shows the
test can see look-ahead. Bounded look-ahead left in causal mode: resample_poly (~40 ms) and the dropout repair's
right neighbour (1-2 samples), both far below one 2-s hop.
Writes results/data_checks/causality_check.md.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

_spec = importlib.util.spec_from_file_location("feat", Path(__file__).resolve().parent / "03_features.py")
F = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F)
OUT = L.ROOT / "results/data_checks/causality_check.md"


def dev_run(prefer=None):
    """A dev run with eeg, ecg, emg and mov files on disk (never sub-097..125)."""
    dev = {s for v in L.load_splits()["dev_folds"].values() for s in v}
    for p in sorted((F.RAW).glob("sub-*/ses-01/eeg/*_eeg.edf")):
        sub = p.name.split("_")[0]
        if sub not in dev or (prefer and sub != prefer):
            continue
        run = int(p.name.split("run-")[1][:2])
        stem = str(p).replace("_eeg.edf", "").replace("\\eeg\\", "\\{m}\\").replace("/eeg/", "/{m}/")
        if all(Path(stem.format(m=m) + f"_{m}.edf").exists() for m in ("ecg", "emg", "mov")):
            return sub, run
    raise SystemExit("no dev run with all modalities found")


def raw(sub, run, m, seconds, picks=None):
    import mne
    p = F.RAW / f"{sub}/ses-01/{m}/{sub}_ses-01_task-szMonitoring_run-{run:02d}_{m}.edf"
    r = mne.io.read_raw_edf(p, preload=False, verbose=False)
    fs = r.info["sfreq"]
    n = min(r.n_times, int(seconds * fs))
    if m == "mov" and picks is None:
        idx = [i for i, nme in enumerate(r.ch_names) if nme.startswith("EEG SD ACC")][:3] or [0, 1, 2]
        picks = idx
    return r.get_data(picks=picks, start=0, stop=n) * 1e6, fs


def feats(kind, x, fs):
    """Same calls as 03_features.features_for_run, for one modality and one chunk."""
    if kind == "eeg":
        h = L.harmonise(x, fs, "eeg", mains=F.MAINS)
        return L.add_relative_eeg(L.eeg_features(h))
    if kind == "ecg":
        return L.ecg_features(L.harmonise(x, fs, "ecg", mains=F.MAINS)[0])
    if kind == "emg":
        return L.emg_features(L.harmonise(x, fs, "emg", mains=F.MAINS)[0])
    acc = L.harmonise(L.acc_to_g(x, fs), fs, "acc", fs_out=L.CFG["fs_acc"])
    return L.acc_features(acc)


def causality(sub, run, T=1800, full=2400, margin=1.0):
    rows = []
    for causal in (True, False):
        L.set_mode(causal)
        for kind, m, picks in (("eeg", "eeg", [0, 1]), ("ecg", "ecg", [0]), ("emg", "emg", [0]), ("acc", "mov", None)):
            x, fs = raw(sub, run, m, full, picks)
            nT = int(T * fs)
            A = feats(kind, x, fs); B = feats(kind, x[:, :nT], fs)
            end = np.arange(len(B)) * L.CFG["step_s"] + L.CFG["win_s"]
            keep = end <= T - margin
            a, b = A.iloc[: len(B)][keep].to_numpy(float), B[keep].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
            d = np.abs(a - b)[both]
            maxd = float(d.max()) if d.size else 0.0
            bad = np.flatnonzero(((np.abs(a - b) > 1e-9) & both).any(1) | (np.isfinite(a) != np.isfinite(b)).any(1))
            first_bad_end = float(end[keep][bad[0]]) if len(bad) else np.nan
            rows.append(dict(mode="causal" if causal else "offline", modality=kind, windows_compared=int(keep.sum()),
                             max_abs_diff=maxd, nan_mismatch=nan_mismatch, windows_differing=int(len(bad)),
                             first_differing_window_end_s=first_bad_end))
            print(rows[-1], flush=True)
    L.set_mode(False)
    return pd.DataFrame(rows)


def regression(sub, run):
    """Offline features of one whole run recomputed in memory vs the parquet in data/features/ (made by lib-...e)."""
    L.set_mode(False)
    ev_path = F.RAW / f"{sub}/ses-01/eeg/{sub}_ses-01_task-szMonitoring_run-{run:02d}_events.tsv"
    if not ev_path.exists():
        ev_path = L.ROOT / "data/raw/ds005873_meta" / ev_path.relative_to(F.RAW)
    ev = pd.read_csv(ev_path, sep="\t").rename(columns={"onset": "onset_s", "duration": "duration_s", "eventType": "type"})
    old = pd.read_parquet(L.ROOT / f"data/features/{sub}_run-{run:02d}.parquet")
    new = F.features_for_run(sub, run, ev, old.montage.iloc[0])
    cols = [c for c in old.columns if c in new.columns]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(old[c]) and not pd.api.types.is_bool_dtype(old[c])]
    a, b = old[num].to_numpy(float), new[num].to_numpy(float)
    same_nan = bool((np.isnan(a) == np.isnan(b)).all())
    maxd = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    other = [c for c in cols if c not in num]
    other_equal = all((old[c].astype(str).values == new[c].astype(str).values).all() for c in other)
    return dict(sub=sub, run=run, windows=len(old), columns=len(cols), missing_in_new=sorted(set(old.columns) - set(new.columns)),
                numeric_max_abs_diff=maxd, nan_pattern_equal=same_nan, non_numeric_equal=other_equal)


def hr_agreement(sub, run, seconds=2400):
    x, fs = raw(sub, run, "ecg", seconds, [0])
    out = {}
    for causal in (False, True):
        L.set_mode(causal)
        out[causal] = L.heart_rate_series(L.harmonise(x, fs, "ecg", mains=F.MAINS)[0])[0]
    L.set_mode(False)
    a, b = out[False], out[True]
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a - b)[ok]
    return dict(sub=sub, run=run, seconds=len(a), both_finite=int(ok.sum()), offline_finite=int(np.isfinite(a).sum()),
                causal_finite=int(np.isfinite(b).sum()), median_abs_diff_bpm=float(np.median(d)) if d.size else np.nan,
                p90_abs_diff_bpm=float(np.percentile(d, 90)) if d.size else np.nan,
                within_5_bpm=float((d <= 5).mean()) if d.size else np.nan,
                mean_hr_offline=float(np.nanmean(a)), mean_hr_causal=float(np.nanmean(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression", action="store_true"); ap.add_argument("--causality", action="store_true")
    ap.add_argument("--hr", action="store_true"); ap.add_argument("--sub"); ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    md = [f"# Causal-mode checks ({L.LIB_VERSION})", ""]
    sub, run = dev_run(a.sub)
    if a.regression:
        r = regression(sub, run); print(r)
        md += ["## Offline path unchanged", "", "| " + " | ".join(r) + " |", "|" + "---|" * len(r), "| " + " | ".join(str(v) for v in r.values()) + " |", ""]
    if a.causality:
        C = causality(sub, run)
        md += [f"## Truncation test ({sub} run-{run:02d}, first 40 min, cut at T = 30 min, windows ending <= T - 1 s)", "",
               C.to_markdown(index=False), "",
               "Causal mode must show max_abs_diff 0 and no differing windows; offline mode differing is expected and shows the "
               "test detects look-ahead.", ""]
    if a.hr:
        subs = sorted({s for v in L.load_splits()["dev_folds"].values() for s in v})
        R = []
        for s in subs:
            try:
                ss, rr = dev_run(s)
            except SystemExit:
                continue
            R.append(hr_agreement(ss, rr)); print(R[-1], flush=True)
            if len(R) >= a.runs:
                break
        md += ["## Heart rate: causal R peaks vs neurokit (first 40 min of each run)", "", pd.DataFrame(R).round(3).to_markdown(index=False), ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    OUT.write_text("\n".join(md) + "\n" + ("\n---\n\n" + prev if prev else ""), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

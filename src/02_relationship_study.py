"""Exp R: how do brain (EEG), heart (ECG) and muscle (EMG, ACC) signals relate in time around a seizure?

This is ICTRA's version of SlopeAI's wetting-front-delay analysis. It runs BEFORE any classifier and uses only the
development patients (sub-001..096). For every seizure it builds one curve per signal from -PRE s to +POST s around
the annotated onset, finds when each signal first leaves its own baseline, and measures the lead/lag between signals.

    python src/02_relationship_study.py --plot --workers 6                  # all dev seizures with downloaded files
    python src/02_relationship_study.py --subjects sub-001 sub-002 --plot
    python src/02_relationship_study.py --from-curves --z-thr 2 --min-dur 5 --tag z2_5s   # robustness, no signal re-read

Inputs : data/raw/ds005873/ (01_download_seizeit2.py), data/catalog/seizeit2_seizures.csv, data/catalog/splits.json
Outputs: results/expR/[<tag>/]curves.parquet    one row per (seizure, second): per-second value of each signal
         results/expR/[<tag>/]onsets.csv        per seizure: first-change time of each signal, lags, HR rise
         results/expR/[<tag>/]summary_by_group.csv  medians + IQR of the lags by seizure group, share responding
         results/expR/[<tag>/]figures/          timeline figures (with --plot)

Signal processing goes through ictra_lib (dropout repair, filters, R-peak heart rate, ACC calibrated to 1 g), the same
code path as the features.

Definitions (fixed before looking at data; change only via a dated note in docs/EVALUATION_PROTOCOL.md):
  baseline        = the window [-PRE, -BASE_GAP] s before onset (default -300..-60 s), per signal, per seizure
  "signal changes" = first second >= -30 s at which the 1-s feature exceeds baseline mean + Z_THR * baseline SD for at
                     least MIN_DUR consecutive seconds (negative direction counted too for EEG suppression)
  lag_X_Y         = change time of Y minus change time of X (positive: Y later)
  HR rise         = max HR in [onset, onset+60] minus median baseline HR, in bpm and %
                    (+ hr_rise_bpm_med5: same with the 5-s rolling median HR, added after the first run; both reported)
  "responds"      = a change time exists; the share is taken over seizures where that signal was recorded
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, json, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

ROOT = L.ROOT
RAW = ROOT / "data/raw/ds005873"
OUT = ROOT / "results/expR"
FS = L.CFG["fs"]
PRE, POST, BASE_GAP = 300, 180, 60
PAD = 10                       # extra seconds read on each side so filter edges fall outside the curve
Z_THR, MIN_DUR = 3.0, 3
FEATURES = ["eeg_linelength", "eeg_power_1_25", "ecg_hr", "emg_rms", "emg_hf_power", "acc_mag"]
SIGNAL_OF = {"eeg_linelength": "eeg", "eeg_power_1_25": "eeg", "ecg_hr": "ecg", "emg_rms": "emg", "emg_hf_power": "emg", "acc_mag": "mov"}


def read_segment(path, t0, t1, picks=None):
    """Raw segment [t0, t1) s (clipped to the file) with its start offset in s; uV for eeg/ecg/emg (header unit)."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    fs = raw.info["sfreq"]
    a, b = max(0, int(t0 * fs)), min(raw.n_times, int(t1 * fs))
    return raw.get_data(picks=picks, start=a, stop=b) * 1e6, fs, a / fs, raw


def per_second(x, fn, fs):
    fs = int(fs); n = len(x) // fs
    return np.array([fn(x[i * fs:(i + 1) * fs]) for i in range(n)])


def place(values, start_s, t0):
    """Put a per-second series that starts at absolute time start_s onto the grid t0 .. t0+PRE+POST (NaN elsewhere)."""
    out = np.full(PRE + POST, np.nan)
    k0 = int(round(start_s - t0))
    lo, hi = max(0, k0), min(PRE + POST, k0 + len(values))
    if hi > lo:
        out[lo:hi] = values[lo - k0: hi - k0]
    return out


def seizure_curves(sub, run, onset):
    """DataFrame indexed by second relative to onset (-PRE..POST-1), one column per feature (NaN where not recorded)."""
    stem = f"{sub}/ses-01/{{m}}/{sub}_ses-01_task-szMonitoring_run-{run:02d}_{{m}}.edf"
    t0 = int(np.floor(onset)) - PRE
    out = {}
    p = RAW / stem.format(m="eeg")
    x, fs, s0, _ = read_segment(p, t0 - PAD, t0 + PRE + POST + PAD, picks=[0, 1])
    h = L.harmonise(x, fs, "eeg").mean(0)                     # mean of the two channels (definition fixed above)
    out["eeg_linelength"] = place(per_second(h, lambda w: np.sum(np.abs(np.diff(w))), FS), s0, t0)
    out["eeg_power_1_25"] = place(per_second(h, lambda w: np.log(np.mean(w ** 2) + 1e-9), FS), s0, t0)
    p = RAW / stem.format(m="ecg")
    if p.exists():
        x, fs, s0, _ = read_segment(p, t0 - PAD, t0 + PRE + POST + PAD, picks=[0])
        hr, _ = L.heart_rate_series(L.harmonise(x, fs, "ecg")[0])
        out["ecg_hr"] = place(hr, s0, t0)
    p = RAW / stem.format(m="emg")
    if p.exists():
        x, fs, s0, _ = read_segment(p, t0 - PAD, t0 + PRE + POST + PAD, picks=[0])
        e = L.harmonise(x, fs, "emg")[0]
        out["emg_rms"] = place(per_second(e, lambda w: np.sqrt(np.mean(w ** 2)), FS), s0, t0)
        out["emg_hf_power"] = place(per_second(e, lambda w: np.log(np.mean(w ** 2) + 1e-9), FS), s0, t0)
    p = RAW / stem.format(m="mov")
    if p.exists():
        import mne
        raw = mne.io.read_raw_edf(p, preload=False, verbose=False)
        names = raw.ch_names
        idx = [i for i, n in enumerate(names) if n.startswith("EEG SD ACC")][:3]
        if len(idx) < 3:
            idx = [i for i, n in enumerate(names) if "acc" in n.lower()][:3]
        fs = raw.info["sfreq"]
        acc = L.acc_to_g(raw.get_data(picks=idx) * 1e6, fs)    # whole run (small): calibration uses the run's median |a|
        a, b = max(0, int((t0 - PAD) * fs)), min(acc.shape[1], int((t0 + PRE + POST + PAD) * fs))
        seg = L.repair_dropouts(acc[:, a:b], fs)
        mag = np.sqrt((seg ** 2).sum(0))
        out["acc_mag"] = place(per_second(mag - np.median(mag), np.std, fs), a / fs, t0)
    df = pd.DataFrame(index=np.arange(-PRE, POST))
    for f in FEATURES:
        df[f] = out.get(f, np.full(PRE + POST, np.nan))
    return df


def first_change(series, z_thr, min_dur, two_sided=False):
    base = series.loc[-PRE:-BASE_GAP].dropna()
    if len(base) < 60:
        return np.nan, np.nan
    mu, sd = base.mean(), base.std() + 1e-9
    z = (series - mu) / sd
    hit = ((z.abs() if two_sided else z) > z_thr).loc[-30:]
    run = 0
    for t, h in hit.items():
        run = run + 1 if h else 0
        if run >= min_dur:
            return t - min_dur + 1, float(z.loc[t - min_dur + 1:t].mean())
    return np.nan, np.nan


def onset_row(meta, df, z_thr, min_dur):
    r = dict(meta)
    for f in FEATURES:
        r[f"has_{f}"] = bool(df[f].loc[-PRE:-BASE_GAP].notna().sum() >= 60)
        r[f"t_{f}"], r[f"z_{f}"] = first_change(df[f], z_thr, min_dur, two_sided=f.startswith("eeg"))
    base_hr = df.ecg_hr.loc[-PRE:-BASE_GAP].median()
    peak_hr = df.ecg_hr.loc[0:60].max()
    r["hr_baseline"], r["hr_peak_0_60"] = base_hr, peak_hr
    r["hr_rise_bpm"] = peak_hr - base_hr
    r["hr_rise_pct"] = 100 * (peak_hr - base_hr) / base_hr if base_hr else np.nan
    # artefact-robust variant (added 2026-10-17 after the first real run, reported next to the pre-set one): the peak of
    # the 5-s rolling median HR, so single mis-detected R-peaks during movement cannot make the peak
    r["hr_rise_bpm_med5"] = df.ecg_hr.rolling(5, center=True, min_periods=3).median().loc[0:60].max() - base_hr
    r["has_eeg"] = r["has_eeg_linelength"]; r["has_hr"] = r["has_ecg_hr"]
    r["has_emg"] = r["has_emg_rms"]; r["has_acc"] = r["has_acc_mag"]
    r["t_eeg"] = np.nanmin([r["t_eeg_linelength"], r["t_eeg_power_1_25"]])
    r["t_hr"] = r["t_ecg_hr"]
    r["t_emg"] = np.nanmin([r["t_emg_rms"], r["t_emg_hf_power"]])
    r["t_acc"] = r["t_acc_mag"]
    r["lag_eeg_to_hr"] = r["t_hr"] - r["t_eeg"]
    r["lag_eeg_to_emg"] = r["t_emg"] - r["t_eeg"]
    r["lag_eeg_to_acc"] = r["t_acc"] - r["t_eeg"]
    r["lag_emg_to_acc"] = r["t_acc"] - r["t_emg"]
    return r


def curve_job(x):
    sub, run, onset = x["sub"], int(x["run"]), float(x["onset_s"])
    try:
        df = seizure_curves(sub, run, onset)
    except Exception as e:
        return x, None, f"{type(e).__name__}: {e}"
    return x, df, None


def summarise(O, out, z_thr, min_dur):
    def q(s):
        s = s.dropna()
        return f"{s.median():.0f} [{s.quantile(.25):.0f}, {s.quantile(.75):.0f}] n={len(s)}" if len(s) else "n=0"
    def share(G, sig):
        h = G[f"has_{sig}"]
        return f"{G.loc[h, f't_{sig}'].notna().mean():.0%} of {int(h.sum())}" if h.any() else "n=0"
    summ = []
    for g, G in list(O.groupby("group")) + [("all", O)]:
        H = G[G.has_hr]
        summ.append(dict(group=g, n=len(G), n_patients=G["sub"].nunique(),
            responds_eeg=share(G, "eeg"), responds_hr=share(G, "hr"), responds_emg=share(G, "emg"), responds_acc=share(G, "acc"),
            hr_rise_ge10bpm=f"{(H.hr_rise_bpm >= 10).mean():.0%}", hr_rise_ge20bpm=f"{(H.hr_rise_bpm >= 20).mean():.0%}",
            hr_rise_ge50bpm=f"{(H.hr_rise_bpm >= 50).mean():.0%}", hr_rise_bpm=q(H.hr_rise_bpm),
            med5_hr_rise_ge20bpm=f"{(H.hr_rise_bpm_med5 >= 20).mean():.0%}", med5_hr_rise_ge50bpm=f"{(H.hr_rise_bpm_med5 >= 50).mean():.0%}",
            med5_hr_rise_bpm=q(H.hr_rise_bpm_med5),
            t_eeg=q(G.t_eeg), t_hr=q(G.t_hr), t_emg=q(G.t_emg), t_acc=q(G.t_acc),
            lag_eeg_to_hr=q(G.lag_eeg_to_hr), lag_eeg_to_emg=q(G.lag_eeg_to_emg), lag_eeg_to_acc=q(G.lag_eeg_to_acc),
            lag_emg_to_acc=q(G.lag_emg_to_acc), share_eeg_before_hr=f"{(G.lag_eeg_to_hr > 0).sum()}/{G.lag_eeg_to_hr.notna().sum()}",
            z_thr=z_thr, min_dur_s=min_dur))
    S = pd.DataFrame(summ)
    S.to_csv(out / "summary_by_group.csv", index=False)
    print(S.set_index("group").T.to_string())
    return S


def write_lab_json(O, C, tag, z_thr, min_dur):
    """results/lab/data/expR.json: per definition (tag), median/IQR z-curves per group and signal (-120..+180 s), the
    per-seizure lags and response flags. Raw numbers only; the Lab page draws them."""
    import json
    groups = {}
    for g, G in list(O.groupby("group")) + [("all", O)]:
        Z = C[C.seizure_id.isin(set(G.seizure_id))]
        sig = {}
        for f in FEATURES:
            piv = Z.pivot(index="t", columns="seizure_id", values=f)
            base = piv.loc[-PRE:-BASE_GAP]
            z = ((piv - base.mean()) / (base.std() + 1e-9)).loc[:, base.notna().sum() >= 60].loc[-120:POST - 1]
            sig[f] = dict(n=int(z.shape[1]), t=z.index.tolist(), median=z.median(1).tolist(),
                          q25=z.quantile(.25, axis=1).tolist(), q75=z.quantile(.75, axis=1).tolist())
        stats = {}
        for c in ["t_eeg", "t_hr", "t_emg", "t_acc", "lag_eeg_to_hr", "lag_eeg_to_emg", "lag_eeg_to_acc", "lag_emg_to_acc",
                  "hr_rise_bpm", "hr_rise_bpm_med5"]:
            v = G[c].dropna()
            stats[c] = dict(n=int(len(v)), median=float(v.median()) if len(v) else None,
                            q25=float(v.quantile(.25)) if len(v) else None, q75=float(v.quantile(.75)) if len(v) else None)
        resp = {}
        for sg in ["eeg", "hr", "emg", "acc"]:
            h = G[f"has_{sg}"].astype(bool)
            k = int(G.loc[h, f"t_{sg}"].notna().sum())
            resp[sg] = dict(n_recorded=int(h.sum()), n_respond=k, share=k / int(h.sum()) if h.sum() else None)
        H = G[G.has_hr.astype(bool)]
        hr = {f"ge{k}": dict(n=int(len(H)), k=int((H.hr_rise_bpm >= k).sum()), k_med5=int((H.hr_rise_bpm_med5 >= k).sum()),
                             share=float((H.hr_rise_bpm >= k).mean()) if len(H) else None,
                             share_med5=float((H.hr_rise_bpm_med5 >= k).mean()) if len(H) else None) for k in (10, 20, 50)}
        groups[g] = dict(n_seizures=int(len(G)), n_patients=int(G["sub"].nunique()), curves=sig, stats=stats, respond=resp, hr_rise=hr)
    cols = ["seizure_id", "sub", "group", "type", "duration_s", "prev_sz_in_baseline", "has_eeg", "has_hr", "has_emg", "has_acc",
            "t_eeg", "t_hr", "t_emg", "t_acc", "lag_eeg_to_hr", "lag_eeg_to_emg", "lag_eeg_to_acc", "lag_emg_to_acc",
            "hr_rise_bpm", "hr_rise_bpm_med5"]
    p = L.LAB_DATA / "expR.json"
    J = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    J.setdefault("definitions", {})[tag] = dict(z_thr=z_thr, min_dur_s=min_dur, baseline_s=[-PRE, -BASE_GAP], groups=groups,
                                               seizures=O[cols].to_dict("records"))
    J["written"] = pd.Timestamp.now().isoformat(timespec="seconds")
    L.lab_write_json("expR.json", L.json_clean(J))


def plot(O, C, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    (out / "figures").mkdir(exist_ok=True)
    for g, G in O.groupby("group"):
        ids = set(G.seizure_id); Z = C[C.seizure_id.isin(ids)]
        fig, ax = plt.subplots(len(FEATURES), 1, figsize=(9, 11), sharex=True)
        for k, f in enumerate(FEATURES):
            piv = Z.pivot(index="t", columns="seizure_id", values=f)
            base = piv.loc[-PRE:-BASE_GAP]
            z = (piv - base.mean()) / (base.std() + 1e-9)
            z = z.loc[:, base.notna().sum() >= 60]
            med = z.median(axis=1); lo, hi = z.quantile(.25, axis=1), z.quantile(.75, axis=1)
            ax[k].fill_between(med.index, lo, hi, alpha=.25); ax[k].plot(med.index, med, lw=1.5)
            ax[k].axvline(0, color="k", ls="--", lw=.8); ax[k].axhline(0, color="gray", lw=.4)
            ax[k].set_ylabel(f"{f}\n(z, n={z.shape[1]})", fontsize=8); ax[k].set_xlim(-120, POST)
        ax[-1].set_xlabel("seconds from annotated onset")
        fig.suptitle(f"{g}: median z-score vs own -300..-60 s baseline across {len(ids)} seizures, {G['sub'].nunique()} patients (IQR shaded)", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "figures" / f"timeline_{g}.png", dpi=150); plt.close(fig)
    print("figures in", out / "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--z-thr", type=float, default=Z_THR)
    ap.add_argument("--min-dur", type=int, default=MIN_DUR)
    ap.add_argument("--from-curves", action="store_true", help="re-use results/expR/curves.parquet (robustness runs)")
    ap.add_argument("--tag", default="", help="write to results/expR/<tag>/")
    a = ap.parse_args()
    out = OUT / a.tag if a.tag else OUT
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(ROOT / "data/catalog/seizeit2_seizures.csv")
    split = json.load(open(ROOT / "data/catalog/splits.json"))
    dev = {s for f in split["dev_folds"].values() for s in f}
    S = S[S["sub"].isin(dev)]                         # never the locked test
    if a.subjects:
        S = S[S["sub"].isin(a.subjects)]
    S = S[S.duration_s >= 5].sort_values(["sub", "run", "onset_s"])   # seizures under 5 s have no measurable dynamics
    # flag seizures whose baseline window contains the end of an earlier seizure of the same run (baseline not clean)
    allS = pd.read_csv(ROOT / "data/catalog/seizeit2_seizures.csv")
    allS = allS[allS["sub"].isin(dev)]
    def prev_in_baseline(x):
        o = allS[(allS["sub"] == x.sub) & (allS.run == x.run) & (allS.onset_s < x.onset_s)]
        return bool(((o.onset_s + o.duration_s) > x.onset_s - PRE).any())
    S["prev_sz_in_baseline"] = [prev_in_baseline(x) for x in S.itertuples()]
    S["seizure_id"] = [f"{s}_r{int(r):02d}_{int(o)}" for s, r, o in zip(S["sub"], S.run, S.onset_s)]

    if a.from_curves:
        C = pd.read_parquet(OUT / "curves.parquet")
        S = S[S.seizure_id.isin(set(C.seizure_id))]
        rows = []
        for x in S.itertuples():
            df = C[C.seizure_id == x.seizure_id].set_index("t")[FEATURES]
            meta = dict(seizure_id=x.seizure_id, sub=x.sub, run=x.run, type=x.type, group=x.group, duration_s=x.duration_s,
                        prev_sz_in_baseline=x.prev_sz_in_baseline)
            rows.append(onset_row(meta, df, a.z_thr, a.min_dur))
    else:
        complete = lambda s, r: (RAW / f"{s}/ses-01/eeg/{s}_ses-01_task-szMonitoring_run-{int(r):02d}_eeg.edf").exists()
        S = S[[complete(s, r) for s, r in zip(S["sub"], S.run)]]
        print(f"{len(S)} dev seizures (>= 5 s) with downloaded EEG; {a.workers} workers")
        curves, rows, failed = [], [], []
        recs = S.to_dict("records")
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (x, df, err) in enumerate(ex.map(curve_job, recs, chunksize=4), 1):
                if err:
                    failed.append(f"{x['seizure_id']}: {err}"); print("FAILED", x["seizure_id"], err, flush=True); continue
                c = df.copy(); c["seizure_id"] = x["seizure_id"]; c["t"] = c.index
                curves.append(c)
                meta = {k: x[k] for k in ["seizure_id", "sub", "run", "type", "group", "duration_s", "prev_sz_in_baseline"]}
                rows.append(onset_row(meta, df, a.z_thr, a.min_dur))
                if i % 25 == 0:
                    print(f"  {i}/{len(recs)}", flush=True)
        if not rows:
            print("No downloaded seizure files found under", RAW); return
        C = pd.concat(curves, ignore_index=True); C.to_parquet(out / "curves.parquet")
        (out / "failed.txt").write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    O = pd.DataFrame(rows); O.to_csv(out / "onsets.csv", index=False)
    summarise(O, out, a.z_thr, a.min_dur)
    write_lab_json(O, C, a.tag or "main", a.z_thr, a.min_dur)
    if a.plot:
        plot(O, C, out)


if __name__ == "__main__":
    main()

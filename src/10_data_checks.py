"""Data checks on real SeizeIT2 files (Tasks 1-3): EDF headers, units, amplitudes, onset alignment, impd epochs,
and the feature-table summary. Dev subjects only (sub-001..096); refuses anything else.

    python src/10_data_checks.py headers  sub-001:3 sub-015:4 sub-016:3      # raw EDF header of every modality
    python src/10_data_checks.py spot --n 5 --seed 0                          # Task 1 spot check of random downloaded files
    python src/10_data_checks.py amplitudes sub-001:3 sub-015:4 sub-016:3    # EEG std, ECG R amplitude, EMG rest RMS, ACC
    python src/10_data_checks.py onsets  --group FBTC --n 3                   # +-120 s plots around annotated onsets
    python src/10_data_checks.py impd sub-001:3                               # impedance-check epochs vs surrounding signal
    python src/10_data_checks.py features                                     # results/data_checks/feature_summary.md

Figures and tables go to results/data_checks/. Signal processing for the plots uses ictra_lib (same filters as features).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

RAW = L.ROOT / "data/raw/ds005873"
OUT = L.ROOT / "results/data_checks"
FEAT = L.ROOT / "data/features"
MODS = ["eeg", "ecg", "emg", "mov"]
DEV = {s for f in L.load_splits()["dev_folds"].values() for s in f}


def edf_path(sub, run, m):
    if sub not in DEV:
        sys.exit(f"Refusing {sub}: not a development subject.")
    return RAW / f"{sub}/ses-01/{m}/{sub}_ses-01_task-szMonitoring_run-{run:02d}_{m}.edf"


def parse_runs(items):
    return [(s.split(":")[0], int(s.split(":")[1])) for s in items]


def raw_header(p):
    """Read the EDF(+) header bytes directly (no MNE), so units and ranges are exactly what the file says."""
    with open(p, "rb") as fh:
        b = fh.read(256); ns = int(b[252:256]); h = fh.read(ns * 256)
    f = lambda off, w: [h[off * ns + i * w: off * ns + (i + 1) * w].decode("latin1").strip() for i in range(ns)]
    lab, dim, pmin, pmax, dmin, dmax, nsr = f(0, 16), f(96, 8), f(104, 8), f(112, 8), f(120, 8), f(128, 8), f(216, 8)
    n_rec, rec_dur = int(b[236:244]), float(b[244:252])
    chans = [dict(label=lab[i], unit=dim[i], phys_min=float(pmin[i]), phys_max=float(pmax[i]), dig_min=int(dmin[i]),
                  dig_max=int(dmax[i]), fs=int(nsr[i]) / rec_dur) for i in range(ns) if lab[i] != "EDF Annotations"]
    return dict(edf_type=b[192:236].decode("latin1").strip() or "EDF", start=b[168:184].decode("latin1"),
                n_records=n_rec, record_s=rec_dur, duration_h=n_rec * rec_dur / 3600, channels=chans)


def cmd_headers(a):
    R = pd.read_csv(L.ROOT / "data/catalog/seizeit2_runs.csv").set_index(["sub", "run"])
    rows = []
    for sub, run in parse_runs(a.runs):
        for m in MODS:
            p = edf_path(sub, run, m)
            if not p.exists():
                continue
            h = raw_header(p)
            for c in h["channels"]:
                rows.append(dict(sub=sub, run=run, montage=R.loc[(sub, run), "eeg_montage"], modality=m, edf_type=h["edf_type"],
                                 start=h["start"], duration_h=round(h["duration_h"], 3), catalog_h=R.loc[(sub, run), "hours"], **c))
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "edf_headers.csv", index=False)
    print(T.to_string(index=False))


def cmd_spot(a):
    """Task 1: random complete files, opened with MNE: fs, channel count, duration vs catalog."""
    import mne
    F = pd.read_csv(L.ROOT / "data/catalog/seizeit2_files.csv")
    F = F[F["sub"].isin(DEV) & F.run_has_seizure]
    F = F[[(RAW / p).exists() and (RAW / p).stat().st_size == b for p, b in zip(F.path, F.bytes)]]
    R = pd.read_csv(L.ROOT / "data/catalog/seizeit2_runs.csv").set_index(["sub", "run"])
    rows = []
    for x in F.sample(a.n, random_state=a.seed).itertuples():
        raw = mne.io.read_raw_edf(RAW / x.path, preload=False, verbose=False)
        rows.append(dict(path=x.path, sfreq=raw.info["sfreq"], n_channels=len(raw.ch_names), channels=";".join(raw.ch_names),
                         duration_h=round(raw.n_times / raw.info["sfreq"] / 3600, 3), catalog_h=R.loc[(x.sub, x.run), "hours"]))
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "spot_check.csv", index=False)
    print(T.to_string(index=False))


def read(sub, run, m, start_s=None, stop_s=None, picks=None):
    import mne
    raw = mne.io.read_raw_edf(edf_path(sub, run, m), preload=False, verbose=False)
    fs = raw.info["sfreq"]
    a = 0 if start_s is None else max(0, int(start_s * fs))
    b = raw.n_times if stop_s is None else min(raw.n_times, int(stop_s * fs))
    return raw.get_data(picks=picks, start=a, stop=b) * 1e6, fs, raw.ch_names   # MNE returns V; header unit is uV


def cmd_amplitudes(a):
    """Typical amplitudes on 30 min from the middle of the run (away from seizures where possible)."""
    import neurokit2 as nk
    rows = []
    for sub, run in parse_runs(a.runs):
        dur = raw_header(edf_path(sub, run, "eeg"))["duration_h"] * 3600
        t0 = max(0, dur / 2 - 900); t1 = t0 + 1800
        r = dict(sub=sub, run=run, segment=f"{t0:.0f}-{t1:.0f} s")
        x, fs, names = read(sub, run, "eeg", t0, t1)
        h = L.harmonise(x[:2], fs, "eeg")
        sd = L.frame(h, 256).std(-1)                                      # 4-s window std per channel
        r.update(eeg_channels=";".join(names[:2]), eeg_std_uv_median_ch0=np.median(sd[0]), eeg_std_uv_median_ch1=np.median(sd[1]))
        if edf_path(sub, run, "ecg").exists():
            x, fs, names = read(sub, run, "ecg", t0, t1)
            e = L.harmonise(x[:1], fs, "ecg")[0].astype(np.float64)
            _, info = nk.ecg_peaks(nk.ecg_clean(e, sampling_rate=256), sampling_rate=256)
            pk = np.asarray(info["ECG_R_Peaks"])
            amp = np.abs(e[pk]) / 1000                                  # mV
            r.update(ecg_channel=names[0], ecg_r_amp_mv_median=np.median(amp), ecg_hr_bpm=60 * 256 / np.median(np.diff(pk)))
        if edf_path(sub, run, "emg").exists():
            x, fs, names = read(sub, run, "emg", t0, t1)
            m = L.harmonise(x[:1], fs, "emg")[0]
            rms = np.sqrt((L.frame(m[None], 256)[0] ** 2).mean(-1))
            r.update(emg_channel=names[0], emg_rms_uv_p10=np.percentile(rms, 10), emg_rms_uv_median=np.median(rms))
        if edf_path(sub, run, "mov").exists():
            x, fs, names = read(sub, run, "mov")
            x = x / 1e6                                                   # undo the uV scaling: mov unit in the header is g
            hdr = raw_header(edf_path(sub, run, "mov"))["channels"]
            for dev in ["EEG SD", "ECGEMG SD"]:
                idx = [i for i, n in enumerate(names) if n.startswith(dev + " ACC")]
                if len(idx) == 3:
                    mag = np.sqrt((x[idx] ** 2).sum(0))
                    at_lim = np.mean([np.isclose(x[i], hdr[i]["phys_max"], atol=1e-4) | np.isclose(x[i], hdr[i]["phys_min"], atol=1e-4) for i in idx])
                    r[f"acc_{dev.split()[0].lower()}_mag_g_median"] = np.median(mag)
                    r[f"acc_{dev.split()[0].lower()}_share_at_phys_limit"] = at_lim
        rows.append(r)
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "amplitudes.csv", index=False)
    print(T.T.to_string())


def _panel(sub, run, onset, dur, title, fname, pre=120, post=120):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t0, t1 = onset - pre, onset + post
    tr = []
    x, fs, names = read(sub, run, "eeg", t0 - 10, t1 + 10)
    h = L.harmonise(x[:2], fs, "eeg"); cut = int(10 * 256)
    tr += [(f"EEG {names[0]} (uV)", h[0, cut:-cut], 256), (f"EEG {names[1]} (uV)", h[1, cut:-cut], 256)]
    if edf_path(sub, run, "ecg").exists():
        x, fs, names = read(sub, run, "ecg", t0 - 10, t1 + 10)
        e = L.harmonise(x[:1], fs, "ecg")[0]
        hr, _ = L.heart_rate_series(e)
        tr += [("ECG (uV)", e[cut:-cut], 256), ("HR (bpm)", hr[10:-10], 1)]
    if edf_path(sub, run, "emg").exists():
        x, fs, names = read(sub, run, "emg", t0 - 10, t1 + 10)
        tr.append(("EMG 20-100 Hz (uV)", L.harmonise(x[:1], fs, "emg")[0, cut:-cut], 256))
    if edf_path(sub, run, "mov").exists():
        x, fs, names = read(sub, run, "mov", t0, t1)
        idx = [i for i, n in enumerate(names) if n.startswith("EEG SD ACC")]
        if len(idx) == 3:
            tr.append(("neck ACC |a| (g, calibrated)", np.sqrt((L.acc_to_g(x[idx], fs) ** 2).sum(0)), fs))
    fig, ax = plt.subplots(len(tr), 1, figsize=(11, 1.6 * len(tr)), sharex=True)
    for k, (lab, y, f) in enumerate(tr):
        t = np.arange(len(y)) / f - pre
        ax[k].plot(t, y, lw=0.4 if f > 1 else 1.2, color="k")
        ax[k].axvline(0, color="r", lw=1); ax[k].axvline(dur, color="r", lw=1, ls="--")
        ax[k].set_ylabel(lab, fontsize=7, rotation=0, ha="right")
        if f > 1 and "ACC" not in lab:
            lim = np.percentile(np.abs(y), 99.5) * 1.2; ax[k].set_ylim(-lim, lim)
    ax[-1].set_xlabel("seconds from annotated onset (red line; dashed = annotated end)")
    fig.suptitle(title, fontsize=9); fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
    print("wrote", OUT / fname)


def cmd_onsets(a):
    S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")
    S = S[S["sub"].isin(DEV) & (S.group == a.group)]
    S = S[[edf_path(s, int(r), "eeg").exists() for s, r in zip(S["sub"], S.run)]]
    for x in S.head(a.n).itertuples():
        _panel(x.sub, int(x.run), x.onset_s, x.duration_s, f"{x.sub} run-{x.run:02d} {x.type} onset {x.onset_s:.0f} s, duration {x.duration_s:.0f} s",
               f"onset_{a.group}_{x.sub}_run-{x.run:02d}_{int(x.onset_s)}.png")


def cmd_impd(a):
    rows = []
    for sub, run in parse_runs(a.runs):
        ev = pd.read_csv(edf_path(sub, run, "eeg").with_name(f"{sub}_ses-01_task-szMonitoring_run-{run:02d}_events.tsv"), sep="\t")
        for k, e in enumerate(ev[ev.eventType == "impd"].itertuples()):
            x, fs, _ = read(sub, run, "eeg", e.onset - 60, e.onset + e.duration + 60)
            h = L.harmonise(x[:2], fs, "eeg")
            i0, i1 = int(60 * 256), int((60 + e.duration) * 256)
            inside, outside = h[:, i0:i1], np.c_[h[:, :i0], h[:, i1:]]
            rows.append(dict(sub=sub, run=run, onset_s=e.onset, duration_s=e.duration,
                             std_inside_uv=inside.std(1).round(1).tolist(), std_around_uv=outside.std(1).round(1).tolist(),
                             ptp_inside_uv=np.ptp(inside, 1).round(0).tolist()))
            if k < a.plots:
                _panel(sub, run, e.onset, e.duration, f"{sub} run-{run:02d} impedance check (impd) at {e.onset:.0f} s, {e.duration:.0f} s",
                       f"impd_{sub}_run-{run:02d}_{int(e.onset)}.png", pre=30, post=30)
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "impd_check.csv", index=False)
    print(T.to_string(index=False))


def cmd_dropouts(a):
    """Share of samples flagged by ictra_lib.dropout_mask, per file and channel, on 12 x 60 s segments spread over each run,
    plus a before/after figure of the first run."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = []
    for sub, run in parse_runs(a.runs):
        dur = raw_header(edf_path(sub, run, "eeg"))["duration_h"] * 3600
        for m in MODS:
            if not edf_path(sub, run, m).exists():
                continue
            flagged, total = None, 0
            for t in np.linspace(60, dur - 120, 12):
                x, fs, names = read(sub, run, m, t, t + 60)
                x = x[:3] if m == "mov" else x
                k = np.array([L.dropout_mask(c, fs).sum() for c in x])
                flagged = k if flagged is None else flagged + k; total += x.shape[1]
            for c, nme in enumerate(names[: len(flagged)]):
                rows.append(dict(sub=sub, run=run, modality=m, channel=nme, share_flagged=flagged[c] / total))
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "dropouts.csv", index=False)
    print(T.pivot_table(index=["sub", "run"], columns="channel", values="share_flagged").map(lambda v: f"{v:.3%}").to_string())
    sub, run = parse_runs(a.runs)[0]
    dur = raw_header(edf_path(sub, run, "eeg"))["duration_h"] * 3600; t = dur / 2
    fig, ax = plt.subplots(4, 1, figsize=(12, 8))
    for k, m in enumerate(MODS):
        if not edf_path(sub, run, m).exists():
            continue
        d = 30 if m == "mov" else 8
        x, fs, names = read(sub, run, m, t - 10, t + d + 10)
        kind = "acc" if m == "mov" else m
        raw0 = x[:1]
        mk = L.dropout_mask(raw0[0], fs)
        # "as stored" = the same filters as ictra_lib.harmonise but without the dropout repair
        x0 = raw0 - np.median(raw0)
        before = {"eeg": lambda: L.bandpass(x0, *L.CFG["eeg_band"], fs), "ecg": lambda: L.bandpass(x0, *L.CFG["ecg_band"], fs),
                  "emg": lambda: L.notch(L.bandpass(x0, *L.CFG["emg_band"], fs), 50, fs), "mov": lambda: x0}[m]()
        after = L.harmonise(raw0, fs, kind, fs_out=int(fs)) if m != "mov" else L.repair_dropouts(raw0, fs) - np.median(raw0)
        c = slice(int(10 * fs), int((10 + d) * fs)); tt = np.arange(c.stop - c.start) / fs
        ax[k].plot(tt, before[0, c], lw=.5, color="tab:red", label="as stored")
        ax[k].plot(tt, after[0, c], lw=.5, color="k", label="dropouts repaired (ictra_lib.harmonise)")
        ax[k].set_title(f"{sub} run-{run:02d} {names[0]}: {mk[c].sum()} samples flagged in this {d} s", fontsize=8)
        ax[k].legend(fontsize=6, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / f"dropouts_{sub}_run-{run:02d}.png", dpi=120); plt.close(fig)


def cmd_features(a):
    """Task 3 summary over every parquet in data/features/ (streamed: one file at a time)."""
    files = sorted(FEAT.glob("*.parquet"))
    tot = dict(runs=0, windows=0, seizure_windows=0, edge_windows=0, impd_windows=0, flat_eeg=0, flat_ecg=0, flat_emg=0,
               has_ecg=0, has_emg=0, has_acc=0, hours=0.0)
    nan, present, n_seiz = None, None, set()
    for p in files:
        d = pd.read_parquet(p)
        tot["runs"] += 1; tot["windows"] += len(d); tot["hours"] += len(d) * L.CFG["step_s"] / 3600
        tot["seizure_windows"] += int(d.y.sum()); tot["edge_windows"] += int(d.edge.sum()); tot["impd_windows"] += int(d.impd.sum())
        for m in ["eeg", "ecg", "emg"]:
            tot[f"flat_{m}"] += int(d.get(f"flat_{m}", pd.Series(False)).fillna(False).astype(bool).sum())
        for m in ["ecg", "emg", "acc"]:
            tot[f"has_{m}"] += int(d[f"has_{m}"].sum())
        n_seiz |= {(d["sub"].iloc[0], int(d["run"].iloc[0]), int(s)) for s in d.sz_id[d.sz_id >= 0].unique()}
        feats = [c for c in d.columns if c.split("_")[0] in ("eeg", "ecg", "emg", "acc")]
        # NaN rate counted only inside runs where that modality exists and outside flat/impd windows
        ok = ~d.impd.values
        cnt = {}
        for c in feats:
            m = c.split("_")[0]
            mask = ok & (d[f"has_{m}"].values if m != "eeg" else True)
            if f"flat_{m}" in d:
                mask = mask & ~d[f"flat_{m}"].fillna(False).astype(bool).values
            cnt[c] = (int(d[c][mask].isna().sum()), int(mask.sum()))
        if nan is None:
            nan = {c: [0, 0] for c in cnt}
        for c, (k, n) in cnt.items():
            nan.setdefault(c, [0, 0]); nan[c][0] += k; nan[c][1] += n
    W = tot["windows"]
    lines = ["# Feature summary (Task 3)", "", f"Computed by `python src/10_data_checks.py features` from {tot['runs']} parquet files in `data/features/`.", "",
             "| quantity | value |", "|---|---|",
             f"| runs | {tot['runs']} |", f"| hours (windows x 2 s) | {tot['hours']:,.1f} |", f"| seizures with >= 1 window | {len(n_seiz)} |",
             f"| total windows | {W:,} |", f"| seizure windows (overlap >= 0.5) | {tot['seizure_windows']:,} ({tot['seizure_windows'] / W:.3%}) |",
             f"| edge windows (0 < overlap < 0.5) | {tot['edge_windows']:,} ({tot['edge_windows'] / W:.3%}) |",
             f"| impd windows | {tot['impd_windows']:,} ({tot['impd_windows'] / W:.3%}) |"]
    for m in ["eeg", "ecg", "emg"]:
        lines.append(f"| flat_{m} windows | {tot['flat_' + m]:,} ({tot['flat_' + m] / W:.3%}) |")
    for m in ["ecg", "emg", "acc"]:
        lines.append(f"| windows with {m} present | {tot['has_' + m] / W:.1%} |")
    lines += ["", "## NaN rate per feature", "",
              "Denominator: windows of runs where the modality exists, excluding impd and that modality's flat windows. "
              "Flag = NaN rate > 5%.", "", "| feature | NaN rate | flag |", "|---|---|---|"]
    for c, (k, n) in sorted(nan.items()):
        r = k / n if n else np.nan
        lines.append(f"| {c} | {r:.2%} | {'**> 5%**' if r > 0.05 else ''} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ["headers", "amplitudes", "dropouts"]:
        s = sp.add_parser(name); s.add_argument("runs", nargs="+")
    s = sp.add_parser("spot"); s.add_argument("--n", type=int, default=5); s.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("onsets"); s.add_argument("--group", default="FBTC"); s.add_argument("--n", type=int, default=3)
    s = sp.add_parser("impd"); s.add_argument("runs", nargs="+"); s.add_argument("--plots", type=int, default=2)
    sp.add_parser("features")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

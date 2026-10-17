"""Exp P: pre-ictal drift. EXPLORATORY study (docs/PREREGISTRATION_DRAFT.md): no hypothesis, no accept/reject.

Question (Allen, 2026-10-17): in the 2 minutes before the annotated onset, do heart rate, HRV, EMG level or movement drift
systematically compared with the 3 minutes before that? Development patients only (sub-001..096).

    python src/13_preictal_study.py --workers 2
    python src/13_preictal_study.py --from-curves          # re-summarise / re-plot from results/expP/curves.parquet

Design (fixed before the run):
  seizures   dev seizures with onset >= 600 s after the start of their recording and no other seizure (any type) starting
             or ending in the 600 s before onset (same run). Counted per group in summary.csv.
  controls   for every seizure, 10 pseudo-onsets t drawn uniformly (seeded per seizure) from the SAME patient's
             seizure-containing runs, where t >= 600 s, t <= run end - 30 s and the window [t-300, t] is >= 1800 s from
             every annotated seizure of that run. Sleep/wake matching is NOT possible: vigilance is annotated for
             seizures only and the recording clock is de-identified (every run starts 2000-01-01 00:00). Seizure-free
             runs are not used (stage 2 was still downloading; using only stage-1 runs keeps the control pool fixed).
  signals    1-s series over -330..+30 s (30-s pad), through ictra_lib:
               hr      ictra_lib.heart_rate_series on harmonised ECG (bpm)
               rmssd   ictra_lib.ecg_features "ecg_rmssd" (RMS of successive differences of the 1-s HR inside each 4-s
                       window, bpm; the model's HRV feature, NOT the clinical RR-interval RMSSD), each window held 2 s
               emg     log RMS per second of harmonised EMG (same quantity as ictra_lib emg_logrms, 1-s blocks)
               acc     SD per second of the neck |a| after ictra_lib.acc_to_g + repair_dropouts (movement, as Exp R)
             seconds inside an impd epoch are set to NaN.
  drift      z = (mean[pre] - mean[ref]) / SD[ref], per seizure and signal; main: ref = -300..-120, pre = -120..-30 s;
             robustness: ref = -300..-60, pre = -60..-30 s. Needs >= 50% finite seconds in each window.
             slope = least-squares slope over -300..-30 s, in SD[ref] per minute.
  summary    per group and signal: median z of seizures and of controls (patient-bootstrap 95% CI, 1,000 resamples of
             patients), share z > 2 for seizures vs controls and the difference with CI. A signal "shows pre-ictal
             drift" only if the seizure share of z > 2 exceeds the control share with the CI of the difference above 0.
             The share z < -2 is reported as a secondary column (falling HRV), not as the criterion.
Outputs: results/expP/curves.parquet, drift.csv, summary.csv, eligibility.csv, figures/preictal_<group>.png
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, hashlib, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

ROOT = L.ROOT
RAW = ROOT / "data/raw/ds005873"
OUT = ROOT / "results/expP"
FS = L.CFG["fs"]
SIGNALS = ["hr", "rmssd", "emg", "acc"]
LABEL = {"hr": "heart rate (bpm)", "rmssd": "HRV: ecg_rmssd (bpm)", "emg": "EMG log RMS", "acc": "movement: SD of |a| (g)"}
GROUPS = ["FBTC", "focal_motor", "focal_nonmotor", "focal_unknown_motor"]
PRE_MIN, GAP_S, N_CTRL, PAD, T0, T1 = 600, 1800, 10, 30, -300, 0
DEFS = {"main": dict(ref=(-300, -120), pre=(-120, -30)), "robust": dict(ref=(-300, -60), pre=(-60, -30))}


def dev_subjects():
    return {s for v in L.load_splits()["dev_folds"].values() for s in v}


def stem(sub, run, m):
    return RAW / sub / "ses-01" / m / f"{sub}_ses-01_task-szMonitoring_run-{run:02d}_{m}.edf"


def read_seg(path, t0, t1, picks):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    fs = raw.info["sfreq"]; a, b = max(0, int(t0 * fs)), min(raw.n_times, int(t1 * fs))
    return raw.get_data(picks=picks, start=a, stop=b) * 1e6, fs, a / fs


def place(values, start_abs, grid0, n):
    out = np.full(n, np.nan); k0 = int(round(start_abs - grid0))
    lo, hi = max(0, k0), min(n, k0 + len(values))
    if hi > lo:
        out[lo:hi] = values[lo - k0: hi - k0]
    return out


def run_job(task):
    """All segments (seizures + controls) of one run: dict(sub, run, items=[(seg_id, t)], impd=[(a, b)])."""
    sub, run, items, impd = task["sub"], task["run"], task["items"], task["impd"]
    n = (T1 - T0)                                         # seconds -300..-1 relative to t
    res, errs = [], []
    acc_g, acc_fs = None, None
    if stem(sub, run, "mov").exists():
        try:
            import mne
            raw = mne.io.read_raw_edf(stem(sub, run, "mov"), preload=False, verbose=False)
            idx = [i for i, c in enumerate(raw.ch_names) if c.startswith("EEG SD ACC")][:3]
            if len(idx) < 3:
                idx = [i for i, c in enumerate(raw.ch_names) if "acc" in c.lower()][:3]
            acc_fs = raw.info["sfreq"]
            acc_g = L.acc_to_g(raw.get_data(picks=idx) * 1e6, acc_fs)    # whole run: calibration = the run's median |a|
        except Exception as e:
            errs.append(f"{sub} run-{run:02d} mov: {e}")
    for seg_id, t in items:
        lo, hi = t + T0 - PAD, t + T1 + PAD
        grid0 = t + T0
        cur = {}
        try:
            if stem(sub, run, "ecg").exists():
                x, fs, s0 = read_seg(stem(sub, run, "ecg"), lo, hi, [0])
                g = L.harmonise(x, fs, "ecg")[0]
                hr, _ = L.heart_rate_series(g)
                cur["hr"] = place(hr, s0, grid0, n)
                E = L.ecg_features(g)
                rm = np.repeat(E.ecg_rmssd.values, L.CFG["step_s"])            # window i covers 2i..2i+4 s: hold 2 s
                cur["rmssd"] = place(rm, s0, grid0, n)
            if stem(sub, run, "emg").exists():
                x, fs, s0 = read_seg(stem(sub, run, "emg"), lo, hi, [0])
                g = L.harmonise(x, fs, "emg")[0]; k = len(g) // FS
                cur["emg"] = place(np.log(np.sqrt((g[: k * FS].reshape(k, FS) ** 2).mean(1)) + 1e-9), s0, grid0, n)
            if acc_g is not None:
                a_, b_ = max(0, int(lo * acc_fs)), min(acc_g.shape[1], int(hi * acc_fs))
                seg = L.repair_dropouts(acc_g[:, a_:b_], acc_fs); mag = np.sqrt((seg ** 2).sum(0))
                f = int(acc_fs); k = len(mag) // f
                cur["acc"] = place(mag[: k * f].reshape(k, f).std(1), a_ / acc_fs, grid0, n)
        except Exception as e:
            errs.append(f"{seg_id}: {type(e).__name__}: {e}"); continue
        rel = np.arange(T0, T1)
        bad = np.zeros(n, bool)
        for a, b in impd:
            bad |= (rel + t >= a) & (rel + t < b)
        for s in SIGNALS:
            v = cur.get(s, np.full(n, np.nan)); v = np.where(bad, np.nan, v)
            res.append(pd.DataFrame({"seg_id": seg_id, "signal": s, "t": rel, "value": v.astype(np.float32)}))
    return (pd.concat(res, ignore_index=True) if res else None), errs


def build_tasks():
    S = pd.read_csv(ROOT / "data/catalog/seizeit2_seizures.csv")
    R = pd.read_csv(ROOT / "data/catalog/seizeit2_runs.csv")
    dev = dev_subjects()
    S = S[S["sub"].isin(dev)].copy(); R = R[R["sub"].isin(dev) & (R.n_seizures > 0)].copy()
    assert not set(S["sub"]) - dev and not set(R["sub"]) - dev      # never anything outside sub-001..096
    S["seizure_id"] = [f"{a}_r{int(b):02d}_{int(c)}" for a, b, c in zip(S["sub"], S.run, S.onset_s)]
    S["end_s"] = S.onset_s + S.duration_s
    elig = []
    for e in S.itertuples():
        same = S[(S["sub"] == e.sub) & (S.run == e.run) & (S.seizure_id != e.seizure_id)]
        prior = same[(same.onset_s < e.onset_s) & (same.end_s > e.onset_s - PRE_MIN)]
        ok_files = stem(e.sub, int(e.run), "ecg").exists() or stem(e.sub, int(e.run), "emg").exists() or stem(e.sub, int(e.run), "mov").exists()
        why = "ok" if (e.onset_s >= PRE_MIN and prior.empty and ok_files) else \
              ("onset < 600 s into the recording" if e.onset_s < PRE_MIN else "another seizure in the preceding 600 s" if not prior.empty else "no ECG/EMG/MOV file")
        elig.append(dict(seizure_id=e.seizure_id, sub=e.sub, run=int(e.run), group=e.group, onset_s=e.onset_s, eligible=why == "ok", reason=why))
    El = pd.DataFrame(elig)
    Q = S.merge(El[["seizure_id", "eligible"]], on="seizure_id")
    Q = Q[Q.eligible]
    # control pool per patient: eligible seconds of the patient's seizure-containing runs
    pools = {}
    for sub, Rs in R.groupby("sub"):
        segs = []
        for r in Rs.itertuples():
            dur = r.hours * 3600
            ev = S[(S["sub"] == sub) & (S.run == r.run)]
            ok = [(PRE_MIN, dur - PAD)]
            for a, b in zip(ev.onset_s, ev.end_s):          # t is allowed if [t-300, t] is >= GAP_S from [a, b]
                cut_lo, cut_hi = a - GAP_S, b + GAP_S - T0   # forbid t in (a - GAP_S, b + GAP_S + 300)
                ok = [(x, y) for lo_, hi_ in ok for (x, y) in [(lo_, min(hi_, cut_lo)), (max(lo_, cut_hi), hi_)] if y > x]
            segs += [(int(r.run), x, y) for x, y in ok]
        pools[sub] = segs
    items, ctrl_rows = {}, []
    for e in Q.itertuples():
        items.setdefault((e.sub, int(e.run)), []).append((e.seizure_id, float(e.onset_s)))
        pool = pools.get(e.sub, [])
        tot = sum(y - x for _, x, y in pool)
        if tot <= 0:
            continue
        rng = np.random.default_rng(int(hashlib.sha256(e.seizure_id.encode()).hexdigest()[:8], 16))
        w = np.array([y - x for _, x, y in pool]) / tot
        for k in range(N_CTRL):
            j = rng.choice(len(pool), p=w); run, x, y = pool[j]
            t = float(np.floor(rng.uniform(x, y)))
            cid = f"{e.seizure_id}_c{k}"
            items.setdefault((e.sub, run), []).append((cid, t))
            ctrl_rows.append(dict(seg_id=cid, seizure_id=e.seizure_id, sub=e.sub, run=run, t=t, group=e.group))
    impd = {}
    for (sub, run) in items:
        tsv = stem(sub, run, "eeg").with_name(f"{sub}_ses-01_task-szMonitoring_run-{run:02d}_events.tsv")
        if not tsv.exists():
            tsv = ROOT / "data/raw/ds005873_meta" / tsv.relative_to(RAW)
        Ev = pd.read_csv(tsv, sep="\t") if tsv.exists() else pd.DataFrame(columns=["onset", "duration", "eventType"])
        Ev = Ev[Ev.eventType == "impd"]
        impd[(sub, run)] = list(zip(Ev.onset, Ev.onset + Ev.duration))
    tasks = [dict(sub=s, run=r, items=v, impd=impd[(s, r)]) for (s, r), v in items.items()]
    seg = pd.concat([Q.assign(seg_id=Q.seizure_id, kind="seizure", t=Q.onset_s)[["seg_id", "seizure_id", "sub", "run", "t", "group", "kind"]],
                     pd.DataFrame(ctrl_rows).assign(kind="control")], ignore_index=True)
    return tasks, seg, El


def drift_table(C, seg):
    rows = []
    for (sid, s), d in C.groupby(["seg_id", "signal"], sort=False):
        v = pd.Series(d.value.values, index=d.t.values)
        r = dict(seg_id=sid, signal=s)
        for name, D in DEFS.items():
            ref, pre = v.loc[D["ref"][0]: D["ref"][1] - 1], v.loc[D["pre"][0]: D["pre"][1] - 1]
            ok = ref.notna().mean() >= 0.5 and pre.notna().mean() >= 0.5 and ref.std() > 0
            r[f"z_{name}"] = (pre.mean() - ref.mean()) / ref.std() if ok else np.nan
        x = v.loc[-300:-31].dropna(); ref = v.loc[-300:-121]
        r["slope_sd_per_min"] = (np.polyfit(x.index.values / 60, x.values, 1)[0] / ref.std()) if len(x) >= 135 and ref.std() > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).merge(seg, on="seg_id", how="left")


def boot(D, fn, n=1000, seed=0):
    subs = D["sub"].unique(); rng = np.random.default_rng(seed); by = {s: g for s, g in D.groupby("sub")}; vals = []
    for _ in range(n):
        pick = rng.choice(subs, len(subs))
        vals.append(fn(pd.concat([by[s] for s in pick])))
    v = np.array(vals, float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def summarise(Dr):
    rows = []
    for name in DEFS:
        z = f"z_{name}"
        for g in GROUPS + ["all"]:
            for s in SIGNALS:
                D = Dr[(Dr.signal == s) & ((Dr.group == g) | (g == "all")) & Dr[z].notna()]
                sz, ct = D[D.kind == "seizure"], D[D.kind == "control"]
                if sz.empty or ct.empty:
                    continue
                med = lambda X, k: X.loc[X.kind == k, z].median()
                share = lambda X, k, f: f(X.loc[X.kind == k, z]).mean()
                up = lambda x: x > 2; dn = lambda x: x < -2
                diff_up = lambda X: share(X, "seizure", up) - share(X, "control", up)
                diff_dn = lambda X: share(X, "seizure", dn) - share(X, "control", dn)
                lo_d, hi_d = boot(D, diff_up); lo_dn, hi_dn = boot(D, diff_dn)
                rows.append(dict(definition=name, ref=f"{DEFS[name]['ref'][0]}..{DEFS[name]['ref'][1]}", pre=f"{DEFS[name]['pre'][0]}..{DEFS[name]['pre'][1]}",
                                 group=g, signal=s, n_seizures=len(sz), n_patients=sz["sub"].nunique(), n_controls=len(ct),
                                 median_z_seizure=med(D, "seizure"), median_z_seizure_ci=boot(D, lambda X: med(X, "seizure")),
                                 median_z_control=med(D, "control"), median_z_control_ci=boot(D, lambda X: med(X, "control")),
                                 share_z_gt2_seizure=share(D, "seizure", up), share_z_gt2_control=share(D, "control", up),
                                 diff_share_gt2=diff_up(D), diff_share_gt2_lo=lo_d, diff_share_gt2_hi=hi_d,
                                 share_z_lt_m2_seizure=share(D, "seizure", dn), share_z_lt_m2_control=share(D, "control", dn),
                                 diff_share_lt_m2=diff_dn(D), diff_share_lt_m2_lo=lo_dn, diff_share_lt_m2_hi=hi_dn,
                                 median_slope_seizure=sz.slope_sd_per_min.median(), median_slope_control=ct.slope_sd_per_min.median(),
                                 shows_preictal_drift=bool(lo_d > 0)))
    T = pd.DataFrame(rows)
    for c in ["median_z_seizure_ci", "median_z_control_ci"]:
        T[c + "_lo"] = [x[0] for x in T[c]]; T[c + "_hi"] = [x[1] for x in T[c]]
    return T.drop(columns=["median_z_seizure_ci", "median_z_control_ci"])


def figures(C, seg):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    C = C.merge(seg[["seg_id", "kind", "group"]], on="seg_id")
    # each segment's curve as z vs its own reference window (-300..-120 s), so signals and patients are comparable
    st = C[(C.t >= -300) & (C.t < -120)].groupby(["seg_id", "signal"]).value.agg(["mean", "std"]).reset_index()
    C = C.merge(st, on=["seg_id", "signal"]); C["z"] = (C.value - C["mean"]) / C["std"].where(C["std"] > 0)
    for g in GROUPS:
        fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)
        for ax, s in zip(axs, SIGNALS):
            for kind, ls, col in [("seizure", "-", "#2a78d6"), ("control", "--", "#52514e")]:
                d = C[(C.group == g) & (C.signal == s) & (C.kind == kind)]
                if d.empty:
                    continue
                q = d.groupby("t").z.quantile([.25, .5, .75]).unstack()
                n = d.seg_id.nunique()
                ax.plot(q.index, q[0.5], ls, color=col, lw=2 if kind == "seizure" else 1.5, label=f"{kind}s (n={n})")
                if kind == "seizure":
                    ax.fill_between(q.index, q[0.25], q[0.75], color=col, alpha=.15, lw=0)
            for x0, x1 in [(-300, -120), (-120, -30)]:
                ax.axvspan(x0, x1, color="#e4e3df" if x0 == -300 else "#f3e9d8", alpha=.5, lw=0)
            ax.axhline(0, color="#b5b4af", lw=.8); ax.set_title(LABEL[s], fontsize=9); ax.set_xlim(-300, 0)
            ax.set_xlabel("s before annotated onset (shaded: reference / pre window)", fontsize=8)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
        axs[0].set_ylabel("median z vs own -300..-120 s", fontsize=8); axs[0].legend(fontsize=8, frameon=False)
        nsz = seg[(seg.group == g) & (seg.kind == "seizure")].shape[0]
        fig.suptitle(f"Exp P, {g}: median 1-s curves before onset, seizures (solid, IQR shaded) vs same-patient pseudo-onsets "
                     f"(dashed); {nsz} eligible seizures, dev patients", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "figures" / f"preictal_{g}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--from-curves", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, seg, El = build_tasks()
    El.to_csv(OUT / "eligibility.csv", index=False); seg.to_csv(OUT / "segments.csv", index=False)
    print(El.groupby("group").eligible.agg(["sum", "size"]).rename(columns={"sum": "eligible", "size": "seizures"}).to_string())
    print(f"{(seg.kind == 'seizure').sum()} seizures, {(seg.kind == 'control').sum()} control pseudo-onsets, {len(tasks)} runs, {a.workers} workers", flush=True)
    if a.from_curves:
        C = pd.read_parquet(OUT / "curves.parquet")
    else:
        parts, errs = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (df, e) in enumerate(ex.map(run_job, tasks), 1):
                if df is not None:
                    parts.append(df)
                errs += e
                if i % 25 == 0:
                    print(f"  {i}/{len(tasks)} runs", flush=True)
        C = pd.concat(parts, ignore_index=True); C.to_parquet(OUT / "curves.parquet", index=False)
        (OUT / "failed.txt").write_text("\n".join(errs), encoding="utf-8")
        print(f"{len(errs)} segment errors (failed.txt)")
    Dr = drift_table(C, seg); Dr.to_csv(OUT / "drift.csv", index=False)
    T = summarise(Dr); T.to_csv(OUT / "summary.csv", index=False)
    cols = ["definition", "group", "signal", "n_seizures", "n_controls", "median_z_seizure", "median_z_control", "share_z_gt2_seizure",
            "share_z_gt2_control", "diff_share_gt2", "diff_share_gt2_lo", "diff_share_gt2_hi", "shows_preictal_drift"]
    print(T[cols].round(3).to_string(index=False))
    figures(C, seg)
    print("figures in", OUT / "figures")


if __name__ == "__main__":
    main()

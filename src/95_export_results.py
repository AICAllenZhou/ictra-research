"""Results hand-off to the website (docs/DELIVERABLES_SPEC.md, Task 15).

    python src/95_export_results.py --stage dev --version v01_baseline          # writes + zips
    python src/09_independent_check.py --zip results/export/ictra_results_<date>_dev_v01_baseline.zip   # before sending

Writes results/export/ictra_results_<YYYYMMDD>_<stage>_<version>/ and the .zip next to it. Only --stage dev exists before
pre-registration; --stage locked_test / external refuse to run without docs/PREREGISTRATION.sha256 (same guard as 03).

Everything comes from files the pipeline already wrote (results/dev/<version>/, data/catalog/, results/expR/); nothing is
re-trained. The only signal processing (example_events) goes through ictra_lib.harmonise / heart_rate_series / acc_to_g;
the 64 Hz and 16 Hz columns are display decimations of the harmonised 256 Hz signals.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, json, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

ROOT = L.ROOT
RAW = ROOT / "data/raw/ds005873"
# feature directory: L.features_dir() (data/features_causal for causal versions; set by L.mode_for_version in main)
STEP = L.CFG["step_s"]
GROUPS = ["FBTC", "focal_motor", "focal_nonmotor", "focal_unknown_motor"]
# exported prediction sets: the three fusion levels (xgb, the primary family) and the four single-input rules
EXPORT_MODELS = [("A_eeg", "xgb"), ("B_eeg_ecg", "xgb"), ("C_eeg_ecg_emg_acc", "xgb"),
                 ("rule", "rule_hr"), ("rule", "rule_emg"), ("rule", "rule_eeg"), ("rule", "rule_acc")]
RULE_UNITS = {"rule_hr": "ecg_hr_rise_pct (% HR rise over the causal 5-min baseline)",
              "rule_emg": "emg_logrms_rel (log EMG RMS minus its causal 5-min median)",
              "rule_eeg": "eeg_ll_max_rel (max-channel log line length minus its causal 5-min median)",
              "rule_acc": "acc_logsd_rel (log ACC SD minus its causal 5-min median)"}
FUSION = ("C_eeg_ecg_emg_acc", "xgb")


def folder_name(s, m):
    return m if s == "rule" else f"{s}__{m}"


def guard(stage):
    if stage != "dev" and not (ROOT / "docs/PREREGISTRATION.sha256").exists():
        sys.exit(f"--stage {stage} needs docs/PREREGISTRATION.sha256 (pre-registration, Task 10); refusing.")


def run_masks(sub, run, cache={}):
    """Window mask (impd | flat EEG, as in 04/05) and has_* flags of one run, from its feature file."""
    if (sub, run) not in cache:
        q = pd.read_parquet(L.features_dir() / f"{sub}_run-{run:02d}.parquet", columns=["impd", "flat_eeg", "has_ecg", "has_emg", "has_acc"])
        cache[(sub, run)] = (q.impd.values | q.flat_eeg.fillna(False).astype(bool).values,
                             dict(ecg=bool(q.has_ecg.iloc[0]), emg=bool(q.has_emg.iloc[0]), acc=bool(q.has_acc.iloc[0])))
    return cache[(sub, run)]


def thresholds(vdir):
    rows = []
    for p in sorted(vdir.glob("fold_*.csv")):
        s, m, rest = p.stem[len("fold_"):].split("__"); f, sd = rest[1:].split("_s")
        rows.append(dict(set=s, model=m, fold=int(f), seed=int(sd), thr=float(pd.read_csv(p, usecols=["thr"]).thr.iloc[0])))
    return pd.DataFrame(rows)


def load_preds(vdir, s, m):
    """All held-out window probabilities of one (set, model): DataFrame sub, run, t0, fold, prob_s<seed>... (folds are
    disjoint by patient, so each seed's column is the concatenation of its 5 fold files)."""
    by_seed = {}
    for p in sorted((vdir / "preds").glob(f"{s}__{m}__f*_s*.parquet")):
        f, sd = p.stem.split("__")[2][1:].split("_s")
        d = pd.read_parquet(p).rename(columns={"prob": f"prob_s{sd}"}); d["fold"] = int(f)
        by_seed.setdefault(int(sd), []).append(d)
    if not by_seed:
        return None
    out = None
    for sd in sorted(by_seed):
        d = pd.concat(by_seed[sd], ignore_index=True)
        out = d if out is None else out.merge(d, on=["sub", "run", "t0", "fold"], how="outer", validate="one_to_one")
    return out


# ------------------------------------------------------------------ predictions_val/
def export_predictions(vdir, ex, S, T):
    """1 row per second, each window's value held for its 2-s hop (window i covers t0 = 2i .. 2i+4 s; seconds 2i, 2i+1 get
    window i). Columns: subject, session, run, t_sec, fold, prob (mean over seeds; rules: the raw rule score), prob_s<k>,
    alert_s<k> (threshold of that fold/seed + mask + 3-of-5, exactly as scored), label (second inside an annotated
    seizure), missing (impd or flat EEG: can never alarm, excluded from the FA hours)."""
    counts = {}
    for s, m in EXPORT_MODELS:
        P = load_preds(vdir, s, m)
        if P is None:
            print("  no preds for", s, m); continue
        seeds = sorted(int(c.split("_s")[1]) for c in P.columns if c.startswith("prob_s"))
        d_out = ex / "predictions_val" / folder_name(s, m); d_out.mkdir(parents=True, exist_ok=True)
        n_rows = 0
        for sub, Ps in P.groupby("sub"):
            parts = []
            for run, d in Ps.groupby("run"):
                d = d.sort_values("t0"); nw = len(d)
                assert (d.t0.values == np.arange(nw) * STEP).all(), (sub, run)
                mask, _ = run_masks(sub, int(run))
                fold = int(d.fold.iloc[0])
                o = pd.DataFrame({"subject": sub, "session": "ses-01", "run": int(run), "t_sec": np.arange(nw * STEP), "fold": fold})
                rep = lambda v: np.repeat(np.asarray(v), STEP)
                probs = []
                for sd in seeds:
                    p = d[f"prob_s{sd}"].values.astype(float)
                    if s == "rule":
                        p = np.where(p <= -1e8, np.nan, p)            # Rule() codes a missing feature as -1e9
                    thr = T[(T.set == s) & (T.model == m) & (T.fold == fold) & (T.seed == sd)].thr.iloc[0]
                    alarm = L.k_of_n((np.nan_to_num(p, nan=-np.inf) > thr) & ~mask)
                    o[f"prob_s{sd}"] = rep(p); o[f"alert_s{sd}"] = rep(alarm.astype(np.int8)); probs.append(p)
                o.insert(5, "prob", rep(np.nanmean(np.vstack(probs), 0) if len(probs) > 1 else probs[0]))
                ev = S[(S["sub"] == sub) & (S.run == run)]
                lab = np.zeros(nw * STEP, np.int8); t = np.arange(nw * STEP)
                for e in ev.itertuples():
                    lab[(t >= e.onset_s) & (t < e.onset_s + e.duration_s)] = 1   # second [t, t+1) starts inside the seizure
                o["label"] = lab; o["missing"] = rep(mask.astype(np.int8))
                parts.append(o)
            D = pd.concat(parts, ignore_index=True); n_rows += len(D)
            D.to_csv(d_out / f"{sub}.csv.gz", index=False, float_format="%.9g",   # %.9g round-trips the float32 probabilities exactly
                     compression={"method": "gzip", "compresslevel": 6})
        counts[folder_name(s, m)] = dict(rows=n_rows, seeds=seeds, patients=int(P["sub"].nunique()))
        print(f"  predictions_val/{folder_name(s, m)}: {n_rows:,} s, seeds {seeds}", flush=True)
    return counts


def export_events(ex, S, version_runs):
    ev_json = json.loads((RAW / "events.json").read_text(encoding="utf-8"))
    levels = ev_json["eventType"]["Levels"]
    rows = []
    for (sub, run), G in S.groupby(["sub", "run"]):
        if (sub, int(run)) not in version_runs:
            continue
        tsv = RAW / sub / "ses-01" / "eeg" / f"{sub}_ses-01_task-szMonitoring_run-{int(run):02d}_events.tsv"
        E = pd.read_csv(tsv, sep="\t") if tsv.exists() else None
        mask, has = run_masks(sub, int(run))
        for k, e in enumerate(G.sort_values("onset_s").itertuples(), 1):
            vig = np.nan
            if E is not None:
                hit = E[(E.onset.round(2) == round(e.onset_s, 2)) & (E.eventType == e.type)]
                vig = hit.vigilance.iloc[0] if len(hit) and hit.vigilance.iloc[0] not in ("n/a", None) else np.nan
            w0, w1 = int(e.onset_s // STEP), int(np.ceil((e.onset_s + e.duration_s) / STEP))
            flat = pd.read_parquet(L.features_dir() / f"{sub}_run-{int(run):02d}.parquet", columns=["flat_ecg", "flat_emg"]).iloc[w0:w1]
            mods = ["eeg"] + [m for m, ok in [("ecg", has["ecg"] and not flat.flat_ecg.fillna(False).all()),
                                              ("emg", has["emg"] and not flat.flat_emg.fillna(False).all()), ("acc", has["acc"])] if ok]
            rows.append(dict(event_id=f"{sub}_ses-01_run-{int(run):02d}_e{k:02d}", subject=sub, session="ses-01", run=int(run),
                             onset_s=e.onset_s, duration_s=e.duration_s, offset_s=e.onset_s + e.duration_s, seizure_type=e.type,
                             seizure_type_ilae=levels.get(e.type, np.nan), seizure_group=e.group, vigilance=vig,
                             modalities_available=";".join(mods)))
    E = pd.DataFrame(rows)
    E.to_csv(ex / "predictions_val" / "events_val.csv", index=False)
    return E


# ------------------------------------------------------------------ metrics/
def score_no_kofn(vdir, s, m, S, T):
    """fusion_no_kofn: the same held-out probabilities and thresholds, alarms = threshold only (no 3-of-5)."""
    per = []
    for p in sorted((vdir / "preds").glob(f"{s}__{m}__f*_s*.parquet")):
        f, sd = p.stem.split("__")[2][1:].split("_s")
        thr = T[(T.set == s) & (T.model == m) & (T.fold == int(f)) & (T.seed == int(sd))].thr.iloc[0]
        for (sub, run), d in pd.read_parquet(p).groupby(["sub", "run"]):
            mask, _ = run_masks(sub, int(run))
            pos = (d.sort_values("t0").prob.values > thr) & ~mask
            hyp = L.windows_to_events(pos)
            ev = S[(S["sub"] == sub) & (S.run == run)]
            sc = L.score_events(list(zip(ev.onset_s, ev.onset_s + ev.duration_s)), hyp, (len(d) - int(mask.sum())) * STEP / 3600)
            sc.update(sub=sub, seed=int(sd)); per.append(sc)
    P = pd.DataFrame(per)
    pat = P.groupby(["seed", "sub"]).agg(n_ref=("n_ref", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), n_hyp=("n_hyp", "sum"),
                                         hours=("hours", "sum"), median_delay_s=("median_delay_s", "median")).reset_index()
    pooled = pd.DataFrame([L.pool_scores(g.to_dict("records")) for _, g in pat.groupby("seed")]).mean(numeric_only=True)
    d0 = pat[pat.seed == pat.seed.min()].to_dict("records")
    ci = {k: L.patient_bootstrap(d0, k) for k in ["sensitivity", "fa_per_24h", "f1"]}
    return dict(set=s, model=f"{m}_no_kofn", n_seeds=pat.seed.nunique(), n_patients=pat["sub"].nunique(), hours=pooled.hours,
                seizures=pooled.n_ref, sensitivity=pooled.sensitivity, fa_per_24h=pooled.fa_per_24h, precision=pooled.precision,
                f1=pooled.f1, median_delay_s=pooled.median_delay_s, sens_lo=ci["sensitivity"][0], sens_hi=ci["sensitivity"][1],
                fa_lo=ci["fa_per_24h"][0], fa_hi=ci["fa_per_24h"][1], f1_lo=ci["f1"][0], f1_hi=ci["f1"][1])


def export_metrics(vdir, ex, S, T):
    mdir = ex / "metrics"; mdir.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(vdir / "summary.csv")
    A = pd.concat([A, pd.DataFrame([dict(score_no_kofn(vdir, *FUSION, S, T), config="fusion_no_kofn")])], ignore_index=True)
    A["config"] = A.get("config", pd.Series(index=A.index, dtype=object)).fillna(A.set + "__" + A.model)
    A["post_processing"] = np.where(A.model.str.endswith("_no_kofn"), "threshold only", "3-of-5 windows")
    keep = ["config", "set", "model", "post_processing", "n_seeds", "n_patients", "seizures", "hours", "sensitivity", "sens_lo", "sens_hi",
            "fa_per_24h", "fa_lo", "fa_hi", "precision", "f1", "f1_lo", "f1_hi", "median_delay_s"] + [c for c in A if c.startswith("sens_") and c[5:] in GROUPS]
    A = A[keep]
    for c in ["sensitivity", "sens_lo", "sens_hi", "precision"] + [c for c in A if c.startswith("sens_") and c[5:] in GROUPS]:
        A[c.replace("sensitivity", "sensitivity_pct") if c == "sensitivity" else c + "_pct"] = 100 * A[c]
    A = A.drop(columns=["sensitivity", "sens_lo", "sens_hi", "precision"] + [c for c in A if c.startswith("sens_") and c[5:] in GROUPS and not c.endswith("_pct")])
    A.to_csv(mdir / "ablation.csv", index=False)
    T.to_csv(mdir / "thresholds.csv", index=False)

    if (vdir / "operating_curve.csv").exists():
        C = pd.read_csv(vdir / "operating_curve.csv")
        C["sensitivity_pct"] = 100 * C.sensitivity
        for g in GROUPS:
            C[f"sens_{g}_pct"] = 100 * C[f"sens_{g}"]
        C.drop(columns=["sensitivity"] + [f"sens_{g}" for g in GROUPS]).to_csv(mdir / "operating_curve.csv", index=False)
        shutil.copy(vdir / "operating_points.csv", mdir / "operating_points.csv")

    G = pd.read_csv(vdir / "scores_by_group.csv")
    pt = G.groupby(["set", "model", "seed", "group"]).agg(seizures=("n_ref", "sum"), detected=("tp", "sum")).reset_index()
    pt = pt.groupby(["set", "model", "group"]).agg(seizures=("seizures", "first"), detected_mean=("detected", "mean"), n_seeds=("seed", "nunique")).reset_index()
    pt["sensitivity_pct"] = 100 * pt.detected_mean / pt.seizures
    pt.to_csv(mdir / "per_type.csv", index=False)

    pat = pd.read_csv(vdir / "scores_by_patient.csv")
    pat["sensitivity_pct"] = 100 * pat.tp / pat.n_ref.where(pat.n_ref > 0)
    pat["fa_per_24h"] = pat.fp / pat.hours * 24
    pat.rename(columns={"sub": "subject", "n_ref": "seizures", "tp": "detected", "fp": "false_alarms", "n_hyp": "alarms"}).to_csv(mdir / "per_patient.csv", index=False)

    D = pd.read_csv(vdir / "detections_by_seizure.csv")
    O = pd.read_csv(ROOT / "results/expR/onsets.csv")[["seizure_id", "t_eeg"]]
    D = D.merge(O, on="seizure_id", how="left")
    D["latency_s"] = D.delay_s; D["latency_from_bte_change_s"] = D.delay_s - D.t_eeg
    D.rename(columns={"sub": "subject", "t_eeg": "bte_eeg_change_s"}).drop(columns=["delay_s"]).to_csv(mdir / "latency.csv", index=False)
    shutil.copy(vdir / "summary_by_montage.csv", mdir / "per_montage.csv")
    shutil.copy(vdir / "delay_by_group.csv", mdir / "latency_by_group.csv")

    curves = []
    for p in sorted(vdir.glob("curve_*.csv")):
        s, m, rest = p.stem[len("curve_"):].split("__"); f, sd = rest[1:].split("_s")
        c = pd.read_csv(p); c.insert(0, "seed", int(sd)); c.insert(0, "fold", int(f)); c.insert(0, "model", m); c.insert(0, "set", s)
        curves.append(c)
    pd.concat(curves).to_csv(mdir / "training_curves.csv", index=False)

    imp = []
    for p in sorted(vdir.glob(f"importance_{FUSION[0]}__{FUSION[1]}__*.csv")):
        imp.append(pd.read_csv(p))
    I = pd.concat(imp).groupby("feature").gain.agg(["mean", "std", "count"]).reset_index().rename(
        columns={"mean": "gain_mean", "std": "gain_sd", "count": "n_models"})
    I["gain_share_pct"] = 100 * I.gain_mean / I.gain_mean.sum()
    for mod in ["eeg", "ecg", "emg", "acc"]:
        I[I.feature.str.startswith(mod + "_")].sort_values("gain_mean", ascending=False).to_csv(mdir / f"feature_importance_{mod}.csv", index=False)

    fa_ctx, missed = [], []
    for p in sorted((vdir / "preds").glob(f"{FUSION[0]}__{FUSION[1]}__f*_s0.parquet")):
        f = int(p.stem.split("__")[2][1:].split("_s")[0])
        thr = T[(T.set == FUSION[0]) & (T.model == FUSION[1]) & (T.fold == f) & (T.seed == 0)].thr.iloc[0]
        for (sub, run), d in pd.read_parquet(p).groupby(["sub", "run"]):
            d = d.sort_values("t0"); mask, _ = run_masks(sub, int(run))
            hyp = L.windows_to_events(L.k_of_n((d.prob.values > thr) & ~mask))
            ev = S[(S["sub"] == sub) & (S.run == run)]
            ref = list(zip(ev.onset_s, ev.onset_s + ev.duration_s))
            ov = lambda hs, he, a, b: hs < b + L.CFG["tol_post_s"] and he > a - L.CFG["tol_pre_s"]
            for hs, he in hyp:
                if not any(ov(hs, he, a, b) for a, b in ref):
                    w = d.prob.values[int(hs // STEP): int(np.ceil(he / STEP))]
                    fa_ctx.append(dict(subject=sub, run=int(run), fold=f, start_s=hs, end_s=he, duration_s=he - hs, max_prob=float(w.max()) if len(w) else np.nan, note=""))
            for e, (a, b) in zip(ev.itertuples(), ref):
                if not any(ov(hs, he, a, b) for hs, he in hyp):
                    w = d.prob.values[int(a // STEP): int(np.ceil(b / STEP))]
                    missed.append(dict(subject=sub, run=int(run), fold=f, onset_s=a, duration_s=b - a, seizure_type=e.type, seizure_group=e.group,
                                       max_prob_during=float(w.max()) if len(w) else np.nan, threshold=thr, note=""))
    pd.DataFrame(fa_ctx).to_csv(mdir / "false_alarm_context.csv", index=False)
    pd.DataFrame(missed).to_csv(mdir / "missed_events.csv", index=False)

    F = pd.read_csv(vdir / "scores_by_fold.csv")
    rt = F.groupby(["set", "model", "fold", "seed"]).agg(minutes=("minutes", "first"), peak_rss_gb=("peak_rss_gb", "first")).reset_index()
    json.dump(dict(note="wall-clock minutes and peak working set per training job (inner fit + threshold sweep + refit + held-out "
                        "prediction + training-curve monitor), 3 jobs in parallel on one laptop (24 threads, RTX GPU for xgboost)",
                   per_set_model=rt.groupby(["set", "model"]).agg(minutes_median=("minutes", "median"), minutes_max=("minutes", "max"),
                                                                    peak_rss_gb_max=("peak_rss_gb", "max")).reset_index().to_dict("records")),
              open(mdir / "runtime.json", "w"), indent=1, default=float)
    return A


# ------------------------------------------------------------------ figures/example_events/
def pick_examples(vdir, S, E):
    """Three cases, chosen by a fixed rule (first by seizure_id among those satisfying it, after v01, before looking at plots):
    1. FBTC detected by C (seed 0);  2. focal impaired-awareness (sz_foc_ia*) detected by C;
    3. 'EEG+ECG' case: a seizure detected by B (EEG+ECG) with no EMG or ACC change in Exp R (t_emg and t_acc empty)."""
    D = pd.read_csv(vdir / "detections_by_seizure.csv")
    D0 = D[D.seed == 0]
    hitC = set(D0[(D0.set == FUSION[0]) & (D0.model == "xgb") & D0.hit].seizure_id)
    hitB = set(D0[(D0.set == "B_eeg_ecg") & (D0.model == "xgb") & D0.hit].seizure_id)
    O = pd.read_csv(ROOT / "results/expR/onsets.csv").set_index("seizure_id")
    S = S.assign(seizure_id=[f"{a}_r{int(b):02d}_{int(c)}" for a, b, c in zip(S["sub"], S.run, S.onset_s)])
    S = S[S.seizure_id.isin(set(D0.seizure_id))].sort_values("seizure_id")
    c1 = S[(S.group == "FBTC") & S.seizure_id.isin(hitC)]
    c2 = S[S.type.str.startswith("sz_foc_ia") & S.seizure_id.isin(hitC)]
    quiet = [i for i in S.seizure_id if i in O.index and pd.isna(O.loc[i, "t_emg"]) and pd.isna(O.loc[i, "t_acc"]) and O.loc[i, "has_hr"]]
    c3 = S[S.seizure_id.isin(hitB) & S.seizure_id.isin(quiet)]
    return [("fbtc", c1.iloc[0] if len(c1) else None), ("focal_impaired_awareness", c2.iloc[0] if len(c2) else None),
            ("eeg_ecg_only", c3.iloc[0] if len(c3) else None)]


def read_seg(path, t0, t1, picks):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    fs = raw.info["sfreq"]; a, b = max(0, int(t0 * fs)), min(raw.n_times, int(t1 * fs))
    return raw.get_data(picks=picks, start=a, stop=b) * 1e6, fs, a / fs, raw


def export_examples(vdir, ex, S, E, T):
    from scipy.signal import resample_poly
    out = ex / "figures" / "example_events"; out.mkdir(parents=True, exist_ok=True)
    PRE, POST, PAD = 60, 60, 30
    chosen = []
    for case, e in pick_examples(vdir, S, E):
        if e is None:
            print("  example", case, ": no seizure satisfies the rule"); continue
        sub, run, on = e["sub"], int(e.run), float(e.onset_s)
        ev_row = E[(E.subject == sub) & (E.run == run) & (E.onset_s == on)].iloc[0]
        stem = f"{sub}/ses-01/{{m}}/{sub}_ses-01_task-szMonitoring_run-{run:02d}_{{m}}.edf"
        t_lo, t_hi = on - PRE, on + POST
        n64 = (PRE + POST) * 64
        df = pd.DataFrame({"t_rel_s": np.arange(n64) / 64 - PRE})
        def put(name, x, fs_x, start_abs):
            k0 = int(round((start_abs - t_lo) * fs_x)); col = np.full((PRE + POST) * int(fs_x), np.nan)
            lo, hi = max(0, k0), min(len(col), k0 + len(x))
            if hi > lo: col[lo:hi] = x[lo - k0: hi - k0]
            stride = 64 // int(fs_x)
            full = np.full(n64, np.nan); full[::stride] = col; df[name] = full
        x, fs, s0, raw = read_seg(RAW / stem.format(m="eeg"), t_lo - PAD, t_hi + PAD, [0, 1])
        h = L.harmonise(x, fs, "eeg")
        for c in range(2):
            put(f"eeg_{c + 1}_uv", resample_poly(h[c], 1, 4), 64, s0)
        p = RAW / stem.format(m="ecg")
        if p.exists():
            x, fs, s0, _ = read_seg(p, t_lo - PAD, t_hi + PAD, [0])
            g = L.harmonise(x, fs, "ecg")[0]
            put("ecg_uv", resample_poly(g, 1, 4), 64, s0)
            hr, _ = L.heart_rate_series(g); put("hr_bpm", hr, 1, s0)
        p = RAW / stem.format(m="emg")
        if p.exists():
            x, fs, s0, _ = read_seg(p, t_lo - PAD, t_hi + PAD, [0])
            g = L.harmonise(x, fs, "emg")[0]; n = len(g) // 16
            put("emg_env_uv", np.sqrt((g[: n * 16].reshape(-1, 16) ** 2).mean(1)), 16, s0)   # RMS over 1/16 s
        for s_, m_ in [("A_eeg", "xgb"), ("B_eeg_ecg", "xgb"), FUSION]:
            P = load_preds(vdir, s_, m_); d = P[(P["sub"] == sub) & (P.run == run)].sort_values("t0")
            seeds = [c for c in d if c.startswith("prob_s")]
            put(f"prob_{s_.split('_')[0]}", np.repeat(d[seeds].mean(1).values, STEP), 1, 0.0)
            if (s_, m_) == FUSION:
                fold = int(d.fold.iloc[0]); mask, _ = run_masks(sub, run)
                thr = T[(T.set == s_) & (T.model == m_) & (T.fold == fold) & (T.seed == 0)].thr.iloc[0]
                put("alert_C_s0", np.repeat(L.k_of_n((d.prob_s0.values > thr) & ~mask).astype(float), STEP), 1, 0.0)
        lab = ((df.t_rel_s >= 0) & (df.t_rel_s < e.duration_s)).astype(float)
        lab[np.arange(n64) % 64 != 0] = np.nan                                           # 1 Hz column, like the probs
        df["label"] = lab
        f = out / f"example_{ev_row.event_id}.csv"; df.to_csv(f, index=False, float_format="%.4g")
        chosen.append(dict(case=case, event_id=ev_row.event_id, seizure_type=e.type, group=e.group, duration_s=e.duration_s, file=f.name))
        print(f"  example {case}: {ev_row.event_id} ({e.type})", flush=True)
    return chosen


# ------------------------------------------------------------------ dataset_stats.json, config/, README
def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def export_stats(vdir, ex, version, runs_mode, E, pred_counts, examples):
    R = pd.read_csv(ROOT / "data/catalog/seizeit2_runs.csv")
    dev = R[R.split == "train"]
    used = {(s, int(r)) for s, r in E[["subject", "run"]].drop_duplicates().itertuples(index=False)}
    P = load_preds(vdir, *FUSION)
    used_runs = {(s, int(r)) for s, r in P[["sub", "run"]].drop_duplicates().itertuples(index=False)}
    excl = []
    for r in dev.itertuples():
        if (r.sub, int(r.run)) in used_runs:
            continue
        has_feat = (L.features_dir() / f"{r.sub}_run-{int(r.run):02d}.parquet").exists()
        why = ("seizure-free run, not part of v01 (--runs seizure; added in v02)" if r.n_seizures == 0 else
               "no feature file" if not has_feat else "not in the held-out predictions")
        excl.append(dict(subject=r.sub, run=int(r.run), hours=r.hours, n_seizures=int(r.n_seizures), reason=why))
    X = pd.DataFrame(excl)
    S = pd.read_csv(vdir / "summary.csv")
    st = dict(version=version, stage="dev", label="development (cross-validation), offline pipeline",
              runs=("seizure-containing runs only" if runs_mode == "seizure" else "all runs"),
              lib_version=L.LIB_VERSION, git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "src")),
              exported=pd.Timestamp.now().isoformat(timespec="seconds"),
              patients=int(P["sub"].nunique()), runs_used=len(used_runs), hours_used=float(R.set_index(["sub", "run"]).loc[sorted(used_runs)].hours.sum()),
              hours_scored_c_xgb=float(S[(S.set == FUSION[0]) & (S.model == FUSION[1])].hours.iloc[0]),
              seizures=int(len(E)), seizures_by_group=E.seizure_group.value_counts().to_dict(),
              seizures_by_vigilance=E.vigilance.fillna("unknown").value_counts().to_dict(),
              folds={k: len(v) for k, v in L.load_splits()["dev_folds"].items()},
              excluded_runs=len(X), excluded_hours=float(X.hours.sum()) if len(X) else 0.0,
              excluded_by_reason=X.groupby("reason").agg(runs=("run", "size"), hours=("hours", "sum"), seizures=("n_seizures", "sum")).reset_index().to_dict("records") if len(X) else [],
              masked_windows="impd (impedance checks) and flat EEG: can never alarm, removed from the false-alarm hours",
              windows=dict(win_s=L.CFG["win_s"], hop_s=STEP, post_processing=f"{L.CFG['post_k']}-of-{L.CFG['post_n']} windows",
                           merge_s=L.CFG["merge_s"], split_s=L.CFG["split_s"], tol_pre_s=L.CFG["tol_pre_s"], tol_post_s=L.CFG["tol_post_s"]),
              predictions_val=pred_counts, example_events=examples)
    json.dump(L.json_clean(st), open(ex / "dataset_stats.json", "w"), indent=1)
    X.to_csv(ex / "metrics" / "excluded_runs.csv", index=False)
    return st


def export_config(ex):
    c = ex / "config"; c.mkdir(exist_ok=True)
    shutil.copy(ROOT / "data/catalog/splits.json", c / "splits.json")
    shutil.copy(ROOT / "data/catalog/splits.sha256", c / "splits.sha256")
    (c / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout, encoding="utf-8")
    (c / "git.txt").write_text(f"HEAD {git('rev-parse', 'HEAD')}\n\n{git('log', '--oneline', '-15')}\n\nsrc changes not committed:\n{git('status', '--porcelain', '--', 'src')}\n", encoding="utf-8")
    (c / "ictra_cfg.json").write_text(json.dumps(dict(LIB_VERSION=L.LIB_VERSION, CFG=L.CFG, FEATURE_SETS=L.FEATURE_SETS), indent=1, default=str), encoding="utf-8")


README = """# ICTRA results export: {version} ({stage})

> **DEVELOPMENT ONLY. Cross-validation on SeizeIT2 dev patients sub-001..096, offline pipeline.**
> These are out-of-fold predictions: each patient is scored by a model that never saw that patient. They are not the
> locked test (sub-097..125, after pre-registration, ~Nov 14) and not an external test. Label every chart made from this
> zip "development (cross-validation), offline pipeline". {runs_banner}

Exported {exported} from git {commit} (`{lib}`), by `python src/95_export_results.py --stage {stage} --version {version}`.
Spec: `docs/DELIVERABLES_SPEC.md`. Every number in `metrics/` is re-derived from `predictions_val/` by
`python src/09_independent_check.py --zip <this zip>` (result in `INDEPENDENT_CHECK.md` inside the zip).

## Contents

| Path | What |
|---|---|
| `predictions_val/<model>/sub-XXX.csv.gz` | 1 row per second per run. Columns: `subject, session, run, t_sec, fold, prob, prob_s<k>, alert_s<k>, label, missing`. Models: {models}. |
| `predictions_val/events_val.csv` | {n_events} annotated seizures of the exported runs (`event_id = sub-XXX_ses-01_run-NN_eNN`, BIDS code + ILAE 2017 text, vigilance, modalities available) |
| `metrics/ablation.csv` | one row per (set, model) incl. the single-input rules and `fusion_no_kofn` (C xgb, threshold only); mean over 3 seeds, patient-bootstrap 95% CI (seed 0) |
| `metrics/thresholds.csv` | the threshold of every (set, model, fold, seed), tuned on an inner 80/20 patient split of the training folds |
| `metrics/operating_curve.csv`, `operating_points.csv` | 150-threshold sweep per (set, model, seed) over the pooled held-out folds, `sensitivity_pct, fa_per_24h, latency_median_s`. The threshold is swept on held-out data, so this is an operating curve, not a deployable threshold. |
| `metrics/per_type.csv`, `per_patient.csv`, `per_montage.csv` | sensitivity by seizure group; counts per patient; by EEG montage |
| `metrics/latency.csv`, `latency_by_group.csv` | one row per (seizure, model, seed): hit, `latency_s` = first overlapping alarm start − annotated onset (can be negative, down to −30 s); `latency_from_bte_change_s` = the same minus the first behind-the-ear EEG change found by Exp R (`bte_eeg_change_s`) |
| `metrics/training_curves.csv` | per (set, model, fold, seed): `iteration, val_logloss, val_auprc` on the held-out fold (≤ 1M windows, natural class ratio, display only), `val_event_f1` at the last iteration only |
| `metrics/feature_importance_<eeg/ecg/emg/acc>.csv` | gain importance of C (xgb), mean and SD over 5 folds × 3 seeds, share of total gain |
| `metrics/false_alarm_context.csv`, `missed_events.csv` | C xgb seed 0: every false alarm and every missed seizure. `note` is for the human reviewer (empty). |
| `metrics/runtime.json`, `excluded_runs.csv` | job time and peak RAM; dev runs not in this export and why |
| `dataset_stats.json` | patients, runs, hours, seizures by group/vigilance, exclusions, LIB_VERSION, git commit, scoring constants |
| `figures/example_events/example_<event_id>.csv` | 3 cases, −60..+60 s around the onset. 64 Hz rows: `eeg_1_uv, eeg_2_uv, ecg_uv` (harmonised, then decimated 256→64 Hz for display); `emg_env_uv` on every 4th row (16 Hz RMS envelope); `hr_bpm, prob_A, prob_B, prob_C, alert_C_s0, label` on every 64th row (1 Hz). Empty = not sampled at that row / not recorded. |
| `config/` | splits.json + hash, pip freeze, git log, ictra_lib CFG |

### Column conventions
- `prob`: mean over the seeds of the window probability, held for the 2 s of the window hop.
  - `prob_s<k>` / `alert_s<k>`: seed k's probability, and its alarm state after the fold's threshold, the impd/flat-EEG
    mask and 3-of-5 windows, exactly as scored.
  - **Rules** (`rule_*` folders) have one seed. Their `prob` is the rule's raw score, not a probability: {rule_units}.
    Empty = feature unavailable.
- `label` = 1 if the second starts inside an annotated seizure.
- `missing` = 1 in impd (impedance check) or flat-EEG windows. They can never alarm and are not counted in the
  false-alarm hours.
- Probabilities are 0–1 and percentages 0–100 (`_pct`). Missing values are empty.
- FA is reported per 24 h only.

## Example events (chosen by a fixed rule, not by looking at plots)
{examples}

## Deviations from the checklist (stated, not silently "complied with")

1. **Preprocessing is not matched to the firmware rates.**
   - The checklist (§D4) asks for EEG 256 Hz / 0.5–29.5 Hz, ECG 125 Hz / 0.5–44.5 Hz and EMG 500 Hz / 60–100 Hz.
   - The pipeline resamples every signal to 256 Hz and uses EEG 1–25 Hz, ECG 0.5–40 Hz, EMG 20–100 Hz, a 50 Hz notch on EMG,
     and ACC at 25 Hz (`ictra_lib.harmonise`).
   - SeizeIT2 was recorded in Europe, hence the 50 Hz notch; the North-American firmware needs 60 Hz. It is the firmware
     that should change (TASKS stage 4).
   - Until then, the statement is "same code path, firmware update pending", not "the evaluated model is the model on the
     device".
2. **Normalisation.** Each `*_rel` feature is the feature minus its own causal rolling 5-min median (shifted by one window;
   `ictra_lib.add_relative_eeg`, `emg_features`, `acc_features`). The ECG HR rise uses a causal 5-min median HR baseline.
   - The absolute features are kept unscaled.
   - There is **no per-recording robust scaling** and no z-score on the first 5 min. (The spec's wording "per-recording
     robust scaling" does not describe the code.)
   - Same intent: no cross-recording statistics.
3. **Dropout repair, ACC calibration and zero-phase filters are offline (non-causal).**
   - `repair_dropouts` looks 1–2 samples (≤ 1 s) ahead.
   - `acc_to_g` divides by the whole recording's median |a|.
   - `bandpass` / `notch` use `sosfiltfilt` / `filtfilt`.
   - See `docs/EVALUATION_PROTOCOL.md` amendment 2026-10-17. A causal pipeline is candidate v11.
   - **All exported numbers are offline evaluations.**
4. **The window hop is 2 s, exported at 1 s** by holding each window's value for both seconds (window i covers 2i..2i+4 s).
   Alarms come from 3-of-5 windows, then events are merged (gaps < 90 s) and split (> 300 s).
5. **No EEG CNN, so no Grad-CAM saliency.** SHAP time-courses of the tree model (`figures/explain/`) are planned. **This
   zip does not contain them yet.**
6. **Test predictions (sub-097..125) exist only after pre-registration** (Task 10/11). This zip ships `predictions_val/`
   (dev out-of-fold) only. There is no `predictions/` folder and no `events_test.csv`.
7. **Cross-dataset:** `metrics/cross_dataset.csv` is not in this zip. The external tests run once, after the locked test
   (Task 12). The planned sets are Siena, Billeci, szdb and CHB-MIT (+ Seer/OSDB if access arrives), all through the
   T7–P7 / T8–P8 behind-the-ear proxy.
8. **Single-modality thresholds.**
   - In this export, every model and rule threshold is tuned on the inner split to maximise event F1, the same rule as
     fusion.
   - The ≤ 1 FA/24h operating point is in `operating_points.csv` (`sens_at_fa_le_1`).
   - The single-signal learned models (ecg_only, emg_only, acc_only, emg_acc) are not in this zip; they run after v01.
9. **FA counting.** {fa_note}
10. **Early fusion, not late fusion.** v01 is one tree model over the concatenated features of each set (A: EEG; B: + ECG;
    C: + EMG + ACC). There are no per-modality sub-models, so `predictions_val/fusion/` with `prob_eeg, prob_ecg,
    prob_emg` does not exist. The closest equivalents are the A / B / C folders and the rules.
11. **No saved model files.** v01 keeps predictions, thresholds, curves and importances, not the fitted boosters, so
    `config/` has no `models/*.meta.json`. The frozen final model will be saved (Task 10).
12. **Seeds.** GPU xgboost is not bit-reproducible. The reported numbers are the mean over 3 seeds (`ablation.csv`); the
    CIs bootstrap patients with seed 0.
"""


def write_readme(ex, version, stage, runs_mode, st, examples, A):
    ex_lines = "\n".join(f"- **{c['case']}**: `{c['event_id']}` ({c['seizure_type']}, {c['group']}, {c['duration_s']:.0f} s)" for c in examples) or "- none"
    fa = ("v01 uses **seizure-containing runs only** (`--runs seizure`): the seizure-free runs are added in v02. So FA/24h "
          "here is measured on a harder, smaller subset of hours, not the full monitoring time.") if runs_mode == "seizure" else \
         "FA counting covers all exported hours, including the seizure-free runs."
    txt = README.format(version=version, stage=stage, exported=st["exported"], commit=st["git_commit"][:10], lib=L.LIB_VERSION,
                        runs_banner=("**v01: seizure-containing runs only; FA/24h is not the full-monitoring figure.**" if runs_mode == "seizure" else ""),
                        models=", ".join(f"`{folder_name(s, m)}`" for s, m in EXPORT_MODELS), n_events=st["seizures"],
                        rule_units="; ".join(f"`{k}` = {v}" for k, v in RULE_UNITS.items()), examples=ex_lines, fa_note=fa)
    (ex / "README.md").write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="dev", choices=["dev", "locked_test", "external"])
    ap.add_argument("--version", required=True)
    ap.add_argument("--skip-predictions", action="store_true", help="re-export metrics/README only (predictions_val/ kept)")
    a = ap.parse_args()
    guard(a.stage)
    L.mode_for_version(a.version)                  # v11: causal features and alarm timestamps
    vdir = ROOT / "results/dev" / a.version
    if not (vdir / "summary.csv").exists():
        sys.exit(f"{vdir}/summary.csv missing: run 04_train_ablation.py first")
    tag = a.version.split("_")[0]
    ex = ROOT / "results/export" / f"ictra_results_{time.strftime('%Y%m%d')}_{a.stage}_{a.version}"
    ex.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(ROOT / "data/catalog/seizeit2_seizures.csv")
    T = thresholds(vdir)
    runs_mode = json.loads((ROOT / "results/lab/data" / f"dev_{a.version}.json").read_text(encoding="utf-8")).get("runs") if \
        (ROOT / "results/lab/data" / f"dev_{a.version}.json").exists() else None
    runs_mode = runs_mode or ("seizure" if a.version.startswith("v01") else "all")
    t = time.time()
    P = load_preds(vdir, *FUSION)
    version_runs = {(s, int(r)) for s, r in P[["sub", "run"]].drop_duplicates().itertuples(index=False)}
    (ex / "predictions_val").mkdir(exist_ok=True)
    E = export_events(ex, S, version_runs); print(f"events_val.csv: {len(E)} seizures", flush=True)
    counts = export_predictions(vdir, ex, S, T) if not a.skip_predictions else json.load(open(ex / "dataset_stats.json")).get("predictions_val", {})
    A = export_metrics(vdir, ex, S, T); print("metrics/ done", flush=True)
    examples = export_examples(vdir, ex, S, E, T)
    st = export_stats(vdir, ex, a.version, runs_mode, E, counts, examples)
    export_config(ex)
    write_readme(ex, a.version, a.stage, runs_mode, st, examples, A)
    z = ex.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:      # the .csv.gz inside are already compressed
        for p in sorted(ex.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(ex.parent))
    print(f"{z} ({z.stat().st_size / 1e6:.0f} MB) in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()

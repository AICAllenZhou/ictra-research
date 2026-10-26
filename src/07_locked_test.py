"""Task 11: the locked internal test, run ONCE, exactly as pre-registered (docs/PREREGISTRATION.md).

    python src/07_locked_test.py                 # THE locked test (sub-097..125); refuses if results/locked_test/RESULTS.md exists
    python src/07_locked_test.py --dry-run-dev   # format/run check on sub-001..096 only -> results/locked_test_dryrun/

The frozen models were trained on all 96 development patients, so dry-run numbers are IN-SAMPLE: they only show that
the script runs end to end and writes every table; they are not results of any kind.

What it does (section names refer to docs/PREREGISTRATION.md):
  * Guards: RESULTS.md must not exist (locked mode); docs/PREREGISTRATION.md must match docs/PREREGISTRATION.sha256;
    every frozen model, metadata and code file must match the SHA-256 recorded in the pre-registration; locked mode reads
    only sub-097..125, dry-run only sub-001..096; nothing is trained, no threshold is chosen.
  * Models: both frozen lineages (v02g offline: data/features; v11 causal: data/features_causal). A, B, C = mean
    probability of the three frozen seed models; the four rules = their frozen feature; frozen tau and 3-of-5 from the
    metadata. Alarms, merging, splitting and masks through ictra_lib in each lineage's mode.
  * Metrics: per patient, then averaged (primary); counts summed over patients (secondary); patient bootstrap 1,000
    resamples, paired for every comparison; operating curve swept on this set (a curve, not a deployable number).
  * Hypotheses H1-H4 (text in the pre-registration) decided by its "Analysis plan", in each lineage; H3a with the
    Exp R functions of src/02_relationship_study.py (offline processing, as in Exp R).
  * Every table with all seizures (primary: the pre-registered set) and without the 15 seizures lacking ECG/EMG.
"""
import argparse, hashlib, importlib.util, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

_s = importlib.util.spec_from_file_location("trainer", Path(__file__).resolve().parent / "04_train_ablation.py")
T = importlib.util.module_from_spec(_s); _s.loader.exec_module(T)
_s = importlib.util.spec_from_file_location("expr", Path(__file__).resolve().parent / "02_relationship_study.py")
R2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(R2)

PREREG_SHA = "bc5048c88ae3524348215e4c62801ec66fedbe5c4e4ac616bcbb3c3944fedca6"
LINEAGES = ("v02g", "v11")
LEARNED = ("A_eeg/xgb", "B_eeg_ecg/xgb", "C_eeg_ecg_emg_acc/xgb")
SHORT = {"A_eeg/xgb": "A", "B_eeg_ecg/xgb": "B", "C_eeg_ecg_emg_acc/xgb": "C"}
GROUPS = ["FBTC", "focal_motor", "focal_nonmotor", "focal_unknown_motor"]
DUR_BINS = [(0, 15, "< 15 s"), (15, 60, "15-60 s"), (60, 1e9, "> 60 s")]
MONTAGES = ["left", "right", "generalized"]
N_BOOT, BOOT_SEED, N_THR = 1000, 0, 150
FA_BUDGETS = [1, 2, 5, 10]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ guards
def integrity(dry):
    pre = L.ROOT / "docs/PREREGISTRATION.md"
    rec = (L.ROOT / "docs/PREREGISTRATION.sha256").read_text().split()[0]
    if not (sha(pre) == rec == PREREG_SHA):
        sys.exit("refusing: docs/PREREGISTRATION.md does not match its recorded SHA-256")
    text = pre.read_text(encoding="utf-8")
    bad = []
    for path, h in re.findall(r"`((?:models|src)/[^`]+)` \| `([0-9a-f]{64})`", text):
        if sha(L.ROOT / path) != h:
            bad.append(path)
    if bad:
        sys.exit(f"refusing: frozen files changed since the pre-registration: {bad}")
    n = len(re.findall(r"`((?:models|src)/[^`]+)` \| `([0-9a-f]{64})`", text))
    return f"pre-registration SHA-256 {PREREG_SHA} OK; {n} frozen model/metadata/code files match their recorded SHA-256"


def subjects(dry):
    sp = L.load_splits()
    dev = sorted(s for v in sp["dev_folds"].values() for s in v)
    locked = sorted(sp["locked_test"])
    if dry:
        return dev, locked
    return locked, dev


# ------------------------------------------------------------------ scoring
def load_models(tag):
    import xgboost as xgb
    meta = json.loads((L.ROOT / f"models/ictra_{tag}.meta.json").read_text(encoding="utf-8"))
    models = {}
    for key in LEARNED:
        c = meta["cells"][key]; ms = []
        for f in c["files"]:
            m = xgb.XGBClassifier(); m.load_model(str(L.ROOT / "models" / f)); m.get_booster().set_param({"device": "cpu", "nthread": 4})
            ms.append(m)
        models[key] = (ms, c["columns"], c["thr"], c["k"], c["n"])
    for r, c in meta["rules"].items():
        models[r] = (T.Rule(c["feature"]), [c["feature"]], c["thr"], c["k"], c["n"])
    return meta, models


def score_runs(tag, subs, S):
    """Per run: window scores of every frozen model, mask, hours, montage. Reads only the given subjects' feature files."""
    meta, models = load_models(tag)
    L.set_mode(meta["causal"])
    files = sorted(p for p in L.features_dir().glob("*.parquet") if p.name.split("_run")[0] in set(subs))
    assert files and all(p.name.split("_run")[0] in set(subs) for p in files)
    runs = []
    for f in files:
        d = pd.read_parquet(f)
        sub, run = d["sub"].iloc[0], int(d["run"].iloc[0])
        mask = d.impd.values | d.get("flat_eeg", pd.Series(False, index=d.index)).fillna(False).astype(bool).values
        sc = {}
        for key, (m, cols, thr, k, n) in models.items():
            X = d.reindex(columns=cols)
            sc[key] = (np.mean([mm.predict_proba(X)[:, 1] for mm in m], axis=0) if isinstance(m, list) else m.predict_proba(X)[:, 1]).astype(np.float64)
        ev = S[(S["sub"] == sub) & (S.run == run)]
        runs.append(dict(sub=sub, run=run, montage=str(d["montage"].iloc[0]), nw=len(d), mask=mask, scores=sc,
                         hours=(len(d) - int(mask.sum())) * L.CFG["step_s"] / 3600,
                         sz=ev.reset_index(drop=True)))
    return meta, models, runs


def alarms(p, mask, thr, k, n):
    pos = (p > thr) & ~mask
    if pos.sum() < k:
        return []
    return L.windows_to_events(L.k_of_n(pos, k, n))


def score_at(runs, key, thr, k, n):
    """Frozen-tau scoring: one row per seizure (hit, delay) and one per run (alarms, true alarms, false alarms, hours)."""
    tol_pre, tol_post = L.CFG["tol_pre_s"], L.CFG["tol_post_s"]
    sz_rows, run_rows = [], []
    for r in runs:
        hyp = alarms(r["scores"][key], r["mask"], thr, k, n)
        a = r["sz"].onset_s.values; b = a + r["sz"].duration_s.values
        if hyp:
            hs, he = np.array(hyp).T
            ov = (hs[None, :] < (b + tol_post)[:, None]) & (he[None, :] > (a - tol_pre)[:, None])
        else:
            hs = np.zeros(0); ov = np.zeros((len(a), 0), bool)
        hit = ov.any(1) if len(hyp) else np.zeros(len(a), bool)
        tp_hyp = int(ov.any(0).sum()) if len(a) else 0
        run_rows.append(dict(sub=r["sub"], run=r["run"], montage=r["montage"], hours=r["hours"], n_hyp=len(hyp), tp_hyp=tp_hyp,
                             fp=len(hyp) - tp_hyp))
        for i, s in r["sz"].iterrows():
            d = float(np.min(np.where(ov[i], hs, np.inf)) - a[i]) if hit[i] else np.nan
            sz_rows.append(dict(sub=r["sub"], run=r["run"], montage=r["montage"], onset_s=a[i], duration_s=s.duration_s, group=s.group,
                                has_ecg_emg=bool(s.has_ecg_emg), hit=bool(hit[i]), delay_s=d))
    return pd.DataFrame(sz_rows), pd.DataFrame(run_rows)


def sweep(runs, key, k, n):
    """Operating curve on this set: tau grid as in 05_operating_points (quantiles of all unmasked scores). Returns the
    grid, per-seizure hits [seizure x tau] and per-run false alarms [run x tau]."""
    allp = np.concatenate([r["scores"][key][~r["mask"]] for r in runs])
    thrs = np.unique(np.quantile(allp, 1 - np.logspace(-5.5, np.log10(0.5), N_THR)))
    tol_pre, tol_post = L.CFG["tol_pre_s"], L.CFG["tol_post_s"]
    hits, fps = [], []
    for r in runs:
        a = r["sz"].onset_s.values; b = a + r["sz"].duration_s.values
        H = np.zeros((len(a), len(thrs)), bool); FP = np.zeros(len(thrs), int)
        for j, thr in enumerate(thrs):
            hyp = alarms(r["scores"][key], r["mask"], thr, k, n)
            if not hyp:
                continue
            hs, he = np.array(hyp).T
            ov = (hs[None, :] < (b + tol_post)[:, None]) & (he[None, :] > (a - tol_pre)[:, None])
            H[:, j] = ov.any(1) if len(a) else H[:, j]
            FP[j] = len(hyp) - (int(ov.any(0).sum()) if len(a) else 0)
        hits.append(H); fps.append(FP)
    return thrs, np.vstack(hits) if hits else np.zeros((0, len(thrs)), bool), np.vstack(fps)


# ------------------------------------------------------------------ per-patient metrics
class Patients:
    """Per-patient values for one model at one tau, with any seizure subset; aggregate = mean over eligible patients
    (delay: median of per-patient medians); bootstrap via index arrays over the patient list."""

    def __init__(self, pats, sz, rn, sel=None, runsel=None):
        self.pats = pats
        z = sz if sel is None else sz[sel]
        r = rn if runsel is None else rn[runsel]
        g = z.groupby("sub"); gr = r.groupby("sub")
        self.n = np.array([g.size().get(p, 0) for p in pats], float)
        self.det = np.array([g.hit.sum().get(p, 0) for p in pats], float)
        self.hours = np.array([gr.hours.sum().get(p, 0.0) for p in pats])
        self.fp = np.array([gr.fp.sum().get(p, 0) for p in pats], float)
        self.n_hyp = np.array([gr.n_hyp.sum().get(p, 0) for p in pats], float)
        self.tp_hyp = np.array([gr.tp_hyp.sum().get(p, 0) for p in pats], float)
        dl = z[z.hit].groupby("sub").delay_s.median()
        self.delay = np.array([dl.get(p, np.nan) for p in pats])
        self.delays_all = z.loc[z.hit, "delay_s"].values
        with np.errstate(invalid="ignore", divide="ignore"):
            self.sens = np.where(self.n > 0, self.det / self.n, np.nan)
            self.fa = np.where(self.hours > 0, self.fp * 24 / self.hours, np.nan)
            self.prec = np.where(self.n_hyp > 0, self.tp_hyp / self.n_hyp, np.nan)
            p0 = np.nan_to_num(self.prec)
            f1 = np.where((p0 + np.nan_to_num(self.sens)) > 0, 2 * p0 * np.nan_to_num(self.sens) / (p0 + np.nan_to_num(self.sens)), 0.0)
            self.f1 = np.where(self.n > 0, f1, np.nan)

    def agg(self, idx=None):
        i = slice(None) if idx is None else idx
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return dict(sensitivity=np.nanmean(self.sens[i]), fa_per_24h=np.nanmean(self.fa[i]), precision=np.nanmean(self.prec[i]),
                            f1=np.nanmean(self.f1[i]), delay_s=np.nanmedian(self.delay[i]))

    def pooled(self):
        n, d, hy, tp = self.n.sum(), self.det.sum(), self.n_hyp.sum(), self.tp_hyp.sum()
        s = d / n if n else np.nan; pr = tp / hy if hy else np.nan
        return dict(sensitivity=s, fa_per_24h=self.fp.sum() * 24 / self.hours.sum() if self.hours.sum() else np.nan, precision=pr,
                    f1=2 * s * pr / (s + pr) if (n and hy and (s + pr) > 0) else (0.0 if n else np.nan),
                    delay_s=float(np.median(self.delays_all)) if len(self.delays_all) else np.nan, seizures=int(n), detected=int(d))


def boot_idx(n_pat):
    return np.random.default_rng(BOOT_SEED).integers(0, n_pat, size=(N_BOOT, n_pat))


def ci(vals):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


# ------------------------------------------------------------------ analysis
def curve_per_patient(pats, sz_meta, run_meta, H, FP, sel):
    """Per-patient sensitivity [patient x tau] on the selected seizures and FA/24h [patient x tau]."""
    pi = {p: i for i, p in enumerate(pats)}
    n_t = H.shape[1]
    det = np.zeros((len(pats), n_t)); n = np.zeros(len(pats))
    for i in np.flatnonzero(sel):
        det[pi[sz_meta.at[i, "sub"]]] += H[i]; n[pi[sz_meta.at[i, "sub"]]] += 1
    fp = np.zeros((len(pats), n_t)); hrs = np.zeros(len(pats))
    for j, row in run_meta.iterrows():
        fp[pi[row["sub"]]] += FP[j]; hrs[pi[row["sub"]]] += row["hours"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.where(n[:, None] > 0, det / np.maximum(n, 1)[:, None], np.nan)
        fa = np.where(hrs[:, None] > 0, fp * 24 / np.maximum(hrs, 1e-9)[:, None], np.nan)
    return sens, fa


def nanmean0(M, idx=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(M if idx is None else M[idx], axis=0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")            # labels contain U+2212 etc.; the Windows console default (cp1252) cannot print them
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-dev", action="store_true", help="sub-001..096 only; output results/locked_test_dryrun/ (in-sample)")
    a = ap.parse_args()
    dry = a.dry_run_dev
    out = L.ROOT / ("results/locked_test_dryrun" if dry else "results/locked_test")
    if not dry and (out / "RESULTS.md").exists():
        sys.exit("refusing: results/locked_test/RESULTS.md exists - the locked test is run once")
    check = integrity(dry)
    subs, other = subjects(dry)
    for d in ("data/features", "data/features_causal"):
        present = {p.name.split("_run")[0] for p in (L.ROOT / d).glob("*.parquet")}
        if not dry and not set(subs) <= present:
            sys.exit(f"refusing: {d} lacks feature files of {sorted(set(subs) - present)}")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{'DRY RUN on development patients (in-sample)' if dry else 'LOCKED TEST'}: {len(subs)} patients; {check}", flush=True)
    S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")
    S = S[S["sub"].isin(subs)].copy()
    S["duration_bin"] = pd.cut(S.duration_s, [b[0] for b in DUR_BINS] + [DUR_BINS[-1][1]], right=False, labels=[b[2] for b in DUR_BINS]).astype(str)

    md = [f"# {'DRY RUN (development patients, in-sample — NOT a result)' if dry else 'Locked internal test — results'}", ""]
    if dry:
        md += ["**Dry run of `src/07_locked_test.py` on sub-001..096.** The frozen models were trained on these patients, so every "
               "number here is in-sample. This file only shows that the script runs end to end. It is not a result.", ""]
    else:
        md += ["**These are the results of the locked internal test (sub-097..125), run once as pre-registered in "
               "`docs/PREREGISTRATION.md`. From now on this set is development data and cannot be used for an independent test.**", ""]
    md += [f"Run {t_start}. {check}.", ""]

    all_tables, hyp_rows = [], []
    per_lineage = {}
    for tag in LINEAGES:
        meta, models, runs = score_runs(tag, subs, S)
        pats = sorted({r["sub"] for r in runs})
        idx = boot_idx(len(pats))
        res = {}
        for key, (m, cols, thr, k, n) in models.items():
            sz, rn = score_at(runs, key, thr, k, n)
            thrs, H, FP = sweep(runs, key, k, n)
            sz_meta = pd.concat([r["sz"].assign(sub=r["sub"], run=r["run"], montage=r["montage"]) for r in runs], ignore_index=True)
            sz_meta["duration_bin"] = pd.cut(sz_meta.duration_s, [b[0] for b in DUR_BINS] + [DUR_BINS[-1][1]], right=False,
                                             labels=[b[2] for b in DUR_BINS]).astype(str)
            run_meta = pd.DataFrame([dict(sub=r["sub"], run=r["run"], montage=r["montage"], hours=r["hours"]) for r in runs])
            sz["duration_bin"] = sz_meta.duration_bin.values
            res[key] = dict(sz=sz, rn=rn, thrs=thrs, H=H, FP=FP, sz_meta=sz_meta, run_meta=run_meta, thr=thr)
            print(f"  {tag} {key}: tau {thr:.6g}, {int(sz.hit.sum())}/{len(sz)} seizures hit, {int(rn.fp.sum())} false alarms", flush=True)
        per_lineage[tag] = (meta, res, pats, idx)
        L.set_mode(False)

    # ---------------- tables per lineage and variant
    def variant_sel(z, v):
        return np.ones(len(z), bool) if v == "all" else z.has_ecg_emg.values

    for v, vname in (("all", "all seizures (pre-registered set)"), ("ecg_emg", "without the seizures lacking ECG/EMG")):
        for tag in LINEAGES:
            meta, res, pats, idx = per_lineage[tag]
            rows = []
            for key, R in res.items():
                sel = variant_sel(R["sz"], v)
                P = Patients(pats, R["sz"], R["rn"], sel)
                pt = P.agg(); bs = [P.agg(i) for i in idx]; pl = P.pooled()
                row = dict(variant=v, lineage=tag, model=SHORT.get(key, key), tau=R["thr"], patients=len(pats),
                           seizures=int(P.n.sum()), detected=int(P.det.sum()))
                for mtr in ("sensitivity", "fa_per_24h", "precision", "f1", "delay_s"):
                    row[mtr] = pt[mtr]; row[mtr + "_lo"], row[mtr + "_hi"] = ci([b[mtr] for b in bs])
                    row["pooled_" + mtr] = pl[mtr]
                rows.append(row)
                if v == "all":
                    pp = pd.DataFrame(dict(sub=pats, seizures=P.n, detected=P.det, hours=P.hours, false_alarms=P.fp, alarms=P.n_hyp,
                                           sensitivity=P.sens, fa_per_24h=P.fa, precision=P.prec, f1=P.f1, median_delay_s=P.delay))
                    pp.insert(1, "model", SHORT.get(key, key))
                    all_tables.append((f"per_patient_{tag}.csv", pp))
            all_tables.append((f"overall_{v}.csv", pd.DataFrame(rows)))
            # subgroups at frozen tau: seizure group, duration (sensitivity, delay); montage (sensitivity, FA, F1)
            sub_rows = []
            for key, R in res.items():
                for kind, col, vals in (("group", "group", GROUPS), ("duration", "duration_bin", [b[2] for b in DUR_BINS]),
                                        ("montage", "montage", MONTAGES)):
                    for val in vals:
                        sel = variant_sel(R["sz"], v) & (R["sz"][col].values == val)
                        runsel = (R["rn"].montage.values == val) if kind == "montage" else None
                        P = Patients(pats, R["sz"], R["rn"], sel, runsel)
                        if P.n.sum() == 0 and kind != "montage":
                            continue
                        pt = P.agg(); bs = [P.agg(i) for i in idx]
                        r = dict(variant=v, lineage=tag, model=SHORT.get(key, key), kind=kind, subgroup=val, seizures=int(P.n.sum()),
                                 patients=int((P.n > 0).sum()), sensitivity=pt["sensitivity"], delay_s=pt["delay_s"],
                                 pooled_sensitivity=P.pooled()["sensitivity"])
                        r["sensitivity_lo"], r["sensitivity_hi"] = ci([b["sensitivity"] for b in bs])
                        if kind == "montage":
                            r["fa_per_24h"], r["f1"] = pt["fa_per_24h"], pt["f1"]
                        sub_rows.append(r)
            all_tables.append((f"subgroups_{v}.csv", pd.DataFrame(sub_rows)))
            # operating curve points: per-patient-average sensitivity at per-patient-average FA/24h <= budget
            cur_rows = []
            for key, R in res.items():
                sel_all = variant_sel(R["sz_meta"], v)
                for gname, gsel in [("all", sel_all)] + [(g, sel_all & (R["sz_meta"].group.values == g)) for g in GROUPS]:
                    if gsel.sum() == 0:
                        continue
                    sens, fa = curve_per_patient(pats, R["sz_meta"], R["run_meta"], R["H"], R["FP"], gsel)
                    ms, mf = nanmean0(sens), nanmean0(fa)
                    for bdg in FA_BUDGETS:
                        ok = mf <= bdg
                        cur_rows.append(dict(variant=v, lineage=tag, model=SHORT.get(key, key), group=gname, fa_budget=bdg,
                                             sensitivity=float(np.nanmax(ms[ok])) if ok.any() else np.nan))
            all_tables.append((f"operating_curve_points_{v}.csv", pd.DataFrame(cur_rows)))

            # ---------------- hypotheses (this lineage, this variant)
            A, B, C = res["A_eeg/xgb"], res["B_eeg_ecg/xgb"], res["C_eeg_ecg_emg_acc/xgb"]
            # H1
            fm = lambda R: variant_sel(R["sz"], v) & np.isin(R["sz"].group.values, ["FBTC", "focal_motor"])
            PA = Patients(pats, A["sz"], A["rn"], fm(A))
            selC = variant_sel(C["sz_meta"], v) & np.isin(C["sz_meta"].group.values, ["FBTC", "focal_motor"])
            sensC, faC = curve_per_patient(pats, C["sz_meta"], C["run_meta"], C["H"], C["FP"], selC)

            def h1(ix):
                tgt = PA.agg(ix)["sensitivity"]; faA = PA.agg(ix)["fa_per_24h"]
                ms, mf = nanmean0(sensC, ix), nanmean0(faC, ix)
                ok = ms >= tgt
                if not ok.any() or not np.isfinite(tgt):
                    return np.nan, tgt, np.nan, faA
                j = np.flatnonzero(ok)[np.argmin(mf[ok])]
                return mf[j] - faA, tgt, mf[j], faA
            d0, tgt0, faCm, faA0 = h1(None)
            bsd = np.array([h1(i)[0] for i in idx])
            p_h1 = float(np.mean(np.nan_to_num(bsd, nan=np.inf) < 0))
            hyp_rows.append(dict(variant=v, lineage=tag, hypothesis="H1", decision="held" if (np.isfinite(d0) and d0 < 0 and p_h1 >= 0.90) else "not held",
                                 estimate=f"ΔFA/24h (C at matched sensitivity − A at frozen τ) = {d0:+.2f} (C {faCm:.2f} vs A {faA0:.2f}; target sensitivity {tgt0:.3f})",
                                 ci="[{:+.2f}, {:+.2f}]".format(*ci(bsd)), p=f"P(ΔFA < 0) = {p_h1:.3f}"))
            # H2
            fnm = lambda R: variant_sel(R["sz"], v) & (R["sz"].group.values == "focal_nonmotor")
            PA2, PB2 = Patients(pats, A["sz"], A["rn"], fnm(A)), Patients(pats, B["sz"], B["rn"], fnm(B))
            d2 = PB2.agg()["sensitivity"] - PA2.agg()["sensitivity"]
            bs2 = np.array([PB2.agg(i)["sensitivity"] - PA2.agg(i)["sensitivity"] for i in idx])
            p2 = float(np.mean(bs2 > 0))
            # secondary: matched per-patient-average FA/24h = 1
            def at_fa1(R, ix):
                s_, f_ = curve_per_patient(pats, R["sz_meta"], R["run_meta"], R["H"], R["FP"],
                                           variant_sel(R["sz_meta"], v) & (R["sz_meta"].group.values == "focal_nonmotor"))
                ms, mf = nanmean0(s_, ix), nanmean0(f_, ix); ok = mf <= 1
                return float(np.nanmax(ms[ok])) if ok.any() else np.nan
            d2b = at_fa1(B, None) - at_fa1(A, None)
            hyp_rows.append(dict(variant=v, lineage=tag, hypothesis="H2", decision="held" if (d2 > 0 and p2 >= 0.90) else "not held",
                                 estimate=f"Δsensitivity on focal non-motor (B − A, frozen τ) = {d2:+.3f} (B {PB2.agg()['sensitivity']:.3f}, A {PA2.agg()['sensitivity']:.3f}); "
                                          f"secondary at FA/24h ≤ 1: {d2b:+.3f}",
                                 ci="[{:+.3f}, {:+.3f}]".format(*ci(bs2)), p=f"P(Δ > 0) = {p2:.3f}"))
            # H3b: non-FBTC focal seizures
            nf = lambda R: variant_sel(R["sz"], v) & np.isin(R["sz"].group.values, ["focal_motor", "focal_nonmotor", "focal_unknown_motor"])
            PA3, PB3 = Patients(pats, A["sz"], A["rn"], nf(A)), Patients(pats, B["sz"], B["rn"], nf(B))
            dA = A["sz"].loc[nf(A) & A["sz"].hit.values, "delay_s"]; dB = B["sz"].loc[nf(B) & B["sz"].hit.values, "delay_s"]
            dd = float(np.median(dB) - np.median(dA)) if len(dA) and len(dB) else np.nan
            both = nf(A) & A["sz"].hit.values & B["sz"].hit.values
            dd_both = float(np.median(B["sz"].delay_s.values[both]) - np.median(A["sz"].delay_s.values[both])) if both.any() else np.nan
            ds3 = PB3.agg()["sensitivity"] - PA3.agg()["sensitivity"]
            pi = {p: i for i, p in enumerate(pats)}
            subA = A["sz"]["sub"].map(pi).values
            def h3b_boot(ix):
                w = np.bincount(ix, minlength=len(pats))[subA]
                ma, mb = nf(A) & A["sz"].hit.values, nf(B) & B["sz"].hit.values
                ra, rb = np.repeat(A["sz"].delay_s.values[ma], w[ma]), np.repeat(B["sz"].delay_s.values[mb], w[mb])
                return (np.median(rb) - np.median(ra)) if len(ra) and len(rb) else np.nan
            bs3 = np.array([h3b_boot(i) for i in idx]); bs3s = np.array([PB3.agg(i)["sensitivity"] - PA3.agg(i)["sensitivity"] for i in idx])
            hyp_rows.append(dict(variant=v, lineage=tag, hypothesis="H3b",
                                 decision="held" if (np.isfinite(dd) and dd >= -2 and ds3 > 0) else "not held",
                                 estimate=f"Δdelay (median B − median A, each model's detected non-FBTC focal seizures) = {dd:+.1f} s "
                                          f"(over seizures both detect: {dd_both:+.1f} s); Δsensitivity (B − A) = {ds3:+.3f}",
                                 ci="Δdelay [{:+.1f}, {:+.1f}] s; Δsens [{:+.3f}, {:+.3f}]".format(*ci(bs3), *ci(bs3s)),
                                 p=f"for reference only: P(Δdelay ≥ −2) = {np.mean(np.nan_to_num(bs3, nan=-np.inf) >= -2):.3f}; P(Δsens > 0) = {np.mean(bs3s > 0):.3f}"))
            # H4
            P_all = {key: Patients(pats, R["sz"], R["rn"], variant_sel(R["sz"], v)) for key, R in res.items()}
            f1 = {key: P.agg()["f1"] for key, P in P_all.items()}
            for model, rules in (("A_eeg/xgb", ["rule_eeg"]), ("B_eeg_ecg/xgb", ["rule_eeg", "rule_hr"]),
                                 ("C_eeg_ecg_emg_acc/xgb", ["rule_eeg", "rule_hr", "rule_emg", "rule_acc"])):
                best = max(rules, key=lambda r: f1[r] if np.isfinite(f1[r]) else -1)
                dd4 = f1[model] - f1[best]
                bs4 = np.array([P_all[model].agg(i)["f1"] - P_all[best].agg(i)["f1"] for i in idx])
                p4 = float(np.mean(bs4 > 0))
                hyp_rows.append(dict(variant=v, lineage=tag, hypothesis=f"H4 ({SHORT[model]} vs {best})",
                                     decision="held" if (dd4 > 0 and p4 >= 0.90) else "not held",
                                     estimate=f"ΔF1 = {dd4:+.3f} ({SHORT[model]} {f1[model]:.3f}, {best} {f1[best]:.3f}; best of {', '.join(rules)})",
                                     ci="[{:+.3f}, {:+.3f}]".format(*ci(bs4)), p=f"P(ΔF1 > 0) = {p4:.3f}"))

        # offline - causal difference for C (this variant), paired over patients
        (_, ra, pats_a, idx_a), (_, rb, pats_b, _) = per_lineage["v02g"], per_lineage["v11"]
        assert pats_a == pats_b
        Pg = Patients(pats_a, ra["C_eeg_ecg_emg_acc/xgb"]["sz"], ra["C_eeg_ecg_emg_acc/xgb"]["rn"], variant_sel(ra["C_eeg_ecg_emg_acc/xgb"]["sz"], v))
        Pc = Patients(pats_a, rb["C_eeg_ecg_emg_acc/xgb"]["sz"], rb["C_eeg_ecg_emg_acc/xgb"]["rn"], variant_sel(rb["C_eeg_ecg_emg_acc/xgb"]["sz"], v))
        for mtr in ("f1", "sensitivity", "fa_per_24h", "delay_s"):
            d = Pc.agg()[mtr] - Pg.agg()[mtr]
            bs = np.array([Pc.agg(i)[mtr] - Pg.agg(i)[mtr] for i in idx_a])
            hyp_rows.append(dict(variant=v, lineage="v11 − v02g", hypothesis=f"offline−causal Δ{mtr} (C)", decision="reported",
                                 estimate=f"{d:+.3f}", ci="[{:+.3f}, {:+.3f}]".format(*ci(bs)), p=""))

    # ---------------- H3a (signals; Exp R definitions, offline processing; once, all FBTC seizures of this set)
    L.set_mode(False)
    fb = S[S.group == "FBTC"]
    h3 = []
    for z_thr, min_dur in ((R2.Z_THR, R2.MIN_DUR), (2.0, 5)):
        rows = []
        for x in fb.itertuples():
            meta = dict(sub=x.sub, run=int(x.run), onset_s=float(x.onset_s), has_ecg_emg=bool(x.has_ecg_emg))
            try:
                df = R2.seizure_curves(x.sub, int(x.run), float(x.onset_s))
                rows.append(R2.onset_row(meta, df, z_thr, min_dur))
            except Exception as e:
                rows.append(dict(meta, error=f"{type(e).__name__}: {e}"))
        O = pd.DataFrame(rows)
        all_tables.append((f"h3a_onsets_z{z_thr:g}_{min_dur}s.csv", O))
        for v in ("all", "ecg_emg"):
            Ov = O if v == "all" else O[O.has_ecg_emg]
            pats3 = sorted(Ov["sub"].unique()); idx3 = boot_idx(len(pats3)); pi3 = {p: i for i, p in enumerate(pats3)}
            ests = {}
            for X in ("hr", "acc", "emg"):
                lag = Ov[f"lag_eeg_to_{X}"].values.astype(float) if f"lag_eeg_to_{X}" in Ov else np.array([])
                ok = np.isfinite(lag); subs3 = Ov["sub"].map(pi3).values[ok]; lg = lag[ok]
                med = float(np.median(lg)) if ok.any() else np.nan
                bs = []
                for ix in idx3:
                    w = np.bincount(ix, minlength=len(pats3))[subs3]
                    rr = np.repeat(lg, w); bs.append(np.median(rr) if len(rr) else np.nan)
                ests[X] = (med, ci(bs), int(ok.sum()))
            held = all(np.isfinite(ests[X][0]) and ests[X][0] > 0 for X in ests)
            h3.append(dict(variant=v, criterion=f"z > {z_thr:g}, {min_dur} s", decision=("held" if held else "not held") if z_thr == R2.Z_THR else "reported",
                           **{f"lag_eeg_to_{X}": f"{ests[X][0]:+.1f} s [{ests[X][1][0]:+.1f}, {ests[X][1][1]:+.1f}], n = {ests[X][2]}" for X in ests}))
    all_tables.append(("h3a.csv", pd.DataFrame(h3)))

    # ---------------- write
    tables = {}
    for name, df in all_tables:                       # several lineages / models write to the same table: concatenate
        tables.setdefault(name, []).append(df)
    for name, dfs in tables.items():
        pd.concat(dfs, ignore_index=True).to_csv(out / name, index=False)
    H = pd.DataFrame(hyp_rows); H.to_csv(out / "hypotheses.csv", index=False)
    ov = {v: pd.read_csv(out / f"overall_{v}.csv") for v in ("all", "ecg_emg")}
    f3 = lambda x: "–" if not np.isfinite(x) else f"{x:.3f}"
    f2 = lambda x: "–" if not np.isfinite(x) else f"{x:.2f}"
    md += ["## Data", "", f"- {len(subs)} patients; {sum(r['hours'] for r in per_lineage['v02g'][1]['A_eeg/xgb']['run_meta'].to_dict('records')):,.0f} unmasked hours; "
           f"{len(S)} seizures ({', '.join(f'{g} {int((S.group == g).sum())}' for g in GROUPS)}); {int((~S.has_ecg_emg).sum())} without ECG/EMG.", ""]
    md += ["## Hypotheses (decided as in the pre-registration's analysis plan)", "",
           "Decisions use all seizures of the pre-registered set and per-patient averages (the pre-registered primary "
           "aggregation). The same hypotheses are also shown without the seizures that lack ECG/EMG.", ""]
    for v, vname in (("all", "All seizures (pre-registered set)"), ("ecg_emg", "Without the seizures lacking ECG/EMG")):
        md += [f"### {vname}", "", "| hypothesis | lineage | decision | estimate | 95% CI | bootstrap |", "|---|---|---|---|---|---|"]
        for r in H[H.variant == v].itertuples():
            md.append(f"| {r.hypothesis} | {r.lineage} | **{r.decision}** | {r.estimate} | {r.ci} | {r.p} |")
        for r in pd.DataFrame(h3)[lambda d: d.variant == v].itertuples():
            rule = "decision rule: median > 0 for all three" if r.decision != "reported" else "robustness, not decisive"
            md.append(f"| H3a ({r.criterion}) | signals (Exp R) | **{r.decision}** | EEG→HR {r.lag_eeg_to_hr}; EEG→ACC {r.lag_eeg_to_acc}; EEG→EMG {r.lag_eeg_to_emg} | | {rule} |")
        md += ["", "H3b is decided by the point estimates, as the hypothesis is worded in the pre-registration (Δdelay ≥ −2 s and B's "
               "sensitivity higher than A's; no bootstrap threshold). Its bootstrap probabilities and 95% CIs are reported for "
               "reference only. H1, H2 and H4 use P ≥ 0.90 as pre-registered.", ""]
    for v, vname in (("all", "all seizures"), ("ecg_emg", "without the seizures lacking ECG/EMG")):
        prim = "per-patient average (primary)" if v == "all" else "per-patient average"
        md += [f"## Frozen-τ results — {vname}, {prim}", "",
               "Pre-registered primary aggregation: each metric per patient, then averaged over patients; patient-bootstrap 95% CIs.", "",
               "| lineage | model | sensitivity [CI] | FA/24h [CI] | precision | F1 [CI] | median delay s |", "|---|---|---|---|---|---|---|"]
        for r in ov[v].itertuples():
            md.append(f"| {r.lineage} | {r.model} | {f3(r.sensitivity)} [{f3(r.sensitivity_lo)}, {f3(r.sensitivity_hi)}] | {f2(r.fa_per_24h)} [{f2(r.fa_per_24h_lo)}, {f2(r.fa_per_24h_hi)}] | "
                      f"{f3(r.precision)} | {f3(r.f1)} [{f3(r.f1_lo)}, {f3(r.f1_hi)}] | {f2(r.delay_s)} |")
        md += ["", f"## Frozen-τ results — {vname}, pooled counts (secondary)", "",
               "Secondary: counts summed over all patients, then the rates computed; for comparison with the development tables.", "",
               "| lineage | model | seizures | detected | sensitivity | FA/24h | precision | F1 | median delay s |", "|---|---|---|---|---|---|---|---|---|"]
        for r in ov[v].itertuples():
            md.append(f"| {r.lineage} | {r.model} | {r.seizures} | {r.detected} | {f3(r.pooled_sensitivity)} | {f2(r.pooled_fa_per_24h)} | "
                      f"{f3(r.pooled_precision)} | {f3(r.pooled_f1)} | {f2(r.pooled_delay_s)} |")
        md.append("")
    for v, vname in (("all", "all seizures"), ("ecg_emg", "without the seizures lacking ECG/EMG")):
        sg = pd.read_csv(out / f"subgroups_{v}.csv")
        sg = sg[sg.model.isin(["A", "B", "C"])]
        md += [f"## Subgroups at the frozen τ (models A, B, C) — {vname}", "",
               "Sensitivity per patient, then averaged (primary, with patient-bootstrap 95% CI) next to the pooled sensitivity "
               "(secondary: detected / all seizures of the subgroup). The two can differ a lot when a few patients have many "
               "seizures. Montage rows also give FA/24h and F1 per patient. Rules: `subgroups_*.csv`.", "",
               "| lineage | model | subgroup | seizures | patients | sensitivity [CI] (per patient, primary) | pooled sensitivity (secondary) | "
               "median delay s | FA/24h (montage) | F1 (montage) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sg.itertuples():
            fa_m = f2(r.fa_per_24h) if "fa_per_24h" in sg and np.isfinite(getattr(r, "fa_per_24h", np.nan)) else ""
            f1_m = f3(r.f1) if "f1" in sg and np.isfinite(getattr(r, "f1", np.nan)) else ""
            md.append(f"| {r.lineage} | {r.model} | {r.kind}: {r.subgroup} | {r.seizures} | {r.patients} | {f3(r.sensitivity)} "
                      f"[{f3(r.sensitivity_lo)}, {f3(r.sensitivity_hi)}] | {f3(r.pooled_sensitivity)} | {f2(r.delay_s)} | {fa_m} | {f1_m} |")
        md.append("")
    md += ["Subgroups (seizure group, duration, montage), operating-curve points (sensitivity at FA/24h ≤ 1, 2, 5, 10; τ swept on "
           "this set — a curve, not a deployable number), per-patient tables and the H3a onsets are in the CSV files next to this "
           "file.", ""]
    md += ["*" + ("Dry run on development patients: in-sample, not a result." if dry else
                  "This is the locked test result. From now on sub-097..125 are development data.") + "*"]
    (out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(L.ROOT) / 'RESULTS.md'}", flush=True)
    for r in H[H.variant == "all"].itertuples():
        print(f"  {r.lineage:10s} {r.hypothesis:22s} {r.decision}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Independent re-computation of the locked internal test (CLAUDE.md: every number is re-computed before it goes into a
document, slide or website), plus the development-set numbers in the same aggregation.

Run only after 07_locked_test.py has run once (results/locked_test/RESULTS.md exists). This script never changes a
threshold, model, feature or rule: it reads the frozen models and their frozen tau and recomputes the frozen-tau numbers
along a different code path than 07, so that an error in either one shows up as a mismatch:

  07                                      this script
  seizures from data/catalog/seizeit2_seizures.csv   seizures from each run's events.tsv (hashed in
                                                     results/freeze/locked_events_sha256.txt)
  xgboost XGBClassifier.predict_proba     xgboost Booster.predict on a DMatrix
  own event matching (score_at)           ictra_lib.score_recording (k_of_n -> windows_to_events -> score_events)
  Patients class                          own per-patient aggregation (pandas)

Shared on purpose (one signal-processing code path, CLAUDE.md): the feature files and ictra_lib's post-processing and
scoring. Not recomputed here: H1 (curve point at matched sensitivity) and H3a (Exp R onsets).

    python src/28_recompute_locked.py      # -> results/locked_test_recompute/
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

OUT = L.ROOT / "results/locked_test_recompute"
LINEAGES = {"v02g": "v02g_grid_fix", "v11": "v11_causal"}          # frozen tag -> dev version (same pipeline)
SHORT = {"A_eeg/xgb": "A", "B_eeg_ecg/xgb": "B", "C_eeg_ecg_emg_acc/xgb": "C"}
RULES = ("rule_hr", "rule_emg", "rule_eeg", "rule_acc")
H4 = {"A": ("rule_eeg",), "B": ("rule_eeg", "rule_hr"), "C": ("rule_eeg", "rule_hr", "rule_emg", "rule_acc")}


def guard():
    if not (L.ROOT / "docs/PREREGISTRATION.sha256").exists():
        sys.exit("refusing: docs/PREREGISTRATION.sha256 does not exist yet")
    if not (L.ROOT / "results/locked_test/RESULTS.md").exists():
        sys.exit("refusing: 07_locked_test.py has not run; this script only re-computes its numbers afterwards")


def locked_subjects():
    sp = L.load_splits()
    dev = {s for f in sp["dev_folds"].values() for s in f}
    subs = sorted(sp["locked_test"])
    assert not dev & set(subs) and len(subs) == 29
    return subs


def load(tag):
    import xgboost as xgb
    meta = json.loads((L.ROOT / f"models/ictra_{tag}.meta.json").read_text(encoding="utf-8"))
    m = {}
    for key, c in meta["cells"].items():
        boosters = []
        for f in c["files"]:
            b = xgb.Booster(); b.load_model(str(L.ROOT / "models" / f)); b.set_param({"nthread": 4})
            boosters.append(b)
        m[SHORT[key]] = ("xgb", boosters, c["columns"], c["thr"], c["k"], c["n"])
    for r, c in meta["rules"].items():
        m[r] = ("rule", None, [c["feature"]], c["thr"], c["k"], c["n"])
    return meta, m


def window_scores(kind, boosters, cols, d):
    import xgboost as xgb
    X = d.reindex(columns=cols)
    if kind == "rule":
        return X[cols[0]].fillna(-1e9).to_numpy(np.float64)
    dm = xgb.DMatrix(X, missing=np.nan)
    return np.mean([b.predict(dm) for b in boosters], axis=0).astype(np.float64)


def score_lineage(tag, subs, groups):
    """One row per run and model (counts from ictra_lib.score_recording) and one row per seizure and model (hit, delay)."""
    meta, models = load(tag)
    L.set_mode(meta["causal"])
    assert (L.CFG["post_k"], L.CFG["post_n"]) == (3, 5) and all(v[4:] == (3, 5) for v in models.values())
    runs, szs = [], []
    files = sorted(p for p in L.features_dir().glob("*.parquet") if p.name.split("_run")[0] in set(subs))
    assert len(files) == 778, len(files)
    for f in files:
        d = pd.read_parquet(f)
        sub, run = str(d["sub"].iloc[0]), int(d["run"].iloc[0])
        ev = pd.read_csv(L.ROOT / f"data/raw/ds005873/{sub}/ses-01/eeg/{sub}_ses-01_task-szMonitoring_run-{run:02d}_events.tsv",
                         sep="\t").rename(columns={"onset": "onset_s", "duration": "duration_s", "eventType": "type"})
        mask = d.impd.to_numpy(bool) | d.get("flat_eeg", pd.Series(False, index=d.index)).fillna(False).astype(bool).to_numpy()
        ref = [(e.onset_s, e.onset_s + e.duration_s) for e in ev.itertuples() if str(e.type).startswith("sz")]
        for name, (kind, boosters, cols, thr, k, n) in models.items():
            p = window_scores(kind, boosters, cols, d)
            s, hyp = L.score_recording(p > thr, ev, len(d), mask=mask)
            runs.append(dict(lineage=tag, model=name, sub=sub, run=run, n_ref=s["n_ref"], tp=s["tp"], fp=s["fp"],
                             n_hyp=s["n_hyp"], hours=s["hours"]))
            for a, b in ref:
                hits = [hs for hs, he in hyp if hs < b + L.CFG["tol_post_s"] and he > a - L.CFG["tol_pre_s"]]
                szs.append(dict(lineage=tag, model=name, sub=sub, run=run, onset_s=a, group=groups.get((sub, run, round(a, 3)), "?"),
                                hit=bool(hits), delay_s=(min(hits) - a) if hits else np.nan))
    L.set_mode(False)
    return pd.DataFrame(runs), pd.DataFrame(szs)


def per_patient(rn, sz):
    """Per-patient metrics as pre-registered: sensitivity over patients with seizures; FA/24h over all patients;
    precision over patients with alarms; F1 per patient with seizures (0 when it has no alarm); delay per patient = median."""
    g = rn.groupby("sub")[["n_ref", "tp", "fp", "n_hyp", "hours"]].sum()
    out = pd.DataFrame(index=g.index)
    out["sens"] = (g.tp / g.n_ref).where(g.n_ref > 0)
    out["fa"] = g.fp * 24 / g.hours
    out["prec"] = ((g.n_hyp - g.fp) / g.n_hyp).where(g.n_hyp > 0)
    s, pr = out.sens.fillna(0), out.prec.fillna(0)
    out["f1"] = (2 * s * pr / (s + pr)).where(s + pr > 0, 0.0).where(g.n_ref > 0)
    out["delay"] = sz[sz.hit].groupby("sub").delay_s.median().reindex(out.index)
    return out


def agg(pp):
    return dict(sensitivity=pp.sens.mean(), fa_per_24h=pp.fa.mean(), precision=pp.prec.mean(), f1=pp.f1.mean(),
                delay_s=pp.delay.median())


def pooled(rn, sz):
    n, tp, fp, hy, h = rn.n_ref.sum(), rn.tp.sum(), rn.fp.sum(), rn.n_hyp.sum(), rn.hours.sum()
    s, pr = tp / n, (hy - fp) / hy if hy else np.nan
    return dict(sensitivity=s, fa_per_24h=fp * 24 / h, precision=pr, f1=2 * s * pr / (s + pr) if hy and s + pr > 0 else 0.0,
                delay_s=sz.loc[sz.hit, "delay_s"].median(), seizures=int(n), detected=int(tp), false_alarms=int(fp))


def boot_ci(pp, rng, B=1000):
    """Patient bootstrap of the per-patient averages (own generator; CIs are expected to differ slightly from 07's)."""
    idx = rng.integers(0, len(pp), size=(B, len(pp)))
    out = {}
    for c, name in (("sens", "sensitivity"), ("fa", "fa_per_24h"), ("f1", "f1")):
        v = pp[c].to_numpy()
        m = np.array([np.nanmean(v[i]) for i in idx])
        out[name] = (np.nanpercentile(m, 2.5), np.nanpercentile(m, 97.5))
    return out


def dev_same_aggregation():
    """Development numbers (out-of-fold, per-fold tau, every seed) in the locked test's per-patient aggregation, and pooled."""
    rows = []
    for tag, ver in LINEAGES.items():
        P = pd.read_csv(L.ROOT / f"results/dev/{ver}/scores_by_patient.csv")
        P = P[(P.model == "xgb") | (P.set == "rule")]
        for (st, model), G in P.groupby(["set", "model"]):
            name = model if st == "rule" else {"A_eeg": "A", "B_eeg_ecg": "B", "C_eeg_ecg_emg_acc": "C"}[st]
            per_seed = []
            for seed, S in G.groupby("seed"):
                g = S.set_index("sub")
                sens = (g.tp / g.n_ref).where(g.n_ref > 0)
                prec = ((g.n_hyp - g.fp) / g.n_hyp).where(g.n_hyp > 0)
                s0, p0 = sens.fillna(0), prec.fillna(0)
                f1 = (2 * s0 * p0 / (s0 + p0)).where(s0 + p0 > 0, 0.0).where(g.n_ref > 0)
                n, tp, fp, hy, h = g.n_ref.sum(), g.tp.sum(), g.fp.sum(), g.n_hyp.sum(), g.hours.sum()
                ps, pp_ = tp / n, (hy - fp) / hy
                per_seed.append(dict(pp_sens=sens.mean(), pp_fa=(g.fp * 24 / g.hours).mean(), pp_prec=prec.mean(), pp_f1=f1.mean(),
                                     pool_sens=ps, pool_fa=fp * 24 / h, pool_prec=pp_, pool_f1=2 * ps * pp_ / (ps + pp_),
                                     patients=len(g), seizures=int(n)))
            d = pd.DataFrame(per_seed)
            rows.append(dict(lineage=tag, model=name, seeds=len(d), patients=int(d.patients.iloc[0]), seizures=int(d.seizures.iloc[0]),
                             **{c: d[c].mean() for c in d.columns if c.startswith(("pp_", "pool_"))},
                             pp_sens_seed_min=d.pp_sens.min(), pp_sens_seed_max=d.pp_sens.max()))
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    guard()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    subs = locked_subjects()
    S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")
    groups = {(s, int(r), round(o, 3)): g for s, r, o, g in zip(S["sub"], S.run, S.onset_s, S.group)}   # group labels only
    rn_all, sz_all = [], []
    for tag in LINEAGES:
        rn, sz = score_lineage(tag, subs, groups)
        rn_all.append(rn); sz_all.append(sz)
        print(f"{tag}: scored {rn.run.nunique()} runs x {rn.model.nunique()} models ({time.time() - t0:.0f} s)", flush=True)
    rn, sz = pd.concat(rn_all), pd.concat(sz_all)
    assert (sz.group != "?").all(), "a seizure in events.tsv has no catalog entry"
    rn.to_csv(OUT / "per_run.csv", index=False); sz.to_csv(OUT / "per_seizure.csv", index=False)

    rng = np.random.default_rng(20261028)
    tab, pps = [], {}
    for (tag, model), R in rn.groupby(["lineage", "model"], sort=False):
        Z = sz[(sz.lineage == tag) & (sz.model == model)]
        pp = per_patient(R, Z); pps[(tag, model)] = pp
        a, p, ci = agg(pp), pooled(R, Z), boot_ci(pp, rng)
        tab.append(dict(lineage=tag, model=model, **{f"pp_{k}": v for k, v in a.items()},
                        pp_sens_lo=ci["sensitivity"][0], pp_sens_hi=ci["sensitivity"][1], pp_fa_lo=ci["fa_per_24h"][0],
                        pp_fa_hi=ci["fa_per_24h"][1], pp_f1_lo=ci["f1"][0], pp_f1_hi=ci["f1"][1],
                        **{f"pool_{k}": v for k, v in p.items()}))
    T = pd.DataFrame(tab)
    T.to_csv(OUT / "frozen_tau_recomputed.csv", index=False)

    # hypothesis point estimates that do not need the operating curve
    H = []
    for tag in LINEAGES:
        f1 = {m: pps[(tag, m)].f1.mean() for m in ("A", "B", "C") + RULES}
        for m, rules in H4.items():
            best = max(rules, key=lambda r: f1[r])
            H.append(dict(lineage=tag, hypothesis=f"H4 ({m} vs best of {', '.join(rules)})", estimate=f1[m] - f1[best], detail=f"{m} {f1[m]:.3f}, {best} {f1[best]:.3f}"))
        sub_sz = lambda m, sel: sz[(sz.lineage == tag) & (sz.model == m) & sel(sz)]
        nm = lambda z: z.group == "focal_nonmotor"
        sens_nm = {m: sub_sz(m, nm).groupby("sub").hit.mean().mean() for m in ("A", "B")}
        H.append(dict(lineage=tag, hypothesis="H2 (focal non-motor sensitivity B - A, per-patient average)", estimate=sens_nm["B"] - sens_nm["A"],
                      detail=f"B {sens_nm['B']:.3f}, A {sens_nm['A']:.3f}"))
        nf = lambda z: z.group.str.startswith("focal")
        za, zb = sub_sz("A", nf), sub_sz("B", nf)
        H.append(dict(lineage=tag, hypothesis="H3b delay (median B - median A over each model's detected non-FBTC focal seizures)",
                      estimate=zb.loc[zb.hit, "delay_s"].median() - za.loc[za.hit, "delay_s"].median(), detail=""))
        H.append(dict(lineage=tag, hypothesis="H3b sensitivity (B - A, non-FBTC focal, per-patient average)",
                      estimate=zb.groupby("sub").hit.mean().mean() - za.groupby("sub").hit.mean().mean(), detail=""))
    c = {t: pps[(t, "C")] for t in LINEAGES}
    for k, col in (("f1", "f1"), ("sensitivity", "sens"), ("fa_per_24h", "fa")):
        H.append(dict(lineage="v11 - v02g", hypothesis=f"offline-causal delta {k} (C)", estimate=c["v11"][col].mean() - c["v02g"][col].mean(), detail=""))
    pd.DataFrame(H).to_csv(OUT / "hypotheses_point_estimates.csv", index=False)

    D = dev_same_aggregation()
    D.to_csv(OUT / "dev_same_aggregation.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(T[["lineage", "model", "pp_sensitivity", "pp_fa_per_24h", "pp_precision", "pp_f1", "pp_delay_s", "pool_detected", "pool_seizures", "pool_false_alarms"]].round(4).to_string(index=False))
    print(pd.DataFrame(H)[["lineage", "hypothesis", "estimate"]].round(4).to_string(index=False))
    print(D[["lineage", "model", "seeds", "pp_sens", "pp_fa", "pp_f1", "pool_sens", "pool_fa", "pool_f1"]].round(4).to_string(index=False))
    print(f"DONE ({time.time() - t0:.0f} s) -> {OUT.relative_to(L.ROOT).as_posix()}/")


if __name__ == "__main__":
    main()

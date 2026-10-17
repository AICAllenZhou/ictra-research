"""Official results table of one version (Allen, 2026-10-17): every cell as seed-mean AND seed-median.

    python src/15_version_report.py --version v01_baseline

Reads only what 04_train_ablation.py and 05_operating_points.py wrote in results/dev/<version>/ (no re-training, no new
scoring rules: per-patient counts are pooled with ictra_lib.pool_scores, CIs with ictra_lib.patient_bootstrap).
Writes results/dev/<version>/REPORT.md and report_overall.csv, report_by_group.csv, report_by_montage.csv,
report_sweep.csv. Order: seizure groups with FBTC first. FBTC sensitivity < 0.80 for any learned model is flagged as a
possible pipeline problem (to be checked before the next version).
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

GROUPS = ["FBTC", "focal_motor", "focal_nonmotor", "focal_unknown_motor"]
SETS = ["A_eeg", "B_eeg_ecg", "C_eeg_ecg_emg_acc"]
RULES = ["rule_eeg", "rule_hr", "rule_emg", "rule_acc"]
FBTC_FLAG = 0.80


def order(df):
    key = {s: i for i, s in enumerate(SETS + ["rule"])}
    return df.assign(_k=df.set.map(key)).sort_values(["_k", "model"]).drop(columns="_k")


def mm(v):
    v = np.asarray(v, float)
    return np.nanmean(v), np.nanmedian(v)


def fmt(m, md, d=2):
    return f"{m:.{d}f} / {md:.{d}f}" if np.isfinite(m) else "–"


def overall(out):
    pat = pd.read_csv(out / "scores_by_patient.csv")
    rows = []
    for (s, m), d in pat.groupby(["set", "model"]):
        per = pd.DataFrame([L.pool_scores(g.to_dict("records")) for _, g in d.groupby("seed")])
        d0 = d[d.seed == d.seed.min()].to_dict("records")
        r = dict(set=s, model=m, n_seeds=len(per), patients=d["sub"].nunique(), seizures=int(per.n_ref.iloc[0]), hours=per.hours.iloc[0])
        for k in ["sensitivity", "precision", "f1", "fa_per_24h", "median_delay_s"]:
            r[f"{k}_mean"], r[f"{k}_median"] = mm(per[k])
        for k in ["sensitivity", "fa_per_24h", "f1"]:
            r[f"{k}_lo"], r[f"{k}_hi"] = L.patient_bootstrap(d0, k)
        rows.append(r)
    return order(pd.DataFrame(rows))


def by_group(out):
    """Sensitivity per seizure group: per seed pooled over patients; CI = patient bootstrap of seed 0."""
    G = pd.read_csv(out / "scores_by_group.csv")
    P = G.groupby(["set", "model", "seed", "group", "sub"])[["n_ref", "tp"]].sum().reset_index()
    rows = []
    rng = np.random.default_rng(0)
    for (s, m, g), d in P.groupby(["set", "model", "group"]):
        per = d.groupby("seed")[["n_ref", "tp"]].sum(); sens = per.tp / per.n_ref
        d0 = d[d.seed == d.seed.min()]; n = len(d0); vals = []
        for _ in range(1000):
            idx = rng.integers(0, n, n); x = d0.iloc[idx]; vals.append(x.tp.sum() / x.n_ref.sum())
        r = dict(set=s, model=m, group=g, seizures=int(per.n_ref.iloc[0]), patients=d["sub"].nunique())
        r["sensitivity_mean"], r["sensitivity_median"] = mm(sens)
        r["sensitivity_lo"], r["sensitivity_hi"] = np.nanpercentile(vals, [2.5, 97.5])
        rows.append(r)
    T = pd.DataFrame(rows)
    T["group"] = pd.Categorical(T.group, GROUPS, ordered=True)
    return order(T).sort_values(["group"], kind="stable")


def by_montage(out):
    F = pd.read_csv(out / "scores_by_fold.csv")
    R = pd.read_csv(L.ROOT / "data/catalog/seizeit2_runs.csv")[["sub", "run", "eeg_montage"]]
    F = F.merge(R, on=["sub", "run"], how="left")
    rows = []
    for (s, m, mo), d in F.groupby(["set", "model", "eeg_montage"]):
        per = pd.DataFrame([L.pool_scores(g.to_dict("records")) for _, g in d.groupby("seed")])
        r = dict(set=s, model=m, montage=mo, patients=d["sub"].nunique(), seizures=int(per.n_ref.iloc[0]), hours=per.hours.iloc[0])
        for k in ["sensitivity", "fa_per_24h", "f1"]:
            r[f"{k}_mean"], r[f"{k}_median"] = mm(per[k])
        rows.append(r)
    return order(pd.DataFrame(rows))


def sweep(out):
    """Best sensitivity at FA/24h <= 1, 2, 5 on the pooled held-out folds (operating curve; threshold swept on held-out
    data, so these are what the model CAN trade off, not deployable numbers), per seed, then mean and median."""
    p = out / "operating_curve.csv"
    if not p.exists():
        return pd.DataFrame()
    C = pd.read_csv(p); rows = []
    for (s, m), d in C.groupby(["set", "model"]):
        for g in ["all"] + GROUPS:
            col = "sensitivity" if g == "all" else f"sens_{g}"
            r = dict(set=s, model=m, group=g)
            for b in [1, 2, 5]:
                v = [ds.loc[ds.fa_per_24h <= b, col].max() for _, ds in d.groupby("seed")]
                v = [x if np.isfinite(x) else 0.0 for x in v]
                r[f"sens_at_fa{b}_mean"], r[f"sens_at_fa{b}_median"] = mm(v)
            r["n_seeds"] = d.seed.nunique(); rows.append(r)
    T = pd.DataFrame(rows)
    T["group"] = pd.Categorical(T.group, ["all"] + GROUPS, ordered=True)
    return order(T).sort_values("group", kind="stable")


def name(s, m):
    return m if s == "rule" else f"{s.split('_')[0]} {m}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    out = L.ROOT / "results/dev" / a.version
    O, G, M, S = overall(out), by_group(out), by_montage(out), sweep(out)
    O.to_csv(out / "report_overall.csv", index=False); G.to_csv(out / "report_by_group.csv", index=False)
    M.to_csv(out / "report_by_montage.csv", index=False); S.to_csv(out / "report_sweep.csv", index=False)
    md = [f"# {a.version}: official dev table", "",
          "Dev cross-validation (sub-001..096), offline pipeline. Each cell = **seed-mean / seed-median** over the seeds "
          "(3 for learned models, 1 for rules). [95% CI] = patient bootstrap (1,000 resamples) of seed 0. Threshold tuned "
          "on the inner split of the training folds; 3-of-5 windows; SzCORE scoring; impd and flat-EEG masked.", ""]
    md += ["## Overall", "", "| model | seizures | hours | sensitivity | [CI] | FA/24h | [CI] | precision | F1 | [CI] | median delay s |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in O.itertuples():
        md.append(f"| {name(r.set, r.model)} | {r.seizures} | {r.hours:,.0f} | {fmt(r.sensitivity_mean, r.sensitivity_median)} | "
                  f"[{r.sensitivity_lo:.2f}, {r.sensitivity_hi:.2f}] | {fmt(r.fa_per_24h_mean, r.fa_per_24h_median, 1)} | "
                  f"[{r.fa_per_24h_lo:.1f}, {r.fa_per_24h_hi:.1f}] | {fmt(r.precision_mean, r.precision_median)} | "
                  f"{fmt(r.f1_mean, r.f1_median)} | [{r.f1_lo:.2f}, {r.f1_hi:.2f}] | {fmt(r.median_delay_s_mean, r.median_delay_s_median, 0)} |")
    md += ["", "## Sensitivity by seizure type (FBTC first)", "",
           "| model | " + " | ".join(f"{g} (n={int(G[G.group == g].seizures.iloc[0])})" if (G.group == g).any() else g for g in GROUPS) + " |",
           "|---|" + "---|" * len(GROUPS)]
    flags = []
    for (s, m), d in G.groupby(["set", "model"], sort=False):
        cells = []
        for g in GROUPS:
            x = d[d.group == g]
            if x.empty:
                cells.append("–"); continue
            x = x.iloc[0]
            cells.append(f"{fmt(x.sensitivity_mean, x.sensitivity_median)} [{x.sensitivity_lo:.2f}, {x.sensitivity_hi:.2f}]")
            if g == "FBTC" and s != "rule" and min(x.sensitivity_mean, x.sensitivity_median) < FBTC_FLAG:
                flags.append(f"{name(s, m)}: FBTC sensitivity {x.sensitivity_mean:.2f} (mean) / {x.sensitivity_median:.2f} (median) < {FBTC_FLAG}")
        md.append(f"| {name(s, m)} | " + " | ".join(cells) + " |")
    if len(S):
        md += ["", "## Sensitivity at a fixed false-alarm budget (operating-point sweep)", "",
               "Threshold swept over the pooled held-out folds, so this is an operating curve, not a deployable threshold.", "",
               "| model | group | FA/24h ≤ 1 | ≤ 2 | ≤ 5 |", "|---|---|---|---|---|"]
        keep = S[(S.set.isin(SETS) & S.model.isin(["xgb", "lgbm"])) | (S.set == "rule")]
        for r in keep.itertuples():
            md.append(f"| {name(r.set, r.model)} | {r.group} | {fmt(r.sens_at_fa1_mean, r.sens_at_fa1_median)} | "
                      f"{fmt(r.sens_at_fa2_mean, r.sens_at_fa2_median)} | {fmt(r.sens_at_fa5_mean, r.sens_at_fa5_median)} |")
    md += ["", "## By EEG montage", "", "| model | montage | patients | seizures | hours | sensitivity | FA/24h | F1 |", "|---|---|---|---|---|---|---|---|"]
    for r in M.itertuples():
        md.append(f"| {name(r.set, r.model)} | {r.montage} | {r.patients} | {r.seizures} | {r.hours:,.0f} | {fmt(r.sensitivity_mean, r.sensitivity_median)} | "
                  f"{fmt(r.fa_per_24h_mean, r.fa_per_24h_median, 1)} | {fmt(r.f1_mean, r.f1_median)} |")
    md += ["", "## FBTC check (sensitivity < 0.80 = possible pipeline problem, check before the next version)", ""]
    md += [f"- **FLAG** {f}" for f in flags] or ["- no learned model below 0.80 on FBTC"]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

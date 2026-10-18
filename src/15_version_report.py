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
DUR_BINS = [("< 15 s", 0, 15), ("15-60 s", 15, 60), ("> 60 s", 60, np.inf)]     # Allen 2026-10-17: in every table
SHORT_TRIO = ["sub-039", "sub-047", "sub-035"]       # named subgroup: 194 of 702 dev seizures, median 10-13 s (v01 NOTES)


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


def by_duration(out):
    """Sensitivity by annotated seizure duration (< 15 s, 15-60 s, > 60 s), seed-mean / seed-median, from
    detections_by_seizure.csv (same hit rule as the pooled scores); CI = patient bootstrap of seed 0."""
    D = pd.read_csv(out / "detections_by_seizure.csv")
    S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")
    S["seizure_id"] = [f"{a}_r{int(b):02d}_{int(c)}" for a, b, c in zip(S["sub"], S.run, S.onset_s)]
    D = D.merge(S[["seizure_id", "duration_s"]], on="seizure_id", how="left")
    rows, rng = [], np.random.default_rng(0)
    for (s_, m_), d in D.groupby(["set", "model"]):
        for lab, lo, hi in DUR_BINS:
            x = d[(d.duration_s >= lo) & (d.duration_s < hi)]
            if x.empty:
                continue
            per = x.groupby("seed").hit.mean()
            x0 = x[x.seed == x.seed.min()]; P = x0.groupby("sub").hit.agg(["sum", "size"]); n = len(P); vals = []
            for _ in range(1000):
                idx = rng.integers(0, n, n); vals.append(P["sum"].values[idx].sum() / P["size"].values[idx].sum())
            r = dict(set=s_, model=m_, duration=lab, seizures=int(len(x0)), patients=int(n))
            r["sensitivity_mean"], r["sensitivity_median"] = mm(per)
            r["sensitivity_lo"], r["sensitivity_hi"] = np.nanpercentile(vals, [2.5, 97.5])
            rows.append(r)
    return order(pd.DataFrame(rows))


def fig_per_patient(out, version):
    """Per-patient sensitivity, A vs C (xgb, hits summed over seeds), dot area ~ seizures; the short-seizure trio is a
    named subgroup (orange, labelled) so its weight in the pooled numbers is visible."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pat = pd.read_csv(out / "scores_by_patient.csv")
    pp = pat[pat.model == "xgb"].groupby(["set", "sub"])[["n_ref", "tp"]].sum().reset_index()
    a = pp[pp.set == "A_eeg"].set_index("sub"); c = pp[pp.set == "C_eeg_ecg_emg_acc"].set_index("sub")
    J = a.join(c, lsuffix="_a", rsuffix="_c").dropna(); J = J[J.n_ref_a > 0]
    J["sa"], J["sc"], J["n"] = J.tp_a / J.n_ref_a, J.tp_c / J.n_ref_c, J.n_ref_a / pat.seed.nunique()
    trio = J.index.isin(SHORT_TRIO)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="#b5b4af", lw=1, zorder=1)
    ax.scatter(J.sa[~trio], J.sc[~trio], s=12 + 6 * J.n[~trio], color="#2a78d6", alpha=.55, edgecolor="white", lw=.8, zorder=2,
               label=f"other patients (n={int((~trio).sum())}, {int(J.n[~trio].sum())} seizures)")
    ax.scatter(J.sa[trio], J.sc[trio], s=12 + 6 * J.n[trio], color="#eb6834", alpha=.9, edgecolor="white", lw=.8, zorder=3,
               label=f"short-seizure trio (sub-039/047/035, {int(J.n[trio].sum())} seizures, median 10-13 s)")
    for sub, r in J[trio].iterrows():
        ax.annotate(f"{sub} ({int(r.n)} sz)", (r.sa, r.sc), xytext=(8, 4), textcoords="offset points", fontsize=8, color="#0b0b0b")
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.set_aspect("equal")
    ax.set_xlabel("sensitivity with A (EEG only)", fontsize=9, color="#52514e"); ax.set_ylabel("sensitivity with C (EEG+ECG+EMG+ACC)", fontsize=9, color="#52514e")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#e4e3df", lw=.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=8, frameon=False)
    ax.set_title(f"{version}: per-patient event sensitivity, xgb\ndot area ~ seizures; above the dashed line C finds more", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_per_patient.png", dpi=150); plt.close(fig)


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
    O, G, M, S, Du = overall(out), by_group(out), by_montage(out), sweep(out), by_duration(out)
    fig_per_patient(out, a.version)
    Du.to_csv(out / "report_by_duration.csv", index=False)
    O.to_csv(out / "report_overall.csv", index=False); G.to_csv(out / "report_by_group.csv", index=False)
    M.to_csv(out / "report_by_montage.csv", index=False); S.to_csv(out / "report_sweep.csv", index=False)
    md = [f"# {a.version}: official dev table", "",
          "Dev cross-validation (sub-001..096), offline pipeline. Each cell = **seed-mean / seed-median** over the seeds "
          "(3 for learned models, 1 for rules). [95% CI] = patient bootstrap (1,000 resamples) of seed 0. Threshold tuned "
          "on the inner split of the training folds; 3-of-5 windows; SzCORE scoring; impd and flat-EEG masked.", "",
          "**Pooling:** counts (seizures, hits, false alarms, hours) are summed over all patients of all 5 held-out folds, then "
          "the rates are computed, then averaged over seeds. This is pooled over patients, not the mean over folds that "
          "EVALUATION_PROTOCOL.md section 1 names. Per-fold numbers are in scores_by_fold.csv and compare_*_by_fold.csv.", ""]
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
    md += ["", "## Sensitivity by seizure duration", "", "| model | " + " | ".join(
        f"{lab} (n={int(Du[Du.duration == lab].seizures.iloc[0])})" if (Du.duration == lab).any() else lab for lab, _, _ in DUR_BINS) + " |",
        "|---|" + "---|" * len(DUR_BINS)]
    for (s_, m_), d in Du.groupby(["set", "model"], sort=False):
        cells = []
        for lab, _, _ in DUR_BINS:
            x = d[d.duration == lab]
            cells.append("–" if x.empty else f"{fmt(x.sensitivity_mean.iloc[0], x.sensitivity_median.iloc[0])} [{x.sensitivity_lo.iloc[0]:.2f}, {x.sensitivity_hi.iloc[0]:.2f}]")
        md.append(f"| {name(s_, m_)} | " + " | ".join(cells) + " |")
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
    sys.stdout.reconfigure(encoding="utf-8"); print("\n".join(md))


if __name__ == "__main__":
    main()

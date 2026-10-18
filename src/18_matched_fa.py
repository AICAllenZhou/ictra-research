"""Operating point at a matched false-alarm budget (prospective design note, docs/PREREGISTRATION_DRAFT.md; Allen
2026-10-17: v03b reports it next to the section-1 comparison).

    python src/18_matched_fa.py --ref v01_baseline --new v02_full_hours --fa 1 2 5 10

From each version's operating_curve.csv (05_operating_points.py: threshold swept over the pooled held-out folds), the
best sensitivity reachable at FA/24h <= budget, per seed, then mean and median over seeds, for every (set, model) and
seizure group, plus each version's deployable point (summary.csv). Writes results/dev/<new>/matched_fa_vs_<ref>.csv/.md.
This describes the curve at one FA budget; the accept/reject decision stays the section-1 rule.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

GROUPS = ["all", "FBTC", "focal_motor", "focal_nonmotor", "focal_unknown_motor"]


def at_budget(version, fa):
    """Per (set, model, group): at FA/24h <= fa, the threshold with the highest sensitivity (ties: higher F1), per seed;
    then mean and median over seeds of that sensitivity and of the pooled event F1 at that threshold (F1 only for the
    'all' group: precision is not group-specific). Rows without precision/f1 columns (older sweeps) give F1 = NaN."""
    C = pd.read_csv(L.ROOT / "results/dev" / version / "operating_curve.csv")
    if "f1" not in C:
        C["f1"] = np.nan
    rows = []
    for (s, m), d in C.groupby(["set", "model"]):
        for g in GROUPS:
            col = "sensitivity" if g == "all" else f"sens_{g}"
            sv, fv = [], []
            for _, ds in d.groupby("seed"):
                sub = ds[ds.fa_per_24h <= fa]
                if sub.empty:
                    sv.append(0.0); fv.append(np.nan); continue
                best = sub.sort_values([col, "f1"]).iloc[-1]
                sv.append(float(best[col])); fv.append(float(best["f1"]) if g == "all" else np.nan)
            sv, fv = np.array(sv), np.array(fv)
            rows.append(dict(set=s, model=m, group=g, fa_budget=fa, sens_mean=sv.mean(), sens_median=float(np.median(sv)),
                             f1_mean=np.nanmean(fv) if np.isfinite(fv).any() else np.nan,
                             f1_median=float(np.nanmedian(fv)) if np.isfinite(fv).any() else np.nan, n_seeds=len(sv)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--fa", type=float, nargs="+", required=True, help="FA/24h budgets, e.g. 1 2 5 10 (and the reference's deployable FA)")
    a = ap.parse_args()
    R = pd.concat([at_budget(a.ref, f) for f in a.fa]); N = pd.concat([at_budget(a.new, f) for f in a.fa])
    M = R.merge(N, on=["set", "model", "group", "fa_budget"], suffixes=("_ref", "_new"))
    M["d_sens_mean"] = M.sens_mean_new - M.sens_mean_ref; M["d_f1_mean"] = M.f1_mean_new - M.f1_mean_ref
    M.insert(0, "new", a.new); M.insert(0, "ref", a.ref)
    out = L.ROOT / "results/dev" / a.new
    M.to_csv(out / f"matched_fa_vs_{a.ref}.csv", index=False)
    dep = {v: pd.read_csv(L.ROOT / "results/dev" / v / "summary.csv").set_index(["set", "model"]) for v in (a.ref, a.new)}
    f = lambda x: "–" if not np.isfinite(x) else f"{x:.3f}"
    md = [f"# Fixed-FA comparison: {a.new} vs {a.ref}", "",
          "Best held-out sensitivity on each version's operating curve at FA/24h <= budget, and the event F1 at that "
          "threshold; seed-mean / seed-median. Threshold swept on held-out data: an operating curve, not a deployable "
          "threshold. The section-1 decision (single tuned tau) is reported separately.", ""]
    for g in GROUPS:
        D = M[M.group == g]
        if D.empty:
            continue
        md += [f"## {g}", "", f"| model | FA/24h ≤ | sens {a.ref} | sens {a.new} | Δ sens | F1 {a.ref} | F1 {a.new} | Δ F1 |", "|---|---|---|---|---|---|---|---|"]
        for r in D.sort_values(["set", "model", "fa_budget"]).itertuples():
            md.append(f"| {r.set} {r.model} | {r.fa_budget:g} | {f(r.sens_mean_ref)} / {f(r.sens_median_ref)} | {f(r.sens_mean_new)} / {f(r.sens_median_new)} | "
                      f"{r.d_sens_mean:+.3f} | {f(r.f1_mean_ref)} / {f(r.f1_median_ref)} | {f(r.f1_mean_new)} / {f(r.f1_median_new)} | "
                      + (f"{r.d_f1_mean:+.3f}" if np.isfinite(r.d_f1_mean) else "–") + " |")
        md.append("")
    md += ["Deployable points (single tuned tau; sensitivity / FA/24h / F1):", ""]
    for k in dep[a.ref].index:
        if k in dep[a.new].index:
            x, y = dep[a.ref].loc[k], dep[a.new].loc[k]
            md.append(f"- {k[0]} {k[1]}: {a.ref} {x.sensitivity:.3f} / {x.fa_per_24h:.2f} / {x.f1:.3f}; {a.new} {y.sensitivity:.3f} / {y.fa_per_24h:.2f} / {y.f1:.3f}")
    (out / f"matched_fa_vs_{a.ref}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(M[M.group == "all"][["set", "model", "fa_budget", "sens_mean_ref", "sens_mean_new", "f1_mean_ref", "f1_mean_new"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Operating point at a matched false-alarm budget (prospective design note, docs/PREREGISTRATION_DRAFT.md; Allen
2026-10-17: v03b reports it next to the section-1 comparison).

    python src/18_matched_fa.py --ref v01_baseline --new v03b_thr_sens_floor --fa 9.67

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
    C = pd.read_csv(L.ROOT / "results/dev" / version / "operating_curve.csv")
    rows = []
    for (s, m), d in C.groupby(["set", "model"]):
        for g in GROUPS:
            col = "sensitivity" if g == "all" else f"sens_{g}"
            v = [ds.loc[ds.fa_per_24h <= fa, col].max() for _, ds in d.groupby("seed")]
            v = np.array([x if np.isfinite(x) else 0.0 for x in v])
            rows.append(dict(set=s, model=m, group=g, sens_mean=v.mean(), sens_median=float(np.median(v)), n_seeds=len(v)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--fa", type=float, required=True, help="FA/24h budget, e.g. the reference's deployable FA")
    a = ap.parse_args()
    R, N = at_budget(a.ref, a.fa), at_budget(a.new, a.fa)
    M = R.merge(N, on=["set", "model", "group"], suffixes=(f"_{a.ref}", f"_{a.new}"))
    M["d_sens_mean"] = M[f"sens_mean_{a.new}"] - M[f"sens_mean_{a.ref}"]
    out = L.ROOT / "results/dev" / a.new
    M.to_csv(out / f"matched_fa_vs_{a.ref}.csv", index=False)
    dep = {v: pd.read_csv(L.ROOT / "results/dev" / v / "summary.csv").set_index(["set", "model"]) for v in (a.ref, a.new)}
    md = [f"# Matched-FA operating point: {a.new} vs {a.ref} at FA/24h <= {a.fa}", "",
          "Best sensitivity on the held-out operating curve at this FA budget (seed-mean / seed-median). Deployable "
          "points (summary.csv: own tuned threshold) are given for context. The section-1 decision is unchanged by this table.", "",
          "| model | group | " + f"{a.ref} | {a.new} | Δ (mean) |", "|---|---|---|---|---|"]
    for r in M.itertuples():
        md.append(f"| {r.set} {r.model} | {r.group} | {getattr(r, f'sens_mean_{a.ref}'):.3f} / {getattr(r, f'sens_median_{a.ref}'):.3f} | "
                  f"{getattr(r, f'sens_mean_{a.new}'):.3f} / {getattr(r, f'sens_median_{a.new}'):.3f} | {r.d_sens_mean:+.3f} |")
    md += ["", "Deployable points (sensitivity / FA/24h):", ""]
    for k in dep[a.ref].index:
        if k in dep[a.new].index:
            x, y = dep[a.ref].loc[k], dep[a.new].loc[k]
            md.append(f"- {k[0]} {k[1]}: {a.ref} {x.sensitivity:.3f} / {x.fa_per_24h:.2f}; {a.new} {y.sensitivity:.3f} / {y.fa_per_24h:.2f}")
    (out / f"matched_fa_vs_{a.ref}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(M[M.group == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

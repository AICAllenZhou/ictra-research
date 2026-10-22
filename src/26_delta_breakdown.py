"""Where does a difference between two versions come from? Sensitivity difference by seizure group and by duration.

    python src/26_delta_breakdown.py --ref v02g_grid_fix --new v11_causal

For every (set, model) and subgroup (seizure group; duration < 15 s / 15-60 s / > 60 s), from each version's
detections_by_seizure.csv (one row per seizure and repeat):
  - sensitivity ref / new = hits / (seizures x repeats), i.e. the mean over repeats of the pooled sensitivity;
  - delta with a paired patient-bootstrap 95% CI (1,000 resamples of patients; both versions use the same patients);
  - repeat noise: SD over repeats of the per-repeat delta (how much the difference moves between seeds/draws);
  - median alarm delay of detected seizures (all repeats pooled), ref and new.
FA/24h does not depend on the seizure group; it is in summary.csv. Writes results/dev/<new>/delta_vs_<ref>.csv and .md.
Allen 2026-10-21: if the loss came from the heart-rate detector it should concentrate where heart rate matters most,
not spread evenly over groups.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

BINS = [(0, 15, "< 15 s"), (15, 60, "15-60 s"), (60, 1e9, "> 60 s")]


def load(v):
    D = pd.read_csv(L.ROOT / "results/dev" / v / "detections_by_seizure.csv")
    S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")[["sub", "run", "onset_s", "duration_s"]]
    D = D.merge(S, on=["sub", "run", "onset_s"], how="left")
    D["dur"] = pd.cut(D.duration_s, [b[0] for b in BINS] + [BINS[-1][1]], right=False, labels=[b[2] for b in BINS])
    D["hit"] = D.hit.astype(bool)
    return D


def compare(R, N, n_boot=1000, seed=0):
    key = ["set", "model", "seed", "seizure_id"]
    M = R.merge(N[key + ["hit", "delay_s"]], on=key, suffixes=("_ref", "_new"))
    rng = np.random.default_rng(seed)
    rows = []
    for (s, m), d in M.groupby(["set", "model"]):
        subsets = [("all", "all", d)] + [("group", g, x) for g, x in d.groupby("group")] + \
                  [("duration", str(b), x) for b, x in d.groupby("dur", observed=True)]
        for kind, name, x in subsets:
            per = x.groupby("sub").agg(h_ref=("hit_ref", "sum"), h_new=("hit_new", "sum"), n=("hit_ref", "size"))
            sr, sn = per.h_ref.sum() / per.n.sum(), per.h_new.sum() / per.n.sum()
            idx = rng.integers(0, len(per), size=(n_boot, len(per)))
            H_r, H_n, Nn = per.h_ref.values[idx].sum(1), per.h_new.values[idx].sum(1), per.n.values[idx].sum(1)
            bd = H_n / Nn - H_r / Nn
            ps = x.groupby("seed").agg(r=("hit_ref", "mean"), n_=("hit_new", "mean"))
            rows.append(dict(set=s, model=m, kind=kind, subgroup=name, seizures=x.seizure_id.nunique(), patients=len(per),
                             repeats=x.seed.nunique(), sens_ref=sr, sens_new=sn, d_sens=sn - sr,
                             ci_lo=float(np.percentile(bd, 2.5)), ci_hi=float(np.percentile(bd, 97.5)),
                             repeat_sd=float((ps.n_ - ps.r).std()) if len(ps) > 1 else np.nan,
                             delay_ref=float(x.loc[x.hit_ref, "delay_s_ref"].median()) if x.hit_ref.any() else np.nan,
                             delay_new=float(x.loc[x.hit_new, "delay_s_new"].median()) if x.hit_new.any() else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--new", required=True)
    a = ap.parse_args()
    O = compare(load(a.ref), load(a.new))
    out = L.ROOT / "results/dev" / a.new
    O.to_csv(out / f"delta_vs_{a.ref}.csv", index=False)
    f = lambda v: f"{v:.3f}"
    md = [f"# Sensitivity difference {a.new} − {a.ref}, by seizure group and duration", "",
          "Δ = new − ref (hits / seizures × repeats); [95% CI] = paired patient bootstrap; repeat SD = SD over repeats of the "
          "per-repeat Δ. Delay = median alarm delay of detected seizures (s). FA/24h is group-independent (summary.csv).", ""]
    for kind, title in (("group", "By seizure group"), ("duration", "By seizure duration")):
        md += [f"## {title}", "", "| model | subgroup | seizures | sens ref → new | Δ [95% CI] | repeat SD | delay ref → new |",
               "|---|---|---|---|---|---|---|"]
        for r in O[O.kind.isin(["all", kind])].sort_values(["set", "model", "kind"]).itertuples():
            md.append(f"| {r.set} {r.model} | {r.subgroup} | {r.seizures} | {f(r.sens_ref)} → {f(r.sens_new)} | {r.d_sens:+.3f} "
                      f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.repeat_sd:.3f} | {r.delay_ref:.0f} → {r.delay_new:.0f} |")
        md.append("")
    (out / f"delta_vs_{a.ref}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(O[O.kind != "duration"][["set", "model", "subgroup", "sens_ref", "sens_new", "d_sens", "ci_lo", "ci_hi", "repeat_sd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

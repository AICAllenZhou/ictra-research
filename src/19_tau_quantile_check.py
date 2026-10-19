"""Diagnostic for v03b (Allen 2026-10-18): is the held-out operating point far above the sensitivity floor because the
floor did not bind (the inner F1 peak sits at a low tau), or because the inner out-of-fold models and the final model put
probabilities on different scales (the same tau falls on different quantiles)?

    python src/19_tau_quantile_check.py --version v03b_thr_sens_floor --ref v02_full_hours --workers 2

Per job (default subset: repeat 0 of every learned (set, model) x 5 folds, plus every rule x 5 folds):
  - re-creates the inner leave-patients-out OOF predictions exactly as 04_train_ablation.job() does (same rng sequence,
    same inner groups, same models) and re-selects tau with the same objective; tau_reproduced must equal the logged tau;
  - P_inner = share of unmasked inner OOF windows with prob >= tau; P_final = the same share of the final model's
    held-out predictions (saved preds/*.parquet); both also for background windows only (y = 0);
  - inner sensitivity / FA/24h / F1 at tau (pooled OOF) next to the held-out ones (fold CSV);
  - tau_equiv = the held-out threshold that gives the final model the inner exceedance P_inner.
P_final >> P_inner with inner sensitivity near the floor = scale drift; inner sensitivity >> floor and P_final ~ P_inner
= the floor did not bind. Rules have no fitted scale (their score is one feature), so they are the control.
Dev data only, post hoc, display only: nothing here feeds a threshold, a model or a version decision.
Writes results/dev/<version>/tau_quantile_check.csv and .md.
"""
import argparse, importlib.util, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

_spec = importlib.util.spec_from_file_location("trainer", Path(__file__).resolve().parent / "04_train_ablation.py")
T = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T)


def objective(ref, s, m):
    """Same as 04_train_ablation.main().obj() with --thr-objective f1_sens_floor --thr-ref-matched."""
    r = pd.read_csv(L.ROOT / "results/dev" / ref / "summary.csv").set_index(["set", "model"]).loc[(s, m)]
    C = pd.read_csv(L.ROOT / "results/dev" / ref / "operating_curve.csv")
    d = C[(C.set == s) & (C.model == m)]
    v = [ds.loc[ds.fa_per_24h <= float(r.fa_per_24h), "sensitivity"].max() for _, ds in d.groupby("seed")]
    s_ref = float(np.mean([x if np.isfinite(x) else 0.0 for x in v]))
    return dict(kind="f1_sens_floor", s_floor=s_ref - 0.02, fa_ref=float(r.fa_per_24h))


def windows(subs, cols, model=None):
    """Per run: (prob or None, y, mask) in run_files order (the order predict_runs and the saved preds use)."""
    for f in T.run_files(subs):
        d = pd.read_parquet(f)
        mask = d.impd.values | d.get("flat_eeg", pd.Series(False, index=d.index)).fillna(False).astype(bool).values
        p = model.predict_proba(d.reindex(columns=cols))[:, 1] if model is not None else None
        yield d["sub"].iloc[0], int(d["run"].iloc[0]), p, d.y.values.astype(int), mask


def check(args):
    fs_name, kind, fold, seed, version, ref, device = args
    T.DEVICE, T.RUNS = device, "all"
    T.THR_GRID = "legacy"                              # v03b ran before the grid correction (2026-10-19)
    T.SMOKE_GUARD["v"] = "_diag"                       # never write Lab job states
    draws = fs_name == "C_eeg_ecg_emg_acc" and kind == "xgb"
    mseed = seed // 10 if draws else seed
    splits = L.load_splits(); rng = np.random.default_rng(seed)
    val_subs = splits["dev_folds"][str(fold)]
    train_subs = [s for k, v in splits["dev_folds"].items() if k != str(fold) for s in v]
    mods = L.FEATURE_SETS[fs_name] if fs_name in L.FEATURE_SETS else [T.RULES[kind].split("_")[0]]
    cols = T.feature_columns(T.all_columns(), mods)
    tr = np.array(sorted(train_subs)); rng.shuffle(tr)
    groups = np.array_split(tr, T.N_INNER); pooled, oof = [], []
    for k in range(T.N_INNER):                         # identical sequence to job(), thr_mode inner_cv
        inner_val = list(groups[k]); inner_tr = [x for j, g in enumerate(groups) if j != k for x in g]
        D = T.load_training(inner_tr, cols, seed)
        m = T.predict_on_cpu(T.make_model(kind, mseed, D.y).fit(T.apply_dropout(D[cols], mods, rng), D.y), kind)
        pooled += list(T.predict_runs(m, inner_val, cols))
        oof += [(p, y, mk) for _, _, p, y, mk in windows(inner_val, cols, m)]
    obj = objective(ref, "rule" if kind in T.RULES else fs_name, kind)
    thr_re, kn, (f1_in, _), rule = T.tune_threshold(None, None, cols, runs=pooled, objective=obj)
    q = L.pool_scores([T.score_run(p > thr_re, ev, nw, mk, kn)[0] for _, _, p, ev, nw, mk, _ in pooled])
    F = pd.read_csv(L.ROOT / "results/dev" / version / f"fold_{fs_name}__{kind}__f{fold}_s{seed}.csv")
    thr = float(F.thr.iloc[0])
    P = pd.read_parquet(L.ROOT / "results/dev" / version / "preds" / f"{fs_name}__{kind}__f{fold}_s{seed}.parquet")
    hold = []
    for sub, run, _, y, mk in windows(val_subs, cols):
        p = P[(P["sub"] == sub) & (P.run == run)].prob.values
        assert len(p) == len(y), (sub, run)
        hold.append((p, y, mk))
    cat = lambda W, i: np.concatenate([w[i] for w in W])
    pi, yi, mi = cat(oof, 0), cat(oof, 1), cat(oof, 2).astype(bool)
    pf, yf, mf = cat(hold, 0), cat(hold, 1), cat(hold, 2).astype(bool)
    pi, yi, pf, yf = pi[~mi], yi[~mi], pf[~mf], yf[~mf]
    p_in, p_fin = float((pi >= thr).mean()), float((pf >= thr).mean())
    return dict(set=fs_name, model=kind, fold=fold, seed=seed, thr=thr, thr_reproduced=thr_re, thr_rule=rule,
                s_floor=obj["s_floor"], P_inner=p_in, P_final=p_fin, ratio=p_fin / p_in if p_in else np.nan,
                P_inner_bg=float((pi[yi == 0] >= thr).mean()), P_final_bg=float((pf[yf == 0] >= thr).mean()),
                P_inner_sz=float((pi[yi == 1] >= thr).mean()), P_final_sz=float((pf[yf == 1] >= thr).mean()),
                tau_equiv=float(np.quantile(pf, 1 - p_in)) if p_in else np.nan,
                inner_sens=float(q["sensitivity"]), inner_fa=float(q["fa_per_24h"]), inner_f1=float(np.nan_to_num(q["f1"])),
                held_sens=F.tp.sum() / F.n_ref.sum(), held_fa=F.fp.sum() / F.hours.sum() * 24,
                held_f1=2 * F.tp.sum() / (F.n_ref.sum() + F.n_hyp.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v03b_thr_sens_floor"); ap.add_argument("--ref", default="v02_full_hours")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    out = L.ROOT / "results/dev" / a.version
    learned = [(s, m) for s in L.FEATURE_SETS for m in ("xgb", "lgbm")]
    jobs = [(s, m, f, 0, a.version, a.ref, a.device) for s, m in learned for f in a.folds]
    jobs += [("rule", r, f, 0, a.version, a.ref, a.device) for r in T.RULES for f in a.folds]
    jobs = [j for j in jobs if (out / f"fold_{j[0]}__{j[1]}__f{j[2]}_s{j[3]}.csv").exists()]
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(check, jobs):
            rows.append(r)
            print(f"  {r['set']} {r['model']} f{r['fold']}: tau {r['thr']:.4f} (re {r['thr_reproduced']:.4f}) P_inner {r['P_inner']:.4f} "
                  f"P_final {r['P_final']:.4f} inner sens {r['inner_sens']:.2f} held {r['held_sens']:.2f} floor {r['s_floor']:.2f}", flush=True)
    R = pd.DataFrame(rows).sort_values(["set", "model", "fold"])
    R.to_csv(out / "tau_quantile_check.csv", index=False)
    G = R.groupby(["set", "model"]).agg(folds=("fold", "size"), tau_match=("thr", lambda s: int((abs(s - R.loc[s.index, "thr_reproduced"]) < 1e-9).sum())),
                                        s_floor=("s_floor", "first"), inner_sens=("inner_sens", "median"), held_sens=("held_sens", "median"),
                                        inner_fa=("inner_fa", "median"), held_fa=("held_fa", "median"),
                                        P_inner=("P_inner", "median"), P_final=("P_final", "median"), ratio=("ratio", "median"),
                                        folds_final_gt_inner=("ratio", lambda s: int((s > 1).sum())),
                                        P_inner_bg=("P_inner_bg", "median"), P_final_bg=("P_final_bg", "median")).reset_index()
    md = ["# tau quantile check: " + a.version, "",
          "Medians over folds (repeat 0). P = share of unmasked windows with prob >= tau: inner = pooled inner OOF "
          "predictions, final = final model on the held-out fold. Rules (no fitted scale) are the control.", "",
          G.round(4).to_markdown(index=False), ""]
    (out / "tau_quantile_check.md").write_text("\n".join(md), encoding="utf-8")
    print(G.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

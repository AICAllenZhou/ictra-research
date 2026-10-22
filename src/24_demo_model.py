"""Demo model for the replay (Stage 4): re-create ONE held-out fold model of an evaluated version and save it to disk.

    python src/24_demo_model.py --version v02g_grid_fix --fold 4 --seed 0          # C xgb, as evaluated
    python src/24_demo_model.py --version v11_causal --fold 4 --seed 0             # causal pipeline (after v11)

The trainer does not keep fitted models (only their held-out predictions). This script repeats
04_train_ablation.job() for one (set, model, fold, repeat) with the same random sequence, inner split, τ grid and
objective, then saves the final model + its τ / k-of-n / version / pipeline mode to models/demo/. It checks that the
saved model reproduces the version's stored held-out predictions for that fold (max |Δprob| reported), so the replay
shows the model that was evaluated, not a new one. The demo runs (docs/DEMO_RUNS.txt) are all in the held-out fold.
Dev patients only.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

_spec = importlib.util.spec_from_file_location("trainer", Path(__file__).resolve().parent / "04_train_ablation.py")
T = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True); ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="repeat id as in the fold CSV name (C xgb: model_seed*10 + draw)")
    ap.add_argument("--set", default="C_eeg_ecg_emg_acc"); ap.add_argument("--model", default="xgb")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    vdir = L.ROOT / "results/dev" / a.version
    stem = f"{a.set}__{a.model}__f{a.fold}_s{a.seed}"
    F = pd.read_csv(vdir / f"fold_{stem}.csv")
    causal = L.mode_for_version(a.version)
    T.DEVICE, T.RUNS = a.device, "all"
    T.THR_GRID = str(F.thr_grid.iloc[0]) if "thr_grid" in F else "legacy"
    draws = int(F.model_seed.iloc[0]) != a.seed if "model_seed" in F else False
    mseed = a.seed // 10 if draws else a.seed
    if str(F.thr_mode.iloc[0]) != "inner_split":
        sys.exit("only inner_split versions are supported (v02 / v02g / v11)")
    # --- same sequence as 04_train_ablation.job(), thr_mode inner_split, objective max_f1
    splits = L.load_splits(); rng = np.random.default_rng(a.seed)
    val_subs = splits["dev_folds"][str(a.fold)]
    train_subs = [s for k, v in splits["dev_folds"].items() if k != str(a.fold) for s in v]
    mods = L.FEATURE_SETS[a.set]
    cols = T.feature_columns(T.all_columns(), mods)
    tr = np.array(sorted(train_subs)); rng.shuffle(tr)
    n_in = max(2, int(0.2 * len(tr))); inner_val, inner_tr = list(tr[:n_in]), list(tr[n_in:])
    D = T.load_training(inner_tr, cols, a.seed)
    m = T.predict_on_cpu(T.make_model(a.model, mseed, D.y).fit(T.apply_dropout(D[cols], mods, rng), D.y), a.model)
    thr, kn, (f1_in, _), rule = T.tune_threshold(m, inner_val, cols)
    D = T.load_training(train_subs, cols, a.seed)
    m = T.predict_on_cpu(T.make_model(a.model, mseed, D.y).fit(T.apply_dropout(D[cols], mods, rng), D.y), a.model)
    print(f"tau {thr:.6f} (logged {float(F.thr.iloc[0]):.6f}), k-of-n {kn}, inner F1 {f1_in:.3f}", flush=True)
    # --- check against the stored held-out predictions of this fold
    P = pd.read_parquet(vdir / "preds" / f"{stem}.parquet")
    diffs = []
    for f in T.run_files(val_subs)[:6]:
        d = pd.read_parquet(f)
        p = m.predict_proba(d.reindex(columns=cols))[:, 1]
        q = P[(P["sub"] == d["sub"].iloc[0]) & (P.run == int(d["run"].iloc[0]))].prob.values
        diffs.append(float(np.max(np.abs(p.astype(np.float32) - q))))
    out = L.ROOT / "models/demo"; out.mkdir(parents=True, exist_ok=True)
    name = f"{a.version}__{stem}"
    m.save_model(str(out / f"{name}.json"))
    meta = dict(version=a.version, set=a.set, model=a.model, fold=a.fold, repeat=a.seed, model_seed=mseed, thr=float(thr),
                thr_logged=float(F.thr.iloc[0]), post_k=int(kn[0]), post_n=int(kn[1]), thr_grid=T.THR_GRID, causal=causal,
                pipeline="causal pipeline" if causal else "offline pipeline", lib_version=L.LIB_VERSION,
                features_dir=str(L.features_dir().relative_to(L.ROOT)), columns=cols, held_out_subjects=sorted(val_subs),
                check_max_abs_dprob_first6runs=max(diffs))
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved models/demo/{name}.json; max |prob - stored held-out prob| over 6 runs: {max(diffs):.2e}")


if __name__ == "__main__":
    main()

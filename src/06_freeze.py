"""Task 10: freeze the two reference lineages for the single locked-set opening (ITERATION_LOG 2026-10-21 policy).

    python src/keep_awake.py -- python -u src/06_freeze.py                # both: v02g (offline) and v11 (causal)
    python src/06_freeze.py --lineage v02g                                  # one of them
    python src/06_freeze.py --check-tau v02g_grid_fix A_eeg xgb 0 0          # reproduction check of a logged dev tau

LOCKED-TEST GUARD. This script reads development patients only (splits.json dev_folds, sub-001..096). It refuses to run
if any sub-097..125 feature file exists, checks every file it reads, and never imports or calls 95_export_results.py.

For each lineage (same configuration as its evaluated version; only the training data and the averaging differ):
  * v02g = v02g_grid_fix: offline features (data/features), alarms timestamped at the window start;
  * v11  = v11_causal:    causal features (data/features_causal), alarms at the window end (ictra_lib causal mode).
Cells frozen: A_eeg, B_eeg_ecg, C_eeg_ecg_emg_acc with xgboost (the primary family) and the four single-signal rules,
i.e. every model named in H1-H4 (docs/PREREGISTRATION_DRAFT.md).
Procedure per learned cell (seeds 0, 1, 2; the deployed score is the mean probability of the three seed models):
  1. tau: one inner patient split of the 96 dev patients (numpy default_rng(0) shuffle, first 20 % = 19 patients for
     validation, as in 04_train_ablation inner_split); three inner models (seeds 0-2) trained on the other 77 patients;
     their MEAN probability on the 19 inner-validation patients is swept over the corrected tau grid
     (1 - logspace(-1, -5, 81)) and tau = the max pooled event F1 point, k-of-n 3/5 (04_train_ablation.tune_threshold).
  2. final models: seeds 0-2 trained on all 96 dev patients (5:1 background sampling, modality dropout 0.25, same
     hyper-parameters as the evaluated versions), saved as xgboost JSON.
Rules: tau tuned the same way on the same inner split (no training). Everything is written to models/ with SHA-256s.
"""
import argparse, hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

_spec = importlib.util.spec_from_file_location("trainer", Path(__file__).resolve().parent / "04_train_ablation.py")
T = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T)

LINEAGES = {"v02g": "v02g_grid_fix", "v11": "v11_causal"}
CELLS = [("A_eeg", "xgb"), ("B_eeg_ecg", "xgb"), ("C_eeg_ecg_emg_acc", "xgb")]
SEEDS = (0, 1, 2)
OUT = L.ROOT / "models"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard(dev, locked):
    """Locked-test guard: dev and locked sets disjoint; no locked feature file on disk; nothing outside dev is read."""
    assert not set(dev) & set(locked), "dev and locked_test overlap"
    for d in ("data/features", "data/features_causal"):
        bad = [p.name for p in (L.ROOT / d).glob("*.parquet") if p.name.split("_run")[0] in set(locked)]
        if bad:
            sys.exit(f"refusing: {d} contains locked-test feature files ({bad[:3]} ...)")
    if "95_export_results" in sys.modules:
        sys.exit("refusing: 95_export_results must not be loaded here")


def checked_files(subs, dev):
    files = T.run_files(subs)
    assert all(p.name.split("_run")[0] in set(dev) for p in files), "a non-dev file was selected"
    return files


def mean_runs(run_lists):
    """Average the per-window probabilities of several models over the same runs (predict_runs tuples)."""
    out = []
    for parts in zip(*run_lists):
        sub, run, _, ev, nw, mask, grp = parts[0]
        assert all(p[0] == sub and p[1] == run for p in parts)
        out.append((sub, run, np.mean([p[2] for p in parts], axis=0), ev, nw, mask, grp))
    return out


def freeze(tag, log):
    version = LINEAGES[tag]
    causal = L.mode_for_version(version)
    T.RUNS, T.THR_GRID = "all", "tail_log"
    T.DEVICE = "cuda"
    splits = L.load_splits()
    dev = sorted(s for v in splits["dev_folds"].values() for s in v)
    guard(dev, splits["locked_test"])
    assert len(dev) == 96
    rng = np.random.default_rng(0); tr = np.array(dev); rng.shuffle(tr)
    n_in = max(2, int(0.2 * len(tr))); inner_val, inner_tr = sorted(tr[:n_in]), sorted(tr[n_in:])
    checked_files(dev, dev)
    meta = dict(lineage=tag, version=version, pipeline="causal pipeline" if causal else "offline pipeline", causal=causal,
                lib_version=L.LIB_VERSION, features_dir=str(L.features_dir().relative_to(L.ROOT)),
                post_processing=dict(k=L.CFG["post_k"], n=L.CFG["post_n"], win_s=L.CFG["win_s"], step_s=L.CFG["step_s"],
                                     merge_s=L.CFG["merge_s"], split_s=L.CFG["split_s"], tol_pre_s=L.CFG["tol_pre_s"],
                                     tol_post_s=L.CFG["tol_post_s"], alarm_timestamp="window end" if causal else "window start",
                                     mask="impd and flat-EEG windows never alarm and are removed from the FA hours"),
                tau_grid="1 - logspace(-1, -5, 81) quantiles of the inner-validation scores (04_train_ablation.thr_quantiles, tail_log)",
                training_subjects=dev, inner_validation_subjects=inner_val, inner_training_subjects=inner_tr,
                ensemble="mean probability of the seed models", seeds=list(SEEDS),
                git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=L.ROOT).stdout.strip(),
                xgboost=__import__("xgboost").__version__, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), cells={}, rules={})
    for s, m in CELLS:
        t0 = time.time()
        mods = L.FEATURE_SETS[s]
        cols = T.feature_columns(T.all_columns(), mods)
        inner = []
        for seed in SEEDS:
            D = T.load_training(inner_tr, cols, seed)
            mdl = T.predict_on_cpu(T.make_model(m, seed, D.y).fit(T.apply_dropout(D[cols], mods, np.random.default_rng(seed)), D.y), m)
            inner.append(list(T.predict_runs(mdl, inner_val, cols)))
        thr, kn, (f1_in, neg_fa), rule = T.tune_threshold(None, None, cols, runs=mean_runs(inner))
        files = {}
        for seed in SEEDS:
            D = T.load_training(dev, cols, seed)
            mdl = T.predict_on_cpu(T.make_model(m, seed, D.y).fit(T.apply_dropout(D[cols], mods, np.random.default_rng(seed)), D.y), m)
            f = OUT / f"ictra_{tag}_{s}_{m}_seed{seed}.json"
            mdl.save_model(str(f)); files[f.name] = sha256(f)
        meta["cells"][f"{s}/{m}"] = dict(feature_set=s, modalities=mods, model=m, columns=cols, thr=float(thr), k=int(kn[0]), n=int(kn[1]),
                                          inner_f1=float(f1_in), inner_fa_per_24h=float(-neg_fa), thr_rule=rule, files=files)
        print(log(f"{tag} {s}/{m}: tau {thr:.6f}, inner F1 {f1_in:.3f}, inner FA/24h {-neg_fa:.2f}, 3 models saved ({(time.time() - t0) / 60:.1f} min)"), flush=True)
    for r, col in T.RULES.items():
        mods = [col.split("_")[0]]
        cols = T.feature_columns(T.all_columns(), mods)
        thr, kn, (f1_in, neg_fa), rule = T.tune_threshold(T.Rule(col), inner_val, cols)
        meta["rules"][r] = dict(feature=col, thr=float(thr), k=int(kn[0]), n=int(kn[1]), inner_f1=float(f1_in), inner_fa_per_24h=float(-neg_fa))
        print(log(f"{tag} {r}: feature {col}, tau {thr:.4f}, inner F1 {f1_in:.3f}, inner FA/24h {-neg_fa:.2f}"), flush=True)
    mf = OUT / f"ictra_{tag}.meta.json"
    mf.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(log(f"{tag}: wrote {mf.relative_to(L.ROOT)} sha256 {sha256(mf)}"), flush=True)


def reproduce_tau(version, s, m, fold, seed):
    """Reproduction check (Allen 2026-10-26): re-run one evaluated development job's tau selection (04_train_ablation.job,
    thr_mode inner_split: inner 80/20 split of that fold's training patients, one inner model) with the SAME helpers the
    freeze uses (load_training, make_model, apply_dropout, tune_threshold, tau grid) and compare with the logged tau."""
    F = pd.read_csv(L.ROOT / "results/dev" / version / f"fold_{s}__{m}__f{fold}_s{seed}.csv")
    L.mode_for_version(version)
    T.RUNS, T.DEVICE = "all", "cuda"
    T.THR_GRID = str(F.thr_grid.iloc[0]) if "thr_grid" in F else "legacy"
    mseed = int(F.model_seed.iloc[0]) if "model_seed" in F else seed
    splits = L.load_splits()
    dev = sorted(x for v in splits["dev_folds"].values() for x in v)
    guard(dev, splits["locked_test"])
    train_subs = [x for k, v in splits["dev_folds"].items() if k != str(fold) for x in v]
    rng = np.random.default_rng(seed); tr = np.array(sorted(train_subs)); rng.shuffle(tr)
    n_in = max(2, int(0.2 * len(tr))); inner_val, inner_tr = list(tr[:n_in]), list(tr[n_in:])
    mods = L.FEATURE_SETS[s]; cols = T.feature_columns(T.all_columns(), mods)
    checked_files(inner_tr + inner_val, dev)
    D = T.load_training(inner_tr, cols, seed)
    mdl = T.predict_on_cpu(T.make_model(m, mseed, D.y).fit(T.apply_dropout(D[cols], mods, rng), D.y), m)
    thr, kn, (f1_in, _), rule = T.tune_threshold(mdl, inner_val, cols)
    logged = float(F.thr.iloc[0])
    return thr, logged, float(F.inner_f1.iloc[0]), f1_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", nargs="+", default=list(LINEAGES), choices=list(LINEAGES))
    ap.add_argument("--check-tau", nargs=5, metavar=("VERSION", "SET", "MODEL", "FOLD", "SEED"), action="append",
                    help="reproduction check only: re-select tau for one logged development job and compare; writes nothing to models/")
    a = ap.parse_args()
    if a.check_tau:
        logf = L.ROOT / "results/lab/logs/freeze_tau_check.log"
        for v, s, m, f, sd in a.check_tau:
            thr, logged, f1_log, f1_new = reproduce_tau(v, s, m, int(f), int(sd))
            ok = abs(thr - logged) < 1e-9
            line = (f"{time.strftime('%m-%d %H:%M')} {v} {s}/{m} fold {f} seed {sd}: tau re-selected {thr:.9f} vs logged {logged:.9f} "
                    f"({'IDENTICAL' if ok else 'DIFFERENT'}); inner F1 {f1_new:.6f} vs logged {f1_log:.6f}")
            print(line, flush=True)
            with open(logf, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return
    OUT.mkdir(exist_ok=True)
    logf = L.ROOT / "results/lab/logs/freeze.log"
    def log(msg):
        line = f"{time.strftime('%m-%d %H:%M')} {msg}"
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return line
    for tag in a.lineage:
        existing = list(OUT.glob(f"ictra_{tag}_*")) + list(OUT.glob(f"ictra_{tag}.meta.json"))
        if existing:
            sys.exit(f"refusing to overwrite frozen files: {[p.name for p in existing][:4]}")
        freeze(tag, log)
    print(log("FREEZE DONE"), flush=True)


if __name__ == "__main__":
    main()

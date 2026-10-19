"""Parallel training of the A / B / C ablation (and single-signal branches) on the development folds.

    python src/04_train_ablation.py                                   # sets A,B,C x models xgb,lgbm,logit,rf x 5 folds x 1 seed
    python src/04_train_ablation.py --sets A_eeg B_eeg_ecg C_eeg_ecg_emg_acc ecg_only emg_acc --models xgb lgbm --seeds 0 1 2
    python src/04_train_ablation.py --version v01_baseline --note "first run, default features"

Every job = (feature set, model family, fold, seed). Jobs run in parallel processes (--workers, default 2).
Protocol (docs/EVALUATION_PROTOCOL.md, tier 1):
  * folds come from data/catalog/splits.json (patient-wise); a fold's validation runs are never touched during fitting
  * threshold is tuned on an INNER patient split of the training folds (80/20), then the model is refit on all training
    patients with that threshold frozen; validation runs keep their natural class ratio
  * training background windows are undersampled to BG_RATIO x seizure windows; edge windows (0 < overlap < 0.5) and
    impedance-check windows are excluded from training and from window scoring
  * modality dropout: in sets with >= 2 signals, each signal block is blanked (NaN) in a random DROPOUT share of rows
  * scoring: SzCORE-style events (ictra_lib.score_recording) -> sensitivity, precision, F1, FA/24h, delay, by patient
Outputs: results/dev/<version>/scores_by_fold.csv, scores_by_patient.csv, preds/*.parquet (window probabilities),
         summary.csv (pooled over folds with patient-bootstrap 95% CI, by seizure group), and a line in results/ITERATION_LOG.md
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, json, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

warnings.filterwarnings("ignore")
FEAT = L.ROOT / "data/features"
BG_RATIO, DROPOUT = 5, 0.25
MAX_RAM_FRAC = 0.5      # training workers x per-job peak RSS must stay under half of physical RAM
DEVICE = "cpu"   # set from --device in main() and passed to every worker (xgboost only; other families are CPU-only)
RULES = {"rule_hr": "ecg_hr_rise_pct", "rule_emg": "emg_logrms_rel", "rule_eeg": "eeg_ll_max_rel", "rule_acc": "acc_logsd_rel"}
_S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")
SEIZ = {k: g[["onset_s", "duration_s", "type"]].reset_index(drop=True) for k, g in _S.groupby(["sub", "run"])}
EMPTY_EV = pd.DataFrame({"onset_s": [], "duration_s": [], "type": []})
LABEL_COLS = ["t0", "overlap", "y", "edge", "impd", "sz_id", "sz_type", "sz_group", "sub", "run", "montage", "has_ecg", "has_emg", "has_acc",
              "flat_eeg", "flat_ecg", "flat_emg"]


def lab(update, log=None, src="", version=None):
    """Lab status write, skipped for smoke versions (names starting with "_": never logged, never shown)."""
    if not str(version or SMOKE_GUARD.get("v", "")).startswith("_"):
        L.lab_status(update, log=log, src=src)


SMOKE_GUARD = {}


def feature_columns(all_cols, modalities):
    return [c for c in all_cols if any(c.startswith(L.MODALITY_PREFIX[m]) for m in modalities)]


def all_columns():
    """Union of the columns of every feature file, in first-seen order. A run without an ECG/EMG/MOV file has no columns
    for that signal, so the column list must never come from one file (B would silently lose its ECG block)."""
    import pyarrow.parquet as pq
    seen = {}
    for p in sorted(FEAT.glob("*.parquet")):
        for c in pq.read_schema(p).names:
            seen.setdefault(c, None)
    return list(seen)


def read_cols(f, cols):
    """Read the wanted columns of one feature file; columns of a signal the run does not have come back as NaN."""
    import pyarrow.parquet as pq
    have = set(pq.read_schema(f).names)
    return pd.read_parquet(f, columns=[c for c in cols if c in have]).reindex(columns=cols)


_R = pd.read_csv(L.ROOT / "data/catalog/seizeit2_runs.csv")
SEIZURE_RUNS = {f"{a}_run-{int(b):02d}.parquet" for a, b, n in zip(_R["sub"], _R.run, _R.n_seizures) if n > 0}
RUNS = "all"     # "seizure" = only runs that contain a seizure (stage 1 / v01); set from --runs in every worker


def run_files(subs):
    files = sorted(p for p in FEAT.glob("*.parquet") if p.name.split("_run")[0] in set(subs))
    return [p for p in files if p.name in SEIZURE_RUNS] if RUNS == "seizure" else files


def load_training(subs, cols, seed):
    """All seizure windows + BG_RATIO x as many background windows, sampled uniformly over the training runs."""
    rng = np.random.default_rng(seed)
    files = run_files(subs)
    counts = [(f, pd.read_parquet(f, columns=["y", "edge", "impd", "flat_eeg"])) for f in files]
    for _, c in counts:
        c["impd"] = c.impd | c.flat_eeg.fillna(False).astype(bool)
    n_sz = sum(int(c.y.sum()) for _, c in counts)
    n_bg = sum(int(((c.y == 0) & ~c.edge & ~c.impd).sum()) for _, c in counts)
    p_bg = min(1.0, BG_RATIO * n_sz / max(n_bg, 1))
    parts = []
    for f, c in counts:
        keep = (c.y == 1) | (((c.y == 0) & ~c.edge & ~c.impd) & (rng.random(len(c)) < p_bg))
        if keep.any():
            d = read_cols(f, cols + ["y", "sub"])
            parts.append(d[keep.values])
    return pd.concat(parts, ignore_index=True)


def apply_dropout(X, modalities, rng):
    X = X.copy()
    if len(modalities) < 2:
        return X
    for m in modalities:
        block = [c for c in X.columns if c.startswith(L.MODALITY_PREFIX[m])]
        mask = rng.random(len(X)) < DROPOUT
        X.loc[mask, block] = np.nan
    return X


class Rule:
    """Single-input baseline: the score is one feature (NaN -> very low)."""
    def __init__(self, col): self.col = col
    def fit(self, X, y): return self
    def predict_proba(self, X):
        v = X[self.col].fillna(-1e9).values
        return np.c_[-v, v]


def make_model(kind, seed, y):
    pos = max(int(y.sum()), 1); spw = (len(y) - pos) / pos
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=5, scale_pos_weight=spw, random_state=seed, n_jobs=1, tree_method="hist",
                                 device=DEVICE)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, min_child_samples=20, scale_pos_weight=spw, random_state=seed, n_jobs=1, verbose=-1)
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median", add_indicator=True)
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=seed))
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return make_pipeline(imp, RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=1, random_state=seed))
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(imp, StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300, early_stopping=True, random_state=seed))
    if kind in RULES:
        return Rule(RULES[kind])
    raise ValueError(kind)


def predict_on_cpu(m, kind):
    """xgboost trained on CUDA warns "Falling back to prediction using DMatrix due to mismatched devices" when it predicts
    from pandas (CPU) data. Training stays on the GPU; prediction is switched to CPU (same trees, same outputs)."""
    if kind == "xgb" and DEVICE == "cuda":
        m.get_booster().set_param({"device": "cpu"})
    return m


def predict_runs(model, subs, cols):
    """Yield (sub, run, prob per window, events, nw, impd, group per window) for every run of these patients."""
    for f in run_files(subs):
        d = pd.read_parquet(f)
        prob = model.predict_proba(d.reindex(columns=cols))[:, 1]
        # reference events = the exact annotations from the catalog (not rebuilt from window overlaps, which can hide a
        # short seizure that overlaps a longer one)
        sub, run = d["sub"].iloc[0], int(d["run"].iloc[0])
        ev = SEIZ.get((sub, run), EMPTY_EV)
        mask = d.impd.values | d.get("flat_eeg", pd.Series(False, index=d.index)).fillna(False).astype(bool).values
        yield sub, run, prob, ev, len(d), mask, d.sz_group.values


DEFAULT_KOFN = ((L.CFG["post_k"], L.CFG["post_n"]),)


def score_run(pos, ev, nw, mask, kn):
    """ictra_lib.score_recording with an explicit k-of-n (v08). For the default (3, 5) it IS score_recording; otherwise
    the same chain from ictra_lib pieces: mask -> k_of_n(k, n) -> windows_to_events -> score_events (masked hours)."""
    if tuple(kn) == DEFAULT_KOFN[0]:
        return L.score_recording(pos, ev, nw, mask=mask)
    pos = pos & ~mask
    hyp = L.windows_to_events(L.k_of_n(pos, int(kn[0]), int(kn[1])))
    ref = [(e.onset_s, e.onset_s + e.duration_s) for e in ev.itertuples() if str(e.type).startswith("sz")]
    return L.score_events(ref, hyp, hours=(nw - int(mask.sum())) * L.CFG["step_s"] / 3600), hyp


THR_GRID = "tail_log"   # set from --thr-grid in every worker


def thr_quantiles():
    """Quantiles of the inner predictions tried as tau.
    legacy (v01-v03b): 40 quantiles linear in q from 0.90 to 0.9999; the top two are q = 0.99734 and 0.9999, a 27x jump in
    alarm rate with nothing between, which pinned tau (ITERATION_LOG v03b quantile check).
    tail_log (defect correction 2026-10-19, EVALUATION_PROTOCOL amendment): log-spaced in the tail probability 1 - q,
    because the false-alarm rate is proportional to it; 0.1 -> 1e-5, 20 per decade (81 candidates). The upper end keeps the
    legacy grid's 0.90 start (rules picked tails 0.008-0.036 in v03b)."""
    if THR_GRID == "legacy":
        return np.linspace(0.90, 0.9999, 40)
    return 1 - np.logspace(-1, -5, 81)


def tune_threshold(model, subs, cols, runs=None, kofn=DEFAULT_KOFN, objective=None):
    """Sweep thresholds (and, for v08, the k-of-n candidates) on inner-validation patients.
    objective None / max_f1: maximise pooled event F1 (ties -> fewer false alarms).
    objective dict(kind="f1_sens_floor", s_floor, fa_ref) (v03b, ITERATION_LOG 2026-10-17): maximise F1 subject to inner
    sensitivity >= s_floor; if infeasible, maximise sensitivity subject to FA/24h <= fa_ref; if that is infeasible too,
    the lowest-FA candidate. runs: precomputed predict_runs output (inner_cv pools several inner models).
    Returns (thr, (k, n), (f1, -fa), rule_used)."""
    runs = runs if runs is not None else list(predict_runs(model, subs, cols))
    allp = np.concatenate([r[2] for r in runs])
    cands = np.unique(np.quantile(allp[np.isfinite(allp)], thr_quantiles()))
    res = []
    for kn in kofn:
        for thr in cands:
            per = [score_run(p > thr, ev, nw, impd, kn)[0] for _, _, p, ev, nw, impd, _ in runs]
            q = L.pool_scores(per)
            res.append((float(thr), tuple(kn), float(np.nan_to_num(q["f1"])), float(q["fa_per_24h"]), float(np.nan_to_num(q["sensitivity"]))))
    rule = "max_f1"
    if not objective or objective.get("kind", "max_f1") == "max_f1":
        pick = max(res, key=lambda r: (r[2], -r[3]))
    else:
        ok = [r for r in res if r[4] >= objective["s_floor"]]
        if ok:
            pick, rule = max(ok, key=lambda r: (r[2], -r[3])), "f1_at_sens_floor"
        else:
            ok = [r for r in res if r[3] <= objective["fa_ref"]]
            pick, rule = (max(ok, key=lambda r: (r[4], -r[3])), "max_sens_at_ref_fa") if ok else (min(res, key=lambda r: r[3]), "lowest_fa")
    return pick[0], pick[1], (pick[2], -pick[3]), rule


def monitor(m, kind, val_subs, cols, out, stem, event_f1, seed, n_checkpoints=40, max_windows=1_000_000):
    """Training curves and gain importance for the website export (docs/DELIVERABLES_SPEC.md). Computed AFTER the final
    fit, on the held-out fold, for display only: nothing here feeds back into the model, its threshold or the version
    decision. Windows: held-out fold, edge / impd / flat-EEG excluded, natural class ratio (uniform subsample to
    max_windows if larger, which keeps the ratio). Writes curve_<stem>.csv and importance_<stem>.csv."""
    from sklearn.metrics import log_loss, average_precision_score
    parts = []
    for f in run_files(val_subs):
        d = read_cols(f, cols + ["y", "edge", "impd", "flat_eeg"])
        keep = ~d.edge.astype(bool) & ~d.impd.astype(bool) & ~d.flat_eeg.fillna(False).astype(bool)
        parts.append(d[keep.values])
    V = pd.concat(parts, ignore_index=True)
    if len(V) > max_windows:
        V = V.sample(max_windows, random_state=seed)
    n_iter = m.get_params()["n_estimators"]
    its = sorted(set(np.linspace(1, n_iter, n_checkpoints).astype(int)))
    rows, raw, prev = [], np.zeros(len(V)), 0
    X = V[cols]
    for k in its:
        if kind == "xgb":
            p = m.predict_proba(X, iteration_range=(0, k))[:, 1]
        else:   # LightGBM: add the raw score of trees prev..k only, so every tree is evaluated once per window
            raw += m.booster_.predict(X, start_iteration=prev, num_iteration=k - prev, raw_score=True); prev = k
            p = 1 / (1 + np.exp(-raw))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        rows.append(dict(iteration=k, val_logloss=log_loss(V.y, p, labels=[0, 1]),
                         val_auprc=average_precision_score(V.y, p) if V.y.sum() else np.nan, val_event_f1=np.nan,
                         n_windows=len(V), n_seizure_windows=int(V.y.sum())))
    rows[-1]["val_event_f1"] = event_f1                      # event F1 only at the end of the fold (threshold + k-of-n)
    pd.DataFrame(rows).to_csv(out / f"curve_{stem}.csv", index=False)
    if kind == "xgb":
        g = m.get_booster().get_score(importance_type="gain")
    else:
        g = dict(zip(m.booster_.feature_name(), m.booster_.feature_importance("gain")))
    pd.DataFrame({"feature": cols, "gain": [float(g.get(c, 0.0)) for c in cols]}).to_csv(out / f"importance_{stem}.csv", index=False)


THR_MODES = ("inner_split", "inner_cv")
N_INNER = 5


def job(args):
    fs_name, kind, fold, seed, version, device, runs, thr_mode, kofn, draws, thr_obj, thr_grid = args
    # draws (protocol 3 x 5): `seed` is then a repeat id r = model_seed * 10 + draw; the model seed is r // 10 and the
    # background sample / inner split use r, so 3 model seeds x 5 background draws = 15 scored repeats
    mseed = seed // 10 if draws else seed
    global DEVICE, RUNS, THR_GRID
    DEVICE, RUNS, THR_GRID = device, runs, thr_grid
    SMOKE_GUARD["v"] = version
    t = time.time()
    key = f"{fs_name}|{kind}|{fold}|{seed}"
    lab({"training": {version: {"jobs": {key: {"state": "running", "start": pd.Timestamp.now().isoformat(timespec="seconds")}}}}})
    splits = L.load_splits(); rng = np.random.default_rng(seed)
    val_subs = splits["dev_folds"][str(fold)]
    train_subs = [s for k, v in splits["dev_folds"].items() if k != str(fold) for s in v]
    mods = L.FEATURE_SETS[fs_name] if fs_name in L.FEATURE_SETS else [RULES[kind].split("_")[0]]
    cols = feature_columns(all_columns(), mods)
    tr = np.array(sorted(train_subs)); rng.shuffle(tr)
    if thr_mode == "inner_split":
        # v01: one inner split by patient (80/20) for the threshold
        n_in = max(2, int(0.2 * len(tr))); inner_val, inner_tr = list(tr[:n_in]), list(tr[n_in:])
        D = load_training(inner_tr, cols, seed)
        m = predict_on_cpu(make_model(kind, mseed, D.y).fit(apply_dropout(D[cols], mods, rng), D.y), kind)
        thr, kn, (f1_in, _), thr_rule = tune_threshold(m, inner_val, cols, kofn=kofn, objective=thr_obj)
    else:
        # v03: inner leave-patients-out CV over ALL training patients (N_INNER groups, each inner model trained on the
        # other 80%); the out-of-fold inner predictions of every training patient are pooled and ONE threshold is swept
        # over them, so tau no longer depends on which 20% of patients one seed happened to hold out.
        groups = np.array_split(tr, N_INNER); pooled = []
        for k in range(N_INNER):
            inner_val = list(groups[k]); inner_tr = [x for j, g in enumerate(groups) if j != k for x in g]
            D = load_training(inner_tr, cols, seed)
            m = predict_on_cpu(make_model(kind, mseed, D.y).fit(apply_dropout(D[cols], mods, rng), D.y), kind)
            pooled += list(predict_runs(m, inner_val, cols))
        thr, kn, (f1_in, _), thr_rule = tune_threshold(None, None, cols, runs=pooled, kofn=kofn, objective=thr_obj)
    D = load_training(train_subs, cols, seed)
    m = predict_on_cpu(make_model(kind, mseed, D.y).fit(apply_dropout(D[cols], mods, rng), D.y), kind)
    # validation on the held-out fold, full natural class ratio
    rows, preds = [], []
    for sub, run, prob, ev, nw, impd, grp in predict_runs(m, val_subs, cols):
        s, hyp = score_run(prob > thr, ev, nw, impd, kn)
        s.update(sub=sub, run=run, fold=fold, seed=seed, model=kind, set=fs_name, thr=thr, post_k=kn[0], post_n=kn[1], model_seed=mseed)
        rows.append(s)
        preds.append(pd.DataFrame({"sub": sub, "run": run, "t0": np.arange(nw) * L.CFG["step_s"], "prob": prob.astype(np.float32)}))
    out = L.ROOT / "results/dev" / version
    (out / "preds").mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_parquet(out / "preds" / f"{fs_name}__{kind}__f{fold}_s{seed}.parquet", index=False)
    if kind in ("xgb", "lgbm"):
        monitor(m, kind, val_subs, cols, out, f"{fs_name}__{kind}__f{fold}_s{seed}", L.pool_scores(rows)["f1"], seed)
    res = pd.DataFrame(rows); res["inner_f1"] = f1_in; res["thr_mode"] = thr_mode; res["thr_rule"] = thr_rule; res["thr_grid"] = thr_grid
    res["minutes"] = (time.time() - t) / 60
    import psutil
    mi = psutil.Process().memory_info()
    peak_gb = getattr(mi, "peak_wset", mi.rss) / 1e9        # Windows: peak working set of this worker process so far
    res["peak_rss_gb"] = peak_gb
    res.to_csv(out / f"fold_{fs_name}__{kind}__f{fold}_s{seed}.csv", index=False)
    pooled = L.pool_scores(rows)
    info = {"state": "done", "peak_rss_gb": peak_gb, "sensitivity": pooled["sensitivity"], "fa_per_24h": pooled["fa_per_24h"], "f1": pooled["f1"],
            "thr": thr, "minutes": (time.time() - t) / 60}
    line = f"{fs_name:18s} {kind:9s} fold {fold} seed {seed}: sens {pooled['sensitivity']:.2f} FA/24h {pooled['fa_per_24h']:.2f} F1 {pooled['f1']:.2f} thr {thr:.3f} ({(time.time() - t) / 60:.1f} min)"
    return line, key, info


def seizure_group_scores(version):
    """Re-score every saved prediction by seizure group (sensitivity per group; FA/24h is group-independent)."""
    out = L.ROOT / "results/dev" / version
    S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")
    rows, per_sz = [], []
    for p in sorted((out / "preds").glob("*.parquet")):
        fs_name, kind, fs = p.stem.split("__"); fold, seed = int(fs[1]), int(fs.split("_s")[1])
        fc = pd.read_csv(out / f"fold_{p.stem}.csv"); thr = fc.thr.iloc[0]
        kn = (int(fc.post_k.iloc[0]), int(fc.post_n.iloc[0])) if "post_k" in fc else DEFAULT_KOFN[0]
        P = pd.read_parquet(p)
        for (sub, run), d in P.groupby(["sub", "run"]):
            q = pd.read_parquet(FEAT / f"{sub}_run-{run:02d}.parquet", columns=["impd", "flat_eeg"])
            impd = q.impd.values | q.flat_eeg.fillna(False).astype(bool).values
            alarm = L.k_of_n((d.prob.values > thr) & ~impd, *kn); hyp = L.windows_to_events(alarm)
            ev = S[(S["sub"] == sub) & (S.run == run)]
            for g, G in ev.groupby("group"):
                ref = list(zip(G.onset_s, G.onset_s + G.duration_s))
                sc = L.score_events(ref, hyp, (len(d) - int(impd.sum())) * L.CFG["step_s"] / 3600)
                rows.append(dict(set=fs_name, model=kind, fold=fold, seed=seed, sub=sub, run=run, group=g, n_ref=sc["n_ref"], tp=sc["tp"], median_delay_s=sc["median_delay_s"]))
            # one row per seizure: hit or not, and detection delay (first overlapping alarm start - annotated onset), with
            # the same SzCORE overlap rule as score_events; the delay relative to the first bte-EEG change is added in summarise
            for e in ev.itertuples():
                a, b = e.onset_s, e.onset_s + e.duration_s
                hits = [hs for hs, he in hyp if hs < b + L.CFG["tol_post_s"] and he > a - L.CFG["tol_pre_s"]]
                per_sz.append(dict(set=fs_name, model=kind, fold=fold, seed=seed, sub=sub, run=int(run), onset_s=a, group=e.group,
                                   seizure_id=f"{sub}_r{int(run):02d}_{int(a)}", hit=bool(hits), delay_s=(min(hits) - a) if hits else np.nan))
    G = pd.DataFrame(rows)
    if len(G):
        G.to_csv(out / "scores_by_group.csv", index=False)
    pd.DataFrame(per_sz).to_csv(out / "detections_by_seizure.csv", index=False)
    return G


def summarise(version, note="", log=True):
    out = L.ROOT / "results/dev" / version
    F = pd.concat([pd.read_csv(p) for p in out.glob("fold_*.csv")], ignore_index=True)
    F.to_csv(out / "scores_by_fold.csv", index=False)
    pat = F.groupby(["set", "model", "seed", "sub"]).agg(n_ref=("n_ref", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), n_hyp=("n_hyp", "sum"),
                                                        hours=("hours", "sum"), median_delay_s=("median_delay_s", "median")).reset_index()
    pat.to_csv(out / "scores_by_patient.csv", index=False)
    rows = []
    for (s, m), d in pat.groupby(["set", "model"]):
        per_seed = []
        for seed, ds in d.groupby("seed"):
            pooled = L.pool_scores(ds.to_dict("records")); per_seed.append(pooled)
        pooled = pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()
        d0 = d[d.seed == d.seed.min()].to_dict("records")
        lo_s, hi_s = L.patient_bootstrap(d0, "sensitivity"); lo_f, hi_f = L.patient_bootstrap(d0, "fa_per_24h"); lo_1, hi_1 = L.patient_bootstrap(d0, "f1")
        rows.append(dict(set=s, model=m, n_seeds=d.seed.nunique(), n_patients=d["sub"].nunique(), hours=pooled["hours"], seizures=pooled["n_ref"],
                         sensitivity=pooled["sensitivity"], sens_ci=f"[{lo_s:.2f}, {hi_s:.2f}]", fa_per_24h=pooled["fa_per_24h"], fa_ci=f"[{lo_f:.2f}, {hi_f:.2f}]",
                         precision=pooled["precision"], f1=pooled["f1"], f1_ci=f"[{lo_1:.2f}, {hi_1:.2f}]", median_delay_s=pooled["median_delay_s"],
                         sens_lo=lo_s, sens_hi=hi_s, fa_lo=lo_f, fa_hi=hi_f, f1_lo=lo_1, f1_hi=hi_1))
    T = pd.DataFrame(rows).sort_values(["set", "f1"], ascending=[True, False])
    G = seizure_group_scores(version)
    if len(G):
        gs = G.groupby(["set", "model", "group"]).agg(n=("n_ref", "sum"), tp=("tp", "sum")).reset_index()
        gs["sensitivity"] = gs.tp / gs.n
        piv = gs.pivot_table(index=["set", "model"], columns="group", values="sensitivity")
        T = T.merge(piv.add_prefix("sens_").reset_index(), on=["set", "model"], how="left")
    T.to_csv(out / "summary.csv", index=False)
    M = montage_scores(F); M.to_csv(out / "summary_by_montage.csv", index=False)
    Dl = delay_table(out); Dl.to_csv(out / "delay_by_group.csv", index=False)
    if not version.startswith("_"):
        write_dev_json(version, note, T, pat, G, M, Dl)
    if not log:
        print(T.round(3).to_string()); return T
    with open(L.ROOT / "results/ITERATION_LOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## {version} ({pd.Timestamp.now():%Y-%m-%d %H:%M}) {note}\n\nDev folds, pooled over patients, patient-bootstrap 95% CI.\n\n")
        f.write(T[["set", "model", "seizures", "hours", "sensitivity", "sens_ci", "fa_per_24h", "fa_ci", "f1", "f1_ci", "median_delay_s"]].round(3).to_markdown(index=False) + "\n")
        f.write("\nOffline pipeline (EVALUATION_PROTOCOL.md amendment 2026-10-17).\n")
        if "peak_rss_gb" in F:
            pk = F.groupby(["set", "model"]).peak_rss_gb.max().reset_index()
            f.write("Peak RSS per job (GB; max over folds and seeds of the worker's peak working set): "
                    + ", ".join(f"{r.set}/{r.model} {r.peak_rss_gb:.2f}" for r in pk.itertuples())
                    + f". Overall max {F.peak_rss_gb.max():.2f} GB.\n")
    print(T.round(3).to_string())
    return T


def montage_scores(F):
    """Sensitivity and FA/24h per (set, model, EEG montage): counts pooled over patients per seed, then the mean over seeds."""
    R = pd.read_csv(L.ROOT / "data/catalog/seizeit2_runs.csv")[["sub", "run", "eeg_montage"]]
    F = F.merge(R, on=["sub", "run"], how="left")
    rows = []
    for (s_, m_, mo), d in F.groupby(["set", "model", "eeg_montage"]):
        per = [L.pool_scores(ds.to_dict("records")) for _, ds in d.groupby("seed")]
        P = pd.DataFrame(per).mean(numeric_only=True)
        rows.append(dict(set=s_, model=m_, montage=mo, n_patients=d["sub"].nunique(), seizures=P.n_ref, hours=P.hours,
                         sensitivity=P.sensitivity, fa_per_24h=P.fa_per_24h, f1=P.f1))
    return pd.DataFrame(rows)


def delay_table(out):
    """Detection delay per (set, model, group), twice: from the annotated onset, and from the first behind-the-ear EEG
    change found by Exp R (results/expR/onsets.csv, t_eeg; z > 3 for 3 s). Hits only, all seeds pooled."""
    D = pd.read_csv(out / "detections_by_seizure.csv")
    on = L.ROOT / "results/expR/onsets.csv"
    if on.exists():
        O = pd.read_csv(on)[["seizure_id", "t_eeg"]]
        D = D.merge(O, on="seizure_id", how="left")
        D["delay_vs_bte_s"] = D.delay_s - D.t_eeg
    else:
        D["t_eeg"] = np.nan; D["delay_vs_bte_s"] = np.nan
    rows = []
    for (s_, m_), d in D.groupby(["set", "model"]):
        for g, dg in list(d.groupby("group")) + [("all", d)]:
            h = dg[dg.hit]
            q = lambda v: (v.median(), v.quantile(.25), v.quantile(.75), int(v.notna().sum()))
            a, b = q(h.delay_s.dropna()), q(h.delay_vs_bte_s.dropna())
            rows.append(dict(set=s_, model=m_, group=g, n_hits=int(len(h)), delay_annot_median=a[0], delay_annot_q25=a[1], delay_annot_q75=a[2],
                             delay_bte_median=b[0], delay_bte_q25=b[1], delay_bte_q75=b[2], n_with_bte_change=b[3]))
    return pd.DataFrame(rows)


def write_dev_json(version, note, T, pat, G, M=None, Dl=None):
    """results/lab/data/dev_<version>.json for the Lab page: raw numbers only (formatting happens in the page)."""
    pp = pat.groupby(["set", "model", "sub"])[["n_ref", "tp", "fp", "hours"]].sum().reset_index()   # summed over seeds
    pp["sensitivity"] = pp.tp / pp.n_ref.where(pp.n_ref > 0); pp["fa_per_24h"] = pp.fp / pp.hours * 24
    gs = []
    if len(G):
        g = G.groupby(["set", "model", "group"]).agg(n=("n_ref", "sum"), tp=("tp", "sum")).reset_index()
        g["sensitivity"] = g.tp / g.n; gs = g.to_dict("records")
    p = L.LAB_DATA / f"dev_{version}.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(version=version, note=note, written=pd.Timestamp.now().isoformat(timespec="seconds"),
               ci_note="95% patient-bootstrap CIs (1,000 resamples) from seed 0; point estimates are the mean over seeds",
               summary=T.drop(columns=[c for c in T.columns if c.endswith("_ci")]).to_dict("records"),
               per_patient=pp.to_dict("records"), by_group=gs,
               by_montage=M.to_dict("records") if M is not None else [], delays=Dl.to_dict("records") if Dl is not None else [],
               named_subgroups=NAMED_SUBGROUPS)
    L.lab_write_json(f"dev_{version}.json", L.json_clean(old))


# Allen 2026-10-17: shown as a named subgroup in the per-patient figures (v01 NOTES: 194 of 702 dev seizures, median 10-13 s)
NAMED_SUBGROUPS = {"short-seizure trio": dict(subjects=["sub-039", "sub-047", "sub-035"],
                                              note="sub-039/047/035, 194 of 702 dev seizures, median duration 10-13 s")}


RULE_1 = dict(min_df1=0.01, min_p=0.90, max_dsens=-0.02, max_dfa=0.5)    # docs/EVALUATION_PROTOCOL.md section 1


def compare(ref, new, set_ref="C_eeg_ecg_emg_acc", model_ref="xgb", set_new=None, model_new=None, n=1000, seed=0):
    """Paired patient bootstrap of version `new` against the accepted version `ref` (SlopeAI 09_iterate.py compare).

    Per patient the counts (seizures, hits, false alarms, alarms, hours) are summed over seeds, so each patient enters
    once; both versions are resampled with the SAME patient indices. Point estimates are the pooled rates over all
    patients. Decision: ACCEPT only if dF1 >= +0.01, P(dF1 > 0) >= 0.90, dSens >= -0.02 and dFA/24h <= +0.5."""
    set_new, model_new = set_new or set_ref, model_new or model_ref
    def per_patient(v, st, md):
        P = pd.read_csv(L.ROOT / "results/dev" / v / "scores_by_patient.csv")
        P = P[(P.set == st) & (P.model == md)]
        if P.empty:
            sys.exit(f"{v}: no rows for set {st} model {md}")
        return P.groupby("sub")[["n_ref", "tp", "fp", "n_hyp", "hours"]].sum()
    A, B = per_patient(ref, set_ref, model_ref), per_patient(new, set_new, model_new)
    subs = sorted(set(A.index) & set(B.index))
    if len(subs) < len(set(A.index) | set(B.index)):
        print(f"WARNING: {len(set(A.index) ^ set(B.index))} patients are in only one version; comparing the {len(subs)} shared ones")
    A, B = A.loc[subs], B.loc[subs]
    def rates(X):
        n_ref, tp, fp, n_hyp, h = X.sum()
        sens = tp / n_ref; prec = (n_hyp - fp) / n_hyp if n_hyp else np.nan
        f1 = 2 * sens * prec / (sens + prec) if n_hyp and (sens + prec) > 0 else 0.0
        return np.array([sens, fp / h * 24, f1])
    ra, rb = rates(A), rates(B); d = rb - ra
    rng = np.random.default_rng(seed); Av, Bv = A.values, B.values; boots = []
    for _ in range(n):
        idx = rng.integers(0, len(subs), len(subs))
        boots.append(rates(pd.DataFrame(Bv[idx], columns=A.columns)) - rates(pd.DataFrame(Av[idx], columns=A.columns)))
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)
    p_improve = float(np.mean(boots[:, 2] > 0))
    checks = {"dF1 >= +0.01": d[2] >= RULE_1["min_df1"], "P(dF1>0) >= 0.90": p_improve >= RULE_1["min_p"],
              "dSens >= -0.02": d[0] >= RULE_1["max_dsens"], "dFA/24h <= +0.5": d[1] <= RULE_1["max_dfa"]}
    decision = "ACCEPT" if all(checks.values()) else "REJECT"
    res = dict(ref=f"{ref} {set_ref} {model_ref}", new=f"{new} {set_new} {model_new}", n_patients=len(subs),
               sens_ref=ra[0], sens_new=rb[0], d_sens=d[0], d_sens_ci=f"[{lo[0]:+.3f}, {hi[0]:+.3f}]",
               fa_ref=ra[1], fa_new=rb[1], d_fa=d[1], d_fa_ci=f"[{lo[1]:+.2f}, {hi[1]:+.2f}]",
               f1_ref=ra[2], f1_new=rb[2], d_f1=d[2], d_f1_ci=f"[{lo[2]:+.3f}, {hi[2]:+.3f}]", p_improve=p_improve,
               failed=", ".join(k for k, v in checks.items() if not v) or "none", decision=decision)
    for k, v in res.items():
        print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")
    # fold by fold (display; the decision above is the pooled paired bootstrap over all patients)
    folds = L.load_splits()["dev_folds"]; fr = []
    for f, fsubs in sorted(folds.items()):
        ss = [x for x in subs if x in set(fsubs)]
        if ss:
            a_, b_ = rates(A.loc[ss]), rates(B.loc[ss])
            fr.append(dict(fold=int(f), n_patients=len(ss), sens_ref=a_[0], sens_new=b_[0], fa_ref=a_[1], fa_new=b_[1],
                           f1_ref=a_[2], f1_new=b_[2], d_f1=b_[2] - a_[2]))
    FR = pd.DataFrame(fr)
    # spread of the per-seed thresholds per fold (the quantity v03 is meant to shrink): sample SD and IQR over seeds
    def taus(v, st, md):
        T = []
        for q in (L.ROOT / "results/dev" / v).glob(f"fold_{st}__{md}__f*_s*.csv"):
            f_, sd_ = q.stem.split("__")[2][1:].split("_s")
            T.append(dict(fold=int(f_), seed=int(sd_), thr=float(pd.read_csv(q, usecols=["thr"]).thr.iloc[0])))
        T = pd.DataFrame(T)
        return T.groupby("fold").thr.agg(list=lambda x: " ".join(f"{t:.4f}" for t in x), sd=lambda x: x.std(ddof=1),
                                         iqr=lambda x: x.quantile(.75) - x.quantile(.25), n_seeds="size")
    if len(FR):
        for tag, (v, st, md) in [("ref", (ref, set_ref, model_ref)), ("new", (new, set_new, model_new))]:
            FR = FR.merge(taus(v, st, md).add_prefix(f"tau_").add_suffix(f"_{tag}").reset_index(), on="fold", how="left")
    print(FR.round(4).to_string(index=False))
    out = L.ROOT / "results/dev" / new
    stem = f"compare_vs_{ref}" if (ref, set_ref, model_ref) != (new, set_new, model_new) and ref != new else \
        f"compare_{set_new}_{model_new}_vs_{set_ref}_{model_ref}"
    pd.DataFrame([res]).to_csv(out / f"{stem}.csv", index=False)
    FR.to_csv(out / f"{stem}_by_fold.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["A_eeg", "B_eeg_ecg", "C_eeg_ecg_emg_acc"])
    ap.add_argument("--models", nargs="+", default=["xgb", "lgbm", "logit", "rf"])
    ap.add_argument("--rules", action="store_true", help="also run the single-input rules (rule_hr, rule_emg, rule_eeg, rule_acc)")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--version", default="v01_baseline")
    ap.add_argument("--note", default="")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--runs", choices=["seizure", "all"],
                    help="seizure = only seizure-containing runs (v01, stage 1); all = every feature file (v02 on)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="xgboost device (same model either way)")
    ap.add_argument("--thr-objective", default="max_f1", choices=["max_f1", "f1_sens_floor"],
                    help="f1_sens_floor (v03b): max inner F1 s.t. inner sensitivity >= S_ref - 0.02, S_ref/FA_ref per "
                         "(set, model) from --thr-ref's summary.csv; fallback max sensitivity at FA <= FA_ref")
    ap.add_argument("--thr-ref", default="v01_baseline", help="reference version for --thr-objective f1_sens_floor")
    ap.add_argument("--thr-ref-matched", action="store_true",
                    help="S_ref = the reference's matched-FA sensitivity: best held-out sensitivity on its operating curve "
                         "at FA/24h <= its deployable FA (seed mean), per (set, model) (protocol amendment 2026-10-17 evening)")
    ap.add_argument("--kofn-grid", default="", help='v08: k-of-n candidates tuned with tau on the inner split, e.g. "2/3,3/5,4/6" '
                    "(default: ictra_lib CFG 3/5 only)")
    ap.add_argument("--bg-draws", type=int, default=1, help="protocol 3 x 5: background-sampling draws per seed")
    ap.add_argument("--bg-draws-cells", nargs="*", default=[], help="cells that get --bg-draws, e.g. C_eeg_ecg_emg_acc:xgb (default: all)")
    ap.add_argument("--thr-mode", default="inner_split", choices=THR_MODES,
                    help="inner_split: one 80/20 inner patient split (v01); inner_cv: pooled out-of-fold predictions of a "
                         f"{N_INNER}-fold inner leave-patients-out CV (v03)")
    ap.add_argument("--thr-grid", default="tail_log", choices=["tail_log", "legacy"],
                    help="tau candidates: tail_log = 1 - logspace(-1, -5, 81) (defect correction 2026-10-19); legacy = "
                         "linspace(0.90, 0.9999, 40), only to reproduce v01-v03b")
    ap.add_argument("--no-log", action="store_true", help="smoke runs: do not append to results/ITERATION_LOG.md")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "NEW"), help="paired bootstrap NEW vs accepted REF, then exit")
    ap.add_argument("--compare-sets", nargs=2, metavar=("SET_REF", "SET_NEW"), default=["C_eeg_ecg_emg_acc", "C_eeg_ecg_emg_acc"])
    ap.add_argument("--compare-models", nargs=2, metavar=("MODEL_REF", "MODEL_NEW"), default=["xgb", "xgb"])
    a = ap.parse_args()
    SMOKE_GUARD["v"] = a.version
    if a.compare:
        compare(a.compare[0], a.compare[1], a.compare_sets[0], a.compare_models[0], a.compare_sets[1], a.compare_models[1]); return
    if a.summary_only:
        summarise(a.version, a.note, log=not a.no_log); return
    if a.runs is None:
        ap.error("--runs seizure|all is required for training")
    # memory cap (Allen, 2026-10-17): the training pool may reserve at most MAX_RAM_FRAC of physical RAM. The per-job
    # peak comes from earlier fold CSVs (peak_rss_gb, +25 % margin), else a conservative 2 GB.
    import psutil
    peaks = []
    for p in (L.ROOT / "results/dev").glob("*/fold_*.csv"):
        c = pd.read_csv(p)
        if "peak_rss_gb" in c:
            peaks.append(float(c.peak_rss_gb.max()))
    per_job = max(0.5, 1.25 * max(peaks)) if peaks else 2.0
    cap = max(1, int(MAX_RAM_FRAC * psutil.virtual_memory().total / 1e9 / per_job))
    if a.workers > cap:
        print(f"workers {a.workers} -> {cap}: {per_job:.1f} GB per job x {cap} <= {MAX_RAM_FRAC:.0%} of physical RAM")
        a.workers = cap
    kofn = tuple(tuple(int(x) for x in g.split("/")) for g in a.kofn_grid.split(",")) if a.kofn_grid else DEFAULT_KOFN

    def reps(s, m):
        """Repeat ids: plain seeds, or seed * 10 + draw for the cells that get background draws."""
        if a.bg_draws > 1 and (not a.bg_draws_cells or f"{s}:{m}" in a.bg_draws_cells):
            return [(sd * 10 + d, True) for sd in a.seeds for d in range(a.bg_draws)]
        return [(sd, False) for sd in a.seeds]
    REF = pd.read_csv(L.ROOT / "results/dev" / a.thr_ref / "summary.csv").set_index(["set", "model"]) if a.thr_objective != "max_f1" else None

    CUR = pd.read_csv(L.ROOT / "results/dev" / a.thr_ref / "operating_curve.csv") if (REF is not None and a.thr_ref_matched) else None

    def obj(s, m):
        if REF is None:
            return None
        r = REF.loc[(s, m)]; fa_ref = float(r.fa_per_24h); s_ref = float(r.sensitivity)
        if CUR is not None:                    # matched-FA sensitivity of the reference at its own deployable FA
            d = CUR[(CUR.set == s) & (CUR.model == m)]
            v = [ds.loc[ds.fa_per_24h <= fa_ref, "sensitivity"].max() for _, ds in d.groupby("seed")]
            s_ref = float(np.mean([x if np.isfinite(x) else 0.0 for x in v]))
        print(f"  tau objective {s}/{m}: inner sensitivity >= {s_ref - 0.02:.4f} (S_ref {s_ref:.4f}), fallback FA <= {fa_ref:.3f}")
        return dict(kind=a.thr_objective, s_floor=s_ref - 0.02, fa_ref=fa_ref)
    jobs = [(s, m, f, r, a.version, a.device, a.runs, a.thr_mode, kofn, dr, obj(s, m), a.thr_grid) for s in a.sets for m in a.models for f in a.folds for r, dr in reps(s, m)]
    if a.rules:
        jobs += [(f"rule", r, f, 0, a.version, a.device, a.runs, a.thr_mode, kofn, False, obj("rule", r), a.thr_grid) for r in RULES for f in a.folds]
    done = {p.stem for p in (L.ROOT / "results/dev" / a.version).glob("fold_*.csv")}
    jobs = [j for j in jobs if f"fold_{j[0]}__{j[1]}__f{j[2]}_s{j[3]}" not in done]
    print(f"{len(jobs)} jobs, {a.workers} workers")
    cells = {}
    for st in done:                                                       # resumed: jobs already finished earlier
        fs_, kd_, rest = st[len("fold_"):].split("__"); fo_, sd_ = rest[1:].split("_s")
        cells[f"{fs_}|{kd_}|{fo_}|{sd_}"] = {"state": "done"}
    for j in jobs:
        cells[f"{j[0]}|{j[1]}|{j[2]}|{j[3]}"] = {"state": "queued", "set": j[0], "model": j[1], "fold": j[2], "seed": j[3]}
    lab({"training": {a.version: {"note": a.note, "runs": a.runs, "device": a.device, "workers": a.workers, "thr_mode": a.thr_mode, "kofn_grid": a.kofn_grid or "3/5", "thr_objective": a.thr_objective, "thr_grid": a.thr_grid, "bg_draws": a.bg_draws, "bg_draws_cells": a.bg_draws_cells,
                                           "n_jobs": len(cells), "started": pd.Timestamp.now().isoformat(timespec="seconds"), "jobs": cells}}},
                 log=f"{a.version}: {len(jobs)} jobs queued ({a.workers} workers, xgboost on {a.device})", src="04_train")
    with ProcessPoolExecutor(a.workers) as ex:
        for line, key, info in ex.map(job, jobs):
            print(line, flush=True)
            lab({"training": {a.version: {"jobs": {key: info}}}})
    lab({}, log=f"{a.version}: all jobs done, summarising", src="04_train")
    summarise(a.version, a.note, log=not a.no_log)
    lab({"training": {a.version: {"finished": pd.Timestamp.now().isoformat(timespec="seconds")}}},
                 log=f"{a.version}: summary.csv written", src="04_train")


if __name__ == "__main__":
    main()

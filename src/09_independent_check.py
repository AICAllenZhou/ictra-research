"""Independent re-computation (Task 13). Deliberately does NOT import ictra_lib or any pipeline script: the scoring rules
are re-implemented from docs/EVALUATION_PROTOCOL.md, so a bug in the pipeline's scoring cannot reproduce itself here.

    python src/09_independent_check.py --zip results/export/ictra_results_<date>_dev_v01_baseline.zip

Zip mode (Task 15 "done when"): reads only the zip (plus data/catalog/seizeit2_seizures.csv for the event count) and checks
  1. every per-second file: t_sec contiguous from 0 per run, no duplicate rows, values in range;
  2. events_val.csv: one row per catalog seizure of the exported runs (count and onsets match);
  3. alert_s<k> == 3-of-5( prob_s<k> > threshold(fold, seed) and not missing ), per window;
  4. every number in metrics/ablation.csv (sensitivity, precision, F1, FA/24h, seizures, hours) and metrics/per_type.csv,
     re-scored from alert_s<k> + events_val.csv with its own event building and SzCORE overlap.
Writes INDEPENDENT_CHECK.md next to the export folder and into the zip. Exit code 1 if anything does not reproduce.
"""
import argparse, io, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# EVALUATION_PROTOCOL.md section 1 (SzCORE): 4-s windows, 2-s hop, alarm = 3 of the last 5 windows, merge alarms closer
# than 90 s, split events longer than 300 s, a hit = any overlap with [onset - 30 s, offset + 60 s].
HOP, WIN, K, N, MERGE, SPLIT, PRE, POST = 2, 4, 3, 5, 90, 300, 30, 60
FUSION_DIR = "C_eeg_ecg_emg_acc__xgb"
TOL = dict(pct=0.05, fa=0.005, f1=0.0005, hours=0.01)      # absolute tolerances (printing/rounding only)


def kofn(pos):
    """alarm[i] = at least K of windows i-N+1..i positive (windows before the start count as negative)."""
    c = np.cumsum(np.r_[np.zeros(N, int), pos.astype(int)])
    return (c[N:] - c[:-N]) >= K


def events_from_alarm(alarm):
    out, i, n = [], 0, len(alarm)
    runs = []
    while i < n:
        if alarm[i]:
            j = i
            while j + 1 < n and alarm[j + 1]:
                j += 1
            runs.append([i * HOP, j * HOP + WIN]); i = j + 1
        else:
            i += 1
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < MERGE:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in merged:
        while e - s > SPLIT:
            out.append((s, s + SPLIT)); s += SPLIT
        out.append((s, e))
    return out


def overlaps(h, r):
    return h[0] < r[1] + POST and h[1] > r[0] - PRE


def score_run(alarm, refs, n_eff_windows):
    hyp = events_from_alarm(alarm)
    tp = sum(any(overlaps(h, r) for h in hyp) for r in refs)
    fp = sum(not any(overlaps(h, r) for r in refs) for h in hyp)
    hit_by_ref = [any(overlaps(h, r) for h in hyp) for r in refs]
    return dict(n_ref=len(refs), tp=tp, fp=fp, n_hyp=len(hyp), hours=n_eff_windows * HOP / 3600), hit_by_ref


def pooled(d):
    n_ref, tp, fp, n_hyp, hours = d.n_ref.sum(), d.tp.sum(), d.fp.sum(), d.n_hyp.sum(), d.hours.sum()
    sens = tp / n_ref if n_ref else np.nan
    prec = (n_hyp - fp) / n_hyp if n_hyp else np.nan
    f1 = 2 * sens * prec / (sens + prec) if (n_ref and n_hyp and sens + prec > 0) else (0.0 if n_ref else np.nan)
    return dict(seizures=n_ref, hours=hours, sensitivity_pct=100 * sens, precision_pct=100 * prec, f1=f1, fa_per_24h=fp / hours * 24)


def check_zip(zpath):
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    top = names[0].split("/")[0]
    rd = lambda p, **k: pd.read_csv(io.BytesIO(z.read(f"{top}/{p}")), **k)
    problems, lines = [], []
    E = rd("predictions_val/events_val.csv")
    T = rd("metrics/thresholds.csv")
    A = rd("metrics/ablation.csv")
    PT = rd("metrics/per_type.csv")
    models = sorted({n.split("/")[2] for n in names if n.startswith(f"{top}/predictions_val/") and n.endswith(".csv.gz")})

    # 2. events vs catalog
    S = pd.read_csv(ROOT / "data/catalog/seizeit2_seizures.csv")
    runs_exported = set()
    rows_by_model, per_type, rows_nok = {}, [], []
    for mdl in models:
        files = sorted(n for n in names if n.startswith(f"{top}/predictions_val/{mdl}/") and n.endswith(".csv.gz"))
        rows, n_bad_alert, n_win, n_files = [], 0, 0, 0
        set_, model_ = ("rule", mdl) if mdl.startswith("rule_") else mdl.split("__")
        for fn in files:
            D = pd.read_csv(io.BytesIO(z.read(fn)), compression="gzip")
            n_files += 1
            seeds = sorted(int(c.split("_s")[1]) for c in D if c.startswith("alert_s"))
            for (sub, run), d in D.groupby(["subject", "run"], sort=False):
                runs_exported.add((sub, int(run)))
                t = d.t_sec.values
                # 1. contiguity
                if not (t == np.arange(len(t))).all():
                    problems.append(f"{mdl} {sub} run {run}: t_sec not contiguous from 0"); continue
                if len(t) % HOP:
                    problems.append(f"{mdl} {sub} run {run}: odd number of seconds")
                if not (d.missing.isin([0, 1]).all() and d.label.isin([0, 1]).all()):
                    problems.append(f"{mdl} {sub} run {run}: label/missing not 0/1")
                w = d.iloc[::HOP]                               # window i = second 2i
                miss = w.missing.values.astype(bool); n_win += len(w)
                fold = int(w.fold.iloc[0])
                ev = E[(E.subject == sub) & (E.run == run)]
                refs = list(zip(ev.onset_s, ev.onset_s + ev.duration_s))
                for sd in seeds:
                    alert = w[f"alert_s{sd}"].values.astype(bool)
                    thr = T[(T.set == set_) & (T.model == model_) & (T.fold == fold) & (T.seed == sd)].thr
                    if len(thr) != 1:
                        problems.append(f"{mdl} fold {fold} seed {sd}: no unique threshold"); continue
                    p = w[f"prob_s{sd}"].values.astype(float)
                    re = kofn((np.nan_to_num(p, nan=-np.inf) > thr.iloc[0]) & ~miss)
                    n_bad_alert += int((re != alert).sum())
                    sc, hits = score_run(alert, refs, int((~miss).sum()))
                    sc.update(sub=sub, seed=sd); rows.append(sc)
                    if mdl == FUSION_DIR:            # fusion_no_kofn row: same probabilities and thresholds, no 3-of-5
                        sc2, _ = score_run((np.nan_to_num(p, nan=-np.inf) > thr.iloc[0]) & ~miss, refs, int((~miss).sum()))
                        sc2.update(sub=sub, seed=sd); rows_nok.append(sc2)
                    for e, h in zip(ev.itertuples(), hits):
                        per_type.append(dict(model_dir=mdl, seed=sd, group=e.seizure_group, hit=h))
        R = pd.DataFrame(rows); rows_by_model[mdl] = R
        if rows_nok and mdl == FUSION_DIR:
            rows_by_model[mdl + "_no_kofn"] = pd.DataFrame(rows_nok)
        lines.append(f"- `{mdl}`: {n_files} patient files, {n_win:,} windows; alert re-derived from prob + threshold + mask + "
                     f"3-of-5: **{n_bad_alert} mismatching windows**")
        if n_bad_alert:
            problems.append(f"{mdl}: {n_bad_alert} windows where alert_s != 3-of-5(prob > thr)")

    cat = S[[(s, int(r)) in runs_exported for s, r in zip(S["sub"], S.run)]]
    ok_ev = len(cat) == len(E) and set(zip(cat["sub"], cat.run, cat.onset_s)) == set(zip(E.subject, E.run, E.onset_s))
    lines.insert(0, f"- events_val.csv: {len(E)} rows; catalog seizures in the exported runs: {len(cat)} → **{'match' if ok_ev else 'MISMATCH'}**")
    if not ok_ev:
        problems.append("events_val.csv does not match the catalog for the exported runs")

    # 4. ablation.csv re-scored: per patient over runs, pooled per seed, mean over seeds
    tab = []
    for mdl, R in rows_by_model.items():
        set_, model_ = ("rule", mdl) if mdl.startswith("rule_") else mdl.split("__")
        pat = R.groupby(["seed", "sub"])[["n_ref", "tp", "fp", "n_hyp", "hours"]].sum().reset_index()
        mine = pd.DataFrame([pooled(g) for _, g in pat.groupby("seed")]).mean()
        ref = A[(A.set == set_) & (A.model == model_)]
        if len(ref) != 1:
            problems.append(f"{mdl}: not exactly one row in ablation.csv"); continue
        ref = ref.iloc[0]
        r = dict(model=mdl)
        for k, tol in [("seizures", 0.01), ("hours", TOL["hours"]), ("sensitivity_pct", TOL["pct"]), ("precision_pct", TOL["pct"]),
                       ("f1", TOL["f1"]), ("fa_per_24h", TOL["fa"])]:
            a, b = float(ref[k]), float(mine[k]); same = (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol
            r[k] = f"{a:.3f} / {b:.3f}" + ("" if same else " ✗")
            if not same:
                problems.append(f"{mdl} {k}: ablation.csv {a:.4f} vs re-computed {b:.4f}")
        tab.append(r)
    P = pd.DataFrame(per_type)
    if len(P):
        g = P.groupby(["model_dir", "seed", "group"]).hit.agg(["sum", "size"]).reset_index()
        g = g.groupby(["model_dir", "group"]).agg(n=("size", "first"), det=("sum", "mean")).reset_index()
        n_bad = 0
        for x in g.itertuples():
            set_, model_ = ("rule", x.model_dir) if x.model_dir.startswith("rule_") else x.model_dir.split("__")
            ref = PT[(PT.set == set_) & (PT.model == model_) & (PT.group == x.group)]
            if len(ref) != 1 or abs(float(ref.sensitivity_pct.iloc[0]) - 100 * x.det / x.n) > TOL["pct"] or int(ref.seizures.iloc[0]) != x.n:
                n_bad += 1; problems.append(f"per_type {x.model_dir} {x.group}: {ref.sensitivity_pct.tolist()} vs {100 * x.det / x.n:.2f}")
        lines.append(f"- per_type.csv: {len(g)} (model, group) cells re-computed, **{n_bad} mismatches**")
    md = [f"# Independent check of `{Path(zpath).name}`", "",
          "`src/09_independent_check.py --zip` (own k-of-n, event merge/split and SzCORE overlap from EVALUATION_PROTOCOL §1; "
          "does not import ictra_lib).", "", *lines, "",
          "## metrics/ablation.csv vs re-computed from predictions_val/ (exported / re-computed; ✗ = outside rounding tolerance)", "",
          pd.DataFrame(tab).to_markdown(index=False), "",
          f"**Result: {'every checked number reproduced' if not problems else str(len(problems)) + ' problem(s)'}**", ""]
    md += [f"- {p}" for p in problems[:200]]
    return "\n".join(md), problems, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    md, problems, top = check_zip(a.zip)
    zp = Path(a.zip)
    (zp.parent / top / "INDEPENDENT_CHECK.md").write_text(md, encoding="utf-8")
    with zipfile.ZipFile(zp, "a") as zf:
        if f"{top}/INDEPENDENT_CHECK.md" not in zf.namelist():
            zf.writestr(f"{top}/INDEPENDENT_CHECK.md", md)
    print(md)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

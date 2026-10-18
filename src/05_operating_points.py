"""Operating-point sweep and ablation figure from the saved window probabilities of one version (Tasks 5 and 6).

    python src/05_operating_points.py --version v01_baseline --workers 6

For every (set, model, seed) the held-out-fold probabilities of all 5 folds are pooled and ONE threshold is swept over
them. Because the threshold is chosen on the held-out data itself, these numbers describe an OPERATING CURVE (what the
model can trade off), not a deployable threshold; the deployable numbers are summary.csv (threshold tuned on the inner
split). Post-processing and scoring are ictra_lib's (k-of-n, merge/split, SzCORE tolerances, impd + flat EEG masked
and removed from the hours).

Outputs in results/dev/<version>/:
  operating_curve.csv     one row per (set, model, seed, threshold): sensitivity overall + by seizure group, FA/24h,
                          latency_median_s (first overlapping alarm start - annotated onset, hits only)
  operating_points.csv    per (set, model) and seizure group (mean over seeds): sensitivity at FA/24h <= 0.5/1/2/5,
                          FA/24h at sensitivity >= 0.80/0.90/0.95 (NaN = not reachable)
  fig_operating_curves.png  sensitivity vs FA/24h (log x), one panel per seizure group
  fig_ablation.png        the deployable operating points (summary.csv) of A, B, C (xgb) and the four rules, per group
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

FEAT = L.ROOT / "data/features"
GROUPS = ["FBTC", "focal_motor", "focal_nonmotor", "focal_unknown_motor"]
FA_BUDGETS = [0.5, 1, 2, 5]
SENS_TARGETS = [0.80, 0.90, 0.95]
N_THR = 150
# colours: first three categorical slots of the validated reference palette (dataviz skill), rules in neutral grey
SET_COLOR = {"A_eeg": "#2a78d6", "B_eeg_ecg": "#eb6834", "C_eeg_ecg_emg_acc": "#1baf7a"}
SET_LABEL = {"A_eeg": "A: EEG", "B_eeg_ecg": "B: EEG+ECG", "C_eeg_ecg_emg_acc": "C: EEG+ECG+EMG+ACC"}
RULE_MARK = {"rule_hr": "s", "rule_emg": "^", "rule_eeg": "D", "rule_acc": "v"}
INK, INK2, GRID, RULE_GREY = "#0b0b0b", "#52514e", "#e4e3df", "#7d7c78"


def run_meta(sub, run, cache={}):
    """(mask, seizure refs, seizure groups) for one run; mask = impd | flat EEG, as in 04_train_ablation.py."""
    if (sub, run) not in cache:
        q = pd.read_parquet(FEAT / f"{sub}_run-{run:02d}.parquet", columns=["impd", "flat_eeg"])
        cache[(sub, run)] = (q.impd.values | q.flat_eeg.fillna(False).astype(bool).values,)
    return cache[(sub, run)][0]


def curve_job(args):
    version, set_name, model, seed = args
    out = L.ROOT / "results/dev" / version
    S = pd.read_csv(L.ROOT / "data/catalog/seizeit2_seizures.csv")
    P = pd.concat([pd.read_parquet(p) for p in sorted((out / "preds").glob(f"{set_name}__{model}__f*_s{seed}.parquet"))])
    runs = []
    for (sub, run), d in P.groupby(["sub", "run"], sort=False):
        mask = run_meta(sub, int(run))
        ev = S[(S["sub"] == sub) & (S.run == run)]
        runs.append((d.prob.values, mask, ev.onset_s.values, (ev.onset_s + ev.duration_s).values, ev.group.values))
    allp = np.concatenate([r[0][~r[1]] for r in runs])
    qs = 1 - np.logspace(-5.5, np.log10(0.5), N_THR)            # dense in the far tail, where useful thresholds live
    thrs = np.unique(np.quantile(allp, qs))
    rows = []
    for thr in thrs:
        tp = {g: 0 for g in GROUPS}; n = {g: 0 for g in GROUPS}; fp = 0; n_hyp = 0; hours = 0.0; lat = []
        for p, mask, a, b, grp in runs:
            hyp = L.windows_to_events(L.k_of_n((p > thr) & ~mask))
            hours += (len(p) - mask.sum()) * L.CFG["step_s"] / 3600
            n_hyp += len(hyp)
            if hyp:
                hs, he = np.array(hyp).T
                ov = (hs[None, :] < (b + L.CFG["tol_post_s"])[:, None]) & (he[None, :] > (a - L.CFG["tol_pre_s"])[:, None])
                hit = ov.any(1); fp += int((~ov.any(0)).sum()) if len(a) else len(hyp)
                if hit.any():   # latency = start of the first overlapping alarm - annotated onset (can be < 0 within the 30-s tolerance)
                    lat += list(np.where(ov, hs[None, :], np.inf).min(1)[hit] - a[hit])
            else:
                hit = np.zeros(len(a), bool)
            for g in GROUPS:
                k = grp == g; n[g] += int(k.sum()); tp[g] += int(hit[k].sum())
        sens = sum(tp.values()) / max(sum(n.values()), 1); prec = (n_hyp - fp) / n_hyp if n_hyp else np.nan
        r = dict(set=set_name, model=model, seed=seed, thr=float(thr), fp=fp, n_hyp=n_hyp, hours=hours, fa_per_24h=fp / hours * 24,
                 n_seizures=sum(n.values()), sensitivity=sens, precision=prec,
                 f1=2 * sens * prec / (sens + prec) if (n_hyp and sens + prec > 0) else 0.0,
                 latency_median_s=float(np.median(lat)) if lat else np.nan)
        for g in GROUPS:
            r[f"n_{g}"] = n[g]; r[f"sens_{g}"] = tp[g] / n[g] if n[g] else np.nan
        rows.append(r)
    return rows


def operating_points(C):
    rows = []
    for (s, m), d in C.groupby(["set", "model"]):
        for g in ["all"] + GROUPS:
            col = "sensitivity" if g == "all" else f"sens_{g}"
            r = dict(set=s, model=m, group=g, n_seizures=int(d["n_seizures" if g == "all" else f"n_{g}"].iloc[0]), n_seeds=d.seed.nunique())
            for b in FA_BUDGETS:
                v = [ds.loc[ds.fa_per_24h <= b, col].max() for _, ds in d.groupby("seed")]
                r[f"sens_at_fa_le_{b}"] = np.nanmean([x if np.isfinite(x) else 0.0 for x in v])
            for t in SENS_TARGETS:
                v = [ds.loc[ds[col] >= t, "fa_per_24h"].min() for _, ds in d.groupby("seed")]
                r[f"fa24_at_sens_ge_{t:.2f}"] = np.nan if any(not np.isfinite(x) for x in v) else float(np.mean(v))
            rows.append(r)
    return pd.DataFrame(rows)


def _style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


RULE_DASH = {"rule_hr": (0, (4, 2)), "rule_emg": (0, (1, 1.5)), "rule_eeg": (0, (6, 2, 1, 2)), "rule_acc": (0, (2, 3))}
LABEL_OFF = {"A_eeg": (-14, 6), "B_eeg_ecg": (6, 8), "C_eeg_ecg_emg_acc": (6, -12)}


def frontier(d, col):
    """Best sensitivity reachable at FA/24h <= x (same definition as operating_points.csv), as a step curve."""
    d = d.sort_values("fa_per_24h")
    return d.fa_per_24h.values, np.maximum.accumulate(np.nan_to_num(d[col].values))


def fig_curves(C, out, model="xgb"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.4), sharey=True)
    for ax, g in zip(axs, ["all"] + GROUPS):
        col = "sensitivity" if g == "all" else f"sens_{g}"
        for r, ls in RULE_DASH.items():
            d = C[(C.set == "rule") & (C.model == r)]
            if len(d):
                x, y = frontier(d, col)
                ax.step(np.clip(x, 0.05, None), y, where="post", color=RULE_GREY, lw=1.2, ls=ls, label=r)
        for s in SET_COLOR:
            d = C[(C.set == s) & (C.model == model) & (C.seed == C.seed.min())]
            if len(d):
                x, y = frontier(d, col)
                ax.step(np.clip(x, 0.05, None), y, where="post", color=SET_COLOR[s], lw=2, label=SET_LABEL[s])
        n = int(C[f"n_{g}"].iloc[0]) if g != "all" else int(C.n_seizures.iloc[0])
        ax.set_xscale("log"); ax.set_xlim(0.05, 300); ax.set_ylim(0, 1.02); _style(ax)
        ax.set_title(f"{g} (n={n} seizures)", fontsize=9, color=INK); ax.set_xlabel("false alarms per 24 h (log)", fontsize=8, color=INK2)
    axs[0].set_ylabel("best event sensitivity at this FA/24h", fontsize=8, color=INK2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"{out.name}: operating curves ({model}; single-input rules in grey), held-out folds pooled. "
                 "Threshold swept on the held-out data: an operating curve, not a deployable threshold", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "fig_operating_curves.png", dpi=150); plt.close(fig)


def fig_ablation(out, model="xgb"):
    """Deployable points: summary.csv (threshold from the inner split) of A/B/C and the rules, per seizure group."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = pd.read_csv(out / "summary.csv")
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.4), sharey=True)
    for ax, g in zip(axs, ["all"] + GROUPS):
        col = "sensitivity" if g == "all" else f"sens_{g}"
        if col not in T:
            continue
        for r, mk in RULE_MARK.items():
            d = T[(T.set == "rule") & (T.model == r)]
            if len(d) and np.isfinite(d[col].iloc[0]):
                ax.plot(d.fa_per_24h, d[col], mk, ms=8, mfc="white", mec=RULE_GREY, mew=1.5, label=r)
        for s in SET_COLOR:
            d = T[(T.set == s) & (T.model == model)]
            if len(d) and np.isfinite(d[col].iloc[0]):
                # no letter labels: A/B/C points nearly coincide, so offset letters landed next to the wrong point;
                # identity comes from the legend (colour) and the hover-free PNG keeps one marker per series
                ax.plot(d.fa_per_24h, d[col], "o", ms=9, color=SET_COLOR[s], mec="white", mew=1.5, label=SET_LABEL[s])
        ax.set_xscale("symlog", linthresh=1); ax.set_xlim(0, 300); ax.set_ylim(0, 1.05); _style(ax)
        ax.set_title(g, fontsize=9, color=INK); ax.set_xlabel("false alarms per 24 h", fontsize=8, color=INK2)
    axs[0].set_ylabel("event sensitivity", fontsize=8, color=INK2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"{out.name}: deployable operating points (threshold tuned on the inner split), {model} models and single-input rules. "
                 "FA/24h does not depend on seizure group, so each point keeps its x-position in every panel", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "fig_ablation.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fig-only", action="store_true")
    a = ap.parse_args()
    out = L.ROOT / "results/dev" / a.version
    if not a.fig_only:
        keys = sorted({tuple(p.stem.split("__")[:2]) + (int(p.stem.split("_s")[-1]),) for p in (out / "preds").glob("*.parquet")})
        jobs = [(a.version, s, m, sd) for s, m, sd in keys]
        print(f"{len(jobs)} (set, model, seed) curves, {a.workers} workers")
        rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(curve_job, jobs):
                rows += r; print(f"  {r[0]['set']} {r[0]['model']} seed {r[0]['seed']}", flush=True)
        C = pd.DataFrame(rows); C.to_csv(out / "operating_curve.csv", index=False)
        OP = operating_points(C); OP.to_csv(out / "operating_points.csv", index=False)
        print(OP[OP.group == "all"].round(3).to_string(index=False))
        # Lab page: frontier curve per (set, model) of the first seed + the operating-point table (raw numbers)
        import json
        cur = []
        for (st, md), d in C[C.seed == C.seed.min()].groupby(["set", "model"]):
            row = dict(set=st, model=md)
            for g in ["all"] + GROUPS:
                x, y = frontier(d, "sensitivity" if g == "all" else f"sens_{g}")
                row[g] = dict(fa_per_24h=x.tolist(), sensitivity=y.tolist())
            cur.append(row)
        jp = L.LAB_DATA / f"dev_{a.version}.json"
        J = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {"version": a.version}
        J["operating_curves"] = cur; J["operating_points"] = OP.to_dict("records")
        J["operating_note"] = "threshold swept on the held-out folds: an operating curve, not a deployable threshold"
        L.lab_write_json(f"dev_{a.version}.json", L.json_clean(J))
    C = pd.read_csv(out / "operating_curve.csv")
    fig_curves(C, out)
    if (out / "summary.csv").exists():
        fig_ablation(out)
    print("figures in", out)


if __name__ == "__main__":
    main()

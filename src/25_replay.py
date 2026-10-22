"""Demo replay (Stage 4): play a recorded SeizeIT2 dev run back in time, window by window, through the evaluated model.

    python src/25_replay.py --run sub-070_run-12                      # live window, real time (1x)
    python src/25_replay.py --run sub-070_run-12 --speed 6 --save results/demo/replay_sub-070_run-12.mp4

Not live acquisition: a replay of a stored recording, so the demo is repeatable and explainable. Every 2-s hop the
window that has just ended is scored by the saved demo model (src/24_demo_model.py: the exact held-out model of the
version, here fold 4, which never saw these patients) on that window's ictra_lib features (the version's feature
directory, written by 03_features.py); the alarm needs k of the last n windows above tau and is shown when the window
that completes it ends. The clinical annotation is drawn for the viewer; the model never sees it.
The screen states the pipeline: "offline pipeline" (features computed from the whole recording, zero-phase filters:
not runnable live) until a causal (v11) model is used. Dev runs only (docs/DEMO_RUNS.txt); sub-097..125 are refused.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

RAW = L.ROOT / "data/raw/ds005873"
# dark monitor screen; categorical slots 1-5 of the validated palette (dark), status "critical" for the alarm
BG, INK, INK2, GRID, BAND = "#1a1a19", "#f0efec", "#c3c2b7", "#383835", "#4a4a46"
C_EEG, C_ECG, C_EMG, C_ACC, C_PROB, C_ALARM = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#d03b3b"
VIEW_S = 20          # seconds of signal shown in the waveform panels


def demo_runs():
    rows = {}
    for line in (L.ROOT / "docs/DEMO_RUNS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            p = [x.strip() for x in line.split("|")]
            rows[p[0]] = dict(onset=float(p[1]), duration=float(p[2]), annotation=p[3], group=p[4], note=p[5])
    return rows


def read_edf(sub, run, m, t0, t1, picks=None):
    import mne
    p = RAW / f"{sub}/ses-01/{m}/{sub}_ses-01_task-szMonitoring_run-{run:02d}_{m}.edf"
    r = mne.io.read_raw_edf(p, preload=False, verbose=False)
    fs = r.info["sfreq"]
    if m == "mov":
        picks = [i for i, n in enumerate(r.ch_names) if n.startswith("EEG SD ACC")][:3] or [0, 1, 2]
    a, b = int(max(0, t0) * fs), int(min(r.n_times / fs, t1) * fs)
    return r.get_data(picks=picks, start=a, stop=b) * 1e6, fs, a / fs


def prepare(run_stem, model_path, before, after):
    """Everything the replay needs for one demo run: the demo model and its metadata, the run's ictra_lib features on
    the window grid, and display signals harmonised by ictra_lib in the model's mode. Dev runs only."""
    R = demo_runs()
    if run_stem not in R:
        sys.exit(f"{run_stem} is not in docs/DEMO_RUNS.txt")
    sub, run = run_stem.split("_run-")[0], int(run_stem.split("_run-")[1])
    locked = {s for s in L.load_splits().get("locked_test", [])}
    if sub in locked:
        sys.exit("refusing a locked-test patient")
    meta = json.loads((L.ROOT / model_path).read_text())
    if sub not in meta["held_out_subjects"]:
        sys.exit(f"{sub} was in this model's training folds; use a model whose held-out fold contains it")
    L.set_mode(meta["causal"])
    import xgboost as xgb
    model = xgb.XGBClassifier(); model.load_model(str(L.ROOT / model_path.replace(".meta.json", ".json")))
    model.get_booster().set_param({"device": "cpu"})
    ev = R[run_stem]; on, off = ev["onset"], ev["onset"] + ev["duration"]
    T0, T1 = on - before, off + after

    # ---- features of this run (the version's ictra_lib output) and the window grid
    F = pd.read_parquet(L.ROOT / meta["features_dir"] / f"{run_stem}.parquet")
    step, win = L.CFG["step_s"], L.CFG["win_s"]
    t_end = F.t0.values + win
    mask = F.impd.values | F.flat_eeg.fillna(False).astype(bool).values
    first = int(np.searchsorted(t_end, T0)) - meta["post_n"]          # a few windows of history for k-of-n
    idx = np.arange(max(0, first), int(np.searchsorted(t_end, T1, side="right")))
    X = F.reindex(columns=meta["columns"]).iloc[idx]
    hr = F.ecg_hr.values if "ecg_hr" in F else np.full(len(F), np.nan)

    # ---- display signals, harmonised by ictra_lib in the same mode (display only)
    pad = 30
    eeg, fs_e, s_e = read_edf(sub, run, "eeg", T0 - VIEW_S - pad, T1 + pad, picks=[0, 1])
    ecg, fs_c, s_c = read_edf(sub, run, "ecg", T0 - VIEW_S - pad, T1 + pad, picks=[0])
    emg, fs_m, s_m = read_edf(sub, run, "emg", T0 - VIEW_S - pad, T1 + pad, picks=[0])
    acc, fs_a, s_a = read_edf(sub, run, "mov", T0 - VIEW_S - pad, T1 + pad)
    eeg = L.harmonise(eeg, fs_e, "eeg"); ecg = L.harmonise(ecg, fs_c, "ecg")[0]; emg = L.harmonise(emg, fs_m, "emg")[0]
    accm = np.sqrt((L.harmonise(L.acc_to_g(acc, fs_a), fs_a, "acc", fs_out=L.CFG["fs_acc"]) ** 2).sum(0))
    fs = L.CFG["fs"]
    k = int(fs * 0.05); env = np.convolve(np.abs(emg), np.ones(k) / k, "same")[::4]       # 50-ms EMG envelope, 64 Hz
    sig = {"eeg": (s_e, fs, eeg), "ecg": (s_c, fs, ecg), "emg": (s_m, fs / 4, env), "acc": (s_a, L.CFG["fs_acc"], accm)}

    from types import SimpleNamespace
    return SimpleNamespace(run=run_stem, sub=sub, meta=meta, model=model, ev=ev, on=on, off=off, T0=T0, T1=T1, F=F, step=step,
                           win=win, t_end=t_end, mask=mask, idx=idx, X=X, hr=hr, sig=sig, eeg=eeg, ecg=ecg, env=env, accm=accm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run stem from docs/DEMO_RUNS.txt, e.g. sub-070_run-12")
    ap.add_argument("--model", default="models/demo/v02g_grid_fix__C_eeg_ecg_emg_acc__xgb__f4_s0.meta.json")
    ap.add_argument("--before", type=float, default=150); ap.add_argument("--after", type=float, default=90)
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    ap.add_argument("--save", help="render to an .mp4 instead of a live window")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--preview", type=float, help="write one PNG frame at this many seconds after the replay start, then exit")
    a = ap.parse_args()
    S = prepare(a.run, a.model, a.before, a.after)
    meta, model, ev, on, off, T0, T1 = S.meta, S.model, S.ev, S.on, S.off, S.T0, S.T1
    F, step, win, t_end, mask, idx, X, hr, sig = S.F, S.step, S.win, S.t_end, S.mask, S.idx, S.X, S.hr, S.sig
    eeg, ecg, env, accm = S.eeg, S.ecg, S.env, S.accm

    # ---- figure
    import matplotlib
    if a.save or a.preview is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": BG,
                         "axes.facecolor": BG, "savefig.facecolor": BG})
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = fig.add_gridspec(6, 1, height_ratios=[1.1, 1.1, 0.8, 0.8, 0.8, 2.2], hspace=0.45, left=0.06, right=0.985, top=0.80, bottom=0.07)
    ax_eeg, ax_ecg, ax_emg, ax_acc = (fig.add_subplot(gs[i]) for i in range(4))
    ax_p = fig.add_subplot(gs[5])
    pipe = meta["pipeline"]
    caveat = ("features computed from the whole recording (zero-phase filters): not runnable live" if not meta["causal"]
              else "every window uses only samples up to its end (truncation test passed)")
    fig.text(0.06, 0.955, f"ICTRA replay  ·  {pipe}", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.06, 0.92, f"{meta['version']}  ·  {meta['lib_version']}  ·  model C (EEG+ECG+EMG+ACC) xgboost, fold {meta['fold']} "
             f"(this patient was not in training)  ·  τ = {meta['thr']:.3f}, alarm = {meta['post_k']} of last {meta['post_n']} windows",
             fontsize=11, color=INK2)
    fig.text(0.06, 0.893, f"{pipe}: {caveat}  ·  replay of a stored recording, not live acquisition", fontsize=11, color=INK2)
    clock = fig.text(0.985, 0.955, "", ha="right", fontsize=20, color=INK, family="Consolas")
    badge = fig.text(0.985, 0.905, "", ha="right", fontsize=13, fontweight="bold")
    hr_txt = fig.text(0.985, 0.865, "", ha="right", fontsize=11, color=INK2)
    fig.text(0.985, 0.84, f"{a.run} (SeizeIT2 dev set)  ·  speed {a.speed:g}×", ha="right", fontsize=10, color=INK2)
    lines = {}
    for ax, key, lab, col, n in ((ax_eeg, "eeg", "EEG behind the ear (2 ch, µV)", C_EEG, 2), (ax_ecg, "ecg", "ECG (µV)", C_ECG, 1),
                                 (ax_emg, "emg", "EMG envelope (µV)", C_EMG, 1), (ax_acc, "acc", "movement |a| (g)", C_ACC, 1)):
        ax.set_xlim(-VIEW_S, 0); ax.tick_params(labelsize=9, length=0); ax.grid(axis="x", color=GRID, lw=0.6)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticklabels([]) if key != "acc" else ax.set_xlabel("seconds before now", fontsize=9)
        ax.text(0.0, 1.04, lab, transform=ax.transAxes, fontsize=10, color=INK2)
        lines[key] = [ax.plot([], [], color=col, lw=1.0, alpha=1.0 if i == 0 else 0.6)[0] for i in range(n)]
    # robust fixed y-limits per panel (from the whole segment) so the scale does not jump
    def lim(x, q=99.5, sym=True):
        v = np.nanpercentile(np.abs(x) if sym else x, q); return (-v * 1.1, v * 1.1) if sym else (np.nanpercentile(x, 0.5) * 0.98, v * 1.05)
    ax_eeg.set_ylim(*lim(np.r_[eeg[0], eeg[1]]))
    ax_ecg.set_ylim(*lim(ecg)); ax_emg.set_ylim(0, np.nanpercentile(env, 99.8) * 1.1); ax_acc.set_ylim(*lim(accm, 99.8, sym=False))
    # probability panel over the whole replay segment
    ax_p.set_xlim(T0 - on, T1 - on); ax_p.set_ylim(0, 1.02)
    for s in ax_p.spines.values(): s.set_visible(False)
    ax_p.grid(color=GRID, lw=0.6); ax_p.tick_params(length=0)
    ax_p.set_xlabel("seconds relative to the annotated seizure onset"); ax_p.text(0.0, 1.03, "model output: seizure probability per window",
                                                                                transform=ax_p.transAxes, fontsize=10, color=INK2)
    ax_p.axvspan(0, off - on, color=BAND, lw=0, zorder=0)
    ax_p.text((off - on) / 2, 0.04, f"clinical annotation: {ev['group'].replace('_', ' ')} ({ev['duration']:.0f} s)\nthe model never sees this",
              ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax_p.axhline(meta["thr"], color=INK2, lw=1, ls=(0, (4, 3)))
    ax_p.text(T1 - on, meta["thr"] + 0.015, f"τ = {meta['thr']:.3f}", ha="right", va="bottom", fontsize=9, color=INK2)
    pline, = ax_p.plot([], [], color=C_PROB, lw=2)
    now_line = ax_p.axvline(T0 - on, color=INK, lw=0.8, alpha=0.6)
    for ax in (ax_eeg, ax_ecg, ax_emg, ax_acc):
        ax.axvline(0, color=INK, lw=0.8, alpha=0.6)

    state = dict(scored=0, probs=[], times=[], pos=[], alarm_at=None, last_alarm_win=None)

    def score_until(t):
        """Score every window that has ended by replay time t (one model call per window, in order)."""
        while state["scored"] < len(idx) and t_end[idx[state["scored"]]] <= t:
            j = idx[state["scored"]]
            p = float(model.predict_proba(X.iloc[[state["scored"]]])[:, 1][0])
            state["probs"].append(p); state["times"].append(t_end[j] - on)
            state["pos"].append(bool(p > meta["thr"]) and not mask[j])
            if sum(state["pos"][-meta["post_n"]:]) >= meta["post_k"]:
                state["last_alarm_win"] = t_end[j]
                if state["alarm_at"] is None:
                    state["alarm_at"] = t_end[j]
                    ax_p.axvline(t_end[j] - on, color=C_ALARM, lw=2)
                    ax_p.text(t_end[j] - on, 0.5, f"  ALARM  {t_end[j] - on:+.0f} s", color=C_ALARM, fontsize=11, fontweight="bold", va="center")
            state["scored"] += 1

    def draw(t):
        score_until(t)
        for key, (s0, f, x) in sig.items():
            a0 = int((t - VIEW_S - s0) * f); a1 = int((t - s0) * f)
            a0, a1 = max(0, a0), max(0, a1)
            tt = (np.arange(a0, a1) / f + s0) - t
            seg = x[..., a0:a1]
            for i, ln in enumerate(lines[key]):
                ln.set_data(tt, seg[i] if seg.ndim == 2 else seg)
        pline.set_data(state["times"], state["probs"])
        now_line.set_xdata([t - on, t - on])
        h = hr[min(len(hr) - 1, int(max(0, t - win) // step))]
        hr_txt.set_text(f"heart rate (pipeline): {h:.0f} bpm" if np.isfinite(h) else "heart rate: —")
        rel = t - on
        clock.set_text(time.strftime("%H:%M:%S", time.gmtime(t)) + f"   ({rel:+.0f} s)")
        alarm_on = state["last_alarm_win"] is not None and t - state["last_alarm_win"] < 30
        badge.set_text("  ALARM: carer notified  " if alarm_on else "  monitoring  ")
        badge.set_color(INK if alarm_on else INK2)
        badge.set_bbox(dict(boxstyle="round,pad=0.35", fc=C_ALARM if alarm_on else GRID, ec="none"))
        return []

    if a.preview is not None:
        draw(T0 + a.preview)
        out = L.ROOT / "results/demo" / f"preview_{a.run}_{int(a.preview)}.png"; out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out); print("saved", out); return
    if a.save:
        ff = shutil.which("ffmpeg")
        if not ff:
            import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams["animation.ffmpeg_path"] = ff
        n = int((T1 - T0) / a.speed * a.fps)
        ani = animation.FuncAnimation(fig, lambda i: draw(T0 + i * a.speed / a.fps), frames=n, blit=False)
        out = L.ROOT / a.save; out.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(out), writer=animation.FFMpegWriter(fps=a.fps, bitrate=6000))
        print(f"saved {out} ({n / a.fps:.0f} s at {a.speed:g}x)")
    else:
        start = time.perf_counter()
        ani = animation.FuncAnimation(fig, lambda i: draw(min(T1, T0 + (time.perf_counter() - start) * a.speed)),
                                      interval=1000 / a.fps, blit=False, cache_frame_data=False)
        plt.show()
    alarm = state["alarm_at"]
    print(f"{a.run}: alarm {'at %+.0f s from onset' % (alarm - on) if alarm else 'none'}; {state['scored']} windows scored")


if __name__ == "__main__":
    main()

"""Demo video (website and competition cut): the replay of src/25_replay.py with a visual guide layer.

    python src/27_demo_video.py --run sub-070_run-12 --out results/demo/demo_FBTC_sub-070_run-12.mp4
    python src/27_demo_video.py --run sub-064_run-02 --intro "..." --outro "...|...|..." --out ...

Same data, model and scoring as the replay (25_replay.prepare: the causal v11 fold-4 demo model, which never saw these
patients; each window scored by the model, alarm = k of the last n windows above tau). Added on top, all of it pointing
at things that are in the data (Allen 2026-10-22):
  A  opening card;  B  the four signal bands appear one by one with where each sensor sits (dataset README);
  C  the model output is the largest panel; when it first rises its line thickens and a caption explains it;
  D  "EEG changes here" / "heart rate follows N s later" on the waveforms, ONLY if this seizure's Exp R change points
     (results/expR/onsets.csv, same z > 3 for >= 3 s rule) put EEG strictly before heart rate; N is this seizure's lag;
  E  the alarm moment holds 1.5 s with the delay from the annotated onset;  F  speed badge next to the recording clock;
  G  closing card.  No glow, sound or camera moves: it should read as an instrument.
"""
import argparse, importlib.util, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

_spec = importlib.util.spec_from_file_location("replay", Path(__file__).resolve().parent / "25_replay.py")
RP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RP)
BG, INK, INK2, GRID, BAND = RP.BG, RP.INK, RP.INK2, RP.GRID, RP.BAND
VIEW_S = RP.VIEW_S
# where each sensor sits (data/raw/ds005873/README: EEG SD on the upper back with electrodes on the mastoid behind the
# ear; ECG/EMG SD on the left chest, EMG electrodes on the left deltoid; ACC used = the EEG SD device, docs/DATA_NOTES.md)
BANDS = [("eeg", "EEG", "behind the ear  ·  2 channels, µV", RP.C_EEG, 2), ("ecg", "ECG", "chest  ·  µV", RP.C_ECG, 1),
         ("emg", "EMG", "left shoulder muscle  ·  envelope, µV", RP.C_EMG, 1), ("acc", "movement", "upper back  ·  |a|, g", RP.C_ACC, 1)]
REVEAL = {"eeg": 0.0, "ecg": 3.0, "emg": 4.0, "acc": 5.0, "prob": 6.0}      # video seconds after the opening card
FADE = 0.5
INTRO_S, OUTRO_S, HOLD_S = 3.5, 4.0, 1.5
INTRO = "One seizure, four signals, and the model's output — replayed from a recording the model never trained on."
OUTRO = ("One seizure, not a result.|Median alarm delay across development recordings (causal model): 34.7 s.|"
         "Replay of a stored recording, not live acquisition.")


Z_THR, MIN_DUR, BASE = 3.0, 3, (-300, -60)        # Exp R definitions (src/02_relationship_study.py)
HR_JUMP = 25.0    # bpm between consecutive seconds: above this the heart-rate trace is movement artefact, not heart rate


def first_change(t, z):
    """First second >= -30 s with z > Z_THR for >= MIN_DUR consecutive seconds (Exp R's rule)."""
    ok = (z > Z_THR) & (t >= -30)
    run = 0
    for i in range(len(t)):
        run = run + 1 if ok[i] else 0
        if run >= MIN_DUR:
            return float(t[i - MIN_DUR + 1])
    return np.nan


def change_points(S):
    """Exp R per-second curves of this seizure as z vs the -300..-60 s baseline, and the EEG / heart-rate change times.
    Returns (t, z_eeg_power, z_hr, (t_eeg, t_hr) or None); the change times are recomputed here and must equal
    results/expR/onsets.csv; the order annotation is used only if EEG changes strictly before heart rate."""
    sid = f"{S.sub}_r{int(S.run.split('_run-')[1]):02d}_{int(S.on)}"
    C = pd.read_parquet(L.ROOT / "results/expR/curves.parquet", filters=[("seizure_id", "==", sid)]).sort_values("t")
    if C.empty:
        return None
    t = C.t.values.astype(float)
    base = (t >= BASE[0]) & (t <= BASE[1])
    z = {c: (C[c].values - np.nanmean(C[c].values[base])) / (np.nanstd(C[c].values[base], ddof=1) + 1e-9)
         for c in ("eeg_linelength", "eeg_power_1_25", "ecg_hr")}
    te = np.nanmin([first_change(t, z["eeg_linelength"]), first_change(t, z["eeg_power_1_25"])])
    th = first_change(t, z["ecg_hr"])
    E = pd.read_csv(L.ROOT / "results/expR/onsets.csv").set_index("seizure_id")
    if sid in E.index:
        for mine, theirs in ((te, E.t_eeg[sid]), (th, E.t_hr[sid])):
            if not (np.isnan(mine) and np.isnan(theirs)) and mine != theirs:
                sys.exit(f"change point {mine} differs from Exp R onsets.csv {theirs}: not annotating")
    zeeg = z["eeg_power_1_25"] if first_change(t, z["eeg_power_1_25"]) == te else z["eeg_linelength"]
    # artefact screen: during movement the ECG-derived heart rate jumps by tens of bpm from one second to the next. Such
    # seconds (+-2 s) are not drawn, and the order claim needs a clean heart rate from -30 s to 10 s after its change point
    hr = C.ecg_hr.values.astype(float)
    jump = np.r_[0.0, np.abs(np.diff(hr))]
    bad = pd.Series(jump > HR_JUMP).rolling(5, center=True, min_periods=1).max().fillna(0).values.astype(bool)
    zhr = np.where(bad, np.nan, z["ecg_hr"])
    clean = np.isfinite(th) and not bad[(t >= -30) & (t <= th + 10)].any()
    order = (te, th) if clean and np.isfinite(te) and te < th else None
    why = ("annotated" if order else "not annotated: heart rate is movement artefact around its change point" if not clean
           else "not annotated: EEG does not change before heart rate here")
    seg = (t >= S.T0 - S.on) & (t <= S.T1 - S.on)
    return t, zeeg, zhr, order, bool(bad[seg].any()), (np.nanmin(hr[seg & bad]) if (seg & bad).any() else np.nan,
                                                       np.nanmax(hr[seg & bad]) if (seg & bad).any() else np.nan), why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="models/demo/v11_causal__C_eeg_ecg_emg_acc__xgb__f4_s0.meta.json")
    ap.add_argument("--before", type=float, default=150); ap.add_argument("--after", type=float, default=90)
    ap.add_argument("--speed", type=float, default=6.0); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--intro", default=INTRO); ap.add_argument("--outro", default=OUTRO, help="lines separated by |")
    ap.add_argument("--out", required=True); ap.add_argument("--preview", type=float, nargs="*",
                    help="write PNG frames at these video seconds instead of the video")
    a = ap.parse_args()
    S = RP.prepare(a.run, a.model, a.before, a.after)
    meta, on, off = S.meta, S.on, S.off
    # ---- model output for every window of the segment (one model call per window, in time order, as in the replay)
    P = S.model.predict_proba(S.X)[:, 1]
    tw = S.t_end[S.idx]
    pos = (P > meta["thr"]) & ~S.mask[S.idx]
    k, n = meta["post_k"], meta["post_n"]
    alarm_win = np.convolve(pos.astype(int), np.ones(n, int), "full")[: len(pos)] >= k
    t_alarm = float(tw[np.argmax(alarm_win)]) if alarm_win.any() else None
    t_rise = next((float(t) for t, p in zip(tw, P) if t >= on - 30 and p > 0.2), None)
    CPS = change_points(S)
    cp = CPS[3] if CPS else None
    if CPS and CPS[4]:
        print(f"heart rate partly hidden: movement artefact (jumps > {HR_JUMP:.0f} bpm/s, values {CPS[5][0]:.0f}-{CPS[5][1]:.0f} bpm)")
    print(f"alarm {'%+.0f s' % (t_alarm - on) if t_alarm else 'none'}; first rise {('%+.0f s' % (t_rise - on)) if t_rise else 'none'}; "
          f"Exp R change points {cp} ({CPS[6] if CPS else 'no Exp R curves for this seizure'})")

    # ---- timeline: list of (kind, recording time, video time since playback start)
    step = a.speed / a.fps
    play = list(np.arange(S.T0, S.T1 + 1e-9, step))
    frames = [("intro", S.T0, 0.0)] * int(INTRO_S * a.fps)
    tv = 0.0
    for t in play:
        if t_alarm is not None and t - step < t_alarm <= t:
            frames += [("hold", t_alarm, tv)] * int(HOLD_S * a.fps)
        frames.append(("play", t, tv)); tv += 1 / a.fps
    frames += [("outro", S.T1, tv)] * int(OUTRO_S * a.fps)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": BG,
                         "axes.facecolor": BG, "savefig.facecolor": BG})
    fig = plt.figure(figsize=(16, 9), dpi=120)
    pipe = meta["pipeline"]
    fig.text(0.05, 0.955, f"ICTRA replay  ·  {pipe}", fontsize=20, fontweight="bold")
    fig.text(0.05, 0.922, f"{meta['version']}  ·  {meta['lib_version']}  ·  model C (EEG+ECG+EMG+movement), fold {meta['fold']}: "
             f"this patient was not in training", fontsize=11, color=INK2)
    fig.text(0.05, 0.896, f"alarm when {k} of the last {n} windows exceed τ = {meta['thr']:.3f}  ·  " +
             ("each window uses only data up to its end" if meta["causal"] else
              "features computed from the whole recording: not runnable live"), fontsize=11, color=INK2)
    fig.text(0.05, 0.870, f"{S.run}, SeizeIT2 development set  ·  replay of a stored recording, not live acquisition",
             fontsize=11, color=INK2)
    speed_t = fig.text(0.735, 0.95, f" {a.speed:g}× ", fontsize=18, fontweight="bold", color=INK, va="center",
                       bbox=dict(boxstyle="round,pad=0.3", fc="none", ec=INK2, lw=1.2))
    clock = fig.text(0.975, 0.95, "", ha="right", va="center", fontsize=20, family="Consolas")
    fig.text(0.975, 0.918, f"recording time  ·  1 s on screen = {a.speed:g} s of recording", ha="right", fontsize=10, color=INK2)
    badge = fig.text(0.975, 0.872, "", ha="right", va="center", fontsize=13, fontweight="bold")
    hr_txt = fig.text(0.975, 0.835, "", ha="right", fontsize=11, color=INK2)

    # four signal bands (top) and the model output (bottom, largest)
    tops = [0.805, 0.727, 0.649, 0.571]; h = 0.060
    axes, lines, labels = {}, {}, {}
    for (key, name, where, col, nch), top in zip(BANDS, tops):
        ax = fig.add_axes([0.05, top - h, 0.925, h]); ax.set_xlim(-VIEW_S, 0)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.tick_params(length=0, labelsize=8); ax.set_xticks([]); ax.grid(False)
        lines[key] = [ax.plot([], [], color=col, lw=1.0, alpha=0.0)[0] for _ in range(nch)]
        labels[key] = [ax.text(-VIEW_S, 1.02, name, transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold", color=INK, alpha=0),
                       ax.text(-VIEW_S + 1.6 + 0.35 * len(name), 1.02, where, transform=ax.get_xaxis_transform(), fontsize=10, color=INK2, alpha=0)]
        ax.axvline(0, color=INK, lw=0.8, alpha=0.5)
        axes[key] = ax
    fig.canvas.draw()                                  # place each location note right after its signal name
    for key in axes:
        bb = labels[key][0].get_window_extent()
        x1 = axes[key].transData.inverted().transform((bb.x1, bb.y0))[0]
        labels[key][1].set_x(x1 + 0.25)
    for key, x in (("eeg", np.r_[S.eeg[0], S.eeg[1]]), ("ecg", S.ecg)):
        v = np.nanpercentile(np.abs(x), 99.5) * 1.1; axes[key].set_ylim(-v, v)
    axes["emg"].set_ylim(0, np.nanpercentile(S.env, 99.8) * 1.1)
    axes["acc"].set_ylim(np.nanpercentile(S.accm, 0.5) * 0.98, np.nanpercentile(S.accm, 99.8) * 1.05)
    axz = fig.add_axes([0.05, 0.392, 0.925, 0.066]); axz.set_xlim(-VIEW_S, 0); axz.set_ylim(-2, 12)
    for s_ in axz.spines.values(): s_.set_visible(False)
    axz.tick_params(length=0, labelsize=8); axz.set_yticks([0, 3, 6, 9])
    zt = [axz.text(-VIEW_S, 1.04, "brain and heart, measured as in our relationship study", transform=axz.get_xaxis_transform(),
                   fontsize=11, fontweight="bold", alpha=0),
          axz.text(-VIEW_S + 7.6, 1.04, "SD above each signal's own 5-min baseline  ·  offline analysis", transform=axz.get_xaxis_transform(),
                   fontsize=10, color=INK2, alpha=0)]
    z_thr_ln = axz.axhline(Z_THR, color=INK2, lw=0.9, ls=(0, (4, 3)), alpha=0)
    z_thr_tx = axz.text(-VIEW_S + 0.1, Z_THR + 0.5, "3 SD", ha="left", fontsize=8.5, color=INK2, alpha=0)
    zl_eeg, = axz.plot([], [], color=RP.C_EEG, lw=1.8, alpha=0); zl_hr, = axz.plot([], [], color=RP.C_ECG, lw=1.8, alpha=0)
    from matplotlib.lines import Line2D
    zleg = axz.legend(handles=[Line2D([], [], color=RP.C_EEG, lw=1.8), Line2D([], [], color=RP.C_ECG, lw=1.8)],
                      labels=["EEG power, 1–25 Hz", "heart rate"], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                      frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.6, borderaxespad=0.1)
    zleg.set_visible(False)
    zt_eeg = CPS[0] + on if CPS else np.array([]); zz_eeg = CPS[1] if CPS else np.array([])
    zz_hr = CPS[2] if CPS is not None else np.array([])
    art_note = None
    if CPS is not None and CPS[4]:
        art_note = fig.text(0.975, 0.012, "heart rate hidden where the ECG is dominated by movement artefact "
                            f"(jumps of more than {HR_JUMP:.0f} bpm from one second to the next)", ha="right", fontsize=9.5, color=INK2, alpha=0)
    axz.axvline(0, color=INK, lw=0.8, alpha=0.5)
    tick_ax = axz; tick_ax.set_xticks(np.arange(-VIEW_S, 1, 5)); tick_ax.tick_params(labelsize=9)
    tick_ax.set_xticklabels([f"{x:.0f} s" if x else "now" for x in np.arange(-VIEW_S, 1, 5)])
    for t_ in tick_ax.get_xticklabels(): t_.set_alpha(0)

    axp = fig.add_axes([0.05, 0.075, 0.925, 0.255])
    axp.set_xlim(S.T0 - on, S.T1 - on); axp.set_ylim(0, 1.04)
    for s_ in axp.spines.values(): s_.set_visible(False)
    axp.grid(color=GRID, lw=0.6); axp.tick_params(length=0)
    axp.set_xlabel("seconds from the annotated seizure onset")
    band = axp.axvspan(0, off - on, color=BAND, lw=0, zorder=0)
    ann = axp.text((off - on) / 2, 0.04, f"clinical annotation: {S.ev['group'].replace('_', ' ')}, {S.ev['duration']:.0f} s "
                   f"(the model never sees it)", ha="center", va="bottom", fontsize=10, color=INK2)
    thr_line = axp.axhline(meta["thr"], color=INK2, lw=1, ls=(0, (4, 3)))
    thr_txt = axp.text(S.T1 - on, meta["thr"] + 0.02, f"alarm threshold τ = {meta['thr']:.3f}", ha="right", va="bottom", fontsize=9.5, color=INK2)
    ptitle = axp.text(0.0, 1.04, "model output: seizure probability", transform=axp.transAxes, fontsize=12, fontweight="bold")
    pcap = axp.text(0.205, 1.04, "the model's output — one value per 4-second window", transform=axp.transAxes, fontsize=12,
                    color=RP.C_PROB, alpha=0)
    pline, = axp.plot([], [], color=RP.C_PROB, lw=2)
    now_p = axp.axvline(S.T0 - on, color=INK, lw=0.8, alpha=0.6)
    p_art = [band, ann, thr_line, thr_txt, ptitle, pline, now_p]
    alarm_art = []

    # change-point markers (D)
    marks = []
    if cp:
        te, th = on + cp[0], on + cp[1]
        for t_c, txt, y in ((te, "EEG changes here", 0.9), (th, f"heart rate follows {cp[1] - cp[0]:.0f} s later", 0.9)):
            ln = axz.axvline(0, color=INK, lw=1.4, alpha=0)
            tx = axz.text(0, y, txt, transform=axz.get_xaxis_transform(), fontsize=11, fontweight="bold", color=INK, va="top", alpha=0,
                          bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none"))
            marks.append(("z", t_c, ln, tx))
        cp_note = fig.text(0.05, 0.012, f"change point = first second more than 3 SD above the 5-min baseline for at least 3 s (Exp R rule): "
                           f"EEG {cp[0]:+.0f} s, heart rate {cp[1]:+.0f} s from the annotated onset", fontsize=9.5, color=INK2, alpha=0)
    # opening / closing cards
    card = fig.add_axes([0, 0, 1, 1], zorder=50); card.set_axis_off()
    card_bg = card.add_patch(plt.Rectangle((0, 0), 1, 1, color="#000000", transform=card.transAxes))
    card_txt = card.text(0.5, 0.5, "", ha="center", va="center", fontsize=26, color=INK, wrap=True, linespacing=1.6)

    def alpha_at(tv, start):
        return float(np.clip((tv - start) / FADE, 0, 1))

    def draw(kind, t, tv):
        # cards
        if kind == "intro":
            fi = draw.counter
            fade = float(np.clip((INTRO_S - fi / a.fps) / FADE, 0, 1))
            card.set_visible(True); card_bg.set_alpha(fade); card_txt.set_alpha(fade)
            card_txt.set_text("\n".join(_wrap(a.intro, 60))); card_txt.set_fontsize(26)
        elif kind == "outro":
            fo = draw.counter - draw.outro_start
            fade = float(np.clip(fo / a.fps / FADE, 0, 1))
            card.set_visible(True); card_bg.set_alpha(fade); card_txt.set_alpha(fade)
            card_txt.set_text("\n".join(a.outro.split("|"))); card_txt.set_fontsize(24)
        else:
            card.set_visible(False)
        # signal bands (B)
        for key, name, where, col, nch in BANDS:
            al = alpha_at(tv, REVEAL[key]) if kind != "intro" else 0.0
            s0, f, x = S.sig[key]
            a0 = max(0, int((t - VIEW_S - s0) * f)); a1 = max(0, int((t - s0) * f))
            tt = np.arange(a0, a1) / f + s0 - t
            seg = x[..., a0:a1]
            for i, ln in enumerate(lines[key]):
                ln.set_data(tt, seg[i] if seg.ndim == 2 else seg); ln.set_alpha(al * (1.0 if i == 0 else 0.6))
            for lab in labels[key]:
                lab.set_alpha(al)
            axes[key].set_visible(al > 0)
            if key == "acc":
                for t_ in tick_ax.get_xticklabels(): t_.set_alpha(al)
        # model output (C)
        alp = alpha_at(tv, REVEAL["prob"]) if kind != "intro" else 0.0
        # relationship-study strip (D): Exp R z curves on the same 20-s axis as the waveforms
        axz.set_visible(alp > 0 and CPS is not None); zleg.set_visible(alp > 0 and CPS is not None)
        for art in zt + [z_thr_ln, z_thr_tx]:
            art.set_alpha(alp)
        for tl in axz.get_xticklabels() + axz.get_yticklabels():
            tl.set_alpha(alp)
        if CPS is not None:
            w = (zt_eeg <= t) & (zt_eeg >= t - VIEW_S - 1)
            zl_eeg.set_data(zt_eeg[w] - t, np.clip(zz_eeg[w], -2, 12)); zl_hr.set_data(zt_eeg[w] - t, np.clip(zz_hr[w], -2, 12))
            zl_eeg.set_alpha(alp); zl_hr.set_alpha(alp)
        sel = tw <= t
        pline.set_data(tw[sel] - on, P[sel])
        rising = t_rise is not None and t >= t_rise
        pline.set_linewidth(3.6 if rising else 2.0)
        pcap.set_alpha(alpha_at(t, t_rise) if rising else 0.0)
        now_p.set_xdata([t - on, t - on])
        for art in p_art:
            art.set_alpha(alp if art is not band else alp * 1.0)
        for tl in axp.get_xticklabels() + axp.get_yticklabels():
            tl.set_alpha(alp)
        axp.set_visible(alp > 0)
        axp.xaxis.label.set_alpha(alp)
        # change points (D)
        for key, t_c, ln, tx in marks:
            x = t_c - t
            vis = -VIEW_S <= x <= 0 and kind != "intro"
            ln.set_xdata([x, x]); ln.set_alpha(0.9 if vis else 0); tx.set_alpha(1.0 if vis else 0)
            right = x > -7                              # near "now": write the label to the left of its line
            tx.set_x(x - 0.15 if right else x + 0.15); tx.set_ha("right" if right else "left")
            if tx.get_bbox_patch(): tx.get_bbox_patch().set_alpha(1.0 if vis else 0)
        if cp:
            cp_note.set_alpha(1.0 if t >= on + cp[0] else 0.0)
        if art_note is not None:
            seen = (zt_eeg <= t) & np.isnan(zz_hr) & np.isfinite(zz_eeg)
            art_note.set_alpha(alp if seen.any() else 0.0)
        # alarm (E)
        if t_alarm is not None and t >= t_alarm and not alarm_art:
            alarm_art.append(axp.axvline(t_alarm - on, color=RP.C_ALARM, lw=2.2))
            alarm_art.append(axp.text(t_alarm - on, 0.55, f"  ALARM  {t_alarm - on:+.0f} s", color=RP.C_ALARM, fontsize=12, fontweight="bold", va="center"))
        hold = kind == "hold"
        alarm_on = t_alarm is not None and t_alarm <= t <= t_alarm + 30
        badge.set_text("  ALARM: carer notified  " if alarm_on else "  monitoring  ")
        badge.set_fontsize(22 if hold else 13)
        badge.set_position((0.975, 0.862 if hold else 0.872))
        hr_txt.set_visible(not hold)
        badge.set_color(INK if alarm_on else INK2)
        badge.set_bbox(dict(boxstyle="round,pad=0.35", fc=RP.C_ALARM if alarm_on else GRID, ec="none"))
        if hold and not hasattr(draw, "hold_txt"):
            draw.hold_txt = fig.text(0.975, 0.818, f"+{t_alarm - on:.0f} s after the annotated onset", ha="right", va="center", fontsize=16,
                                     color=INK, fontweight="bold")
        if hasattr(draw, "hold_txt"):
            draw.hold_txt.set_alpha(1.0 if hold else 0.0)
        if t_alarm is None and t >= off and not hasattr(draw, "miss_txt"):
            draw.miss_txt = axp.text(off - on, 0.55, "  no alarm during this seizure", color=INK, fontsize=12, fontweight="bold", va="center")
        # header clock and heart rate (F)
        j = min(len(S.hr) - 1, int(max(0, t - S.win) // S.step))
        hr_txt.set_text(f"heart rate (pipeline): {S.hr[j]:.0f} bpm" if np.isfinite(S.hr[j]) else "heart rate: —")
        hh, mm, ss = int(t // 3600), int(t % 3600 // 60), int(t % 60)
        clock.set_text(f"{hh:02d}:{mm:02d}:{ss:02d}   ({t - on:+.0f} s)")

    def _wrap(s, w):
        import textwrap
        return textwrap.wrap(s, w)

    draw.counter = 0
    draw.outro_start = len(frames) - int(OUTRO_S * a.fps)

    def step_frame(i):
        draw.counter = i
        draw(*frames[i])
        return []

    out = L.ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    if a.preview:
        for sec in a.preview:
            i = min(len(frames) - 1, int(sec * a.fps))
            for j in range(0, i + 1):          # replay state (alarm line, captions) up to this frame
                step_frame(j)
            fig.savefig(out.with_name(out.stem + f"_t{sec:05.1f}.png"))
        print("previews written next to", out); return
    ff = shutil.which("ffmpeg")
    if not ff:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["animation.ffmpeg_path"] = ff
    ani = animation.FuncAnimation(fig, step_frame, frames=len(frames), blit=False)
    ani.save(str(out), writer=animation.FFMpegWriter(fps=a.fps, bitrate=6000))
    print(f"saved {out}: {len(frames) / a.fps:.1f} s ({INTRO_S} s card, playback at {a.speed:g}x"
          f"{', 1.5 s hold at the alarm' if t_alarm else ''}, {OUTRO_S} s card)")


if __name__ == "__main__":
    main()

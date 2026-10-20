"""Shared code for ICTRA: harmonisation, windowing, features, post-processing, SzCORE-style event scoring, bootstrap.

Everything that must be identical between training data and any test data (SeizeIT2 locked test, external datasets,
the team's own board) lives here, so there is exactly one code path (SlopeAI rule: one terrain code path).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch, sosfilt, sosfilt_zi, lfilter, lfilter_zi

ROOT = Path(__file__).resolve().parents[1]
# Bump whenever anything in this file changes; 03_features.py refuses to add files to data/features/ made by another version.
LIB_VERSION = "lib-2026-10-20f"   # a: first real-data version; b: dropout repair, ACC to g, masked FA hours, memory fixes;
                                  # c: vectorised windows_to_events (same output); d: lab_status helper (no signal change);
                                  # e: montage indicator features eeg_mont_* (Allen, 2026-10-17: v01 must see the montage)
                                  # f: causal mode (v11, CFG["causal"], features in data/features_causal/); the default
                                  #    (offline) path is unchanged, checked by src/22_causality_check.py --regression
DEFAULT_PATH_SAME_AS = ("lib-2026-10-17e",)   # feature stamps whose OFFLINE output is identical to this version's
CFG = dict(
    fs=256,                 # every signal is resampled to this
    fs_acc=25,
    win_s=4, step_s=2,      # analysis window and hop
    eeg_band=(1, 25), ecg_band=(0.5, 40), emg_band=(20, 100),
    baseline_s=300,         # rolling baseline for "relative" features (per recording, causal)
    label_overlap=0.5,      # window is seizure if >= this fraction overlaps an annotated seizure
    post_k=3, post_n=5,     # alarm needs k of the last n windows positive
    merge_s=90, split_s=300, tol_pre_s=30, tol_post_s=60,   # SzCORE event rules
    causal=False,           # v11: every step uses only samples up to the window end (set_mode / mode_for_version)
)
EEG_FEATS = ["ll", "logvar", "zcr", "d", "th", "a", "b", "sef", "kurt"]
MODALITY_PREFIX = {"eeg": "eeg_", "ecg": "ecg_", "emg": "emg_", "acc": "acc_"}
FEATURE_SETS = {
    "A_eeg": ["eeg"],
    "B_eeg_ecg": ["eeg", "ecg"],
    "C_eeg_ecg_emg_acc": ["eeg", "ecg", "emg", "acc"],
    "ecg_only": ["ecg"], "emg_only": ["emg"], "acc_only": ["acc"], "emg_acc": ["emg", "acc"],
}


# ---------------------------------------------------------------- offline vs causal mode (v11)
def set_mode(causal):
    """Offline (default) or causal processing for this process; also the ProcessPoolExecutor initializer of 03/04/05."""
    CFG["causal"] = bool(causal)


def features_dir():
    return ROOT / ("data/features_causal" if CFG["causal"] else "data/features")


def write_version_meta(version, **kw):
    d = ROOT / "results/dev" / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "version_meta.json").write_text(json.dumps(dict(causal=CFG["causal"], lib_version=LIB_VERSION, **kw), indent=1))


def mode_for_version(version):
    """Set the mode a version was trained in (results/dev/<version>/version_meta.json; absent = offline)."""
    f = ROOT / "results/dev" / version / "version_meta.json"
    set_mode(json.loads(f.read_text()).get("causal", False) if f.exists() else False)
    return CFG["causal"]


# ---------------------------------------------------------------- signal harmonisation
def _x0(x):
    """First sample of every channel (the causal filters start in steady state at this level)."""
    return np.asarray(x)[..., :1]


def bandpass(x, lo, hi, fs, order=4):
    """Offline: zero-phase (sosfiltfilt, uses future samples). Causal: forward-only sosfilt started in steady state at the
    first sample; same Butterworth design, so it adds the filter's group delay instead of looking ahead."""
    sos = butter(order, [lo, min(hi, fs / 2 - 1)], btype="band", fs=fs, output="sos")
    if CFG["causal"]:
        zi = sosfilt_zi(sos).reshape((sos.shape[0],) + (1,) * (np.ndim(x) - 1) + (2,)) * _x0(x)[None]
        return sosfilt(sos, x, axis=-1, zi=zi)[0]
    return sosfiltfilt(sos, x, axis=-1)


def notch(x, f0, fs):
    b, a = iirnotch(f0, 30, fs)
    if CFG["causal"]:
        zi = lfilter_zi(b, a).reshape((1,) * (np.ndim(x) - 1) + (-1,)) * _x0(x)
        return lfilter(b, a, x, axis=-1, zi=zi)[0]
    return filtfilt(b, a, x, axis=-1)


def dropout_mask(x, fs, k=3.0, frac=0.25, dc_ratio=3.0):
    """Flag SeizeIT2 sample dropouts in a 1-D raw signal that still has its DC level.

    Some SensorDot files contain runs of 1-2 samples pulled toward zero every ~1-4 s (found 2026-10-17, see
    docs/DATA_NOTES.md: ~0.3-0.6 % of EEG/ECG/EMG samples and ~1.7 % of ACC samples in sub-015 run-04; ~0 % in sub-001).
    With raw DC offsets of up to ~1000 uV each dropout becomes a large spike after filtering. A run is flagged only if it
    moves toward zero (at least one sample by > frac of the level), is a V-shape (> k x the difference between its two neighbours), is above
    6 x the median absolute first difference, and the local 1-s level is > dc_ratio x the local MAD (where the level is
    near zero, a dropout is invisible and real signal such as seizure EMG must not be touched)."""
    n = len(x); mask = np.zeros(n, bool)
    if n < 4:
        return mask
    noise = 6 * np.median(np.abs(np.diff(x))) + 1e-9
    fs = int(fs); nb = n // fs
    if nb:
        blk = x[: nb * fs].reshape(nb, fs)
        lvl = np.median(blk, 1); mad = np.median(np.abs(blk - lvl[:, None]), 1) + 1e-9
        if CFG["causal"]:
            # causal: each 1-s block is judged with the PREVIOUS block's level/MAD and the median |diff| of the past
            # 60 blocks (the first block of a signal uses its own: at most 1 s of look-ahead at the very start)
            prev = lambda v: np.r_[v[:1], v[:-1]]
            lvl, mad = prev(lvl), prev(mad)
            bd = np.median(np.abs(np.diff(blk, axis=1)), 1)
            nz = 6 * prev(pd.Series(bd).rolling(60, min_periods=1).median().values) + 1e-9
            noise = np.repeat(nz, fs)
            noise = np.r_[noise, np.full(n - len(noise), noise[-1])]
        far = np.repeat(np.abs(lvl) > dc_ratio * mad, fs)
        far = np.r_[far, np.full(n - len(far), far[-1])]
    else:
        far = np.full(n, np.abs(np.median(x)) > dc_ratio * (np.median(np.abs(x - np.median(x))) + 1e-9))
    for L in (1, 2):
        a = x[: n - L - 1]; b = x[L + 1:]                         # neighbours left (i-1) and right (i+L) of the run i..i+L-1
        ref = (a + b) / 2
        seg = np.stack([x[1 + j: n - L + j] for j in range(L)])
        dev = np.abs(seg - ref).max(0)
        # every sample of the run is closer to zero than the neighbours' level; at least one by more than frac
        flip = seg * ref < 0
        toward0 = np.all((np.abs(seg) < np.abs(ref)) | flip, axis=0) & np.any((np.abs(seg) < (1 - frac) * np.abs(ref)) | flip, axis=0)
        nse = noise[1: n - L] if np.ndim(noise) else noise
        hit = toward0 & (dev > k * np.abs(a - b)) & (dev > nse) & (dev > frac * np.abs(ref)) & far[1: n - L]
        for j in range(L):
            mask[1 + j: n - L + j] |= hit
    return mask


def repair_dropouts(x, fs):
    """(n_ch, n) raw signal with DC -> copy with dropout samples replaced by linear interpolation."""
    x = np.array(x, dtype=np.float64, copy=True)
    i = np.arange(x.shape[-1])
    for c in range(x.shape[0]):
        m = dropout_mask(x[c], fs) & np.isfinite(x[c])
        if m.any() and (~m).sum() > 1:
            if CFG["causal"]:      # hold the last good sample (interpolation would use the next one)
                last = np.maximum.accumulate(np.where(~m, i, 0))
                x[c, m] = x[c, last[m]]
            else:
                x[c, m] = np.interp(i[m], i[~m], x[c, ~m])
    return x


def harmonise(x, fs_in, kind, fs_out=None, mains=50):
    """Repair sample dropouts, resample to CFG fs, band-pass by signal kind, notch mains. x: (n_ch, n) float, raw (with DC)."""
    from scipy.signal import resample_poly
    from math import gcd
    fs_out = fs_out or (CFG["fs_acc"] if kind == "acc" else CFG["fs"])
    x = repair_dropouts(x, fs_in)
    # offline: remove the chunk median; causal: remove the first sample (the filters below start in steady state there).
    # resample_poly is a zero-phase FIR: in causal mode it is the one bounded look-ahead left (half its length, about
    # 40 ms at 250 -> 256 Hz), which a device sampling at 256 Hz does not need
    x = np.nan_to_num(x - (np.nan_to_num(_x0(x)) if CFG["causal"] else np.nanmedian(x, axis=-1, keepdims=True)))
    if int(round(fs_in)) != fs_out:
        g = gcd(int(round(fs_in)), fs_out)
        x = resample_poly(x, fs_out // g, int(round(fs_in)) // g, axis=-1)
    if kind == "eeg":
        x = bandpass(x, *CFG["eeg_band"], fs_out)
    elif kind == "ecg":
        x = bandpass(x, *CFG["ecg_band"], fs_out)
    elif kind == "emg":
        x = notch(bandpass(x, *CFG["emg_band"], fs_out), mains, fs_out)
    return x.astype(np.float32)


def acc_to_g(acc_raw, fs):
    """(3, n) raw accelerometer in any unit -> same signal scaled so the recording's median |a| is 1 g.
    SeizeIT2 mov files say 'g' but the resting magnitude differs between recordings (about 0.5 in most runs, 0.167 in
    sub-019 run-07; docs/DATA_NOTES.md), so gravity is used as the calibration. Dropouts are repaired first."""
    x = repair_dropouts(acc_raw, fs)
    mag = np.sqrt((x ** 2).sum(0))
    if CFG["causal"]:
        # causal: running median |a| over the past 10 min of 1-s block medians (previous blocks only; the first block
        # uses itself)
        f = max(1, int(round(fs))); nb = max(1, len(mag) // f)
        bm = np.array([np.nanmedian(mag[i * f:(i + 1) * f]) for i in range(nb)])
        g = pd.Series(bm).rolling(600, min_periods=1).median().values
        g = np.r_[g[:1], g[:-1]]
        scale = np.repeat(g, f)
        scale = np.r_[scale, np.full(max(0, len(mag) - len(scale)), scale[-1])][: len(mag)]
        return x / (scale + 1e-12)
    return x / (np.nanmedian(mag) + 1e-12)


def bte_proxy_from_scalp(data, ch_names):
    """Behind-the-ear proxy from 10-20 scalp EEG: bipolar T7-P7 and T8-P8 (T3-T5 / T4-T6 in old names).
    Returns (2, n) array or None if the channels are missing. Imperfect proxy: SeizeIT2's electrodes sit near T7/T8 and P9/P10."""
    names = {n.upper().replace("EEG", "").replace("-REF", "").replace("-LE", "").strip(): i for i, n in enumerate(ch_names)}
    def pick(*alts):
        for a in alts:
            if a in names: return names[a]
        return None
    t7, p7, t8, p8 = pick("T7", "T3"), pick("P7", "T5"), pick("T8", "T4"), pick("P8", "T6")
    if None in (t7, p7, t8, p8):
        return None
    return np.stack([data[t7] - data[p7], data[t8] - data[p8]])


# ---------------------------------------------------------------- windowing + features
def n_windows(n_samples, fs):
    w, s = CFG["win_s"] * fs, CFG["step_s"] * fs
    return max(0, (n_samples - w) // s + 1)


def frame(x, fs):
    """(n_ch, n) -> (n_ch, n_win, win_len) view."""
    w, s = CFG["win_s"] * fs, CFG["step_s"] * fs
    nw = n_windows(x.shape[-1], fs)
    idx = np.arange(w)[None, :] + s * np.arange(nw)[:, None]
    return x[:, idx]


def flat_windows(x, fs, min_range_uv=1.0):
    """Per-window lead-off / saturation mask: True where the signal is flat (peak-to-peak < min_range_uv) in every channel.
    Found by Carina's audit: 9 SeizeIT2 ECG files contain flat stretches of >= 1 h (e.g. sub-021 run-08, 8 h at -128)."""
    F = frame(np.atleast_2d(x), fs)
    return ((F.max(-1) - F.min(-1)) < min_range_uv).all(0)


def _bandpowers(F, fs):
    """F: (n_ch, n_win, w). Returns log band powers for 1-4, 4-8, 8-13, 13-25 Hz and spectral edge frequency (95%)."""
    w = F.shape[-1]
    spec = np.abs(np.fft.rfft(F * np.hanning(w), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(w, 1 / fs)
    def bp(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return np.log(spec[..., m].sum(-1) + 1e-9)
    tot = spec[..., (freqs >= 1) & (freqs < 25)]
    cum = np.cumsum(tot, -1) / (tot.sum(-1, keepdims=True) + 1e-12)
    sef = freqs[(freqs >= 1) & (freqs < 25)][np.argmax(cum >= 0.95, axis=-1)]
    return bp(1, 4), bp(4, 8), bp(8, 13), bp(13, 25), sef


def eeg_features(x, fs=None, block=2048):
    """x: (2, n) harmonised EEG -> DataFrame, one row per window. Computed in blocks of windows to bound memory
    (an 18-h run is ~32,000 windows); identical output to a single pass."""
    fs = fs or CFG["fs"]
    nw = n_windows(x.shape[-1], fs)
    s = CFG["step_s"] * fs; w = CFG["win_s"] * fs
    if nw > block:
        parts = [_eeg_features_block(x[:, i * s: (min(i + block, nw) - 1) * s + w], fs) for i in range(0, nw, block)]
        return pd.concat(parts, ignore_index=True)
    return _eeg_features_block(x, fs)


def _eeg_features_block(x, fs):
    F = frame(x, fs)
    ll = np.log(np.abs(np.diff(F, axis=-1)).sum(-1) + 1e-9)
    logvar = np.log(F.var(-1) + 1e-9)
    zcr = (np.diff(np.sign(F), axis=-1) != 0).mean(-1)
    d, th, a, b, sef = _bandpowers(F, fs)
    m, s = F.mean(-1, keepdims=True), F.std(-1, keepdims=True) + 1e-9
    kurt = (((F - m) / s) ** 4).mean(-1)
    out = {}
    for c in range(F.shape[0]):
        for name, v in zip(EEG_FEATS, [ll, logvar, zcr, d, th, a, b, sef, kurt]):
            out[f"eeg_{name}_{c}"] = v[c]
    df = pd.DataFrame(out)
    df["eeg_ll_max"] = df[[f"eeg_ll_{c}" for c in range(F.shape[0])]].max(1)
    df["eeg_asym_ll"] = df["eeg_ll_0"] - df["eeg_ll_1"] if F.shape[0] > 1 else 0.0
    return df


def heart_rate_series(ecg, fs=None):
    """Per-second heart rate (bpm) and beat-quality flag from a 1-channel harmonised ECG."""
    import neurokit2 as nk
    fs = fs or CFG["fs"]
    n_sec = len(ecg) // fs
    hr = np.full(n_sec, np.nan); q = np.zeros(n_sec)
    if CFG["causal"]:
        return _heart_rate_causal(ecg, fs, n_sec)
    try:
        _, info = nk.ecg_peaks(nk.ecg_clean(ecg.astype(np.float64), sampling_rate=fs), sampling_rate=fs)
        peaks = np.asarray(info["ECG_R_Peaks"])
    except Exception:
        return hr, q
    if len(peaks) < 3:
        return hr, q
    rr = np.diff(peaks) / fs
    t = peaks[1:] / fs
    bpm = 60 / rr
    ok = (bpm > 30) & (bpm < 220)
    sec = np.arange(n_sec) + 0.5
    hr = np.interp(sec, t[ok], bpm[ok], left=np.nan, right=np.nan) if ok.sum() > 2 else hr
    # quality: a beat within 2 s of this second (nearest beat by binary search; memory O(n))
    j = np.clip(np.searchsorted(t, sec), 1, len(t) - 1)
    q = (np.minimum(np.abs(sec - t[j - 1]), np.abs(sec - t[j])) < 2).astype(float)
    return hr, q


QRS_REFRACTORY_S = 0.3   # a beat can be replaced by a larger maximum within this time, so it is final only after it


def causal_r_peaks(ecg, fs):
    """Causal QRS detector for v11. neurokit's detector smooths with a centred window and sets its QRS-length threshold
    from the whole recording, so it cannot run live. Every step here uses past samples only: 5-15 Hz causal band-pass,
    derivative, square, trailing 100-ms integration (Pan & Tompkins 1985). A local maximum of that envelope (confirmed one
    sample later, 4 ms) above 35 % of the envelope's maximum over the previous 2 s is a beat; a larger maximum within
    QRS_REFRACTORY_S of the last beat replaces it, so a beat is final QRS_REFRACTORY_S after it. Within 360 ms of the
    previous beat a maximum below half of that beat's is a T wave. Parameters chosen on 12 dev runs against offline
    neurokit HR (results/data_checks/causality_check.md); beat times lag the R peak by a roughly constant delay, which
    leaves RR intervals unchanged."""
    x = np.asarray(ecg, np.float64)
    if len(x) < int(3 * fs):
        return np.zeros(0, int)
    sos = butter(2, [5, 15], btype="band", fs=fs, output="sos")
    y = sosfilt(sos, x - x[0])
    c = np.cumsum(np.r_[0.0, np.diff(y)] ** 2)
    w = max(1, int(0.1 * fs))
    env = (c - np.r_[np.zeros(w), c[:-w]]) / w
    ref = pd.Series(env).rolling(int(2 * fs), min_periods=1).max().shift(1).bfill().values
    cand = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > 0.35 * ref[1:-1])) + 1
    peaks, last = [], -10 ** 9
    R, T = int(QRS_REFRACTORY_S * fs), int(0.36 * fs)
    for i in cand:
        if i - last < R:
            if peaks and env[i] > env[last]:
                peaks[-1] = i; last = i
            continue
        if peaks and i - last < T and env[i] < 0.5 * env[last]:
            continue
        peaks.append(i); last = i
    return np.asarray(peaks, int)


def _heart_rate_causal(ecg, fs, n_sec):
    """Per-second HR from causal R peaks: the RR interval of the last beat that is FINAL at the second's midpoint (beat
    time + QRS_REFRACTORY_S <= midpoint), held rather than interpolated towards the next beat; quality = such a beat within
    the previous 2 s."""
    hr = np.full(n_sec, np.nan); q = np.zeros(n_sec)
    peaks = causal_r_peaks(ecg, fs)
    if len(peaks) < 3:
        return hr, q
    t = peaks[1:] / fs; bpm = 60 / (np.diff(peaks) / fs)
    ok = (bpm > 30) & (bpm < 220); t, bpm = t[ok], bpm[ok]
    if len(t) < 3:
        return hr, q
    sec = np.arange(n_sec) + 0.5
    j = np.searchsorted(t + QRS_REFRACTORY_S, sec, side="right") - 1
    has = j >= 0
    hr[has] = bpm[j[has]]
    q[has] = ((sec[has] - t[j[has]]) < 2).astype(float)
    return hr, q


def ecg_features(ecg, fs=None):
    fs = fs or CFG["fs"]
    hr, q = heart_rate_series(ecg, fs)
    nw = n_windows(len(ecg), fs)
    win, step = CFG["win_s"], CFG["step_s"]
    base = pd.Series(hr).rolling(CFG["baseline_s"], min_periods=60).median().shift(win).values  # causal 5-min baseline
    rows = {k: np.full(nw, np.nan) for k in ["ecg_hr", "ecg_hr_max", "ecg_hr_rise", "ecg_hr_rise_pct", "ecg_rmssd", "ecg_sdnn", "ecg_q", "ecg_hr_slope"]}
    for i in range(nw):
        a = i * step; b = a + win
        h = hr[a:b]; hb = base[min(b, len(base) - 1)]
        rows["ecg_q"][i] = q[a:b].mean()
        if np.isfinite(h).sum() >= 2:
            rows["ecg_hr"][i] = np.nanmean(h); rows["ecg_hr_max"][i] = np.nanmax(h)
            rows["ecg_sdnn"][i] = np.nanstd(h); rows["ecg_rmssd"][i] = np.sqrt(np.nanmean(np.diff(h) ** 2))
            rows["ecg_hr_slope"][i] = (h[-1] - h[0]) / win if np.isfinite(h[0]) and np.isfinite(h[-1]) else np.nan
            if np.isfinite(hb) and hb > 0:
                rows["ecg_hr_rise"][i] = rows["ecg_hr_max"][i] - hb
                rows["ecg_hr_rise_pct"][i] = 100 * rows["ecg_hr_rise"][i] / hb
    return pd.DataFrame(rows)


def emg_features(emg, fs=None):
    fs = fs or CFG["fs"]
    F = frame(emg[None], fs)[0]
    rms = np.log(np.sqrt((F ** 2).mean(-1)) + 1e-9)
    zcr = (np.diff(np.sign(F), axis=-1) != 0).mean(-1)
    env = np.abs(F).reshape(F.shape[0], -1, fs // 8).mean(-1)          # 125 ms envelope
    burst = (env > 2 * np.median(env, axis=-1, keepdims=True)).mean(-1)
    df = pd.DataFrame({"emg_logrms": rms, "emg_zcr": zcr, "emg_burst_frac": burst})
    df["emg_logrms_rel"] = df.emg_logrms - df.emg_logrms.rolling(CFG["baseline_s"] // CFG["step_s"], min_periods=10).median().shift(1)
    return df


def acc_features(acc, fs=None):
    """acc: (3, n) in any unit at CFG fs_acc."""
    fs = fs or CFG["fs_acc"]
    mag = np.sqrt((acc.astype(np.float64) ** 2).sum(0))
    F = frame(mag[None], fs)[0]
    sd = np.log(F.std(-1) + 1e-9)
    jerk = np.log(np.abs(np.diff(F, axis=-1)).mean(-1) + 1e-9)
    spec = np.abs(np.fft.rfft(F - F.mean(-1, keepdims=True), axis=-1)) ** 2
    fr = np.fft.rfftfreq(F.shape[-1], 1 / fs)
    p_motor = np.log(spec[:, (fr >= 1) & (fr <= 8)].sum(-1) + 1e-9)
    df = pd.DataFrame({"acc_logsd": sd, "acc_logjerk": jerk, "acc_p1_8": p_motor})
    df["acc_logsd_rel"] = df.acc_logsd - df.acc_logsd.rolling(CFG["baseline_s"] // CFG["step_s"], min_periods=10).median().shift(1)
    return df


MONTAGES = ["left", "right", "generalized"]


def montage_features(df, montage):
    """One-hot EEG montage per window (eeg_mont_left / _right / _generalized). Channel 1 is the cross-head channel in
    left/right runs but the right behind-the-ear channel in generalized runs (docs/DATA_NOTES.md section 1), so the
    model must know which. External scalp data through bte_proxy_from_scalp count as "generalized" (left + right)."""
    for m in MONTAGES:
        df[f"eeg_mont_{m}"] = np.float32(montage == m)
    return df


def add_relative_eeg(df):
    """Causal per-recording normalisation: feature minus rolling 5-min median (robust to amplifier gain and electrode contact)."""
    n = CFG["baseline_s"] // CFG["step_s"]
    for c in [c for c in df.columns if c.startswith("eeg_ll_") or c.startswith("eeg_logvar_") or c in ("eeg_ll_max",)]:
        df[c + "_rel"] = df[c] - df[c].rolling(n, min_periods=10).median().shift(1)
    return df


def window_labels(nw, events, step=None, win=None):
    """events: DataFrame with onset_s, duration_s, type. Returns overlap fraction, seizure id, impd mask per window."""
    step, win = step or CFG["step_s"], win or CFG["win_s"]
    t0 = np.arange(nw) * step; t1 = t0 + win
    overlap = np.zeros(nw); szid = np.full(nw, -1); impd = np.zeros(nw, bool)
    for j, e in enumerate(events.itertuples()):
        a, b = e.onset_s, e.onset_s + e.duration_s
        ov = np.clip(np.minimum(t1, b) - np.maximum(t0, a), 0, None) / win
        if str(e.type).startswith("sz"):
            upd = ov > overlap
            overlap[upd] = ov[upd]; szid[upd] = j
        elif str(e.type) == "impd":
            impd |= ov > 0
    return overlap, szid, impd


# ---------------------------------------------------------------- post-processing + event scoring
def k_of_n(pos, k=None, n=None):
    k, n = k or CFG["post_k"], n or CFG["post_n"]
    c = np.convolve(pos.astype(int), np.ones(n, int), "full")[: len(pos)]
    return c >= k


def windows_to_events(alarm, step=None, win=None):
    """Boolean per window -> list of (start_s, end_s); merge gaps < merge_s, split events > split_s.
    Vectorised (2026-10-17); identical output to the original loop on 3,000 random alarm sequences."""
    step, win = step or CFG["step_s"], win or CFG["win_s"]
    a = np.asarray(alarm, bool)
    if not a.any():
        return []
    d = np.diff(np.r_[0, a.astype(np.int8), 0])
    i = np.flatnonzero(d == 1); j = np.flatnonzero(d == -1) - 1          # runs of consecutive alarm windows
    # offline: an alarm run spans its windows; causal: the alarm exists from the END of the window that raised it (when
    # the data it needs has arrived) to the end of the last alarm window, so delays are not one window early
    s = (i * step + win) if CFG["causal"] else i * step
    e = j * step + win
    new = np.r_[True, (s[1:] - e[:-1]) >= CFG["merge_s"]]                  # a gap < merge_s joins the previous event
    ms = s[new]; me = np.maximum.reduceat(e, np.flatnonzero(new))
    n = np.maximum(1, np.ceil((me - ms) / CFG["split_s"]).astype(int))    # events longer than split_s are cut
    rep = np.repeat(np.arange(len(ms)), n); m = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ps = ms[rep] + CFG["split_s"] * m; pe = np.minimum(ps + CFG["split_s"], me[rep])
    return list(zip(ps.tolist(), pe.tolist()))


def score_events(ref, hyp, hours):
    """SzCORE-style any-overlap scoring. ref/hyp: lists of (start, end) seconds. Tolerance 30 s before / 60 s after ref."""
    tp_ref = 0
    for a, b in ref:
        if any(hs < b + CFG["tol_post_s"] and he > a - CFG["tol_pre_s"] for hs, he in hyp): tp_ref += 1
    tp_hyp = sum(1 for hs, he in hyp if any(hs < b + CFG["tol_post_s"] and he > a - CFG["tol_pre_s"] for a, b in ref))
    fp = len(hyp) - tp_hyp
    sens = tp_ref / len(ref) if ref else np.nan
    prec = tp_hyp / len(hyp) if hyp else (np.nan if not ref else 0.0)
    f1 = 2 * sens * prec / (sens + prec) if (sens and prec) else (0.0 if ref else np.nan)
    delays = []
    for a, b in ref:
        hits = [hs for hs, he in hyp if hs < b + CFG["tol_post_s"] and he > a - CFG["tol_pre_s"]]
        if hits: delays.append(min(hits) - a)
    return dict(n_ref=len(ref), n_hyp=len(hyp), tp=tp_ref, fp=fp, sensitivity=sens, precision=prec, f1=f1,
                fa_per_24h=fp / hours * 24 if hours else np.nan, hours=hours,
                median_delay_s=float(np.median(delays)) if delays else np.nan)


def score_recording(pred_pos, events, nw, step=None, mask=None):
    """pred_pos: boolean per window after thresholding (before k-of-n). events: run's seizure DataFrame.
    mask: windows that can never alarm (impd, flat EEG); they are removed from pred_pos and from the hours that are the
    false-alarm denominator (EVALUATION_PROTOCOL.md section 1)."""
    step = step or CFG["step_s"]
    if mask is not None:
        pred_pos = pred_pos & ~mask
    alarm = k_of_n(pred_pos)
    hyp = windows_to_events(alarm)
    ref = [(e.onset_s, e.onset_s + e.duration_s) for e in events.itertuples() if str(e.type).startswith("sz")]
    n_eff = nw - (int(mask.sum()) if mask is not None else 0)
    return score_events(ref, hyp, hours=n_eff * step / 3600), hyp


def pool_scores(per_run):
    """Sum counts over runs (of one patient or one fold) and recompute the rates."""
    d = pd.DataFrame(per_run)
    n_ref, tp, fp, n_hyp, hours = d.n_ref.sum(), d.tp.sum(), d.fp.sum(), d.n_hyp.sum(), d.hours.sum()
    sens = tp / n_ref if n_ref else np.nan
    prec = (n_hyp - fp) / n_hyp if n_hyp else np.nan
    f1 = 2 * sens * prec / (sens + prec) if (n_ref and n_hyp and (sens + prec) > 0) else (0.0 if n_ref else np.nan)
    return dict(n_ref=int(n_ref), tp=int(tp), fp=int(fp), n_hyp=int(n_hyp), hours=float(hours), sensitivity=sens, precision=prec,
                f1=f1, fa_per_24h=fp / hours * 24 if hours else np.nan, median_delay_s=np.nanmedian(d.median_delay_s) if d.median_delay_s.notna().any() else np.nan)


def patient_bootstrap(per_patient, metric, n=1000, seed=0):
    """per_patient: list of dicts with counts. Resample patients with replacement, pool, 95% CI of the pooled metric."""
    rng = np.random.default_rng(seed); vals = []
    P = pd.DataFrame(per_patient)
    for _ in range(n):
        idx = rng.integers(0, len(P), len(P))
        vals.append(pool_scores(P.iloc[idx].to_dict("records"))[metric])
    v = np.array(vals, float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def load_splits():
    return json.load(open(ROOT / "data/catalog/splits.json"))


# ---------------------------------------------------------------- ICTRA Lab status (docs/DASHBOARD_SPEC.md)
LAB_DATA = ROOT / "results/lab/data"


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def lab_write_json(name, obj):
    """Atomic write of results/lab/data/<name> (the Lab page only ever reads these files)."""
    import os, time
    LAB_DATA.mkdir(parents=True, exist_ok=True)
    tmp = LAB_DATA / f".{name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(obj, default=_json_default, allow_nan=False), encoding="utf-8")
    for _ in range(50):                                   # Windows: the server may hold the file open for a moment
        try:
            os.replace(tmp, LAB_DATA / name); return
        except PermissionError:
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def json_clean(o):
    """NaN/inf -> None recursively so the files are strict JSON."""
    if isinstance(o, dict): return {str(k): json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [json_clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


def lab_status(update, log=None, src=""):
    """Merge `update` (nested dict) into results/lab/data/status.json under a lock file; optionally append one log line
    (the last 200 are kept). Never raises: a status write must not break a download, a feature run or a training job."""
    import os, time
    from datetime import datetime
    try:
        LAB_DATA.mkdir(parents=True, exist_ok=True)
        lock = LAB_DATA / "status.lock"
        for _ in range(200):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); break
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > 30:  # stale lock from a killed process
                    lock.unlink(missing_ok=True)
                time.sleep(0.05)
        try:
            p = LAB_DATA / "status.json"
            cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            now = datetime.now().isoformat(timespec="seconds")
            _merge(cur, json_clean(update or {}))
            cur["updated"] = now
            if log:
                cur.setdefault("log", []).append({"t": now, "src": src, "msg": str(log)})
                cur["log"] = cur["log"][-200:]
            lab_write_json("status.json", cur)
        finally:
            lock.unlink(missing_ok=True)
    except Exception:
        pass


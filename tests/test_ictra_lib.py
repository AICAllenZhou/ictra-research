"""Unit tests for ictra_lib signal repair, calibration and post-processing.

    python tests/test_ictra_lib.py          # or: python -m pytest tests/

Test signals are small arrays built in memory (never written to data/ or results/, never logged): this is a check of
the code, not data.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ictra_lib as L

FS = 256


def _signal(n_s=60, dc=500.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_s * FS) / FS
    return dc + 20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 3, len(t))


def test_dropouts_found_and_repaired():
    x = _signal()
    clean = x.copy()
    idx = np.arange(300, len(x) - 300, 700)                 # one dropout every ~2.7 s, like the SeizeIT2 files
    x[idx] = 0.05 * x[idx]                                   # 1-sample pull toward zero
    x[idx[::2] + 1] = 0.6 * x[idx[::2] + 1]                  # every second one smeared over 2 samples
    m = L.dropout_mask(x, FS)
    assert m[idx].all(), "every planted dropout must be flagged"
    assert m[idx[::2] + 1].all(), "the smeared second sample must be flagged"
    planted = np.r_[idx, idx[::2] + 1]
    far = [i for i in np.flatnonzero(m) if np.abs(planted - i).min() > 1]
    assert not far, f"flags more than 1 sample away from any dropout: {far[:5]}"   # a 2-sample check may take a neighbour
    assert L.dropout_mask(clean, FS).sum() == 0, "a clean signal must have no flags"
    y = L.repair_dropouts(x[None], FS)[0]
    assert np.abs(y[m] - clean[m]).max() < 25, "every repaired sample must be close to the true signal"
    assert np.array_equal(y[~m], x[~m]), "samples that are not flagged must be unchanged"


def test_no_flags_when_level_is_near_zero():
    x = _signal(dc=0.0)                                      # zero-mean signal: a dropout is invisible, nothing may change
    assert L.dropout_mask(x, FS).sum() == 0


def test_r_peaks_are_not_touched():
    x = _signal(dc=300.0)
    peaks = np.arange(200, len(x) - 200, 220)
    for k in range(-3, 4):                                   # R-peak-like spikes AWAY from zero, 7 samples wide
        x[peaks + k] += 800 * (1 - abs(k) / 4)
    assert not L.dropout_mask(x, FS)[peaks].any()
    x2 = _signal(dc=-300.0)
    for k in range(-3, 4):
        x2[peaks + k] -= 800 * (1 - abs(k) / 4)              # negative DC, spike more negative: still away from zero
    assert not L.dropout_mask(x2, FS)[peaks].any()


def test_acc_calibrated_to_one_g():
    rng = np.random.default_rng(1)
    n = 25 * 600
    g = np.array([0.30, 0.05, -0.40])                        # a resting sensor that stores |a| = 0.5 instead of 1 g
    acc = (g[:, None] + rng.normal(0, 0.002, (3, n))) * 1e6  # stored x 1e6, as 03_features.py reads it
    a = L.acc_to_g(acc, 25)
    assert abs(np.median(np.sqrt((a ** 2).sum(0))) - 1.0) < 1e-3
    acc2 = acc.copy(); acc2[:, ::90] *= 0.1                  # periodic dropouts must not shift the calibration
    a2 = L.acc_to_g(acc2, 25)
    assert abs(np.median(np.sqrt((a2 ** 2).sum(0))) - 1.0) < 1e-3


def _events_loop(alarm, step=2, win=4):
    """The original (pre-vectorisation) windows_to_events, kept here as the reference."""
    ev, i = [], 0
    while i < len(alarm):
        if alarm[i]:
            j = i
            while j + 1 < len(alarm) and alarm[j + 1]: j += 1
            ev.append([i * step, j * step + win]); i = j + 1
        else:
            i += 1
    merged = []
    for s, e in ev:
        if merged and s - merged[-1][1] < L.CFG["merge_s"]: merged[-1][1] = e
        else: merged.append([s, e])
    out = []
    for s, e in merged:
        while e - s > L.CFG["split_s"]:
            out.append((s, s + L.CFG["split_s"])); s += L.CFG["split_s"]
        out.append((s, e))
    return out


def test_windows_to_events_matches_reference():
    rng = np.random.default_rng(2)
    for _ in range(500):
        n = int(rng.integers(1, 2000)); a = rng.random(n) < rng.choice([0.01, 0.1, 0.5, 0.95])
        assert [tuple(map(float, e)) for e in L.windows_to_events(a)] == [tuple(map(float, e)) for e in _events_loop(a)]


def test_score_events_tolerances():
    ref = [(1000, 1060)]
    assert L.score_events(ref, [(975, 980)], 24)["tp"] == 1        # starts 25 s before onset: inside the 30-s tolerance
    assert L.score_events(ref, [(960, 965)], 24)["tp"] == 0        # ends 35 s before onset: outside
    assert L.score_events(ref, [(1115, 1120)], 24)["tp"] == 1      # 55 s after offset: inside the 60-s tolerance
    s = L.score_events(ref, [(5000, 5004)], 24)
    assert s["fp"] == 1 and s["fa_per_24h"] == 1.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok ", t.__name__)
    print(f"{len(tests)} tests passed")

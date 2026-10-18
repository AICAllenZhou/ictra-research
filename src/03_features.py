"""Turn every downloaded SeizeIT2 run into a feature table (one row per 4-s window, 2-s hop).

    python src/03_features.py                 # all downloaded runs of DEV subjects, 2 workers, resumable
    python src/03_features.py --workers 1 --subjects sub-001
    python src/03_features.py --allow-locked  # ONLY after docs/PREREGISTRATION.md is hashed

Output: data/features/<sub>_run-XX.parquet with columns
  t0 (s), eeg_*, ecg_*, emg_*, acc_*, overlap (fraction of window inside a seizure), y (overlap >= 0.5),
  edge (0 < overlap < 0.5), impd (impedance check: masked), flat_eeg/ecg/emg (lead-off or saturated: features blanked),
  sz_id, sz_type, sz_group, sub, run, montage
Runs are processed one modality at a time in 1-hour chunks so a 3 GB machine copes with 18-hour recordings.
A run is skipped if its parquet already exists (delete it to recompute). Failures are logged to data/features/_failed.txt.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, json, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

RAW = L.ROOT / "data/raw/ds005873"
OUTD = L.ROOT / "data/features"
CHUNK_S = 3600
MAINS = 50   # all SeizeIT2 centres are in Europe


def group_of(t):
    if t == "sz_foc_f2b": return "FBTC"
    if "_m" in str(t).replace("_um", ""): return "focal_motor"
    if "_nm" in str(t): return "focal_nonmotor"
    return "focal_unknown_motor"


def read_chunks(path, fs_target, kind, n_ch=None):
    """Yield harmonised (n_ch, n) float32 chunks of CHUNK_S seconds, with 10 s overlap trimmed, from one EDF."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    fs_in = raw.info["sfreq"]; n = raw.n_times; pad = int(10 * fs_in)
    picks = list(range(n_ch)) if n_ch else None
    step = int(CHUNK_S * fs_in)
    for a in range(0, n, step):
        b = min(a + step, n)
        x = raw.get_data(picks=picks, start=max(0, a - pad), stop=min(n, b + pad)) * 1e6
        h = L.harmonise(x, fs_in, kind, fs_out=fs_target, mains=MAINS)
        r = fs_target / fs_in
        lo = int(round((a - max(0, a - pad)) * r)); hi = lo + int(round((b - a) * r))
        yield h[:, lo:hi]


def features_for_run(sub, run, events, montage):
    stem = f"{sub}/ses-01/{{m}}/{sub}_ses-01_task-szMonitoring_run-{run:02d}_{{m}}.edf"
    parts = {}
    p = RAW / stem.format(m="eeg")
    if not p.exists():
        return None
    eeg = np.concatenate(list(read_chunks(p, L.CFG["fs"], "eeg", n_ch=2)), axis=1)
    nw = L.n_windows(eeg.shape[1], L.CFG["fs"])
    parts["eeg"] = L.montage_features(L.add_relative_eeg(L.eeg_features(eeg)), montage)
    flat = {"eeg": L.flat_windows(eeg, L.CFG["fs"])}
    del eeg
    p = RAW / stem.format(m="ecg")
    if p.exists():
        ecg = np.concatenate(list(read_chunks(p, L.CFG["fs"], "ecg", n_ch=1)), axis=1)[0]
        flat["ecg"] = L.flat_windows(ecg, L.CFG["fs"]); parts["ecg"] = L.ecg_features(ecg); del ecg
    p = RAW / stem.format(m="emg")
    if p.exists():
        emg = np.concatenate(list(read_chunks(p, L.CFG["fs"], "emg", n_ch=1)), axis=1)[0]
        flat["emg"] = L.flat_windows(emg, L.CFG["fs"]); parts["emg"] = L.emg_features(emg); del emg
    p = RAW / stem.format(m="mov")
    if p.exists():
        import mne
        names = mne.io.read_raw_edf(p, preload=False, verbose=False).ch_names
        # mov has two SensorDots (docs/DATA_NOTES.md): "EEG SD ACC X/Y/Z" (back of the neck) and "ECGEMG SD ACC X/Y/Z"
        # (left chest). Use the neck device by name; fall back to the first three ACC channels if it is absent.
        idx = [i for i, nme in enumerate(names) if nme.startswith("EEG SD ACC")][:3]
        if len(idx) < 3:
            idx = [i for i, nme in enumerate(names) if "acc" in nme.lower()][:3]
        idx = idx if len(idx) == 3 else [0, 1, 2]
        # ACC is small (3 x 25 Hz), so it is read whole: calibrate to 1 g at rest, then harmonise like every signal
        raw = mne.io.read_raw_edf(p, preload=False, verbose=False)
        fs_mov = raw.info["sfreq"]
        acc = L.harmonise(L.acc_to_g(raw.get_data(picks=idx) * 1e6, fs_mov), fs_mov, "acc", fs_out=L.CFG["fs_acc"])
        parts["acc"] = L.acc_features(acc); del raw, acc
    df = pd.concat([parts[k].iloc[:nw].reset_index(drop=True) for k in parts], axis=1)
    df = df.reindex(range(nw))
    df.insert(0, "t0", np.arange(nw) * L.CFG["step_s"])
    ov, szid, impd = L.window_labels(nw, events)
    df["overlap"] = ov; df["y"] = (ov >= L.CFG["label_overlap"]).astype(np.int8)
    df["edge"] = ((ov > 0) & (ov < L.CFG["label_overlap"])); df["impd"] = impd; df["sz_id"] = szid
    types = events.type.values
    df["sz_type"] = np.where(szid >= 0, types[np.clip(szid, 0, None)], "bckg")
    df["sz_group"] = [group_of(t) if t != "bckg" else "bckg" for t in df.sz_type]
    df["sub"] = sub; df["run"] = run; df["montage"] = montage
    for m in ["ecg", "emg", "acc"]:
        df[f"has_{m}"] = m in parts
    for m in ["eeg", "ecg", "emg"]:
        df[f"flat_{m}"] = np.resize(flat[m], nw) if m in flat else False
    # a flat (lead-off / saturated) signal carries no information: blank its features so the model sees "missing", not "normal"
    for m in ["eeg", "ecg", "emg"]:
        if m in flat:
            cols = [c for c in df.columns if c.startswith(f"{m}_")]
            df.loc[df[f"flat_{m}"].values, cols] = np.nan
    return df


def job(args):
    sub, run, montage = args
    out = OUTD / f"{sub}_run-{run:02d}.parquet"
    if out.exists():
        return (sub, run, "skip")
    try:
        ev_path = RAW / f"{sub}/ses-01/eeg/{sub}_ses-01_task-szMonitoring_run-{run:02d}_events.tsv"
        if not ev_path.exists():
            ev_path = L.ROOT / "data/raw/ds005873_meta" / ev_path.relative_to(RAW)
        ev = pd.read_csv(ev_path, sep="\t").rename(columns={"onset": "onset_s", "duration": "duration_s", "eventType": "type"})
        df = features_for_run(sub, run, ev, montage)
        if df is None:
            return (sub, run, "missing")
        tmp = out.with_suffix(".parquet.tmp")      # atomic: a killed worker never leaves a half-written .parquet
        df.to_parquet(tmp, index=False); os.replace(tmp, out)
        return (sub, run, f"ok {len(df)} win, {int(df.y.sum())} sz-win")
    except Exception as e:
        with open(OUTD / "_failed.txt", "a") as f:
            f.write(f"{sub} run-{run:02d}: {e}\n{traceback.format_exc()}\n")
        return (sub, run, f"FAILED {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--allow-locked", action="store_true")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    stamp = OUTD / "_lib_version.txt"
    have = stamp.read_text().strip() if stamp.exists() else None
    if have and have != L.LIB_VERSION:
        sys.exit(f"data/features/ was made with {have}, ictra_lib is now {L.LIB_VERSION}: delete data/features/ and re-run.")
    if not have and any(OUTD.glob("*.parquet")):
        sys.exit("data/features/ has parquet files without a _lib_version.txt stamp: delete data/features/ and re-run.")
    stamp.write_text(L.LIB_VERSION + "\n")
    R = pd.read_csv(L.ROOT / "data/catalog/seizeit2_runs.csv")
    splits = L.load_splits(); dev = {s for f in splits["dev_folds"].values() for s in f}
    if not a.allow_locked:
        R = R[R["sub"].isin(dev)]
    elif not (L.ROOT / "docs/PREREGISTRATION.sha256").exists():
        sys.exit("Refusing --allow-locked: docs/PREREGISTRATION.sha256 does not exist yet.")
    if a.subjects:
        R = R[R["sub"].isin(a.subjects)]
    # a run is processed only when every catalogued file of it is fully downloaded (size matches) and its events.tsv
    # exists; otherwise a run caught mid-download would be saved without that modality and then skipped forever
    Fc = pd.read_csv(L.ROOT / "data/catalog/seizeit2_files.csv")
    Fc["ok"] = [(RAW / p).exists() and (RAW / p).stat().st_size == b for p, b in zip(Fc.path, Fc.bytes)]
    complete = Fc.groupby(["sub", "run"]).ok.all()
    def ready(sub, run):
        ev = RAW / f"{sub}/ses-01/eeg/{sub}_ses-01_task-szMonitoring_run-{run:02d}_events.tsv"
        return bool(complete.get((sub, run), False)) and ev.exists()
    todo = [(r.sub, int(r.run), r.eeg_montage) for r in R.itertuples() if ready(r.sub, int(r.run))]
    print(f"{len(todo)} complete downloaded runs to process with {a.workers} workers")
    seiz = {(s_, int(r_)) for s_, r_, n_ in zip(R["sub"], R.run, R.n_seizures) if n_ > 0}

    def status(msg=None):
        done = {(f.name.split("_run-")[0], int(f.stem.split("_run-")[1])) for f in OUTD.glob("*.parquet")}
        failed = (OUTD / "_failed.txt").read_text().count(" run-") if (OUTD / "_failed.txt").exists() else 0
        L.lab_status({"features": {
            "updated": pd.Timestamp.now().isoformat(timespec="seconds"), "lib_version": L.LIB_VERSION, "runs_dev_total": int(len(R)), "runs_ready": len(todo), "runs_done": len(done),
            "seizure_runs_total": int(sum(1 for k in zip(R["sub"], R.run) if (k[0], int(k[1])) in seiz)),
            "seizure_runs_done": len(done & seiz), "nonseizure_runs_done": len(done - seiz), "failed": failed, "workers": a.workers}},
            log=msg, src="03_features")

    status(f"{len(todo)} complete runs to process")
    if a.workers == 1:
        for t in todo:
            res = job(t); print(*res); status() if res[2] != "skip" else None
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for res in ex.map(job, todo):
                print(*res, flush=True)
                if res[2] != "skip":
                    status(f"{res[0]} run-{res[1]:02d}: {res[2]}" if str(res[2]).startswith("FAILED") else None)
    status("feature pass finished")


if __name__ == "__main__":
    main()

"""Build the SeizeIT2 catalog (runs + seizures) WITHOUT downloading any signal data.

The OpenNeuro GitHub mirror keeps every small text file (events.tsv, *.json sidecars) in git and stores each .edf as a
git-annex symlink whose name contains the file size and SHA-256. So the whole dataset can be audited in ~1 minute.

    python src/00_seizeit2_catalog.py            # clones to data/raw/ds005873_meta, writes data/catalog/*.csv

Outputs
  data/catalog/seizeit2_runs.csv      one row per recording run: hours, montage, bytes per modality, split, n_seizures
  data/catalog/seizeit2_seizures.csv  one row per seizure: onset, duration, ILAE type, group, split, modality availability
  data/catalog/seizeit2_files.csv     every .edf with its size and SHA-256 (used by 01_download_seizeit2.py)

Split: sub-001..sub-096 = train/dev, sub-097..sub-125 = LOCKED TEST (the authors' suggested split).
Nothing in the locked test is used until the pre-registration in docs/PREREGISTRATION.md is written and hashed.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import glob, json, os, re, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/raw/ds005873_meta"
OUT = ROOT / "data/catalog"
MODS = ["eeg", "ecg", "emg", "mov"]


def split_of(sub):
    return "train" if sub <= "sub-096" else "locked_test"


def group_of(t):
    if t == "sz_foc_f2b":
        return "FBTC"                      # focal to bilateral tonic-clonic: the convulsive, SUDEP-risk seizures
    if "_m" in t.replace("_um", ""):
        return "focal_motor"
    if "_nm" in t:
        return "focal_nonmotor"
    return "focal_unknown_motor"


def write_lab_json():
    """results/lab/data/catalog.json from the committed catalog CSVs (no clone needed): counts only, no signal data."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import ictra_lib as L
    R = pd.read_csv(OUT / "seizeit2_runs.csv"); S = pd.read_csv(OUT / "seizeit2_seizures.csv")
    I = pd.read_csv(OUT / "dataset_inventory.csv"); sp = json.load(open(OUT / "splits.json"))
    by_split = {}
    for k, G in R.groupby("split"):
        by_split[k] = dict(patients=int(G["sub"].nunique()), runs=int(len(G)), hours=float(G.hours.sum()),
                           seizure_runs=int((G.n_seizures > 0).sum()), seizure_run_hours=float(G.hours[G.n_seizures > 0].sum()),
                           runs_with_ecg_emg=int(G.ecg_bytes.notna().sum()), runs_with_mov=int(G.mov_bytes.notna().sum()),
                           montage={m: int(n) for m, n in G.eeg_montage.value_counts().items()})
    sz = S.groupby(["group", "split"]).size().unstack(fill_value=0)
    J = dict(dataset="SeizeIT2 (OpenNeuro ds005873 v1.1.0)", patients=int(R["sub"].nunique()), runs=int(len(R)), hours=float(R.hours.sum()),
             seizures=int(len(S)), by_split=by_split,
             seizures_by_group={g: {k: int(v) for k, v in row.items()} for g, row in sz.iterrows()},
             seizure_duration_s=dict(median=float(S.duration_s.median()), median_by_group={g: float(v) for g, v in S.groupby("group").duration_s.median().items()},
                                     under_10s=int((S.duration_s < 10).sum())),
             seizures_without_ecg_emg={k: int(v) for k, v in S[~S.has_ecg_emg].groupby("split").size().items()},
             dev_folds={k: len(v) for k, v in sp["dev_folds"].items()}, split_rule=sp.get("rule"),
             inventory=I.fillna("").to_dict("records"))
    L.lab_write_json("catalog.json", L.json_clean(J))
    print("wrote", L.LAB_DATA / "catalog.json")


def main():
    import sys
    if "--json" in sys.argv:
        write_lab_json(); return
    if not META.exists():
        META.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/OpenNeuroDatasets/ds005873.git", str(META)], check=True)
    os.chdir(META)
    runs, events, files = [], [], []
    for js in sorted(glob.glob("sub-*/ses-*/eeg/*_eeg.json")):
        sub = js.split("/")[0].split("\\")[0]
        run = int(re.search(r"run-(\d+)", js).group(1))
        base = js[: -len("_eeg.json")]
        j = json.load(open(js))
        r = dict(sub=sub, run=run, hours=round(j["RecordingDuration"] / 3600, 3), eeg_montage=j.get("EEGMontage"),
                 fs_eeg=j["SamplingFrequency"], split=split_of(sub))
        stem = os.path.basename(base)
        for m in MODS:
            p = f"{sub}/ses-01/{m}/{stem}_{m}.edf"
            if os.path.lexists(p):
                target = os.readlink(p) if os.path.islink(p) else ""
                k = re.search(r"SHA256E-s(\d+)--([0-9a-f]{64})", target)
                size, sha = (int(k.group(1)), k.group(2)) if k else (os.path.getsize(p), None)
                r[f"{m}_bytes"] = size
                files.append(dict(sub=sub, run=run, modality=m, path=p, bytes=size, sha256=sha, split=r["split"]))
            else:
                r[f"{m}_bytes"] = None
        e = pd.read_csv(base + "_events.tsv", sep="\t")
        for _, x in e.iterrows():
            events.append(dict(sub=sub, run=run, onset_s=x.onset, duration_s=x.duration, type=x.eventType))
        runs.append(r)
    R, E, F = pd.DataFrame(runs), pd.DataFrame(events), pd.DataFrame(files)
    S = E[E.type.str.startswith("sz")].copy()
    S["group"] = S.type.map(group_of)
    S["split"] = S["sub"].map(split_of)
    Rk = R.set_index(["sub", "run"])
    S["has_ecg_emg"] = [pd.notna(Rk.loc[(s, r), "ecg_bytes"]) for s, r in zip(S["sub"], S.run)]
    S["has_mov"] = [pd.notna(Rk.loc[(s, r), "mov_bytes"]) for s, r in zip(S["sub"], S.run)]
    R["n_seizures"] = R.set_index(["sub", "run"]).index.map(S.groupby(["sub", "run"]).size()).fillna(0).astype(int)
    F["run_has_seizure"] = F.set_index(["sub", "run"]).index.map(R.set_index(["sub", "run"]).n_seizures).values > 0
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "seizeit2_runs.csv", index=False)
    S.to_csv(OUT / "seizeit2_seizures.csv", index=False)
    F.to_csv(OUT / "seizeit2_files.csv", index=False)

    print(f"subjects {R['sub'].nunique()}  runs {len(R)}  hours {R.hours.sum():,.0f}  seizures {len(S)}")
    print(pd.crosstab(S.group, S.split, margins=True))
    for m in MODS:
        print(f"{m}: {F[F.modality == m].bytes.sum() / 1e9:5.1f} GB")
    sub = F[F.run_has_seizure]
    print(f"seizure-containing runs only: {sub.bytes.sum() / 1e9:.1f} GB ({R[R.n_seizures > 0].hours.sum():,.0f} h)")


if __name__ == "__main__":
    main()

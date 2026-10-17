"""Loaders for the external test sets (Task 9). File-format code only: NO seizure labels are read here before
docs/PREREGISTRATION.sha256 exists (EVALUATION_PROTOCOL.md section 3; CLAUDE.md).

Every loader returns the same shape 03_features.py works with:
    dict(signals={"eeg": (2, n) | None, "ecg": (1, n) | None, "emg": (1, n) | None, "acc": (3, n) | None},
         fs={"eeg": .., "ecg": .., "emg": .., "acc": ..}, channels={...}, duration_s=float)
Scalp EEG becomes the behind-the-ear proxy with ictra_lib.bte_proxy_from_scalp (T7-P7, T8-P8). Signals are raw (uV for
EEG/ECG/EMG, any unit for ACC); harmonisation (dropout repair, resampling to 256 Hz, filters, ACC to 1 g) is done
afterwards by the same ictra_lib.harmonise / acc_to_g calls as for SeizeIT2.

    python src/08_external_adapters.py probe --dataset siena      # header + first minute of ONE recording, print shape
    python src/08_external_adapters.py probe --dataset billeci
    python src/08_external_adapters.py probe --dataset szdb
    python src/08_external_adapters.py probe --dataset msg_seer   # needs registration: reports that it cannot run
    python src/08_external_adapters.py probe --dataset osdb       # needs access: reports that it cannot run

The probe fetches only the EDF header and the first 60 s (HTTP range requests; for Billeci one member of the zip is
streamed), patches the record count so the partial file is a valid EDF, and loads it with the adapter. Label files
(Seizures-list-PNxx.txt, Billeci .txt, szdb times / header comments, OSDB event types) are never opened by the probe.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ictra_lib as L

EXT = L.ROOT / "data/raw/external"
SOURCES = {
    "siena": dict(url="https://physionet.org/files/siena-scalp-eeg/1.0.0/PN00/PN00-1.edf", signals="EEG + ECG (EKG)"),
    "billeci": dict(url="https://zenodo.org/records/1415495/files/POne_pat_1-3.zip", signals="EEG + ECG"),
    "szdb": dict(pn_dir="szdb", record="sz01", signals="ECG"),
    "msg_seer": dict(signals="EMG + ACC (Byteflies Sensor Dot)", access="Epilepsy Ecosystem / Seer registration required"),
    "osdb": dict(signals="wrist ACC + HR", access="Open Seizure Database data-sharing agreement required"),
}


def labels_allowed():
    """Seizure labels of external sets may be read only after the pre-registration is hashed."""
    if not (L.ROOT / "docs/PREREGISTRATION.sha256").exists():
        sys.exit("Refusing to read external labels: docs/PREREGISTRATION.sha256 does not exist yet.")


# ------------------------------------------------------------------ generic EDF loader
def _norm(name):
    return name.upper().replace("EEG", "").replace("-REF", "").replace("-LE", "").strip()


def load_edf(path, ecg_names=("EKG", "ECG"), emg_names=("EMG",)):
    """Scalp EDF -> ICTRA signal dict. ECG/EMG picked by channel name; EEG -> BTE proxy (None if T7/P7/T8/P8 missing)."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    fs = raw.info["sfreq"]; names = raw.ch_names
    X = raw.get_data() * 1e6                                           # MNE returns V; EDF unit is uV for EEG/ECG
    out = dict(signals={}, fs={}, channels={}, duration_s=raw.n_times / fs)
    eeg = L.bte_proxy_from_scalp(X, names)
    out["signals"]["eeg"] = eeg; out["fs"]["eeg"] = fs if eeg is not None else None
    out["channels"]["eeg"] = "T7-P7, T8-P8 (proxy)" if eeg is not None else None
    for key, cands in (("ecg", ecg_names), ("emg", emg_names)):
        idx = [i for i, n in enumerate(names) if any(c in n.upper() for c in cands)]
        out["signals"][key] = X[idx[:1]] if idx else None
        out["fs"][key] = fs if idx else None
        out["channels"][key] = names[idx[0]] if idx else None
    out["signals"]["acc"] = None; out["fs"]["acc"] = None; out["channels"]["acc"] = None
    out["all_channels"] = names
    return out


def load_siena(path):
    """Siena Scalp EEG (PhysioNet): PNxx-y.edf, 512 Hz, 10-20 'EEG T3' style names, ECG as 'EKG EKG' (one or two leads)."""
    return load_edf(path, ecg_names=("EKG", "ECG"))


def load_billeci(path):
    """Billeci et al. (Zenodo 1415495): EDF per recording inside POne_pat_*.zip; EEG + ECG channels found by name."""
    return load_edf(path, ecg_names=("EKG", "ECG"))


def load_szdb(record, pn_dir=None, sampto=None):
    """PhysioNet szdb: WFDB ECG (200 Hz). Header comments may contain seizure times, so they are dropped unread."""
    import wfdb
    rec = wfdb.rdrecord(record, pn_dir=pn_dir, sampto=sampto)
    rec.comments = []                                                  # labels live here / in separate files: not read
    i = [k for k, n in enumerate(rec.sig_name) if "ECG" in n.upper() or n.upper().startswith(("MLII", "V", "I"))] or [0]
    scale = {"mV": 1000.0, "uV": 1.0, "V": 1e6}.get(rec.units[i[0]], 1.0)
    x = rec.p_signal[:, i[0]][None] * scale                            # -> uV
    return dict(signals={"eeg": None, "ecg": x, "emg": None, "acc": None},
                fs={"eeg": None, "ecg": rec.fs, "emg": None, "acc": None},
                channels={"eeg": None, "ecg": rec.sig_name[i[0]], "emg": None, "acc": None},
                duration_s=rec.sig_len / rec.fs, all_channels=rec.sig_name)


def load_msg_seer(segment_dir):
    """My Seizure Gauge (Seer export, Byteflies Sensor Dot): one parquet/CSV per channel and segment (format as published
    with the challenge). EMG 250 Hz, ACC 3 axes. UNTESTED: registration not obtained yet."""
    p = Path(segment_dir); files = sorted(p.glob("*.parquet")) + sorted(p.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no Seer export under {p}")
    D = {f.stem: (pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)) for f in files}
    pick = lambda key: [k for k in D if key in k.lower()]
    emg = pick("emg"); acc = [k for k in pick("acc")][:3]
    out = dict(signals={"eeg": None, "ecg": None}, fs={"eeg": None, "ecg": None}, channels={"eeg": None, "ecg": None})
    out["signals"]["emg"] = D[emg[0]].iloc[:, -1].values[None] if emg else None
    out["signals"]["acc"] = np.stack([D[k].iloc[:, -1].values for k in acc]) if len(acc) == 3 else None
    out["fs"]["emg"] = 250.0 if emg else None; out["fs"]["acc"] = 25.0 if len(acc) == 3 else None
    out["channels"]["emg"] = emg[0] if emg else None; out["channels"]["acc"] = acc or None
    n = out["signals"]["emg"].shape[1] / 250.0 if emg else 0.0
    out["duration_s"] = n
    return out


def load_osdb(json_path):
    """Open Seizure Database: JSON list of events, each with 'datapoints' holding 5-s blocks of 25 Hz wrist acceleration
    magnitude ('rawData', 125 values) and heart rate ('hr'). Only ACC is returned; event type (seizure / false alarm)
    is a label and is not read here. UNTESTED: data access not obtained yet."""
    ev = json.load(open(json_path))
    ev = ev if isinstance(ev, list) else [ev]
    blocks = [np.asarray(dp["rawData"], float) for e in ev[:1] for dp in e.get("datapoints", [])]
    mag = np.concatenate(blocks) if blocks else np.zeros(0)
    acc = np.stack([mag, np.zeros_like(mag), np.zeros_like(mag)])       # magnitude only: put it on one axis
    return dict(signals={"eeg": None, "ecg": None, "emg": None, "acc": acc}, fs={"eeg": None, "ecg": None, "emg": None, "acc": 25.0},
                channels={"eeg": None, "ecg": None, "emg": None, "acc": "rawData (magnitude)"}, duration_s=len(mag) / 25.0)


ADAPTERS = {"siena": load_siena, "billeci": load_billeci, "szdb": load_szdb, "msg_seer": load_msg_seer, "osdb": load_osdb}


# ------------------------------------------------------------------ probe: header + first minute only
class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (lets zipfile read one member of a 3 GB zip)."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = requests.head(url, allow_redirects=True, timeout=60); r.raise_for_status()
        self.url = r.url; self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120); r.raise_for_status()
        n = len(r.content); b[:n] = r.content; self.pos += n
        return n


def partial_edf(stream, out, seconds=60):
    """Copy the EDF header + the first `seconds` of records from a byte stream; patch n_records so it is a valid EDF.
    Returns the record count of the ORIGINAL file (for its true duration)."""
    head = stream.read(256); ns = int(head[252:256]); sig = stream.read(256 * ns)
    rec_dur = float(head[244:252]); n_rec_orig = int(head[236:244])
    nsamp = [int(sig[216 * ns + 8 * i: 216 * ns + 8 * (i + 1)]) for i in range(ns)]
    k = max(1, int(np.ceil(seconds / rec_dur))); k = min(k, n_rec_orig) if n_rec_orig > 0 else k
    data = stream.read(k * sum(nsamp) * 2)
    k = len(data) // (sum(nsamp) * 2)
    head = head[:236] + f"{k:<8d}".encode() + head[244:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(head + sig + data[: k * sum(nsamp) * 2])
    return n_rec_orig, rec_dur


def probe(name):
    src = SOURCES[name]
    print(f"== {name}: {src['signals']}")
    if "access" in src:
        print(f"   cannot probe: {src['access']}. Adapter load_{name} is written from the published format and is UNTESTED.")
        return None
    d = EXT / name / "_probe"
    if name == "szdb":
        import wfdb
        hdr = wfdb.rdheader(src["record"], pn_dir=src["pn_dir"]); hdr.comments = []
        r = load_szdb(src["record"], pn_dir=src["pn_dir"], sampto=int(60 * hdr.fs))
        r["duration_s"] = hdr.sig_len / hdr.fs
    elif name == "siena":
        f = d / Path(src["url"]).name
        n_rec, rec_dur = partial_edf(io.BufferedReader(HttpFile(src["url"]), buffer_size=1 << 20), f)
        r = load_siena(f); r["duration_s"] = n_rec * rec_dur
    elif name == "billeci":
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(src["url"]), buffer_size=1 << 20))
        names = z.namelist()
        edfs = [n for n in names if n.lower().endswith(".edf")]
        print(f"   zip members: {len(names)} ({len(edfs)} .edf); first .edf: {edfs[0] if edfs else None}")
        print(f"   member extensions: {sorted({Path(n).suffix.lower() for n in names})}")
        f = d / Path(edfs[0]).name
        with z.open(edfs[0]) as m:
            n_rec, rec_dur = partial_edf(m, f)
        r = load_billeci(f); r["duration_s"] = n_rec * rec_dur
    for k in ["eeg", "ecg", "emg", "acc"]:
        x = r["signals"][k]
        shape = None if x is None else tuple(x.shape)
        print(f"   {k}: channel {r['channels'][k]}, fs {r['fs'][k]}, loaded shape {shape}")
    print(f"   recording duration (from header): {r['duration_s'] / 3600:.2f} h; all channels: {list(r.get('all_channels', []))}")
    # the loaded first minute must survive the shared harmonisation unchanged in shape
    for k in ["eeg", "ecg"]:
        if r["signals"][k] is not None:
            h = L.harmonise(r["signals"][k], r["fs"][k], k)
            print(f"   harmonised {k}: {h.shape} at {L.CFG['fs']} Hz, std {np.std(h):.1f} uV")
    return r


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("probe"); s.add_argument("--dataset", choices=list(SOURCES), required=True)
    a = ap.parse_args()
    probe(a.dataset)


if __name__ == "__main__":
    main()

"""Download SeizeIT2 (OpenNeuro ds005873) .edf files from OpenNeuro's public S3 bucket.

Run 00_seizeit2_catalog.py first (it writes data/catalog/seizeit2_files.csv with every file's size and SHA-256).

    # 1) see what would be downloaded (default: nothing is downloaded)
    python src/01_download_seizeit2.py --runs seizure --modalities eeg ecg emg mov --split train
    # 2) download it
    python src/01_download_seizeit2.py --runs seizure --modalities eeg ecg emg mov --split train --go

Recommended order (see docs/PLAN.md):
  step 1  --runs seizure --split train                  ~26 GB   (all train seizures + the hours around them)
  step 2  --runs nonseizure --split train               ~70 GB   (seizure-free hours -> false alarms per 24 h; disjoint from step 1,
                                                                  so it can run in a second terminal AT THE SAME TIME as step 1)
  step 3  --split locked_test                           only after docs/PREREGISTRATION.md is written and hashed

Every file is streamed to disk (never held in RAM), resumed if interrupted, and verified against its SHA-256.
Files land in data/raw/ds005873/ with the original BIDS paths, plus the small sidecars copied from the metadata clone.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, hashlib, shutil, sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data/catalog/seizeit2_files.csv"
META = ROOT / "data/raw/ds005873_meta"
DEST = ROOT / "data/raw/ds005873"
BASE = "https://s3.amazonaws.com/openneuro.org/ds005873/"


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


BUSY_S = 90        # a .part or .lock touched within this many seconds belongs to another window


def _try(fn):
    """Run a file operation; False instead of PermissionError (Windows: another process has the file open)."""
    try:
        fn(); return True
    except (PermissionError, FileNotFoundError):
        return False


def _recent(path):
    try:
        return time.time() - path.stat().st_mtime < BUSY_S
    except FileNotFoundError:
        return False


def _acquire(lock):
    """Exclusive per-file lock (atomic create). A lock not refreshed for BUSY_S seconds is stale and taken over."""
    import os
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)     # owner PID: the stall watch frees locks of dead PIDs
            return True
        except FileExistsError:
            if _recent(lock) or not _try(lock.unlink):
                return False
        except PermissionError:
            return False
    return False


def fetch(rel, size, sha, tries=4):
    """Download one file to <file>.part, verify size + SHA-256, rename. Returns ok / skip / busy / failed.
    Safe with several windows on the same file list: only the holder of <file>.lock writes the .part; every unlink /
    rename is guarded, and anything another window is touching comes back as "busy" (retried at the end of the run)."""
    import os
    out = DEST / rel
    if out.exists() and out.stat().st_size == size and (sha is None or sha256(out) == sha):
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    lock = out.with_suffix(out.suffix + ".lock")
    if _recent(part) or not _acquire(lock):
        return "busy"          # another window is downloading this file right now
    try:
        for attempt in range(tries):
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                if not _try(part.unlink): return "busy"
                have = 0
            if have < size:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with requests.get(BASE + rel, headers=headers, stream=True, timeout=60) as r:
                    if have and r.status_code != 206:  # server ignored the range: start again
                        if not _try(part.unlink): return "busy"
                        have = 0
                    r.raise_for_status()
                    done, t0, beat = have, time.time(), time.time()
                    with open(part, "ab") as f:
                        for b in r.iter_content(1 << 20):
                            f.write(b); done += len(b)
                            if time.time() - beat > 10:
                                _try(lambda: os.utime(lock)); beat = time.time()   # heartbeat: the lock stays ours
                            rate = (done - have) / max(time.time() - t0, 1e-3) / 1e6
                            print(f"\r  {rel[-58:]:58s} {done / size:6.1%} {rate:5.1f} MB/s", end="", flush=True)
                print()
            got = part.stat().st_size if part.exists() else -1
            if got != size or (sha and sha256(part) != sha):
                why = "size mismatch" if got != size else "checksum mismatch"
                recent = _recent(part)
                if got == size and not _try(part.unlink):   # corrupt full-size .part: delete so the next pass restarts it
                    print(f"  {why}: .part is held open by another process: busy")
                    return "busy"
                print(f"  {why} on attempt {attempt + 1}" + (": busy, left for the second pass" if recent else ": retrying"))
                if recent:
                    return "busy"                  # touched in the last BUSY_S s: do not fight over it now
                time.sleep(5 * (attempt + 1)); continue
            if not _try(lambda: part.replace(out)):
                return "busy"
            return "ok"
    except PermissionError:
        return "busy"
    except Exception as e:      # network hiccup: the .part stays and the next attempt / run resumes from it
        print(f"\n  error: {e}")
        return "busy" if _recent(part) else "failed"
    finally:
        _try(lock.unlink)
    return "failed"


def s3_keys(prefix):
    """All object keys under an S3 prefix (handles the 1,000-key page limit)."""
    import re
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get("https://s3.amazonaws.com/openneuro.org", params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>(.*?)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


def fetch_sidecars(F):
    """Fetch the small BIDS sidecars (events.tsv, *.json, *_channels.tsv) for the selected runs straight from S3.
    Used when the git metadata clone (data/raw/ds005873_meta) is not on this machine. Only subjects in F are listed,
    so with --split train no locked-test path is ever requested."""
    import re
    for p in ["dataset_description.json", "participants.json", "events.json", "README"]:
        if not (DEST / p).exists():
            r = requests.get(BASE + p, timeout=60); r.raise_for_status(); (DEST / p).write_bytes(r.content)
    from concurrent.futures import ThreadPoolExecutor
    want = set(zip(F["sub"], F.run.astype(int)))

    def get(rel):
        out = DEST / rel
        if out.exists():
            return 0
        r = requests.get(BASE + rel, timeout=60); r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_bytes(r.content); tmp.replace(out)   # atomic
        return 1

    n = 0
    for sub in sorted(F["sub"].unique()):
        rels = []
        for key in s3_keys(f"ds005873/{sub}/"):
            rel = key[len("ds005873/"):]
            m = re.search(r"run-(\d+)", rel)
            if not rel.endswith(".edf") and m and (sub, int(m.group(1))) in want:
                rels.append(rel)
        with ThreadPoolExecutor(16) as ex:
            k = sum(ex.map(get, rels))
        n += k
        print(f"  {sub}: {k} new sidecar files", flush=True)
    print(f"sidecars: {n} new files")


def download_status(msg=None):
    """Write download progress (files / GB on disk vs catalogued, train split only) and free disk to the Lab status file.
    Locked-test files are not looked at; the page shows them as locked until docs/PREREGISTRATION.sha256 exists."""
    sys.path.insert(0, str(ROOT / "src"))
    import ictra_lib as L
    F = pd.read_csv(CAT)
    F = F[F.split == "train"]
    have = [(DEST / p).exists() and (DEST / p).stat().st_size == b for p, b in zip(F.path, F.bytes)]
    F = F.assign(have=have)
    out = {}
    for kind, G in (("seizure", F[F.run_has_seizure]), ("nonseizure", F[~F.run_has_seizure])):
        out[kind] = dict(files_total=int(len(G)), files_done=int(G.have.sum()), gb_total=float(G.bytes.sum() / 1e9),
                         gb_done=float(G.bytes[G.have].sum() / 1e9))
    out["locked_test"] = dict(locked=not (ROOT / "docs/PREREGISTRATION.sha256").exists())
    L.lab_status({"download": out, "disk_free_gb": shutil.disk_usage(DEST).free / 1e9}, log=msg, src="01_download")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", choices=["seizure", "nonseizure", "all"], default="seizure",
                    help="seizure = runs containing a seizure; nonseizure = the rest (disjoint, safe to run in parallel with seizure); all = both")
    ap.add_argument("--modalities", nargs="+", default=["eeg", "ecg", "emg", "mov"])
    ap.add_argument("--split", choices=["train", "locked_test", "all"], default="train")
    ap.add_argument("--subjects", nargs="*", help="e.g. sub-001 sub-002 (for a quick first test)")
    ap.add_argument("--go", action="store_true", help="actually download (default is a dry run)")
    ap.add_argument("--shard", default=None, help="i/n: take only every n-th file starting at i (0-based), e.g. 0/4 .. 3/4 in four windows")
    ap.add_argument("--sidecars", action="store_true", help="only fetch events.tsv / json sidecars of the selected runs (KB each)")
    ap.add_argument("--status", action="store_true", help="only write download progress to results/lab/data/status.json")
    a = ap.parse_args()
    if a.status:
        print(download_status()); return

    F = pd.read_csv(CAT)
    F = F[F.modality.isin(a.modalities)]
    if a.runs == "seizure":
        F = F[F.run_has_seizure]
    elif a.runs == "nonseizure":
        F = F[~F.run_has_seizure]
    if a.split != "all":
        F = F[F.split == a.split]
    if a.subjects:
        F = F[F["sub"].isin(a.subjects)]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        F = F.iloc[i::n]
        print(f"shard {i}/{n}")
    todo = F[[not ((DEST / p).exists() and (DEST / p).stat().st_size == b) for p, b in zip(F.path, F.bytes)]]
    need = todo.bytes.sum()
    DEST.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DEST).free
    print(f"{len(F)} files selected, {F.bytes.sum() / 1e9:.1f} GB; still to download: {len(todo)} files, {need / 1e9:.1f} GB")
    print(F.groupby("modality").bytes.sum().div(1e9).round(1).to_string())
    print(f"free disk: {free / 1e9:.0f} GB")
    if a.split in ("locked_test", "all"):
        print("WARNING: includes the LOCKED TEST subjects (sub-097..125). Do not open them before the pre-registration.")
    if a.sidecars:
        if a.split != "train" and not (ROOT / "docs/PREREGISTRATION.sha256").exists():
            sys.exit("Refusing sidecars outside --split train: docs/PREREGISTRATION.sha256 does not exist yet.")
        fetch_sidecars(F)
        return
    if need > free * 0.9:
        sys.exit("Not enough free disk space (keeping a 10% margin).")
    if not a.go:
        print("Dry run only. Add --go to download.")
        return

    # copy small BIDS sidecars (events.tsv, json) so the download folder is a valid BIDS dataset
    if META.exists():
        for p in list(META.glob("*.json")) + list(META.glob("*.tsv")) + [META / "README"]:
            if p.is_file():
                shutil.copy2(p, DEST / p.name)
        for rel in F.path:
            for side in (META / rel).parent.glob("*"):
                if side.suffix in (".json", ".tsv"):
                    (DEST / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(side, DEST / rel.rsplit("/", 1)[0] / side.name)

    stats = {"ok": 0, "skip": 0, "failed": 0, "busy": 0}
    busy = []
    for i, x in enumerate(F.itertuples(), 1):
        print(f"[{i}/{len(F)}] {x.path}")
        r = fetch(x.path, int(x.bytes), x.sha256 if isinstance(x.sha256, str) else None)
        stats[r] += 1
        if r == "busy":
            busy.append(x)
        if r not in ("skip", "busy"):
            try: download_status(f"{x.path.rsplit('/', 1)[-1]}: {r}" if r == "failed" else None)
            except Exception: pass
        if busy and i % 20 == 0:             # in-run retry: a busy file whose .part and .lock went quiet is free again
            quiet = [b for b in busy if not _recent((DEST / b.path).with_suffix(".edf.part")) and not _recent((DEST / b.path).with_suffix(".edf.lock"))]
            for b in quiet:
                r = fetch(b.path, int(b.bytes), b.sha256 if isinstance(b.sha256, str) else None)
                if r != "busy":
                    busy.remove(b); stats["busy"] -= 1; stats[r] += 1
                    print(f"  retried {b.path.rsplit('/', 1)[-1]}: {r}")
    for rnd in range(3):                     # second passes: only the files another window was busy with
        if not busy:
            break
        print(f"{len(busy)} files were busy in another window; checking them again in {BUSY_S + 30} s (pass {rnd + 2})...")
        time.sleep(BUSY_S + 30); again = []
        for x in busy:
            r = fetch(x.path, int(x.bytes), x.sha256 if isinstance(x.sha256, str) else None)
            stats["busy"] -= 1; stats[r] += 1
            if r == "busy":
                again.append(x)
        busy = again
    print(stats)


if __name__ == "__main__":
    main()

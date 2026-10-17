"""Stall watch for the stage-2 (nonseizure) download shards that Claude Code launched (results/lab/logs/nonseizure_pids.json).
Never touches any other download process (Allen's own --runs seizure window is not in the PID file).

    python src/11_download_watch.py           # one check; called every minute by the feature loop

Per shard i: if its process is gone without having printed its final {'ok': ...} line, or its log has not grown for
STALL_S seconds (no .part progress), it is killed and relaunched with the same arguments (resume is automatic: the
.part file is kept). If free disk drops below MIN_FREE_GB every shard is stopped and nothing is relaunched.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import json, shutil, subprocess, sys, time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results/lab/logs"
RAW = ROOT / "data/raw/ds005873"
PIDS = LOGS / "nonseizure_pids.json"
STALL_S, MIN_FREE_GB, N_SHARDS = 300, 15, 3
sys.path.insert(0, str(ROOT / "src"))
import ictra_lib as L


def launch(i):
    out = open(LOGS / f"nonseizure_shard{i}.out.log", "ab"); err = open(LOGS / f"nonseizure_shard{i}.err.log", "ab")
    flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen([sys.executable, "-u", "src/01_download_seizeit2.py", "--runs", "nonseizure", "--split", "train",
                          "--shard", f"{i}/{N_SHARDS}", "--go"], cwd=ROOT, stdout=out, stderr=err, creationflags=flags)
    return p.pid


def finished(i):
    log = LOGS / f"nonseizure_shard{i}.out.log"
    tail = log.read_bytes()[-400:].decode("utf-8", "ignore") if log.exists() else ""
    return "{'ok':" in tail


SEEN = LOGS / "orphans_reported.txt"
ORPHAN_S = 600     # a .lock / .part untouched this long, with no live owner, is an orphan of a killed process


def sweep_orphans():
    """Release .lock files that are idle > ORPHAN_S and whose owner PID (written in the lock) is not running, or that
    carry no PID (locks from before PIDs were recorded). Orphan .part files are KEPT (they hold the resume data); with
    their lock gone, any live shard whose list contains that file resumes it at its next in-run retry. Only locks are
    removed, and only under data/raw/ds005873; Allen's --runs seizure window does not use locks, so it is never touched."""
    events, now = [], time.time()
    seen = set(SEEN.read_text(encoding="utf-8").split()) if SEEN.exists() else set()
    for lock in RAW.rglob("*.lock"):
        try:
            idle = now - lock.stat().st_mtime
            txt = lock.read_text().strip()
        except (FileNotFoundError, PermissionError):
            continue
        pid = int(txt) if txt.isdigit() else None
        if idle > ORPHAN_S and not (pid and psutil.pid_exists(pid)):
            try:
                lock.unlink(); events.append(f"released orphan lock {lock.name} (idle {idle / 60:.0f} min, owner {pid or 'unknown'} gone)")
            except (FileNotFoundError, PermissionError):
                pass
    for part in RAW.rglob("*.part"):
        try:
            idle = now - part.stat().st_mtime
        except FileNotFoundError:
            continue
        if idle > ORPHAN_S and not part.with_suffix(".lock").exists() and part.name not in seen:
            events.append(f"orphan {part.name}: idle {idle / 60:.0f} min, unlocked, resumable by its shard"); seen.add(part.name)
    SEEN.write_text("\n".join(sorted(seen)), encoding="utf-8")
    return events


def main():
    pids = json.loads(PIDS.read_text(encoding="utf-8-sig"))
    free = shutil.disk_usage(ROOT).free / 1e9
    events = []
    for i, pid in sorted(pids.items()):
        alive = psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).is_running()
        if free < MIN_FREE_GB:
            if alive:
                psutil.Process(int(pid)).kill()
            events.append(f"shard {i}: STOPPED, free disk {free:.0f} GB < {MIN_FREE_GB} GB")
            pids[i] = -1; continue
        if pid == -1 or finished(i):
            continue
        log = LOGS / f"nonseizure_shard{i}.out.log"
        idle = time.time() - log.stat().st_mtime if log.exists() else 0
        if alive and idle < STALL_S:
            continue
        if alive:
            try:
                psutil.Process(int(pid)).kill()
            except psutil.NoSuchProcess:
                pass
        why = f"no progress for {idle / 60:.0f} min" if alive else "process exited before finishing"
        pids[i] = launch(int(i))
        events.append(f"shard {i}: {why}; relaunched as PID {pids[i]}")
    PIDS.write_text(json.dumps(pids), encoding="utf-8")
    events += sweep_orphans()
    for e in events:
        print(e, flush=True)
        L.lab_status({}, log=e, src="11_watch")


if __name__ == "__main__":
    main()

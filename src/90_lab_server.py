"""ICTRA Lab, live mode (docs/DASHBOARD_SPEC.md, "Tonight's minimum"). Serves results/lab/index.html and the JSON files
in results/lab/data/ that the pipeline scripts write. Read-only: it never writes a file and never computes a metric.

    python src/90_lab_server.py            # then open http://127.0.0.1:8766
    python src/90_lab_server.py --port 8777

Endpoints
  /                    results/lab/index.html (the page; it draws only what the JSON says)
  /data/<name>.json    status.json, catalog.json, expR.json, dev_<version>.json ... as written by 00/01/02/03/04/05
  /data/index.json     which data files exist right now and their modification times (the page re-fetches a file only
                       when it changed; tabs 3-4 appear as soon as their JSON lands)
  /api/live.json       progress counters read from disk on request: files and GB downloaded per stage (dev split only;
                       the locked test is shown as locked and its files are not looked at), .part files in flight with
                       speed, feature files, disk free. Counting files is progress, not a metric.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per process: otherwise every
    os.environ.setdefault(_v, "1")   # numpy process commits ~1.5 GB of buffers on this 24-thread CPU
import argparse, json, shutil, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "results/lab"
DATA = LAB / "data"
RAW = ROOT / "data/raw/ds005873"
FEAT = ROOT / "data/features"
_files = pd.read_csv(ROOT / "data/catalog/seizeit2_files.csv")
_files = _files[_files.split == "train"].reset_index(drop=True)          # never stat locked-test paths
_size = dict(zip(_files.path, _files.bytes))
_prev = {}                                                               # .part -> (bytes, time) for speed
_lock = threading.Lock()
_cache = {"t": 0, "body": b"{}"}


def live():
    """Progress read from disk (cached 3 s, so many open pages do not re-stat 8,000 files at once)."""
    with _lock:
        if time.time() - _cache["t"] < 3:
            return _cache["body"]
        have = [(RAW / p).exists() and (RAW / p).stat().st_size == b for p, b in zip(_files.path, _files.bytes)]
        F = _files.assign(have=have)
        dl = {}
        for k, G in (("seizure", F[F.run_has_seizure]), ("nonseizure", F[~F.run_has_seizure])):
            dl[k] = dict(files_total=int(len(G)), files_done=int(G.have.sum()), gb_total=float(G.bytes.sum() / 1e9),
                         gb_done=float(G.bytes[G.have].sum() / 1e9))
        dl["locked_test"] = dict(locked=not (ROOT / "docs/PREREGISTRATION.sha256").exists())
        now, flight = time.time(), []
        for p in RAW.rglob("*.part"):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            rel = p.relative_to(RAW).as_posix()[:-5]
            prev = _prev.get(rel); _prev[rel] = (st.st_size, now)
            speed = (st.st_size - prev[0]) / (now - prev[1]) / 1e6 if prev and now > prev[1] else None
            flight.append(dict(file=rel.rsplit("/", 1)[-1], mb=st.st_size / 1e6, total_mb=_size.get(rel, 0) / 1e6,
                               speed_mbs=speed, idle_s=now - st.st_mtime))
        n_feat = len(list(FEAT.glob("*.parquet"))) if FEAT.exists() else 0
        body = json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), download=dl, in_flight=sorted(flight, key=lambda d: d["file"]),
                               feature_files=n_feat, disk_free_gb=shutil.disk_usage(ROOT).free / 1e9)).encode()
        _cache.update(t=now, body=body)
        return body


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                return self.send(200, (LAB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/live.json":
                return self.send(200, live(), "application/json")
            if path == "/data/index.json":
                names = {p.name: p.stat().st_mtime for p in DATA.glob("*.json")} if DATA.exists() else {}
                return self.send(200, json.dumps(names).encode(), "application/json")
            if path.startswith("/data/") and path.endswith(".json"):
                p = (DATA / path[len("/data/"):]).resolve()
                if p.parent == DATA.resolve() and p.exists():
                    for _ in range(5):                      # a writer may be replacing the file this instant
                        try:
                            return self.send(200, p.read_bytes(), "application/json")
                        except PermissionError:
                            time.sleep(0.05)
            return self.send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    print(f"ICTRA Lab: http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

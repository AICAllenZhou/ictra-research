"""Delete raw SeizeIT2 .edf files to free disk, ONLY when nothing downstream can still need them.

    python src/99_cleanup_raw.py            # shows what would be freed and which gates are open/closed
    python src/99_cleanup_raw.py --go       # deletes, only if every gate is open

Raw EDFs are the only thing that cannot be re-created without re-downloading 95-126 GB. Everything that is kept is small:
data/features/ (~3 GB), results/**/preds (~1 GB), models/, data/catalog/, docs/. Feature changes (window length, new
relationship features, montage handling) need the raw files, so the gates below mean "the model is frozen and every test
has been run":
  G1  docs/PREREGISTRATION.sha256 exists                      (model, threshold, hypotheses frozen)
  G2  results/locked_test/RESULTS.md exists                    (tier-2 test done, once)
  G3  results/external/SUMMARY.md exists                       (tier-3 tests done)
  G4  results/INDEPENDENT_REVIEW.md exists and says reproduced (every number re-computed)
  G5  the file docs/DONE_FOR_WAICY.md exists and is signed by Allen (the "good enough" decision, see docs/TASKS.md task 14)
  G6  every run in data/catalog/seizeit2_runs.csv that was downloaded has its parquet in data/features/
--keep-locked keeps sub-097..125 raw files (for the demo replay); --keep-demo keeps the runs listed in docs/DEMO_RUNS.txt.
"""
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ds005873"


def gates():
    g = {
        "G1 pre-registration hashed": (ROOT / "docs/PREREGISTRATION.sha256").exists(),
        "G2 locked test done": (ROOT / "results/locked_test/RESULTS.md").exists(),
        "G3 external tests done": (ROOT / "results/external/SUMMARY.md").exists(),
        "G4 independent review done": (ROOT / "results/INDEPENDENT_REVIEW.md").exists()
                                      and "reproduced" in (ROOT / "results/INDEPENDENT_REVIEW.md").read_text(errors="ignore").lower(),
        "G5 Allen signed docs/DONE_FOR_WAICY.md": (ROOT / "docs/DONE_FOR_WAICY.md").exists()
                                      and "signed" in (ROOT / "docs/DONE_FOR_WAICY.md").read_text(errors="ignore").lower(),
    }
    R = pd.read_csv(ROOT / "data/catalog/seizeit2_runs.csv")
    missing = [f"{r.sub} run-{int(r.run):02d}" for r in R.itertuples()
               if (RAW / f"{r.sub}/ses-01/eeg/{r.sub}_ses-01_task-szMonitoring_run-{int(r.run):02d}_eeg.edf").exists()
               and not (ROOT / f"data/features/{r.sub}_run-{int(r.run):02d}.parquet").exists()]
    g["G6 every downloaded run has features"] = not missing
    return g, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--keep-locked", action="store_true")
    ap.add_argument("--keep-demo", action="store_true")
    a = ap.parse_args()
    g, missing = gates()
    for k, v in g.items():
        print(("OPEN   " if v else "CLOSED ") + k)
    if missing:
        print(f"  runs without features: {len(missing)} (first: {missing[:5]})")
    files = sorted(RAW.rglob("*.edf"))
    keep = set()
    if a.keep_locked:
        keep |= {f for f in files if f.parts[-4] >= "sub-097"}
    if a.keep_demo and (ROOT / "docs/DEMO_RUNS.txt").exists():
        stems = {l.strip() for l in (ROOT / "docs/DEMO_RUNS.txt").read_text().splitlines() if l.strip()}
        keep |= {f for f in files if any(s in f.name for s in stems)}
    todel = [f for f in files if f not in keep]
    gb = sum(f.stat().st_size for f in todel) / 1e9
    print(f"{len(todel)} raw files, {gb:.1f} GB would be deleted; {len(keep)} kept")
    if not all(g.values()):
        sys.exit("Not all gates are open. Nothing deleted.")
    if not a.go:
        print("Dry run. Add --go to delete."); return
    for f in todel:
        f.unlink()
    print("Deleted. Re-download with src/01_download_seizeit2.py if ever needed (SHA-256 catalog is kept).")


if __name__ == "__main__":
    main()

"""Cross-platform runner for the full protocol (Windows, macOS, Linux).

Equivalent to scripts/run_all.sh. Each seed runs in three parts (A trains all
models, B and C reuse them). Parts of different seeds run in parallel.

Usage:
    python scripts/run_all.py                         # test seeds 301-310, 2 workers
    python scripts/run_all.py --workers 4             # 4 seeds at a time (about 2 GB RAM each)
    python scripts/run_all.py --seeds 301 302         # selected seeds
    python scripts/run_all.py --out results_mine      # write into another folder
Finished parts (results/seed_<S>_<P>.json present) are skipped, so the run can be resumed.
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTS = {"A": "seen,unseen,switch", "B": "midswitch,misleading", "C": "oof,robust"}


OUTDIR = os.path.join(ROOT, "results")
RUN_PARTS = list(PARTS)


def run_seed(seed):
    os.makedirs(OUTDIR, exist_ok=True)
    for part in RUN_PARTS:
        conds = PARTS[part]
        out = os.path.join(OUTDIR, f"seed_{seed}_{part}.json")
        if os.path.exists(out):
            print(f"seed {seed} part {part} already done", flush=True)
            continue
        env = dict(os.environ, LPTOM_PART=part, LPTOM_CONDS=conds, LPTOM_OUT=OUTDIR,
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        log = os.path.join(OUTDIR, f"log_seed_{seed}_{part}.txt")
        print(f"seed {seed} part {part} started", flush=True)
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_experiments.py"), str(seed), "3000", "20", "300"],
                               cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"seed {seed} part {part} FAILED, see {log}", flush=True)
            return
        print(f"seed {seed} part {part} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(301, 311)))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results", help="output folder, relative to the repository")
    ap.add_argument("--parts", default="ABC", help="parts to run, e.g. A for the first three conditions only")
    a = ap.parse_args()
    OUTDIR = os.path.join(ROOT, a.out)
    RUN_PARTS = [p for p in "ABC" if p in a.parts.upper()]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run_seed, a.seeds))

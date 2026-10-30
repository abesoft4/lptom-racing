"""Re-run test seeds into a separate folder and compare them with the shipped results.

The shipped results in results/ are never touched. Your run goes to --out
(default results_mine). Afterwards this script prints and saves a comparison of
every agent, condition and race: the win rate in the paper's run, the win rate
in your run and the difference.

Examples (Windows: use "py" instead of "python"):
    python scripts/verify.py --seeds 301 --parts A           # quickest check, about 1.5 h
    python scripts/verify.py --seeds 301 302 --workers 2     # two full seeds, about 5 h
    python scripts/verify.py --workers 4                     # all ten seeds, about 7-10 h
    python scripts/verify.py --compare-only                  # only redo the comparison

Outputs in --out:
    seed_<S>_<P>.json, log_seed_<S>_<P>.txt, models_seed_<S>.pkl   raw results, logs and models of your run
    comparison.txt                                                 your run against the shipped run
If all ten seeds are complete, it also writes
    summary.json, tables/*.tex, figures/*.pdf                      the paper's statistics, tables and figures from your run
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHIPPED = os.path.join(ROOT, "results")


def mean_win(entry):
    return [float(np.mean(r["win"])) for r in entry["races"]]


def compare(out):
    lines, diffs, same, n = [], [], 0, 0
    files = sorted(glob.glob(os.path.join(out, "seed_*_*.json")))
    if not files:
        print("no results found in", out)
        return
    for f in files:
        name = os.path.basename(f)
        ref = os.path.join(SHIPPED, name)
        if not os.path.exists(ref):
            lines.append(f"{name}: no shipped counterpart")
            continue
        mine, theirs = json.load(open(f))["results"], json.load(open(ref))["results"]
        lines.append(f"\n=== {name} ===")
        lines.append(f"{'condition':11s} {'agent':24s} {'paper':>7s} {'yours':>7s} {'diff':>7s}  identical")
        for cond in mine:
            for agent in mine[cond]:
                if agent not in theirs.get(cond, {}):
                    continue
                a = np.mean(mean_win(theirs[cond][agent]))
                b = np.mean(mean_win(mine[cond][agent]))
                ident = all(x["win"] == y["win"] for x, y in zip(mine[cond][agent]["races"], theirs[cond][agent]["races"]))
                same += ident; n += 1; diffs.append(b - a)
                lines.append(f"{cond:11s} {agent:24s} {a:7.3f} {b:7.3f} {b - a:+7.3f}  {'yes' if ident else 'no'}")
    d = np.abs(np.array(diffs)) if diffs else np.zeros(1)
    head = [f"Compared {n} agent-condition pairs over {len(files)} result files.",
            f"Bit-for-bit identical race outcomes: {same} of {n}.",
            f"Win-rate difference, mean absolute {d.mean():.4f}, largest {d.max():.4f} (1.0 = 100 percentage points)."]
    text = "\n".join(head + lines)
    open(os.path.join(out, "comparison.txt"), "w").write(text)
    print("\n".join(head))
    print("Full table written to", os.path.join(out, "comparison.txt"))


def full_outputs(out):
    seeds = {int(os.path.basename(f).split("_")[1]) for f in glob.glob(os.path.join(out, "seed_*_C.json"))}
    if len(seeds) < 10 or len(glob.glob(os.path.join(out, "seed_*_*.json"))) < 30:
        print("Statistics, tables and figures are produced once all ten seeds are complete.")
        return
    env = dict(os.environ, LPTOM_RES=out, LPTOM_TAB=os.path.join(out, "tables"), LPTOM_FIG=os.path.join(out, "figures"))
    for s in ["analyze.py", "make_misspec_figure.py", "make_result_figures.py", "make_architecture_figure.py"]:
        print("running", s, flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", s)], cwd=ROOT, env=env, check=True,
                       stdout=open(os.path.join(out, f"log_{s[:-3]}.txt"), "w"), stderr=subprocess.STDOUT)
    print("Statistics:", os.path.join(out, "summary.json"))
    print("Tables:    ", os.path.join(out, "tables"))
    print("Figures:   ", os.path.join(out, "figures"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(301, 311)))
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results_mine")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    if os.path.abspath(out) == os.path.abspath(SHIPPED):
        sys.exit("--out must differ from results/, which holds the shipped run")
    if not a.compare_only:
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_all.py"), "--out", a.out, "--parts", a.parts,
               "--workers", str(a.workers), "--seeds"] + [str(s) for s in a.seeds]
        subprocess.run(cmd, cwd=ROOT, check=True)
    compare(out)
    full_outputs(out)

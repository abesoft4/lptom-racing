"""Compare a reproduction with the published reference.

Usage
  python scripts/verify_reproduction.py     # after analyze.py and fill_results.py

'tables' compares paper/tables/*.tex, paper/numbers.json and the key
numbers of results/summary.json with reference/. A re-run of the experiments
is compared race by race with results/ by scripts/verify.py.
"""
import os, sys, json, glob, filecmp
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF = os.path.join(ROOT, "reference")


def tables():
    ok = True
    pairs = [(os.path.join(REF, "tables", os.path.basename(f)), f) for f in sorted(glob.glob(os.path.join(ROOT, "paper", "tables", "*.tex")))]
    pairs += [(os.path.join(REF, n), os.path.join(ROOT, "paper", n)) for n in ["numbers.json"]]
    for ref, new in pairs:
        same = filecmp.cmp(ref, new, shallow=False)
        ok &= same
        print(f"{'identical' if same else 'DIFFERENT':10s} {os.path.relpath(new, ROOT)}")
    a = json.load(open(os.path.join(REF, "summary.json")))
    b = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    worst = 0.0
    for c in a["main"]:
        for ag in a["main"][c]:
            worst = max(worst, abs(a["main"][c][ag]["win"] - b["main"][c][ag]["win"]))
    print(f"largest win-rate difference in summary.json: {worst:.2e}")
    for ta, tb in zip(a["tests"], b["tests"]):
        if (ta["p_holm"] < 0.05) != (tb["p_holm"] < 0.05):
            ok = False
            print("significance changed:", ta["cond"], ta["a"], ta["b"])
    print("REPRODUCED" if ok and worst < 1e-9 else "differences found, see above")


if __name__ == "__main__":
    tables()

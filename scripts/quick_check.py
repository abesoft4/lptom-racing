"""Five-minute installation and reproducibility check.

1. Runs the unit tests (autodiff gradients, residual Bayes factor).
2. Runs a tiny version of the full experiment for seed 999 (small data, two
   conditions) into results_quick/, to show that the pipeline runs end to end.
3. Rebuilds every table, figure and quoted number from the stored
   results of the paper and compares them with reference/.

Usage:  python scripts/quick_check.py        (Windows: py scripts\\quick_check.py)
"""
import os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def step(title, args, env=None):
    print(f"\n=== {title} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + args, cwd=ROOT, env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    for line in out[-4:]:
        print("   ", line)
    print(f"    [{'ok' if r.returncode == 0 else 'FAILED'}, {time.time() - t0:.0f} s]", flush=True)
    if r.returncode != 0:
        sys.exit(f"step failed: {title}")


if __name__ == "__main__":
    step("unit test, autodiff gradients", ["tests/test_autograd.py"])
    step("unit test, residual posterior and Bayes factor", ["tests/test_residual.py"])
    env = dict(os.environ, LPTOM_OUT=os.path.join(ROOT, "results_quick"), LPTOM_PPO_ITERS="2",
               LPTOM_PART="A", LPTOM_CONDS="seen", OMP_NUM_THREADS="1")
    step("tiny end-to-end experiment (seed 999, reduced size) -> results_quick/",
         ["scripts/run_experiments.py", "999", "60", "1", "6"], env)
    step("statistics and tables from the stored results", ["scripts/analyze.py"])
    step("Figure 2 and its trace summary", ["scripts/make_misspec_figure.py"])
    step("numbers quoted in the results text and abstract", ["scripts/fill_results.py"])
    step("Figures 3-5", ["scripts/make_result_figures.py"])
    step("Figure 1", ["scripts/make_architecture_figure.py"])
    step("comparison with the published reference", ["scripts/verify_reproduction.py"])
    print("\nAll checks passed. The last step must report REPRODUCED.")

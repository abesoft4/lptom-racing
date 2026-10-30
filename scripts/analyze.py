"""Statistics, tables and summary numbers from results/seed_*.json.

Every number in the manuscript is produced here and written to
results/summary.json and paper/tables/*.tex.
"""
import os, json, glob
import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.environ.get("LPTOM_RES", os.path.join(ROOT, "results"))
TAB = os.environ.get("LPTOM_TAB", os.path.join(ROOT, "paper", "tables"))
os.makedirs(TAB, exist_ok=True)

# a seed may be split into parts (seed_S_A.json, seed_S_B.json). Parts are merged by condition.
by_seed = {}
for f in glob.glob(os.path.join(RES, "seed_*.json")):
    d = json.load(open(f))
    if d["seed"] in by_seed:
        base = by_seed[d["seed"]]
        base["results"].update(d["results"])
        if d.get("part") == "A":
            d["results"] = base["results"]
            by_seed[d["seed"]] = d
    else:
        by_seed[d["seed"]] = d
runs = [by_seed[k] for k in sorted(by_seed)]
seeds = [r["seed"] for r in runs]
print("seeds", seeds)
CONDS = ["seen", "unseen", "switch", "midswitch", "misleading"]
ALLC = CONDS + ["oof"]
AGENTS = list(runs[0]["results"]["seen"].keys())
OOF_AG = list(runs[0]["results"]["oof"].keys())
DEPLOY = ["random", "rule", "agnosticQ", "ppo", "liam", "gt-maxmin", "sim-prior", "sim-bpr", "lptom"]
COMPET = ["agnosticQ", "ppo", "liam", "gt-maxmin", "sim-prior", "sim-bpr"]
RB_AGENTS = list(runs[0]["results"]["robust"].keys())
NAMES = {
    "random": "Random manoeuvre", "rule": "Rule-based racecraft",
    "agnosticQ": "Learned value, opponent-agnostic", "oracleQ": "Learned value, true style$^\\dagger$",
    "ppo": "PPO racer, implicit adaptation", "liam": "LIAM-style opponent embedding",
    "gt-maxmin": "Game-theoretic max-min planner", "lptom-noresid": "w/o residual rival model",
    "sim-prior": "Mental simulation, population prior", "sim-bpr": "Mental simulation, BPR types",
    "lptom": "LP-ToM (ours)", "lptom-structured": "known rival policy family",
    "lptom-nolang": "w/o language prior", "lptom-acc": "accumulation, no forgetting",
    "lptom-const": "constant forgetting", "lptom-single": "single-Gaussian change point",
    "lptom-probe": "with optimism bonus $\\beta=0.5$", "lptom-amortised": "amortised value, no simulation",
    "sim-oracle": "Mental simulation, true rival$^\\dagger$",
}
SHORT = {"sim-prior": "sim. prior", "agnosticQ": "agnostic value", "sim-bpr": "sim. BPR", "lptom-acc": "accumulate",
         "ppo": "PPO", "liam": "LIAM-style", "gt-maxmin": "max-min", "lptom-noresid": "no residual",
         "ppo|noise1": "PPO", "ppo|noise2": "PPO",
         "lptom-const": "const. forget.", "lptom-single": "single Gaussian", "lptom-amortised": "amortised value",
         "lptom-structured": "known family", "lptom-probe": "optimism bonus", "lptom-nolang": "no language",
         "sim-prior|mm-moderate": "sim. prior", "sim-prior|mm-strong": "sim. prior",
         "agnosticQ|noise1": "agnostic value", "agnosticQ|noise2": "agnostic value"}
LAB = {"seen": "Seen", "unseen": "Unseen", "switch": "Switch", "midswitch": "Mid-race sw.", "misleading": "Misleading",
       "oof": "Out-of-family",
       "robust": "Robustness"}
RLAB = {"mm-moderate": "moderate model error", "mm-strong": "strong model error", "noise1": "perception noise", "noise2": "double noise"}


def per_seed(cond, agent, metric, races=None):
    out = []
    for r in runs:
        rr = r["results"][cond][agent]["races"]
        idx = range(len(rr)) if races is None else races
        out.append(np.mean(np.concatenate([np.asarray(rr[k][metric], float).ravel() for k in idx])))
    return np.array(out)


def first_regret(cond, agent, n=2):
    return np.array([np.mean(np.asarray(r["results"][cond][agent]["races"][0]["regret"])[:, :n]) for r in runs])


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def paired(a, b):
    d = a - b
    t = stats.ttest_rel(a, b)
    try:
        pw = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        pw = 1.0
    lo, hi = boot_ci(d)
    return dict(diff=float(d.mean()), sd=float(d.std(ddof=1)), t=float(t.statistic), p_t=float(t.pvalue), p_w=pw,
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), ci=[float(lo), float(hi)], d=d.tolist())


def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


S = {"seeds": seeds, "n_seeds": len(seeds), "n_sessions": len(runs[0]["results"]["seen"]["lptom"]["races"][0]["win"])}
main = {c: {a: dict(win=float(per_seed(c, a, "win").mean()), win_sd=float(per_seed(c, a, "win").std(ddof=1)),
                    tgap=float(per_seed(c, a, "tgap").mean())) for a in AGENTS} for c in CONDS}
main["oof"] = {a: dict(win=float(per_seed("oof", a, "win").mean()), win_sd=float(per_seed("oof", a, "win").std(ddof=1)),
                       tgap=float(per_seed("oof", a, "tgap").mean())) for a in OOF_AG}
# the privileged true-rival reference imagines with the true parametric style or the true out-of-family policy
main["oof"]["sim-oracle"] = main["oof"]["sim-truepolicy"]
main["robust"] = {a: dict(win=float(per_seed("robust", a, "win").mean()), win_sd=float(per_seed("robust", a, "win").std(ddof=1)))
                  for a in RB_AGENTS}
S["main"] = main
for c in CONDS:
    S[f"perrace_{c}"] = {a: [float(per_seed(c, a, "win", [k]).mean()) for k in range(4)] for a in AGENTS}
    S[f"perrace_{c}_sd"] = {a: [float(per_seed(c, a, "win", [k]).std(ddof=1)) for k in range(4)] for a in AGENTS}
# share of the prior-to-oracle gap closed by inference
S["average"] = {a: float(np.mean([main[c][a]["win"] for c in ALLC])) for a in AGENTS if all(a in main[c] for c in ALLC)}
S["gap_closed"] = {c: (main[c]["lptom"]["win"] - main[c]["sim-prior"]["win"]) / (main[c]["sim-oracle"]["win"] - main[c]["sim-prior"]["win"]) for c in ALLC}

plan = []
for ref in COMPET:
    for c in ALLC:
        plan.append((c, "lptom", ref, "win", None))
plan += [("switch", "lptom", "lptom-acc", "win", [2, 3]), ("midswitch", "lptom", "lptom-acc", "win", [1, 2, 3]),
         ("seen", "lptom", "lptom-acc", "win", None),
         ("switch", "lptom", "lptom-const", "win", [2, 3]), ("seen", "lptom", "lptom-const", "win", None),
         ("switch", "lptom", "lptom-single", "win", [2, 3]), ("seen", "lptom", "lptom-single", "win", None),
         ("seen", "lptom", "lptom-amortised", "win", None), ("seen", "lptom", "lptom-structured", "win", None),
         ("oof", "lptom", "lptom-noresid", "win", None), ("seen", "lptom", "lptom-noresid", "win", None),
         ("seen", "lptom", "lptom-nolang", "regret", None), ("misleading", "lptom", "lptom-nolang", "regret", None),
         ("robust", "lptom|mm-moderate", "sim-prior|mm-moderate", "win", None),
         ("robust", "lptom|mm-strong", "sim-prior|mm-strong", "win", None),
         ("robust", "lptom|noise1", "agnosticQ|noise1", "win", None),
         ("robust", "lptom|noise2", "agnosticQ|noise2", "win", None),
         ("robust", "lptom|noise1", "ppo|noise1", "win", None),
         ("robust", "lptom|noise2", "ppo|noise2", "win", None)]
tests = []
for c, a, b, met, races in plan:
    fa, fb = (first_regret(c, a), first_regret(c, b)) if met == "regret" else (per_seed(c, a, met, races), per_seed(c, b, met, races))
    res = paired(fa, fb)
    res.update(cond=c, a=a, b=b, metric=met, races=races, mean_a=float(fa.mean()), mean_b=float(fb.mean()))
    tests.append(res)
for t, p in zip(tests, holm([t["p_t"] for t in tests])):
    t["p_holm"] = float(p)
S["tests"] = tests

S["regret"] = {c: {a: dict(all=float(np.mean([np.mean(r["results"][c][a]["races"][0]["regret"]) for r in runs])),
                          first=float(first_regret(c, a).mean()), first_sd=float(first_regret(c, a).std(ddof=1)))
                   for a in AGENTS} for c in ["seen", "misleading"]}
from lptom_fams import SEEN_NAMES, UNSEEN_NAMES, OOF_NAMES  # noqa: E402
FAM_AG = ["agnosticQ", "ppo", "sim-bpr", "lptom", "sim-oracle"]
fam = {}
for c, names in [("seen", SEEN_NAMES), ("unseen", UNSEEN_NAMES), ("oof", OOF_NAMES)]:
    fam[c] = {}
    for a in FAM_AG:
        key = "sim-truepolicy" if (c == "oof" and a == "sim-oracle") else a
        fam[c][a] = [float(np.mean([np.concatenate([np.asarray(rr["win"])[np.asarray(r["results"][c][key]["fam"]) == k]
                                                    for rr in r["results"][c][key]["races"]]).mean() for r in runs]))
                     for k in range(len(names))]
S["family"] = fam
for key in ["nolang", "lang"]:
    r2 = np.array([np.asarray(r["diag"][key]["r2_curve"]) for r in runs])
    zc = np.array([np.asarray(r["diag"][key]["zcal_curve"]) for r in runs])
    S[f"belief_{key}"] = {"r2_mean": r2.mean(0).tolist(), "zcal_mean": zc.mean(0).tolist()}
S["triggers"] = {c: {a: float(per_seed(c, a, "n_trig").mean()) for a in ["lptom", "lptom-single", "lptom-nolang"]} for c in CONDS}
S["rival_r2"] = [r["rival_r2"] for r in runs]
S["fault"] = {c: {a: float(per_seed(c, a, "fault").mean()) for a in AGENTS} for c in CONDS}
S["average_deploy"] = {a: float(np.mean([main[c][a]["win"] for c in ALLC])) for a in DEPLOY}
S["average_all"] = {a: float(np.mean([main[c][a]["win"] for c in CONDS])) for a in AGENTS}   # five family conditions
S["oof_rows"] = {a: main["oof"][a]["win"] for a in main["oof"]}
S["tau"] = [r["tau"] for r in runs]
S["params"] = runs[0]["params"]
S["latency"] = {k: float(np.mean([r["latency_ms_per_step"][k] for r in runs])) for k in runs[0]["latency_ms_per_step"]}
S["train_seconds"] = float(np.mean([r["train_seconds"] for r in runs]))
json.dump(S, open(os.path.join(RES, "summary.json"), "w"), indent=1)


def fmt(m, s):
    return f"{m:.3f}$\\pm${s:.3f}"


def seed_win(c, a):
    if c == "oof" and a == "sim-oracle":
        a = "sim-truepolicy"
    return per_seed(c, a, "win")


def avg_cond(a, conds):
    v = np.mean([seed_win(c, a) for c in conds], 0)
    return float(v.mean()), float(v.std(ddof=1))


def row(a, conds=ALLC, bold=False):
    nm = ("\\textbf{" + NAMES[a] + "}") if bold else NAMES[a]
    have = [c for c in conds if a in main[c]]
    cells = [fmt(main[c][a]["win"], main[c][a]["win_sd"]) if a in main[c] else "---" for c in conds]
    avg = fmt(*avg_cond(a, conds)) if len(have) == len(conds) else "---"
    return " & ".join([nm] + cells + [avg]) + " \\\\\n"


with open(os.path.join(TAB, "main.tex"), "w") as f:
    for a in DEPLOY[:-1]:
        f.write(row(a))
    f.write(row("lptom", bold=True))
    f.write("\\midrule\n")
    for a in ["oracleQ", "sim-oracle"]:
        f.write(row(a))
ABL = ["lptom", "lptom-noresid", "lptom-structured", "lptom-nolang", "lptom-acc", "lptom-const", "lptom-single", "lptom-amortised"]
with open(os.path.join(TAB, "ablation.tex"), "w") as f:
    for a in ABL:
        nm = ("\\textbf{" + NAMES[a] + "}") if a == "lptom" else NAMES[a]
        cells = [fmt(main[c][a]["win"], main[c][a]["win_sd"]) for c in CONDS] + [fmt(*avg_cond(a, CONDS))]
        f.write(" & ".join([nm] + cells) + " \\\\\n")
with open(os.path.join(TAB, "robust.tex"), "w") as f:
    for lvl in ["mm-moderate", "mm-strong", "noise1", "noise2"]:
        cells = []
        for a in ["agnosticQ", "ppo", "sim-prior", "lptom", "sim-oracle"]:
            k = f"{a}|{lvl}"
            if k in main["robust"]:
                cells.append(fmt(main["robust"][k]["win"], main["robust"][k]["win_sd"]))
            else:
                # a model-free agent has no internal model, so model error leaves it unchanged (same races as Seen)
                cells.append(fmt(main["seen"][a]["win"], main["seen"][a]["win_sd"]) + "$^\\ddagger$")
        f.write(f"{RLAB[lvl]} & " + " & ".join(cells) + " \\\\\n")
with open(os.path.join(TAB, "tests.tex"), "w") as f:
    for t in tests:
        races = "all" if t["races"] is None else ",".join(str(k + 1) for k in t["races"])
        met = "win" if t["metric"] == "win" else "regret"
        pt = f"{t['p_t']:.1e}" if t["p_t"] < 1e-3 else f"{t['p_t']:.3f}"
        ph = f"{t['p_holm']:.1e}" if t["p_holm"] < 1e-3 else f"{t['p_holm']:.3f}"
        cond = RLAB[t["a"].split("|")[1]] if t["cond"] == "robust" else LAB[t["cond"]]
        f.write(f"{cond} & {SHORT[t['b']]} & {met} & {races} & {t['diff']:+.3f} & "
                f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] & {t['dz']:+.2f} & {pt} & {ph} \\\\\n")
with open(os.path.join(TAB, "family.tex"), "w") as f:
    for c, names in [("seen", SEEN_NAMES), ("unseen", UNSEEN_NAMES), ("oof", OOF_NAMES)]:
        for k, nm in enumerate(names):
            f.write(f"{LAB[c] if k == 0 else ''} & {nm} & " + " & ".join(f"{fam[c][a][k]:.3f}" for a in FAM_AG) + " \\\\\n")
        if c != "oof":
            f.write("\\midrule\n")
print(json.dumps({c: {a: round(main[c][a]["win"], 3) for a in main[c]} for c in main}, indent=1))
print("gap closed", {c: round(v, 3) for c, v in S["gap_closed"].items()})
for t in tests:
    print(f"{t['cond']:10s} {t['a']:16s} vs {t['b']:18s} {t['metric']:6s} diff={t['diff']:+.4f} ci=[{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] dz={t['dz']:+.2f} p={t['p_t']:.3g} holm={t['p_holm']:.3g} pw={t['p_w']:.3g}")

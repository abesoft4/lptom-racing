"""Compute every number quoted in the results section and abstract of the paper.

The numbers come from results/summary.json and are written to paper/numbers.json.
The manuscript text is not part of this repository. When its templates
paper/sec_results.tpl and paper/abstract.tpl are present, they are filled as well,
so the text cannot drift from the tables. Qualitative claims made by the text are
checked by assertions and the script stops if the data contradict them.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
S = json.load(open(os.path.join(ROOT, "results/summary.json")))
M, T = S["main"], S["tests"]
C6 = ["seen", "unseen", "switch", "midswitch", "misleading", "oof"]
COMPET = ["agnosticQ", "ppo", "liam", "gt-maxmin", "sim-prior", "sim-bpr"]
NAME = {"agnosticQ": "the opponent-agnostic value network", "ppo": "the PPO racer", "liam": "the LIAM-style agent",
        "gt-maxmin": "the game-theoretic planner", "sim-prior": "the uninformed planner", "sim-bpr": "Bayesian policy reuse"}
pc = lambda x: f"{100 * x:.1f}"
pt = lambda x: f"{100 * x:.1f}"
sg = lambda x: f"{100 * x:+.1f}".replace("-", "$-$")


def test(cond, a, b, metric="win"):
    for t in T:
        if t["cond"] == cond and t["a"] == a and t["b"] == b and t["metric"] == metric:
            return t
    raise KeyError((cond, a, b, metric))


def pexp(p):
    e = int(np.floor(np.log10(max(p, 1e-300)))) + 1
    return f"10^{{{e}}}" if e < -1 else f"{p:.3f}"


def p3(p):
    return f"={p:.3f}" if p >= 0.001 else "<" + pexp(p)


def ci(t, scale=100, nd=1):
    return f"[{scale * t['ci'][0]:.{nd}f},{scale * t['ci'][1]:.{nd}f}]"


w = lambda c, a: M[c][a]["win"]
V = {}
# main comparison
for c in C6:
    best = max(COMPET, key=lambda a: w(c, a))
    assert w(c, "lptom") > w(c, best), f"LP-ToM is not best in {c}"
V["SEEN"] = pc(w("seen", "lptom"))
bs = max(COMPET, key=lambda a: w("seen", a)); V["BESTSEENNAME"] = NAME[bs]; V["BESTSEEN"] = pc(w("seen", bs))
V["BESTSEENNAME"] = V["BESTSEENNAME"][0].upper() + V["BESTSEENNAME"][1:]
avg = S["average_deploy"]
V["AVGLP"] = pc(avg["lptom"])
ba = max(COMPET, key=lambda a: avg[a]); V["BESTAVGNAME"] = NAME[ba]; V["BESTAVG"] = pc(avg[ba])
words = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen twenty").split()
tens = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def num(n):
    if n <= 20:
        return words[n]
    t, u = 10 * (n // 10), n % 10
    return tens[t] + ("" if u == 0 else "-" + words[u])


comp_tests = [test(c, "lptom", b) for b in COMPET for c in C6]
nsig = sum(t["p_holm"] < 0.05 and t["diff"] > 0 for t in comp_tests)
V["NCOMPSIG"] = "All" if nsig == len(comp_tests) else num(nsig).capitalize()
pr = S["perrace_seen"]["ppo"]
V["PPOSEEN"] = pc(w("seen", "ppo")); V["PPOGAIN"] = pt(pr[3] - pr[0])
V["PPOR4GAP"] = pt(S["perrace_seen"]["lptom"][3] - pr[3])
V["LIAMSEEN"] = pc(w("seen", "liam"))
d = w("seen", "liam") - w("seen", "agnosticQ")
V["LIAMVSAGN"] = f"{abs(100 * d):.1f} points {'above' if d >= 0 else 'below'}"
V["GTSEEN"] = pc(w("seen", "gt-maxmin"))
V["PRIORSEEN"] = pc(w("seen", "sim-prior")); t = test("seen", "lptom", "sim-prior")
V["DPRIORSEEN"] = pt(t["diff"]); V["DZPRIORSEEN"] = f"{t['dz']:.2f}"
g = S["gap_closed"]
for k, c in [("GAPSEEN", "seen"), ("GAPUNSEEN", "unseen"), ("GAPSWITCH", "switch"), ("GAPMID", "midswitch"),
             ("GAPMIS", "misleading"), ("GAPOOF", "oof")]:
    V[k] = f"{100 * min(g[c], 1.0):.0f}"
d = [test(c, "lptom", "sim-bpr")["diff"] for c in C6]; V["BPRMIN"], V["BPRMAX"] = pt(min(d)), pt(max(d))
# out of family
V["OOFLP"] = pc(w("oof", "lptom"))
bo = max(COMPET, key=lambda a: w("oof", a)); V["OOFBESTNAME"] = NAME[bo]; V["OOFBEST"] = pc(w("oof", bo))
t = test("oof", "lptom", "lptom-noresid"); V["DRESOOF"] = pt(t["diff"]); V["CIRESOOF"] = ci(t); V["PRESOOF"] = p3(t["p_holm"])
V["RESSENT"] = (f"The residual rival model adds {V['DRESOOF']} points (${V['CIRESOOF']}$, adjusted $p{V['PRESOOF']}$)."
                if t["p_holm"] < 0.05 else
                f"The residual rival model adds {V['DRESOOF']} points (${V['CIRESOOF']}$). The gain is positive in "
                f"{num(int(sum(x > 0 for x in t['d'])))} of {num(len(t['d']))} seeds but not significant after Holm correction "
                f"(adjusted $p{V['PRESOOF']}$).")
V["RESOOF_SIG"] = t["p_holm"] < 0.05
t = test("seen", "lptom", "lptom-noresid"); V["DRESSEEN"] = sg(t["diff"]); V["PRESSEEN"] = p3(t["p_holm"])
V["OOFTRUE"] = pc(M["oof"]["sim-truepolicy"]["win"]); V["GAPOOFPTS"] = pt(M["oof"]["sim-truepolicy"]["win"] - w("oof", "lptom"))
P = json.load(open(os.path.join(ROOT, "results/pmis_trace.json")))
V["POOFR1"] = f"{np.floor(10 * P['oof_race1_min']) / 10:.1f}"
from lptom_fams import OOF_NAMES  # noqa: E402
FO = S["family"]["oof"]
lead = [k for k in range(len(OOF_NAMES)) if FO["lptom"][k] >= max(FO[a][k] for a in ["agnosticQ", "ppo", "sim-bpr"])]
lose = [k for k in range(len(OOF_NAMES)) if k not in lead]
nm_ = lambda k: OOF_NAMES[k].replace("rl", "RL").replace("defender a", "defender A").replace("defender b", "defender B")
if not lose:
    V["OOFKINDSENT"] = "LP-ToM is ahead of the other agents in the table against every out-of-family rival."
else:
    who = {k: max(["agnosticQ", "ppo", "sim-bpr"], key=lambda a: FO[a][k]) for k in lose}
    V["OOFKINDSENT"] = (f"LP-ToM leads the other agents in the table against {num(len(lead))} of the six rivals. " +
                        " ".join(f"Against the {nm_(k)} {NAME[who[k]]} is ahead by {pt(FO[who[k]][k] - FO['lptom'][k])} points."
                                 for k in lose))
V["PSEEN"] = f"{P.get('seen_mean', float('nan')):.2f}"
F = S["family"]["seen"]
V["FAMLPIC"], V["FAMORIC"] = pc(F["lptom"][2]), pc(F["sim-oracle"][2])
V["FAMLPQB"], V["FAMBPRQB"] = pc(F["lptom"][0]), pc(F["sim-bpr"][0])
# inference
bl, bn = np.array(S["belief_lang"]["r2_mean"]), np.array(S["belief_nolang"]["r2_mean"])
V["R2NL"] = f"{bn[-1][1:4].min():.2f} to {bn[-1][1:4].max():.2f}"; V["R2NLEPS"] = f"{bn[-1][4]:.2f}"; V["R2NLU"] = f"{bn[-1][0]:.2f}"
V["R2L0B"], V["R2L0L"] = f"{bl[0][1]:.2f}", f"{bl[0][3]:.2f}"
nh = int((bl[-1] >= bn[-1]).sum())
V["RADIOEND"] = "With radio every end-of-race value is higher." if nh == 5 else \
    f"With radio the end-of-race value is higher for {num(nh)} of the five dimensions."
z = np.array(S["belief_nolang"]["zcal_mean"][-1]); V["ZMIN"], V["ZMAX"] = f"{z.min():.2f}", f"{z.max():.2f}"
# revision
t = test("seen", "lptom", "lptom-acc"); V["DACCSEEN"] = pt(abs(t["diff"])); V["PACCSEEN"] = p3(t["p_holm"])
assert t["p_holm"] >= 0.05, "accumulation differs significantly with stable rivals"
prs = S["perrace_switch"]; V["ACCR3"], V["LPR3"], V["LPR4"] = pc(prs["lptom-acc"][2]), pc(prs["lptom"][2]), pc(prs["lptom"][3])
t = test("switch", "lptom", "lptom-acc"); V["DACCSW"] = pt(t["diff"]); V["CIACCSW"] = ci(t); V["DZACCSW"] = f"{t['dz']:.2f}"
t = test("midswitch", "lptom", "lptom-acc"); V["DACCMID"] = pt(t["diff"]); V["DZACCMID"] = f"{t['dz']:.2f}"
t = test("switch", "lptom", "lptom-const"); V["DCONSTSW"] = pt(t["diff"]); V["PCONSTSW"] = p3(t["p_holm"])
V["CONSTSENT"] = (f"The run-length mixture beats constant forgetting after a switch by {V['DCONSTSW']} points (adjusted $p{V['PCONSTSW']}$)."
                  if t["p_holm"] < 0.05 else
                  f"The run-length mixture is {V['DCONSTSW']} points ahead of constant forgetting after a switch, in "
                  f"{num(int(sum(x > 0 for x in t['d'])))} of {num(len(t['d']))} seeds. This is not significant after correction "
                  f"(adjusted $p{V['PCONSTSW']}$).")
assert test("switch", "lptom", "lptom-single")["p_holm"] >= 0.05
ab = S["average_all"]
rules = {"lptom": "the mixture", "lptom-single": "the single-Gaussian rule", "lptom-const": "constant forgetting",
         "lptom-acc": "accumulation"}
order = sorted(rules, key=lambda a: -ab[a])
if order[0] == "lptom" and ab["lptom"] - ab[order[1]] >= 0.0005:
    V["REVAVG"] = (f"Over the five family conditions the mixture has the highest average of the four revision rules, "
                   f"{pc(ab['lptom'])}\\% (Table~\\ref{{tab:ablation}}).")
else:
    top2 = [a for a in order[:2]]
    assert "lptom" in top2, order
    tie = (f"both reach {pc(ab['lptom'])}\\%" if pc(ab['lptom']) == pc(ab['lptom-single'])
           else f"tie at {pc(ab['lptom'])}\\% and {pc(ab['lptom-single'])}\\%")
    V["REVAVG"] = (f"Over the five family conditions the mixture and the single-Gaussian rule {tie}. Both lead constant forgetting at {pc(ab['lptom-const'])}\\% and accumulation "
                   f"at {pc(ab['lptom-acc'])}\\% (Table~\\ref{{tab:ablation}}).")
# language
R = S["regret"]; V["REGLP"], V["REGNL"] = f"{R['seen']['lptom']['first']:.3f}", f"{R['seen']['lptom-nolang']['first']:.3f}"
t = test("seen", "lptom", "lptom-nolang", "regret"); V["DREG"] = f"{-t['diff']:.3f}"
V["CIREG"] = f"[{-t['ci'][1]:.3f},{-t['ci'][0]:.3f}]"; V["DZREG"] = f"{-t['dz']:.2f}"
V["REGPCT"] = f"{100 * -t['diff'] / R['seen']['lptom-nolang']['first']:.0f}"
V["REGMISLP"], V["REGMISNL"] = f"{R['misleading']['lptom']['first']:.3f}", f"{R['misleading']['lptom-nolang']['first']:.3f}"
assert test("misleading", "lptom", "lptom-nolang", "regret")["p_holm"] >= 0.05
# robustness
rb = M["robust"]
V["MMMOD"], V["MMSTR"] = pc(rb["lptom|mm-moderate"]["win"]), pc(rb["lptom|mm-strong"]["win"])
V["DMMMOD"] = pt(test("robust", "lptom|mm-moderate", "sim-prior|mm-moderate")["diff"])
V["DMMSTR"] = pt(test("robust", "lptom|mm-strong", "sim-prior|mm-strong")["diff"])
V["NZ1"], V["NZ2"] = pc(rb["lptom|noise1"]["win"]), pc(rb["lptom|noise2"]["win"])
V["DNZ1"], V["DNZ2"] = pt(test("robust", "lptom|noise1", "agnosticQ|noise1")["diff"]), pt(test("robust", "lptom|noise2", "agnosticQ|noise2")["diff"])
V["DNZPPO1"], V["DNZPPO2"] = pt(test("robust", "lptom|noise1", "ppo|noise1")["diff"]), pt(test("robust", "lptom|noise2", "ppo|noise2")["diff"])
# ablations
V["RIVR2"] = f"{np.mean(S['rival_r2']):.2f}"
t = test("seen", "lptom", "lptom-structured"); V["DSTRUCT"] = pt(abs(t["diff"])); V["PSTRUCT"] = p3(t["p_holm"])
V["STRUCTOOF"] = pc(M["oof"]["lptom-structured"]["win"])
t = test("seen", "lptom", "lptom-amortised"); V["DAMORT"] = pt(t["diff"]); V["PAMORT"] = p3(t["p_holm"])
# summary
sig = [t for t in T if t["p_holm"] < 0.05]
fav = [t for t in sig if (t["diff"] > 0) != (t["metric"] == "regret")]
V["NSIG"] = num(len(sig)).capitalize()
adverse = [SHORT for SHORT in [t["b"] for t in sig if t not in fav]]
if not adverse:
    V["SIGSENTENCE"] = "All of them favour LP-ToM."
else:
    assert all(a in ("lptom-structured", "lptom-acc", "lptom-const", "lptom-single", "lptom-noresid") for a in adverse), adverse
    V["SIGSENTENCE"] = (f"All but {num(len(adverse))} favour LP-ToM. The exception{'s are' if len(adverse) > 1 else ' is'} "
                        f"{' and '.join(sorted(set(adverse)))} with stable rivals, a variant of LP-ToM itself.")
V["WILCOX"] = "all" if all((t["p_t"] < 0.05) == (t["p_w"] < 0.05) for t in T) else \
    f"{sum((t['p_t'] < 0.05) == (t['p_w'] < 0.05) for t in T)}"
Fs = S["fault"]["seen"]; V["FLP"], V["FPPO"], V["FRND"] = f"{Fs['lptom']:.3f}", f"{Fs['ppo']:.3f}", f"{Fs['random']:.3f}"
V["LAT"] = f"{S['latency']['lptom']:.2f}"
# abstract
V["DBESTSEEN"] = pt(w("seen", "lptom") - w("seen", bs))
V["DOOFBEST"] = pt(w("oof", "lptom") - w("oof", bo))
V["GAPAVG"] = f"{100 * np.mean([min(g[c], 1.0) for c in C6]):.0f}"
json.dump(V, open(os.path.join(ROOT, "paper", "numbers.json"), "w"), indent=1, sort_keys=True)
for name in ["sec_results", "abstract"]:
    if not os.path.exists(os.path.join(ROOT, f"paper/{name}.tpl")):
        continue
    src = open(os.path.join(ROOT, f"paper/{name}.tpl")).read()
    out = re.sub(r"@([A-Z0-9]+)@", lambda m: V[m.group(1)], src)
    open(os.path.join(ROOT, f"paper/{name}.tex"), "w").write(out)
for k in sorted(V):
    print(f"{k:14s} {V[k]}")
print("non-significant tests:")
for t in T:
    if t["p_holm"] >= 0.05:
        print("  ", t["cond"], t["a"], "vs", t["b"], t["metric"], round(t["diff"], 4), round(t["p_holm"], 3))

"""Figure on the misspecification-aware rival model.

(a) Posterior probability of misspecification P_t during a four-race session for
seen rivals and for each out-of-family rival (seed-301 models, 300 sessions each).
(b) Win rate per out-of-family rival over the ten test seeds.
"""
import os, sys, json, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figtext import save_flat_pdf
from lptom.track import TrackSet
from lptom.model import ToMNet
from lptom.language import Vocab, make_message
from lptom.agents import Agent, run
from lptom.opponents import sample_mix, SEEN
from lptom.rival import RivalNet
from lptom.oof import OOFPolicy, RivalPolicy, OOF_KINDS, theta_oof

PRETTY = {"anticipator": "anticipator", "late-chopper": "late chopper", "zone-defender": "zone defender",
          "random-switcher": "random switcher", "rl-defender-a": "RL defender A", "rl-defender-b": "RL defender B"}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED, N = 301, 300
RES = os.environ.get("LPTOM_RES", os.path.join(ROOT, "results"))
FIG = os.environ.get("LPTOM_FIG", os.path.join(ROOT, "paper", "figures"))
os.makedirs(FIG, exist_ok=True)
V = Vocab()
sd = pickle.load(open(os.path.join(RES, f"models_seed_{SEED}.pkl"), "rb"))
tom = ToMNet(SEED, len(V)); tom.load_state_dict(sd["tom"])
rival = RivalNet(SEED); rival.load_state_dict(sd["rival"])
nets = []
for k in ["a", "b"]:
    p = RivalPolicy(0); p.load_state_dict(pickle.load(open(os.path.join(ROOT, "benchmark", f"rl_rival_{k}.pkl"), "rb"))); nets.append(p)
oof = OOFPolicy(nets)
tt = TrackSet(range(100, 104))


def trace(theta, bow, kinds=None):
    a = Agent("mb", tom=tom, source="posterior", belief_mode="bocpd", hazard_bcp=3e-3, n_samples=16,
              rival_net=rival, resid=True, seed=SEED)
    rng = np.random.default_rng(123)
    carry, P = None, []
    for k in range(4):
        r = run(tt, a, theta, rng.integers(0, 4, len(theta)), np.random.default_rng(10 + k), bow=bow, carry=carry,
                record=True, ext=oof if kinds is not None else None, oof_kinds=kinds)
        carry = r["carry"]; P.append(r["trace"]["p_mis"])
    return np.concatenate(P, 0)                    # [32 decisions, E]


rng = np.random.default_rng(7)
CACHE = os.path.join(RES, "pmis_trace.npz")
th, _ = sample_mix(SEEN, N, rng)
kinds = np.repeat(np.arange(len(OOF_KINDS)), N // len(OOF_KINDS))
if os.path.exists(CACHE):
    z = np.load(CACHE); P_seen, P_oof = z["seen"], z["oof"]
else:
    P_seen = trace(th, V.bow([make_message(x, rng, "test") for x in th]))
    P_oof = trace(theta_oof(len(kinds), rng), None, kinds)
    np.savez(CACHE, seen=P_seen, oof=P_oof)

# per-kind win rates over the test seeds
files = glob.glob(os.path.join(RES, "seed_*.json"))
wins = {a: [] for a in ["ppo", "lptom-noresid", "lptom", "sim-truepolicy"]}
for f in files:
    d = json.load(open(f))
    if "oof" not in d["results"]:
        continue
    for a in wins:
        R = d["results"]["oof"][a]
        fam = np.asarray(R["fam"])
        w = np.mean([np.asarray(rr["win"]) for rr in R["races"]], 0)
        wins[a].append([w[fam == j].mean() for j in range(len(OOF_KINDS))])

fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
ax = axs[0]
t = np.arange(32) * 4 / 32 + 4 / 32
cols = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
for j, nm in enumerate(OOF_KINDS):
    ax.plot(np.arange(1, 33), P_oof[:, kinds == j].mean(1), color=cols[j], lw=1.0, label=PRETTY[nm])
ax.plot(np.arange(1, 33), P_seen.mean(1), color="black", lw=1.6, ls="--", label="seen families")
for b in [8.5, 16.5, 24.5]:
    ax.axvline(b, color="#cccccc", lw=0.6, zorder=0)
ax.set_xlabel("decision in session (8 per race)"); ax.set_ylabel(r"$P_t$, misspecification")
ax.set_ylim(-0.03, 1.03); ax.set_xlim(0.5, 32.5)
ax.legend(fontsize=6.0, frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.07), columnspacing=1.0, handlelength=1.6)
ax.set_title("(a) Probability that the learned rival model is wrong", fontsize=8)
ax = axs[1]
x = np.arange(len(OOF_KINDS))
lab = {"ppo": "PPO racer", "lptom-noresid": "LP-ToM w/o residual", "lptom": "LP-ToM", "sim-truepolicy": "simulation, true rival"}
col = {"ppo": "#17becf", "lptom-noresid": "#9ecae1", "lptom": "#1f5fa8", "sim-truepolicy": "#000000"}
nseed = max(len(wins["lptom"]), 1)
for i, a in enumerate(wins):
    W = np.array(wins[a]) if wins[a] else np.zeros((1, len(OOF_KINDS)))
    ax.bar(x + (i - 1.5) * 0.2, W.mean(0), 0.19, yerr=1.96 * W.std(0, ddof=1) / np.sqrt(nseed) if len(W) > 1 else None,
           color=col[a], label=lab[a], capsize=1.5, error_kw=dict(lw=0.6))
ax.set_xticks(x); ax.set_xticklabels([PRETTY[k].replace(" ", "\n", 1) for k in OOF_KINDS], fontsize=6.3)
ax.set_ylabel("win rate"); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(fontsize=6.0, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.8, handlelength=1.2)
ax.set_title("(b) Win rate per out-of-family rival", fontsize=8)
fig.tight_layout()
save_flat_pdf(fig, os.path.join(FIG, "fig_misspec.pdf"))
r1 = [float(P_oof[7, kinds == j].mean()) for j in range(len(OOF_KINDS))]
json.dump({"seen_mean": float(P_seen.mean()), "seen_max_race": float(max(P_seen[k * 8:(k + 1) * 8].mean() for k in range(4))),
           "oof_race1_min": min(r1), "oof_race1": r1}, open(os.path.join(RES, "pmis_trace.json"), "w"))
print("P_t seen mean", P_seen.mean().round(3), "oof end of race 1 per kind", np.round(r1, 2))

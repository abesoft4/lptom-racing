"""Result figures from results/summary.json."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figtext import save_flat_pdf

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.environ.get("LPTOM_RES", os.path.join(ROOT, "results"))
OUT = os.environ.get("LPTOM_FIG", os.path.join(ROOT, "paper", "figures"))
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(RES, "summary.json")))
n = S["n_seeds"]
COL = {"ppo": "#17becf", "agnosticQ": "#7f7f7f", "sim-prior": "#bcbd22", "sim-bpr": "#8c564b", "lptom": "#1f5fa8", "lptom-acc": "#d9772b",
       "lptom-const": "#2ca02c", "sim-oracle": "#000000", "lptom-nolang": "#9467bd"}
LAB = {"ppo": "PPO racer", "agnosticQ": "learned value, agnostic", "sim-prior": "simulation, prior", "sim-bpr": "simulation, BPR",
       "lptom": "LP-ToM", "lptom-acc": "LP-ToM w/o forgetting", "lptom-const": "constant forgetting",
       "sim-oracle": "simulation, true rival", "lptom-nolang": "LP-ToM w/o language"}

# per-race win rate in the stationary and switch conditions
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
ags = ["ppo", "sim-prior", "lptom-acc", "lptom-const", "lptom", "sim-oracle"]
for ax, c, title in [(axs[0], "seen", "Stable rival"), (axs[1], "switch", "Switch between races 2 and 3"),
                     (axs[2], "midswitch", "Hidden switch inside race 2")]:
    for j, a in enumerate(ags):
        m = np.array(S[f"perrace_{c}"][a]); se = np.array(S[f"perrace_{c}_sd"][a]) / np.sqrt(n)
        ax.errorbar(np.arange(1, 5) + 0.04 * (j - 2.5), m, yerr=1.96 * se, color=COL[a], lw=1.1, marker="o", ms=2.8,
                    capsize=1.5, label=LAB[a], ls="--" if a == "sim-oracle" else "-")
    if c != "seen":
        ax.axvspan(2.5 if c == "switch" else 1.5, 4.5 if c == "switch" else 2.5, color="#f6e3d4", lw=0, zorder=0)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("race in session"); ax.set_title(title, fontsize=8.3); ax.set_xlim(0.7, 4.3)
axs[0].set_ylabel("win rate")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, frameon=False, loc="lower center", ncol=6)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save_flat_pdf(fig, os.path.join(OUT, "fig_switch.pdf"))

# Figure 3: belief quality over the first race
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
names = [r"pace $u$", r"blocking $b$", r"delay $\tau_r$", r"lane $l$", r"lapse $\epsilon$"]
cols = ["#999999", "#8c564b", "#2ca02c", "#1f5fa8", "#d62728"]
t = np.arange(len(S["belief_nolang"]["r2_mean"])) * 0.2
for k in range(5):
    axs[0].plot(t, np.array(S["belief_lang"]["r2_mean"])[:, k], color=cols[k], lw=1.1, label=names[k])
    axs[0].plot(t, np.array(S["belief_nolang"]["r2_mean"])[:, k], color=cols[k], lw=0.9, ls=":")
axs[0].set_ylim(-0.3, 1.0); axs[0].set_xlabel("race time [s]"); axs[0].set_ylabel(r"$R^2$ of posterior mean")
axs[0].legend(fontsize=6.3, frameon=False, ncol=3, loc="lower right")
axs[0].set_title("Inference of rival style (solid with radio, dotted without)", fontsize=8)
for k in range(5):
    axs[1].plot(t, np.array(S["belief_nolang"]["zcal_mean"])[:, k], color=cols[k], lw=1.0)
axs[1].axhline(1.0, color="black", lw=0.7, ls="--")
axs[1].set_yscale("log"); axs[1].set_xlabel("race time [s]"); axs[1].set_ylabel(r"mean $z^2$")
axs[1].set_title("Calibration (1 is ideal)", fontsize=8)
fig.tight_layout()
save_flat_pdf(fig, os.path.join(OUT, "fig_belief.pdf"))

# Figure 4: first-decision regret with reliable and misleading radio
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ags = ["agnosticQ", "ppo", "sim-prior", "sim-bpr", "lptom-nolang", "lptom", "sim-oracle"]
x = np.arange(len(ags))
for i, (c, lab, hatch) in enumerate([("seen", "reliable radio", ""), ("misleading", "misleading radio", "////")]):
    m = [S["regret"][c][a]["first"] for a in ags]
    e = [1.96 * S["regret"][c][a]["first_sd"] / np.sqrt(n) for a in ags]
    ax.bar(x + (i - 0.5) * 0.36, m, 0.34, yerr=e, color=[COL[a] for a in ags], alpha=0.85 if i == 0 else 0.45,
           hatch=hatch, edgecolor="white", lw=0.3, capsize=2, label=lab, error_kw=dict(lw=0.7))
ax.set_xticks(x); ax.set_xticklabels(["agnostic\nvalue", "PPO", "sim.\nprior", "sim.\nBPR", "w/o\nlanguage", "LP-ToM", "sim.\ntrue rival"], fontsize=6.3)
ax.set_ylabel("regret, first two decisions\n[0.1 s per option]")
ax.set_ylim(0, 0.5); ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center")
fig.tight_layout()
save_flat_pdf(fig, os.path.join(OUT, "fig_regret.pdf"))
print("figures written")

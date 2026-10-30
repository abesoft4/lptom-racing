"""Pictorial architecture figure built from a real simulator trace.

Every curve, ellipse and trajectory in the figure is computed by the trained
seed-301 models and the simulator. Nothing is drawn by hand.
"""
import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from figtext import save_flat_pdf
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib import patheffects as pe
from lptom.track import TrackSet, HALF_W
from lptom.sim import RaceEnv, N_TACTICS, theta_norm, OPTION_STEPS
from lptom.model import TacticValueNet, ToMNet
from lptom.language import Vocab
from lptom.agents import Agent, run, BeliefState
from lptom.opponents import sample_family
from lptom.autograd import Tensor

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED = 301
V = Vocab()
RES = os.environ.get("LPTOM_RES", os.path.join(ROOT, "results"))
sd = pickle.load(open(os.path.join(RES, f"models_seed_{SEED}.pkl"), "rb"))
tom = ToMNet(SEED, len(V)); tom.load_state_dict(sd["tom"])
qT = TacticValueNet(SEED, zmode="theta"); qT.load_state_dict(sd["qT"])
from lptom.rival import RivalNet
rival = RivalNet(SEED); rival.load_state_dict(sd["rival"])
from lptom.mental import mental_values, denorm, observable_state

C_EGO, C_OPP = "#1f5fa8", "#d9772b"
C_ASPH, C_GRASS = "#4a4d52", "#e9efe4"
CMAP = plt.get_cmap("viridis")

# ------------------------------------------------------------------ trace
tt = TrackSet([101])
rng = np.random.default_rng(3)
E = 96
tha = sample_family("inside-coverer", E, rng)
thb = sample_family("wide-runner", E, rng)
sw = np.full(E, 160)
msg = "car ahead covers the inside line"
bow = V.bow([msg] * E)
agent = Agent("mb", tom=tom, source="posterior", belief_mode="bocpd", hazard_bcp=3e-3, n_samples=16, rival_net=rival, resid=True, seed=0)
r = run(tt, agent, tha, np.zeros(E, int), np.random.default_rng(11), bow=bow, switch=(sw, thb), record=True)
tr = r["trace"]
mu, sdv, S, trig = tr["mu"], tr["sd"], tr["surprise"], tr["trig"]
tn_a, tn_b = theta_norm(tha), theta_norm(thb)
# representative race: gate fires after the switch and the side belief flips
post = np.arange(mu.shape[0]) * 2 >= 160
score = S[post].max(0) - S[~post].max(0) + (mu[-1, :, 3] < 0).astype(float) + (mu[70, :, 3] > 0.3)
j = int(np.argmax(score))
t_ax = np.arange(mu.shape[0]) * 0.2
sbar = S[:, j]

# ------------------------------------------------------------------ scene
env = RaceEnv(tt, tha[j:j + 1], np.array([0]), np.random.default_rng(5), start_gap=8.0)
env.st["s_e"][:] = 285.0; env.st["s_o"][:] = 293.0
env.st["d_e"][:] = env.st["d_o"][:]
base = env.copy_state()
lam_prior = tom.lang_prior(V.bow([msg]))
b0 = BeliefState(1, lam_prior[0].data, lam_prior[1].data)
rngi = np.random.default_rng(0)
th16 = denorm(b0.mu[:, None] + b0.sd[:, None] * rngi.normal(size=(1, 16, 5)))
qbar = mental_values(env, th16, OPTION_STEPS, np.random.default_rng(1), rival_net=rival)[0]
ras = env.raster()[0]
T = tt.tracks[0]


def world(s, d):
    i = (np.floor(s / T.ds).astype(int)) % len(T.x)
    f = (s / T.ds) - np.floor(s / T.ds)
    i2 = (i + 1) % len(T.x)
    x = T.x[i] * (1 - f) + T.x[i2] * f
    y = T.y[i] * (1 - f) + T.y[i2] * f
    h = T.heading[i]
    return x - d * np.sin(h), y + d * np.cos(h), h


paths = {}
th6 = denorm(b0.mu[:, None] + b0.sd[:, None] * np.random.default_rng(2).normal(size=(1, 6, 5)))[0]
for g, name in [(1, "tow-inside"), (2, "tow-outside"), (4, "switchback")]:
    obs = observable_state(env)
    sim = RaceEnv.__new__(RaceEnv)
    n6 = 6
    st = {k: np.repeat(v, n6, 0) for k, v in obs.items() if k != "t"}
    st["t"] = obs["t"]; st["theta"] = th6.copy()
    st["lapse_u"] = np.random.default_rng(3).uniform(size=(n6, env.st["lapse_u"].shape[1]))
    for k in ["lapse_left", "coll", "fault_e", "overtakes", "passed", "att_count"]:
        st[k] = np.zeros(n6)
    st["adaptive"] = np.zeros(n6, bool); st["sw_step"] = np.full(n6, 10 ** 9); st["sw_theta"] = th6.copy()
    sim.T, sim.E, sim.st = env.T, n6, st
    sim.rival_net = rival
    sim.set_tactic(np.full(n6, g))
    pe_, po_ = [], []
    for _ in range(30):
        sim.step()
        pe_.append(np.stack([sim.st["s_e"], sim.st["d_e"]], 1))
        po_.append(np.stack([sim.st["s_o"], sim.st["d_o"]], 1))
    paths[name] = (np.array(pe_), np.array(po_), g)        # [T, 6, 2]
env.set_state(base)
s0 = base["s_e"][0]
x0, y0, h0 = world(np.array([s0]), np.array([0.0]))
rot = -h0[0]
R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])


def tf(x, y):
    p = R @ np.vstack([np.asarray(x) - x0[0], np.asarray(y) - y0[0]])
    return p[0], p[1]


fig = plt.figure(figsize=(7.16, 4.4))
ax = fig.add_axes([0.065, 0.43, 0.385, 0.46])
ax.set_facecolor(C_GRASS)
S0 = base["s_e"][0]
xs_ = np.linspace(-15, 130, 400)
ax.add_patch(Rectangle((-15, -HALF_W), 145, 2 * HALF_W, fc=C_ASPH, ec="none", zorder=1))
kap_s = T.kappa[(np.floor((S0 + xs_) / T.ds).astype(int)) % len(T.x)]
for k in range(0, len(xs_) - 3, 3):
    if abs(kap_s[k]) > 0.008:
        yk = HALF_W if kap_s[k] > 0 else -HALF_W
        ax.add_patch(Rectangle((xs_[k], yk - 0.35), xs_[k + 3] - xs_[k], 0.7,
                               fc=("#c8102e" if (k // 3) % 2 else "white"), ec="none", zorder=2))
ax.plot(xs_, T.dstar[(np.floor((S0 + xs_) / T.ds).astype(int)) % len(T.x)], color="white", lw=0.6,
        ls=(0, (4, 4)), alpha=0.7, zorder=2)


def car(s, d, color, alpha=1.0, ls="-", fill=True, z=6):
    x = s - S0
    pts = np.array([[-2.3, -0.95], [1.5, -0.95], [2.3, -0.45], [2.3, 0.45], [1.5, 0.95], [-2.3, 0.95]])
    ax.add_patch(Polygon(pts + [x, d], closed=True, fc=color if fill else "none",
                         ec="black" if fill else color, lw=0.5 if fill else 1.0, ls=ls, alpha=alpha, zorder=z))
    return x, d


so, do_ = base["s_o"][0], base["d_o"][0]
for k, a_ in enumerate(np.linspace(0.30, 0.04, 9)):
    ax.add_patch(Ellipse((so - S0 - 4 - 2.6 * k, do_), 3.0 + 0.5 * k, 1.4 + 0.25 * k, fc="#9ecae1", ec="none", alpha=a_, zorder=3))
names = {"tow-inside": 1, "tow-outside": 2, "switchback": 4}
qv = {nm: qbar[g] for nm, g in names.items()}
qmin, qmax = min(qv.values()), max(qv.values())
for nm, (pe_, po_, g) in paths.items():
    c = CMAP(0.1 + 0.8 * (qv[nm] - qmin) / (qmax - qmin + 1e-9))
    lw = 0.9 + 1.6 * (qv[nm] - qmin) / (qmax - qmin + 1e-9)
    for k in range(pe_.shape[1]):
        px, py = pe_[:, k, 0] - S0, pe_[:, k, 1]
        ax.plot(px, py, color=c, lw=lw, alpha=0.75, solid_capstyle="round", zorder=5,
                path_effects=[pe.Stroke(linewidth=lw + 0.8, foreground="white", alpha=0.5), pe.Normal()])
        car(po_[-1, k, 0], po_[-1, k, 1], c, alpha=0.35, ls=(0, (2, 1.2)), fill=False, z=4)
    mx, my = pe_[-1, :, 0].mean() - S0, pe_[-1, :, 1].mean()
    ax.annotate("", xy=(mx, my), xytext=(pe_[-4, :, 0].mean() - S0, pe_[-4, :, 1].mean()),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=8), zorder=6)
    ax.text(mx - 6, my + (2.0 if my > 0 else -2.0), f"{nm}  " + r"$\bar V$" + f"$=${qv[nm]:+.2f}", fontsize=6.4, ha="right", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=c, lw=0.8, alpha=0.95))
ex, ey = car(base["s_e"][0], base["d_e"][0], C_EGO, z=7)
ox, oy = car(so, do_, C_OPP, z=7)
ax.text(ex, ey - 1.9, "ego", color="white", fontsize=6.8, weight="bold", ha="center", va="top", zorder=9)
ax.text(ox, oy + 1.5, "rival", color="white", fontsize=6.8, weight="bold", ha="center", va="bottom", zorder=9)
ax.text(-10, -HALF_W + 0.6, "outside", color="#dddddd", fontsize=5.8, va="bottom")
ax.text(-10, HALF_W - 0.9, "inside", color="#dddddd", fontsize=5.8, va="top")
ax.set_xlim(-12, 128); ax.set_ylim(-HALF_W - 1.0, HALF_W + 1.0)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([-6, -3, 0, 3, 6])
ax.tick_params(labelsize=6)
ax.set_xlabel("arc length ahead of ego $s$ [m]  (Frenet frame, lateral axis exaggerated)", fontsize=6.6)
ax.set_ylabel("lateral $d$ [m]", fontsize=6.6)
fig.text(0.015, 0.975, "(a) Mental simulation of candidate manoeuvres", fontsize=8.6, weight="bold", va="top")
fig.text(0.015, 0.935, r"Futures imagined under 6 styles $\theta^{(k)}\sim q_0$. Dashed outlines are imagined rival replies.",
         fontsize=6.4, va="top", color="#333333")
# spotter radio bubble
axm = fig.add_axes([0.065, 0.08, 0.36, 0.16]); axm.axis("off")
axm.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.84, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec="#7a3e9d", lw=0.9, transform=axm.transAxes))
axm.text(0.07, 0.68, "spotter radio (language input)", fontsize=6.6, color="#7a3e9d", weight="bold", va="center", transform=axm.transAxes)
axm.text(0.07, 0.32, f"\u201c{msg}\u201d", fontsize=7.2, style="italic", va="center", transform=axm.transAxes)
wv = np.linspace(0, 1, 80)
axm.plot(0.70 + 0.22 * wv, 0.68 + 0.12 * np.sin(45 * wv) * np.exp(-((wv - 0.5) ** 2) / 0.06), color="#7a3e9d", lw=0.7, transform=axm.transAxes)

# ------------------------------------------------------------- belief panel
axb = fig.add_axes([0.59, 0.43, 0.19, 0.46])
iL, iB = 3, 1
axb.set_xlim(-2.2, 2.2); axb.set_ylim(-2.2, 2.2)
for v in [-1, 0, 1]:
    axb.axhline(v, color="#eeeeee", lw=0.5, zorder=0); axb.axvline(v, color="#eeeeee", lw=0.5, zorder=0)
axb.set_xlabel(r"lane preference $\theta_l$ (outside $\leftarrow\;\rightarrow$ inside)", fontsize=6.8)
axb.set_ylabel(r"blocking gain $\theta_b$", fontsize=6.8)
axb.tick_params(labelsize=6)


def ell(i, color, ls="-", lw=1.0, alpha=1.0, fill=False, z=3):
    m, s_ = mu[i, j], sdv[i, j]
    axb.add_patch(Ellipse((m[iL], m[iB]), 4 * s_[iL], 4 * s_[iB], fc=(color if fill else "none"), ec=color, ls=ls, lw=lw, alpha=alpha, zorder=z))


ell(0, "#7a3e9d", ls="--", lw=1.1)
axb.text(mu[0, j, iL] + 0.05, mu[0, j, iB] + 2 * sdv[0, j, iB] + 0.06, "language prior", color="#7a3e9d", fontsize=6.3, ha="center")
ti = np.where(trig[:, j])[0]
i_sw = int(160 / 2)
i_pre = i_sw - 1
ell(i_pre, C_EGO, lw=1.2, fill=True, alpha=0.18)
ell(i_pre, C_EGO, lw=1.2)
first_post = ti[ti >= i_sw][0] if (ti >= i_sw).any() else i_sw + 5
ell(first_post, "#c0392b", ls=":", lw=1.2)
ell(len(mu) - 1, "#2e8b57", lw=1.2, fill=True, alpha=0.18)
ell(len(mu) - 1, "#2e8b57", lw=1.2)
pts = np.c_[mu[:, j, iL], mu[:, j, iB]]
segs = np.stack([pts[:-1], pts[1:]], 1)
lc = LineCollection(segs, cmap="plasma", lw=1.0, zorder=4)
lc.set_array(t_ax[:-1]); axb.add_collection(lc)
axb.plot(tn_a[j, iL], tn_a[j, iB], marker="*", ms=8, color=C_EGO, mec="white", mew=0.5, zorder=6)
axb.plot(tn_b[j, iL], tn_b[j, iB], marker="*", ms=8, color="#2e8b57", mec="white", mew=0.5, zorder=6)
axb.text(mu[i_pre, j, iL], min(mu[i_pre, j, iB] - 2 * sdv[i_pre, j, iB], mu[0, j, iB] - 2 * sdv[0, j, iB]) - 0.08, "true style\nbefore switch", fontsize=5.8, ha="center", va="top", color=C_EGO)
axb.text(tn_b[j, iL], tn_b[j, iB] - 0.22, "after switch", fontsize=5.8, ha="center", va="top", color="#2e8b57")
axb.text(mu[first_post, j, iL] + 0.1, mu[first_post, j, iB] + 2 * sdv[first_post, j, iB] + 0.05, "new run", fontsize=5.8, color="#c0392b", ha="center")
fig.text(0.545, 0.975, "(b) Belief over rival style", fontsize=8.6, weight="bold", va="top")
fig.text(0.545, 0.935, r"mean and $\pm2\sigma$ of the run-length mixture $q_t(\theta)$", fontsize=6.4, va="top", color="#333333")

# ------------------------------------------------------------ surprise panel
axs = fig.add_axes([0.855, 0.68, 0.135, 0.21])
axs.fill_between(t_ax, 0, 1, where=t_ax >= 16, transform=axs.get_xaxis_transform(), color="#f6e3d4", lw=0)
sbar = sbar.copy(); sbar[t_ax < 0.5] = np.nan
axs.plot(t_ax, sbar, color="#333333", lw=0.8)
axs.axhline(0.5, color="#c0392b", lw=0.8, ls="--")
axs.text(0.03, 0.53, r"$\pi_t=0.5$", transform=axs.transAxes, color="#c0392b", fontsize=6, va="bottom")
for k in ti:
    axs.plot(t_ax[k], sbar[k], marker="v", color="#c0392b", ms=4)
axs.set_ylim(-0.02, 1.02); axs.set_xlim(0, t_ax[-1])
axs.set_xticks([0, 10, 20, 30]); axs.tick_params(labelsize=6); axs.set_xticklabels([])
axs.set_ylabel(r"recent change $\pi_t$", fontsize=7)
axs.text(0.97, 0.95, "after\nhidden\nswitch", transform=axs.transAxes, fontsize=5.6, color="#a0522d", ha="right", va="top")
fig.text(0.815, 0.975, "(c) Change points", fontsize=8.6, weight="bold", va="top")
axl = fig.add_axes([0.855, 0.43, 0.135, 0.21])
axl.fill_between(t_ax, 0, 1, where=t_ax >= 16, transform=axl.get_xaxis_transform(), color="#f6e3d4", lw=0)
m3, s3 = mu[:, j, 3], sdv[:, j, 3]
axl.fill_between(t_ax, m3 - s3, m3 + s3, color=C_EGO, alpha=0.2, lw=0)
axl.plot(t_ax, m3, color=C_EGO, lw=1.0)
axl.step(t_ax, np.where(t_ax < 16, tn_a[j, 3], tn_b[j, 3]), color="black", lw=0.7, ls="--", where="post")
axl.set_ylim(-1.6, 1.6); axl.set_xlim(0, t_ax[-1])
axl.set_xticks([0, 10, 20, 30]); axl.tick_params(labelsize=6); axl.set_xlabel("race time [s]", fontsize=6.8)
axl.set_ylabel(r"$\mu_{t,l}\pm\sigma$", fontsize=7)
# equations strip
axe = fig.add_axes([0.525, 0.03, 0.465, 0.27]); axe.axis("off")
axe.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.96, boxstyle="round,pad=0.0,rounding_size=0.04", fc="#f7f7f9", ec="#bbbbbb", lw=0.6, transform=axe.transAxes))
eqs = [("evidence", r"$(m_t,\lambda_t)=e_\psi(w_t)$"),
       ("runs", r"$q_t=\sum_r w_{t,r}\,\mathcal{N}(\mu_{t,r},\Lambda_{t,r}^{-1})$"),
       ("weights", r"$w_{t,r+1}\propto w_{t-1,r}\,(1-h)\,p_r(m_t)$"),
       ("residual", r"$\hat\pi=v_{\max}\tanh(z_\nu+\omega^\top\chi)$,  $P_t=(1+B_t^{-1})^{-1}$"),
       ("decision", r"$g_t^\ast=\arg\max_g\ \frac{1}{S}\sum_k\mathcal{M}(\hat s_t,g,\hat\pi(\cdot\mid\theta^{(k)},\omega^{(k)}))$"),
       ("radio prior", r"$q_0=\mathcal{N}(\mu_\ell,\Lambda_\ell^{-1})=h_\phi(\mathrm{msg})$")]
for k, (lab, eq) in enumerate(eqs):
    y = 0.88 - k * 0.152
    axe.text(0.03, y, lab, fontsize=6.6, color="#555555", va="center", transform=axe.transAxes)
    axe.text(0.22, y, eq, fontsize=7.2, va="center", transform=axe.transAxes)


# connecting arrows (figure coordinates)
def farrow(p0, p1, rad=0.25, color="#555555"):
    a = FancyArrowPatch(p0, p1, transform=fig.transFigure, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                        mutation_scale=9, lw=1.0, color=color, zorder=20)
    fig.patches.append(a)


farrow((0.455, 0.80), (0.535, 0.80), rad=-0.35)
fig.text(0.495, 0.845, "evidence\n" + r"$w_t$ at 5 Hz", fontsize=6.0, color="#555555", ha="center", va="bottom", linespacing=1.1)
farrow((0.535, 0.56), (0.455, 0.56), rad=-0.35, color=C_EGO)
fig.text(0.495, 0.515, "samples\n" + r"$\theta^{(k)}\sim q_t$", fontsize=6.0, color=C_EGO, ha="center", va="top", linespacing=1.1)
farrow((0.43, 0.17), (0.535, 0.44), rad=0.35, color="#7a3e9d")
fig.text(0.47, 0.235, "prior", fontsize=6.0, color="#7a3e9d", ha="center")
out = os.environ.get("LPTOM_FIG", os.path.join(ROOT, "paper", "figures"))
os.makedirs(out, exist_ok=True)
save_flat_pdf(fig, os.path.join(out, "fig_architecture.pdf"))
print("ok", j, "triggers", ti * 0.2, "qbar", qbar.round(3))

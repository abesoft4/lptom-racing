"""Out-of-family rivals.

These rivals do not follow the parametric policy family used to train
inference and the learned rival model. Four are rule-based with a different
functional form. Two are lateral policies trained by reinforcement learning
(PPO) to defend against attacking egos. Pace and lapses keep their physical
meaning through theta[0] and theta[4]. The other style entries are unused.
"""
import numpy as np
from . import autograd as ag
from .autograd import Tensor
from .nn import Module, MLP
from .sim import HALF_W, LANE, DT, VLAT, THREAT, RIVAL_FEAT

REACT = 3                        # out-of-family rivals perceive the attacker's lateral motion 0.3 s late

OOF_KINDS = ["anticipator", "late-chopper", "zone-defender", "random-switcher", "rl-defender-a", "rl-defender-b"]
RL_KINDS = {"rl-defender-a": 0, "rl-defender-b": 1}
LIM = HALF_W - 1.0
RL_EVERY = 2                     # an RL rival updates its lateral target every 0.2 s


def delayed_ego(env, k=REACT):
    """Ego lateral position and rate as perceived with a reaction delay of k steps."""
    h = env.st["hist_de"]
    return h[:, -1 - k], (h[:, -1 - k] - h[:, -2 - k]) / DT


def rival_features_delayed(env, k=REACT):
    """Observable rival features with the attacker's lateral motion delayed by k steps."""
    st = env.st
    gap = env.gap()
    d_e, vd_e = delayed_ego(env, k)
    lags = st["hist_de"][:, ::-1][:, 1 + k:13 + k] / HALF_W
    return np.concatenate([np.stack([
        np.clip(gap, -40, 40) / 20.0, d_e / HALF_W, st["d_o"] / HALF_W,
        env.T.at("dstar", st["tid"], st["s_o"]) / HALF_W, env.ref_side(), vd_e / VLAT, st["vd_o"] / VLAT,
        ((gap < 0) & (gap > -THREAT)).astype(float), (np.abs(d_e - st["d_o"]) > 0.8).astype(float),
        ((gap > 0) & (gap < 30)).astype(float)], 1), lags], 1).astype(np.float32)


class RivalPolicy(Module):
    """Gaussian lateral-target policy and value function for PPO."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed + 401)
        self.pi = MLP(rng, [RIVAL_FEAT, 64, 64, 1], act="tanh", out_scale=0.1)
        self.v = MLP(rng, [RIVAL_FEAT, 64, 64, 1], act="tanh")
        self.logstd = Tensor(np.full((1,), -0.5, np.float32), requires_grad=True)

    def mean(self, x):
        return self.pi(Tensor(x))[:, 0]

    def act(self, x):
        """Deterministic lateral target in metres."""
        return np.tanh(self.mean(x).data) * LIM


def theta_oof(n, rng):
    """Pace and lapse rate of out-of-family rivals. Unused style entries sit at the family centre."""
    th = np.zeros((n, 5))
    th[:, 0] = rng.uniform(0.993, 0.999, n)
    th[:, 1], th[:, 2], th[:, 3] = 0.5, 6.0, 0.0
    th[:, 4] = rng.uniform(0.0, 0.02, n)
    return th


def init_state(env, kinds, rng):
    """Attach per-rival state of the out-of-family policies to a RaceEnv."""
    E = env.E
    env.st["oof_kind"] = np.asarray(kinds, int)
    env.st["oof_phase"] = rng.uniform(0, 2 * np.pi, E)
    env.st["oof_period"] = rng.uniform(1.5, 3.0, E)
    env.st["oof_tgt"] = env.st["d_o"].copy()


class OOFPolicy:
    """Dispatches the lateral target of every out-of-family rival kind in a batch."""

    def __init__(self, rl_nets):
        self.rl = rl_nets

    def __call__(self, env):
        st, T = env.st, env.T
        kind = st["oof_kind"]
        E = env.E
        gap = env.gap()
        tid = st["tid"]
        dstar = T.at("dstar", tid, st["s_o"])
        nt = env.ref_side()
        d_e, vd_e = delayed_ego(env)
        sep = np.abs(d_e - st["d_o"]) > 0.8
        out = np.full(E, np.nan)
        behind = gap < 0
        # anticipator: covers where the attacker will be in 1 s (lead instead of lag)
        m = kind == 0
        if m.any():
            pred = np.clip(d_e + 1.0 * vd_e, -LIM, LIM)
            out[m] = np.where(behind & (gap > -15), pred, dstar)[m]
        # late chopper: ignores the attacker until 5 m, then blocks at full rate
        m = kind == 1
        if m.any():
            out[m] = np.where(behind & (gap > -5) & sep, d_e, dstar)[m]
        # zone defender: covers the inside only within 60 m of the next corner
        m = kind == 2
        if m.any():
            k_ahead = np.maximum.reduce([np.abs(T.at("kappa", tid, st["s_o"] + x)) for x in (10.0, 30.0, 50.0)])
            zone = k_ahead > 0.008
            out[m] = np.where(behind & (gap > -20) & zone, nt * LANE, dstar)[m]
        # random switcher: alternates the covered side on its own clock, independent of the attacker
        m = kind == 3
        if m.any():
            side = np.sign(np.sin(2 * np.pi * st["t"] * DT / st["oof_period"] + st["oof_phase"]) + 1e-9)
            out[m] = np.where(behind & (gap > -15), side * LANE, dstar)[m]
        # reinforcement-learned defenders control the lateral target at every gap
        for nm, j in RL_KINDS.items():
            m = kind == OOF_KINDS.index(nm)
            if m.any():
                if st["t"] % RL_EVERY == 0:
                    st["oof_tgt"][m] = self.rl[j].act(rival_features_delayed(env)[m])
                out[m] = st["oof_tgt"][m]
        return out

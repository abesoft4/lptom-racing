"""A learned model of the rival's lateral policy.

The internal model of LP-ToM can use this network instead of the parametric
rival policy. It is trained by behaviour cloning on logged races. Inputs are
observable features and the rival style. The target is the rival's observed
lateral velocity. Steps during a lapse are excluded, because lapses keep their
parametric meaning through the lapse rate.
"""
import numpy as np
from . import autograd as ag
from .autograd import Tensor
from .nn import Module, MLP
from .sim import RaceEnv, OPTION_STEPS, RACE_STEPS, N_TACTICS, VLAT, THETA_DIM, rival_features, RIVAL_FEAT, theta_norm, resid_basis

HID = 96
RESID_DIM = HID + 2 * RIVAL_FEAT + 1
from .opponents import sample_mix, SEEN


class RivalNet(Module):
    def __init__(self, seed):
        rng = np.random.default_rng(seed + 31)
        self.f = MLP(rng, [RIVAL_FEAT + THETA_DIM, 96, 96, 1])

    def __call__(self, x, th_n):
        return ag.tanh(self.f(Tensor(np.concatenate([x, th_n.astype(np.float32)], 1))))[:, 0].data

    def hidden(self, x, th_n):
        """Penultimate activations, the basis of the Bayesian last-layer residual."""
        h = Tensor(np.concatenate([x, th_n.astype(np.float32)], 1))
        for l in self.f.layers[:-1]:
            h = ag.relu(l(h))
        return h.data

    def logit_and_basis(self, x, th_n):
        """Pre-activation output and residual basis [hidden, raw features, gated features, bias]."""
        h = self.hidden(x, th_n)
        last = self.f.layers[-1]
        z = (h @ last.W.data + last.b.data)[:, 0]
        return z, np.concatenate([h, resid_basis(x)], 1)


def collect_rival(ts, n_eps, seed, batch=200, every=2):
    rng = np.random.default_rng(seed + 77)
    X, TH, Y = [], [], []
    for b0 in range(0, n_eps, batch):
        n = min(batch, n_eps - b0)
        theta, _ = sample_mix(SEEN, n, rng)
        env = RaceEnv(ts, theta, rng.integers(0, len(ts.tracks), n), rng)
        for t in range(RACE_STEPS):
            if t % OPTION_STEPS == 0:
                env.set_tactic(rng.integers(0, N_TACTICS, n))
            x = rival_features(env)
            env.step()
            if t % every == 0:
                ok = env.st["lapse_left"] <= 0
                X.append(x[ok]); TH.append(theta_norm(env.st["theta"])[ok]); Y.append(env.st["vd_o"][ok] / VLAT)
    return np.concatenate(X), np.concatenate(TH).astype(np.float32), np.concatenate(Y).astype(np.float32)


def train_rival(net, X, TH, Y, epochs, seed, lr=1e-3, batch=512):
    rng = np.random.default_rng(seed + 5)
    opt = ag.Adam(net.parameters(), lr=lr)
    n = len(Y)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = []
        for b0 in range(0, n - batch + 1, batch):
            ix = perm[b0:b0 + batch]
            pred = ag.tanh(net.f(Tensor(np.concatenate([X[ix], TH[ix]], 1))))[:, 0]
            loss = ag.square(pred - Tensor(Y[ix])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append(float(loss.data))
        hist.append(float(np.mean(tot)))
    return hist

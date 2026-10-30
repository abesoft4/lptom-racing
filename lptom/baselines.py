"""Baselines from the literature, adapted to the racecraft benchmark.

PPOEgo. A model-free racer trained by proximal policy optimisation over
four-race sessions. Its input carries session-level summaries of the
interaction, so it can adapt implicitly, as recurrent and RL-squared agents do.

LIAMNet. Opponent modelling with local information as in LIAM. An encoder
embeds the ego's own interaction windows. Its session-level mean is trained
to reconstruct the rival's next response without style labels. A value head
conditioned on the embedding scores every manoeuvre.
"""
import time
import numpy as np
from . import autograd as ag
from .autograd import Tensor
from .nn import Module, MLP
from .model import TacticValueNet, onehot, gauss_nll
from .sim import RaceEnv, RACE_STEPS, OPTION_STEPS, N_TACTICS, THETA_DIM
from .data import step_features, SUB, WIN, FEAT
from .opponents import sample_mix, SEEN
from .ppo import gae, clipped_surrogate, log_softmax

N_RACE = 4
PPO_OBS = 9 + WIN * FEAT + 2 * FEAT + N_RACE
Z_DIM = THETA_DIM


def new_hist(E):
    return dict(zs=np.zeros((E, Z_DIM), np.float32), n=np.zeros(E), fthr=np.zeros((E, FEAT), np.float32),
                nthr=np.zeros(E), fall=np.zeros((E, FEAT), np.float32), nall=np.zeros(E), race=0)


def update_hist(hs, f, W, liam=None):
    hs["fall"] += f
    hs["nall"] += 1
    thr = f[:, 6:7]
    hs["fthr"] += f * thr
    hs["nthr"] += thr[:, 0]
    if liam is not None:
        hs["zs"] += liam.enc(Tensor(W)).data
        hs["n"] += 1


def ppo_obs(kin, W, hs):
    E = kin.shape[0]
    race = np.zeros((E, N_RACE), np.float32)
    race[:, min(hs["race"], N_RACE - 1)] = 1.0
    return np.concatenate([kin, W, hs["fthr"] / np.maximum(hs["nthr"], 1)[:, None],
                           hs["fall"] / np.maximum(hs["nall"], 1)[:, None], race], 1).astype(np.float32)


class PPOEgo(Module):
    def __init__(self, seed):
        rng = np.random.default_rng(seed + 501)
        self.pi = MLP(rng, [PPO_OBS, 128, 128, N_TACTICS], act="tanh", out_scale=0.1)
        self.v = MLP(rng, [PPO_OBS, 128, 128, 1], act="tanh")

    def act(self, obs, rng=None):
        logits = self.pi(Tensor(obs)).data
        if rng is None:
            return logits.argmax(1)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return (rng.uniform(size=(len(p), 1)) > np.cumsum(p, 1)).sum(1).clip(0, N_TACTICS - 1)


def train_ppo_ego(ts, seed, iters, E=256, log=None):
    """PPO over four-race sessions against seen rival families on the training tracks."""
    rng = np.random.default_rng(seed + 503)
    net = PPOEgo(seed)
    opt = ag.Adam(net.parameters(), lr=3e-4)
    t0 = time.time()
    hist = []
    for it in range(iters):
        theta, _ = sample_mix(SEEN, E, rng)
        hs = new_hist(E)
        OBS, ACT, LP, V, R, D = [], [], [], [], [], []
        wins = []
        for k in range(N_RACE):
            hs["race"] = k
            env = RaceEnv(ts, theta, rng.integers(0, len(ts.tracks), E), rng)
            Fbuf = np.zeros((E, WIN, FEAT), np.float32)
            prev = None
            for t in range(RACE_STEPS):
                if t % SUB == 0:
                    f = step_features(env)
                    Fbuf = np.concatenate([Fbuf[:, 1:], f[:, None]], 1)
                    W = Fbuf.reshape(E, -1)
                    update_hist(hs, f, W)
                if t % OPTION_STEPS == 0:
                    if prev is not None:
                        R.append(10.0 * (env.tgap() - prev[0]) - 2.0 * (env.st["fault_e"] - prev[1]))
                        D.append(np.zeros(E))
                    obs = ppo_obs(env.kin_features(), W, hs)
                    logits = net.pi(Tensor(obs)).data
                    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                    a = (rng.uniform(size=(E, 1)) > np.cumsum(p, 1)).sum(1).clip(0, N_TACTICS - 1)
                    OBS.append(obs); ACT.append(a); LP.append(np.log(p[np.arange(E), a] + 1e-12))
                    V.append(net.v(Tensor(obs)).data[:, 0])
                    env.set_tactic(a)
                    prev = (env.tgap(), env.st["fault_e"].copy())
                env.step()
            last = k == N_RACE - 1
            R.append(10.0 * (env.tgap() - prev[0]) - 2.0 * (env.st["fault_e"] - prev[1]) + 1.0 * (env.gap() > 0))
            D.append(np.full(E, 1.0 if last else 0.0))
            wins.append(np.mean(env.gap() > 0))
        OBS, ACT, LP, V, R, D = map(np.array, (OBS, ACT, LP, V, R, D))
        adv, ret = gae(R, V, np.zeros(E), D)
        n = OBS.shape[0] * E
        OBS, ACT, LP = OBS.reshape(n, -1), ACT.reshape(-1), LP.reshape(-1)
        adv, ret = adv.reshape(-1), ret.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for ep in range(4):
            perm = rng.permutation(n)
            for b0 in range(0, n, 2048):
                ix = perm[b0:b0 + 2048]
                lsm = log_softmax(net.pi(Tensor(OBS[ix])))
                oh = Tensor(onehot(ACT[ix]))
                lp = (lsm * oh).sum(-1)
                ent = -(ag.exp(lsm) * lsm).sum(-1).mean()
                loss = clipped_surrogate(lp, LP[ix], adv[ix])
                vloss = ag.square(net.v(Tensor(OBS[ix]))[:, 0] - Tensor(ret[ix].astype(np.float32))).mean()
                tot = loss + 0.5 * vloss - 0.01 * ent
                opt.zero_grad(); tot.backward(); opt.step()
        hist.append(float(np.mean(wins)))
        if log and (it % 25 == 0 or it == iters - 1):
            log(f"PPO it {it} train win {np.mean(wins):.3f} {time.time() - t0:.0f}s")
    return net, hist


class LIAMNet(Module):
    def __init__(self, seed):
        rng = np.random.default_rng(seed + 601)
        self.e = MLP(rng, [WIN * FEAT, 64, 64, Z_DIM])
        self.dec = MLP(rng, [Z_DIM + FEAT + N_TACTICS, 64, 64, 4])
        self.q = TacticValueNet(seed + 602, zmode="theta")

    def enc(self, W):
        return ag.tanh(self.e(W if isinstance(W, Tensor) else Tensor(W)))


def _prefix_mean(z):
    N = z.shape[1]
    return ag.cumsum(z, axis=1) * Tensor((1.0 / np.arange(1, N + 1, dtype=np.float32))[None, :, None])


def train_liam(net, D, epochs_rec, epochs_q, seed, batch=32, log=None):
    """Stage 1 trains the encoder by reconstruction of the rival's response. Stage 2 trains the value head."""
    rng = np.random.default_rng(seed + 603)
    params = net.e.parameters() + net.dec.parameters()
    opt = ag.Adam(params, lr=1e-3)
    n, N = D["W"].shape[:2]
    for ep in range(epochs_rec):
        perm = rng.permutation(n)
        tot = []
        for b0 in range(0, n - batch + 1, batch):
            ix = perm[b0:b0 + batch]
            B = len(ix)
            zbar = _prefix_mean(net.enc(Tensor(D["W"][ix])))
            inp = ag.concat([zbar, Tensor(D["F"][ix]), Tensor(onehot(D["TAC"][ix]))], axis=-1)
            h = net.dec(inp)
            nll = gauss_nll(D["y"][ix], h[..., :2], ag.tanh(h[..., 2:] * 0.5) * 4.0 - 1.0)
            msk = Tensor(D["valid"][ix][..., None].astype(np.float32))
            loss = (nll * msk).sum() * (1.0 / max(float(D["valid"][ix].sum()), 1.0))
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append(float(loss.data))
        if log:
            log(f"LIAM rec epoch {ep + 1}/{epochs_rec} nll={np.mean(tot):.4f}")
    # session embeddings at every decision with the frozen encoder
    K = D["kin"].shape[1]
    Z = np.zeros((n, K, Z_DIM), np.float32)
    for b0 in range(0, n, 200):
        z = net.enc(Tensor(D["W"][b0:b0 + 200])).data
        zbar = np.cumsum(z, 1) / np.arange(1, N + 1)[None, :, None]
        Z[b0:b0 + 200] = zbar[:, np.arange(K) * (OPTION_STEPS // SUB)]
    opt = ag.Adam(net.q.parameters(), lr=1e-3)
    for ep in range(epochs_q):
        perm = rng.permutation(n)
        tot = []
        for b0 in range(0, n - batch + 1, batch):
            ix = perm[b0:b0 + batch]
            B = len(ix)
            feat = net.q.features(D["kin"][ix].reshape(B * K, -1), D["ras"][ix].reshape(B * K, 2, 32, 16))
            q = net.q.qvalues(feat, Tensor(Z[ix].reshape(B * K, Z_DIM)))
            loss = ag.square(q - Tensor(D["ret_all"][ix].reshape(B * K, N_TACTICS))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append(float(loss.data))
        if log:
            log(f"LIAM Q epoch {ep + 1}/{epochs_q} mse={np.mean(tot):.4f}")

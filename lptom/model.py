"""Networks and training for LP-ToM.

Two learned components are trained separately.

TacticValueNet
    Q(c, theta, g) is the counterfactual value of committing to manoeuvre g
    for one option, given the scene c (kinematics and the bird's-eye raster)
    and the opponent style theta. It is trained on simulated outcomes of
    every manoeuvre (privileged counterfactual labels). With zmode="none"
    the same network ignores theta (opponent-agnostic baseline).

ToMNet
    Amortised Bayesian inference over the opponent style theta. Every 5 Hz
    window of interaction features contributes Gaussian evidence
    (m_t, lambda_t). Evidence is fused with a prior in natural parameters.
    The prior is either the population prior or a language prior computed
    from a spotter message. The network also contains an ego-conditioned
    response model p(y | window, g, theta) used for the surprise signal.
"""
import numpy as np
from . import autograd as ag
from .autograd import Tensor
from .nn import Module, MLP, RasterCNN, BagOfWords
from .sim import N_TACTICS, THETA_DIM
from .data import WIN, FEAT, DEC_EVERY

KIN = 9
CNN_OUT = 24
PRIOR_SD = 0.6
PRIOR_PREC = 1.0 / PRIOR_SD ** 2


def onehot(idx, n=N_TACTICS):
    out = np.zeros(np.shape(idx) + (n,), np.float32)
    np.put_along_axis(out, np.asarray(idx)[..., None], 1.0, axis=-1)
    return out


class TacticValueNet(Module):
    def __init__(self, seed, use_vision=True, zmode="theta"):
        rng = np.random.default_rng(seed)
        self.use_vision, self.zmode = use_vision, zmode
        zin = THETA_DIM if zmode == "theta" else 0
        self.cnn = RasterCNN(rng, CNN_OUT)
        self.q = MLP(rng, [KIN + CNN_OUT + zin, 128, 128, N_TACTICS])

    def features(self, kin, ras):
        if self.use_vision:
            r = self.cnn(Tensor(np.asarray(ras, np.float32)))
        else:
            r = Tensor(np.zeros((kin.shape[0], CNN_OUT), np.float32))
        return ag.concat([Tensor(np.asarray(kin, np.float32)), r], axis=-1)

    def qvalues(self, feat, theta=None):
        if self.zmode == "none" or theta is None:
            return self.q(feat)
        return self.q(ag.concat([feat, theta], axis=-1))


class ToMNet(Module):
    def __init__(self, seed, vocab_size):
        rng = np.random.default_rng(seed + 7)
        self.ev = MLP(rng, [WIN * FEAT, 96, 96, 2 * THETA_DIM], out_scale=0.3)
        self.resp = MLP(rng, [WIN * FEAT + N_TACTICS + THETA_DIM, 64, 64, 4])
        self.bow = BagOfWords(rng, vocab_size, 16)
        self.lang = MLP(rng, [16, 32, 2 * THETA_DIM], out_scale=0.3)

    def evidence(self, W):
        """W [...,WIN*FEAT] -> evidence mean m and precision lam."""
        h = self.ev(W if isinstance(W, Tensor) else Tensor(W))
        return h[..., :THETA_DIM], ag.softplus(h[..., THETA_DIM:] - 2.0)

    def lang_prior(self, bow):
        h = self.lang(ag.tanh(self.bow(bow)))
        return h[..., :THETA_DIM], ag.softplus(h[..., THETA_DIM:]) + PRIOR_PREC

    def response(self, W, tac_oh, theta):
        th = theta if isinstance(theta, Tensor) else Tensor(np.asarray(theta, np.float32))
        h = self.resp(ag.concat([Tensor(np.asarray(W, np.float32)), Tensor(tac_oh), th], axis=-1))
        return h[..., :2], ag.tanh(h[..., 2:] * 0.5) * 4.0 - 1.0


def gauss_nll(y, mu, logv):
    return 0.5 * (ag.square(Tensor(y) - mu) * ag.exp(-logv) + logv)


def train_q(net, D, epochs, seed, lr=1e-3, batch=32, log=None):
    """Regress every manoeuvre's simulated outcome (counterfactual supervision)."""
    rng = np.random.default_rng(seed)
    opt = ag.Adam(net.parameters(), lr=lr)
    n, K = D["ret_all"].shape[:2]
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = []
        for b0 in range(0, n - batch + 1, batch):
            ix = perm[b0:b0 + batch]
            B = len(ix)
            feat = net.features(D["kin"][ix].reshape(B * K, -1), D["ras"][ix].reshape(B * K, 2, 32, 16))
            th = None
            if net.zmode == "theta":
                th = Tensor(np.repeat(D["theta_n"][ix][:, None, :], K, 1).reshape(B * K, THETA_DIM).astype(np.float32))
            q = net.qvalues(feat, th)
            loss = ag.square(q - Tensor(D["ret_all"][ix].reshape(B * K, N_TACTICS))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append(float(loss.data))
        hist.append(float(np.mean(tot)))
        if log:
            log(f"Q[{net.zmode},vision={net.use_vision}] epoch {ep + 1}/{epochs} mse={hist[-1]:.4f}")
    return hist


def train_tom(net, D, epochs, seed, lr=1e-3, batch=32, w_resp=0.5, log=None):
    """Neural posterior estimation over theta with product-of-Gaussians aggregation."""
    from .language import Vocab
    rng = np.random.default_rng(seed + 1)
    opt = ag.Adam(net.parameters(), lr=lr)
    BOW = Vocab().bow(D["msg"])
    n, N = D["W"].shape[:2]
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = []
        for b0 in range(0, n - batch + 1, batch):
            ix = perm[b0:b0 + batch]
            B = len(ix)
            th = D["theta_n"][ix].astype(np.float32)                     # [B,5]
            m, lam = net.evidence(Tensor(D["W"][ix]))                       # [B,N,5]
            use_lang = (rng.uniform(size=B) < 0.5).astype(np.float32)[:, None]
            lmu, lprec = net.lang_prior(BOW[ix])
            mask = Tensor(use_lang)
            prec0 = PRIOR_PREC + mask * (lprec - PRIOR_PREC)
            eta0 = mask * lprec * lmu
            # prefix posteriors including the empty prefix (prior only)
            Lam = ag.cumsum(lam, axis=1) + ag.reshape(prec0, (B, 1, -1))
            Eta = ag.cumsum(lam * m, axis=1) + ag.reshape(eta0, (B, 1, -1))
            ns = 24
            ti = rng.integers(0, N, size=(B, ns))
            bi = np.repeat(np.arange(B)[:, None], ns, 1)
            Lt, Et = Lam[bi, ti], Eta[bi, ti]
            mu = Et / Lt
            thb = Tensor(np.repeat(th[:, None, :], ns, 1))
            npe = 0.5 * (Lt * ag.square(thb - mu) - ag.log(Lt))
            npe0 = 0.5 * (prec0 * ag.square(Tensor(th) - eta0 / prec0) - ag.log(prec0))
            l_npe = npe.sum(axis=-1).mean() + 0.25 * npe0.sum(axis=-1).mean()
            # ego-conditioned response model, conditioned on the true style
            ri = rng.integers(0, N, size=(B, ns))
            Wr = D["W"][ix][bi, ri]
            vm = D["valid"][ix][bi, ri].astype(np.float32)
            rm, rlv = net.response(Wr, onehot(D["TAC"][ix][bi, ri]), np.repeat(th[:, None, :], ns, 1))
            nll = gauss_nll(D["y"][ix][bi, ri], rm, rlv).sum(axis=-1)
            l_resp = (nll * Tensor(vm)).sum() * (1.0 / max(vm.sum(), 1.0))
            loss = l_npe + w_resp * l_resp
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append((float(l_npe.data), float(l_resp.data)))
        a = np.mean(tot, 0)
        hist.append({"npe": float(a[0]), "resp": float(a[1])})
        if log:
            log(f"ToM epoch {ep + 1}/{epochs} npe={a[0]:.4f} resp={a[1]:.4f}")
    return hist

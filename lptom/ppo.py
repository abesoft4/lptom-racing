"""Proximal policy optimisation utilities on the NumPy autograd core."""
import numpy as np
from . import autograd as ag
from .autograd import Tensor


def log_softmax(logits):
    mx = Tensor(logits.data.max(-1, keepdims=True))
    z = logits - mx
    return z - ag.log(ag.exp(z).sum(-1, keepdims=True))


def gae(rew, val, last_val, done, gamma=0.99, lam=0.95):
    """rew, val, done [T,E]. Returns advantages and returns [T,E]."""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    nxt_adv = np.zeros(rew.shape[1])
    nxt_val = last_val
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt_val * nonterm - val[t]
        nxt_adv = delta + gamma * lam * nonterm * nxt_adv
        adv[t] = nxt_adv
        nxt_val = val[t]
    return adv, adv + val


def clipped_surrogate(logp_new, logp_old, adv, clip=0.2):
    """PPO-clip objective as a loss. The clipped branch carries no gradient."""
    ratio = ag.exp(logp_new - Tensor(logp_old.astype(np.float32)))
    r = ratio.data
    unclipped = r * adv
    clipped = np.clip(r, 1 - clip, 1 + clip) * adv
    use = (unclipped <= clipped).astype(np.float32)
    return -(ratio * Tensor((use * adv).astype(np.float32))).mean() - float(((1 - use) * clipped).mean())


def gauss_logp(u, mean, logstd):
    """Diagonal Gaussian log density of scalar actions u under mean (Tensor) and logstd (Tensor[1])."""
    var = ag.exp(logstd * 2.0)
    return -0.5 * (ag.square(Tensor(u.astype(np.float32)) - mean) / var) - logstd - 0.5 * np.log(2 * np.pi)

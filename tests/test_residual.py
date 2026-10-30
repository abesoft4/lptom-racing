"""Checks the residual posterior and Bayes factor of Eq. (bf) against direct formulas."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from lptom.agents import residual_posterior


class Cfg:
    resid_lam, resid_prior = 2.0, 0.5


def logpdf(y, C):
    s, ld = np.linalg.slogdet(C)
    return -0.5 * (y @ np.linalg.solve(C, y) + ld + len(y) * np.log(2 * np.pi))


rng = np.random.default_rng(0)
n, M, beta, alpha = 40, 6, 4.0, Cfg.resid_lam
for case in range(5):
    Phi = rng.normal(size=(n, M))
    y = Phi @ rng.normal(size=M) * (case % 2) + rng.normal(size=n) / np.sqrt(beta)
    A = alpha * np.eye(M) + beta * Phi.T @ Phi
    b = beta * Phi.T @ y
    (m, L), p = residual_posterior(A[None], b[None], Cfg)
    direct = logpdf(y, np.eye(n) / beta + Phi @ Phi.T / alpha) - logpdf(y, np.eye(n) / beta)
    if direct < 30:
        assert abs(np.log(p[0] / (1 - p[0])) - direct) < 1e-6, (p, direct)
    else:
        assert p[0] > 1 - 1e-12
    assert np.allclose(m[0], np.linalg.solve(A, b))
    assert np.allclose(L[0] @ L[0].T, np.linalg.inv(A))
print("residual posterior and Bayes factor checks passed")

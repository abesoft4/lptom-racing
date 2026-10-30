"""Finite-difference verification of every autograd operator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from lptom import autograd as ag

ag.DTYPE = np.float64


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check(build, shapes, seed=0, tol=1e-5, positive=False):
    rng = np.random.default_rng(seed)
    xs = [rng.normal(size=s) for s in shapes]
    if positive:
        xs = [np.abs(x) + 0.5 for x in xs]
    ts = [ag.Tensor(x.copy(), requires_grad=True) for x in xs]
    out = build(*ts)
    w = rng.normal(size=out.shape)
    (out * ag.Tensor(w)).sum().backward()
    for t, x in zip(ts, xs):
        def f():
            tt = [ag.Tensor(v) for v in [t2.data for t2 in ts]]
            return float((build(*tt).data * w).sum())
        ng = numgrad(f, t.data)
        err = np.max(np.abs(ng - t.grad)) / (np.max(np.abs(ng)) + 1e-8)
        assert err < tol, (build, err)
    return True


def test_all():
    check(lambda a, b: a + b, [(3, 4), (1, 4)])
    check(lambda a, b: a * b, [(3, 4), (3, 1)])
    check(lambda a, b: a / b, [(3, 4), (3, 4)], positive=True)
    check(lambda a, b: a - b, [(2, 3), (3,)])
    check(lambda a, b: a @ b, [(5, 3), (3, 2)])
    check(lambda a, b: a @ b, [(2, 5, 3), (3, 2)])
    check(lambda a: ag.exp(a), [(3, 3)])
    check(lambda a: ag.log(a), [(3, 3)], positive=True)
    check(lambda a: ag.tanh(a), [(3, 3)])
    check(lambda a: ag.sigmoid(a), [(3, 3)])
    check(lambda a: ag.softplus(a), [(3, 3)])
    check(lambda a: ag.relu(a), [(4, 4)], seed=3)
    check(lambda a: ag.square(a), [(3, 3)])
    check(lambda a: a.sum(axis=1), [(3, 4)])
    check(lambda a: a.mean(axis=0, keepdims=True), [(3, 4)])
    check(lambda a: ag.cumsum(a, axis=1), [(2, 5, 3)])
    check(lambda a: ag.reshape(a, (6, 2)), [(3, 4)])
    check(lambda a: ag.transpose(a, (1, 0, 2)), [(2, 3, 4)])
    check(lambda a: a[np.array([0, 2, 2]), 1:3], [(4, 4)])
    check(lambda a, b: ag.concat([a, b], axis=-1), [(3, 2), (3, 4)])
    check(lambda a: ag.broadcast_to(a, (4, 3)), [(1, 3)])
    check(lambda x, w, b: ag.conv2d(x, w, b, 2, 1), [(2, 2, 8, 6), (3, 2, 3, 3), (3,)])
    check(lambda x, w, b: ag.conv2d(x, w, b, 1, 1), [(1, 2, 5, 5), (2, 2, 3, 3), (2,)])
    print("all autograd gradient checks passed")


if __name__ == "__main__":
    test_all()

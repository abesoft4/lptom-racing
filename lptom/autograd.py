"""Minimal reverse-mode automatic differentiation on NumPy arrays.

The module provides exactly the operators used by the LP-ToM models.
Every operator is verified against central finite differences in
tests/test_autograd.py.
"""
import numpy as np

DTYPE = np.float32


def _unbroadcast(grad, shape):
    """Sum a broadcast gradient back to the original operand shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")
    __array_priority__ = 100

    def __init__(self, data, requires_grad=False, parents=(), backward=None):
        self.data = np.asarray(data, dtype=DTYPE) if not isinstance(data, np.ndarray) else data
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = parents
        self._backward = backward

    # ------------------------------------------------------------------ utils
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numpy(self):
        return self.data

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"

    def detach(self):
        return Tensor(self.data)

    def backward(self, grad=None):
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for t in reversed(order):
            if t._backward is not None and t.grad is not None:
                t._backward(t.grad)

    # --------------------------------------------------------------- algebra
    def __add__(self, other):
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return add(self, neg(as_tensor(other)))

    def __rsub__(self, other):
        return add(as_tensor(other), neg(self))

    def __mul__(self, other):
        return mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return mul(self, reciprocal(as_tensor(other)))

    def __rtruediv__(self, other):
        return mul(as_tensor(other), reciprocal(self))

    def __neg__(self):
        return neg(self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, idx):
        return getitem(self, idx)

    def sum(self, axis=None, keepdims=False):
        return tsum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.data.shape[a] for a in np.atleast_1d(axis)])
        return tsum(self, axis, keepdims) * (1.0 / n)

    def reshape(self, *shape):
        return reshape(self, shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape)


def as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=DTYPE))


def _acc(t, g):
    if not t.requires_grad and t._backward is None:
        return
    t.grad = g if t.grad is None else t.grad + g


def _make(data, parents, backward):
    req = any(p.requires_grad or p._backward is not None for p in parents)
    if not req:
        return Tensor(data)
    return Tensor(data, parents=parents, backward=backward)


# ---------------------------------------------------------------- operators
def add(a, b):
    a, b = as_tensor(a), as_tensor(b)

    def bw(g):
        _acc(a, _unbroadcast(g, a.shape))
        _acc(b, _unbroadcast(g, b.shape))

    return _make(a.data + b.data, (a, b), bw)


def neg(a):
    return _make(-a.data, (a,), lambda g: _acc(a, -g))


def mul(a, b):
    a, b = as_tensor(a), as_tensor(b)

    def bw(g):
        _acc(a, _unbroadcast(g * b.data, a.shape))
        _acc(b, _unbroadcast(g * a.data, b.shape))

    return _make(a.data * b.data, (a, b), bw)


def reciprocal(a):
    out = 1.0 / a.data
    return _make(out, (a,), lambda g: _acc(a, -g * out * out))


def matmul(a, b):
    a, b = as_tensor(a), as_tensor(b)

    def bw(g):
        ga = g @ np.swapaxes(b.data, -1, -2)
        gb = np.swapaxes(a.data, -1, -2) @ g
        _acc(a, _unbroadcast(ga, a.shape))
        _acc(b, _unbroadcast(gb, b.shape))

    return _make(a.data @ b.data, (a, b), bw)


def exp(a):
    out = np.exp(a.data)
    return _make(out, (a,), lambda g: _acc(a, g * out))


def log(a):
    return _make(np.log(a.data), (a,), lambda g: _acc(a, g / a.data))


def relu(a):
    m = a.data > 0
    return _make(a.data * m, (a,), lambda g: _acc(a, g * m))


def tanh(a):
    out = np.tanh(a.data)
    return _make(out, (a,), lambda g: _acc(a, g * (1 - out * out)))


def sigmoid(a):
    out = 0.5 * (1 + np.tanh(0.5 * a.data))
    return _make(out, (a,), lambda g: _acc(a, g * out * (1 - out)))


def softplus(a):
    x = a.data
    out = np.logaddexp(0, x).astype(DTYPE)
    s = 0.5 * (1 + np.tanh(0.5 * x))
    return _make(out, (a,), lambda g: _acc(a, g * s))


def square(a):
    return _make(a.data * a.data, (a,), lambda g: _acc(a, 2 * g * a.data))


def tsum(a, axis=None, keepdims=False):
    out = a.data.sum(axis=axis, keepdims=keepdims)

    def bw(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        _acc(a, np.broadcast_to(g, a.shape).astype(DTYPE))

    return _make(out, (a,), bw)


def cumsum(a, axis):
    out = np.cumsum(a.data, axis=axis)

    def bw(g):
        r = np.flip(np.cumsum(np.flip(g, axis=axis), axis=axis), axis=axis)
        _acc(a, r)

    return _make(out, (a,), bw)


def reshape(a, shape):
    old = a.shape
    return _make(a.data.reshape(shape), (a,), lambda g: _acc(a, g.reshape(old)))


def transpose(a, axes):
    inv = np.argsort(axes)
    return _make(np.transpose(a.data, axes), (a,), lambda g: _acc(a, np.transpose(g, inv)))


def getitem(a, idx):
    def bw(g):
        z = np.zeros_like(a.data)
        np.add.at(z, idx, g)
        _acc(a, z)

    return _make(a.data[idx], (a,), bw)


def concat(ts, axis=-1):
    ts = [as_tensor(t) for t in ts]
    sizes = [t.shape[axis] for t in ts]
    out = np.concatenate([t.data for t in ts], axis=axis)

    def bw(g):
        splits = np.split(g, np.cumsum(sizes)[:-1], axis=axis)
        for t, gg in zip(ts, splits):
            _acc(t, gg)

    return _make(out, tuple(ts), bw)


def broadcast_to(a, shape):
    old = a.shape
    return _make(np.broadcast_to(a.data, shape).copy(), (a,), lambda g: _acc(a, _unbroadcast(g, old)))


# --------------------------------------------------------------- conv2d op
def _im2col(x, k, s, p):
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    ho = (h + 2 * p - k) // s + 1
    wo = (w + 2 * p - k) // s + 1
    cols = np.empty((n, c, k, k, ho, wo), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = xp[:, :, i:i + s * ho:s, j:j + s * wo:s]
    return cols, ho, wo


def conv2d(x, w, b, stride=1, pad=1):
    """x [N,C,H,W], w [O,C,k,k], b [O]."""
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    cols, ho, wo = _im2col(x.data, k, stride, pad)
    cm = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * ho * wo, c * k * k)
    out = (cm @ w.data.reshape(o, -1).T + b.data).reshape(n, ho, wo, o).transpose(0, 3, 1, 2)

    def bw(g):
        gm = g.transpose(0, 2, 3, 1).reshape(n * ho * wo, o)
        _acc(w, (gm.T @ cm).reshape(w.shape))
        _acc(b, gm.sum(axis=0))
        if x.requires_grad or x._backward is not None:
            gc = (gm @ w.data.reshape(o, -1)).reshape(n, ho, wo, c, k, k).transpose(0, 3, 4, 5, 1, 2)
            gx = np.zeros((n, c, h + 2 * pad, wd + 2 * pad), dtype=x.data.dtype)
            for i in range(k):
                for j in range(k):
                    gx[:, :, i:i + stride * ho:stride, j:j + stride * wo:stride] += gc[:, :, i, j]
            _acc(x, gx[:, :, pad:pad + h, pad:pad + wd])

    return _make(out.astype(x.data.dtype), (x, w, b), bw)


# --------------------------------------------------------------- optimiser
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, clip=5.0, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.clip, self.wd = lr, betas[0], betas[1], eps, clip, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        grads = [np.zeros_like(p.data) if p.grad is None else p.grad for p in self.params]
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale + self.wd * p.data
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.data -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p.data.dtype)
        return norm

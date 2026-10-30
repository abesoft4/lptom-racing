"""Small neural building blocks on top of the autograd core."""
import numpy as np
from . import autograd as ag
from .autograd import Tensor, DTYPE


class Module:
    def parameters(self):
        out = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, Module):
                        out.extend(u.parameters())
        return out

    def state_dict(self):
        return [p.data.copy() for p in self.parameters()]

    def load_state_dict(self, sd):
        for p, d in zip(self.parameters(), sd):
            p.data[...] = d

    def n_params(self):
        return int(sum(p.data.size for p in self.parameters()))


def _param(arr):
    return Tensor(arr.astype(DTYPE), requires_grad=True)


class Linear(Module):
    def __init__(self, rng, n_in, n_out, scale=1.0):
        lim = scale * np.sqrt(6.0 / (n_in + n_out))
        self.W = _param(rng.uniform(-lim, lim, size=(n_in, n_out)))
        self.b = _param(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.W + self.b


class MLP(Module):
    def __init__(self, rng, sizes, act="relu", out_scale=1.0):
        self.layers = [Linear(rng, a, b, scale=(out_scale if i == len(sizes) - 2 else 1.0))
                       for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
        self.act = {"relu": ag.relu, "tanh": ag.tanh}[act]

    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


class Conv(Module):
    def __init__(self, rng, c_in, c_out, k=3, stride=2, pad=1):
        lim = np.sqrt(6.0 / (c_in * k * k + c_out))
        self.W = _param(rng.uniform(-lim, lim, size=(c_out, c_in, k, k)))
        self.b = _param(np.zeros(c_out))
        self.stride, self.pad = stride, pad

    def __call__(self, x):
        return ag.conv2d(x, self.W, self.b, self.stride, self.pad)


class RasterCNN(Module):
    """Two strided convolutions and a projection for the 2x32x16 BEV raster."""

    def __init__(self, rng, out_dim=24):
        self.c1 = Conv(rng, 2, 8)
        self.c2 = Conv(rng, 8, 16)
        self.fc = Linear(rng, 16 * 8 * 4, out_dim)

    def __call__(self, x):
        h = ag.relu(self.c1(x))
        h = ag.relu(self.c2(h))
        h = ag.reshape(h, (h.shape[0], -1))
        return ag.tanh(self.fc(h))


class BagOfWords(Module):
    """Mean-pooled learned word embeddings for spotter messages."""

    def __init__(self, rng, vocab, dim):
        self.E = _param(rng.normal(0, 0.3, size=(vocab, dim)))

    def __call__(self, counts):
        # counts [B, V] row-normalised bag of words (constant input)
        return Tensor(counts) @ self.E

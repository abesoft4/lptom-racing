"""Procedurally generated closed tracks in Frenet form."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

M = 1024          # samples per track
HALF_W = 6.0      # track half width [m]
MU_G = 1.5 * 9.81 # lateral grip [m/s^2]
V_TOP = 48.0      # top speed [m/s]
A_ACC = 5.0       # traction-limited acceleration [m/s^2]
A_BRK = 9.0       # braking deceleration [m/s^2]


class Track:
    def __init__(self, seed):
        rng = np.random.default_rng(10_000 + seed)
        for _ in range(200):
            phi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
            r = np.ones_like(phi)
            for k in range(2, 6):
                r += rng.uniform(0.0, 0.42 / k) * np.cos(k * phi + rng.uniform(0, 2 * np.pi))
            r *= rng.uniform(110, 150)
            x, y = r * np.cos(phi), r * np.sin(phi)
            seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
            s = np.concatenate([[0], np.cumsum(seg)])
            L = s[-1]
            su = np.linspace(0, L, M, endpoint=False)
            xu = np.interp(su, s, np.append(x, x[0]))
            yu = np.interp(su, s, np.append(y, y[0]))
            ds = L / M
            dx = (np.roll(xu, -1) - np.roll(xu, 1)) / (2 * ds)
            dy = (np.roll(yu, -1) - np.roll(yu, 1)) / (2 * ds)
            ddx = (np.roll(xu, -1) - 2 * xu + np.roll(xu, 1)) / ds ** 2
            ddy = (np.roll(yu, -1) - 2 * yu + np.roll(yu, 1)) / ds ** 2
            kap = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
            kap = gaussian_filter1d(kap, 3, mode="wrap")
            if np.abs(kap).max() < 1.0 / (HALF_W + 16.0) and np.abs(kap).max() > 1.0 / 45.0:
                break
        self.L, self.ds = L, ds
        self.x, self.y = xu, yu
        self.heading = np.arctan2(dy, dx)
        self.kappa = kap
        # inside-biased racing line (apex hugging) and its speed profile
        kbar = gaussian_filter1d(kap, 10, mode="wrap")
        self.dstar = 0.25 * HALF_W * np.tanh(kbar / 0.012)
        self.kbar = kbar
        kl = np.abs(kap / np.maximum(1 - kap * self.dstar, 0.3))
        v = np.minimum(V_TOP, np.sqrt(MU_G / np.maximum(kl, 1e-5)))
        for _ in range(3):  # periodic forward/backward passes
            for i in range(1, 2 * M):
                j, p = i % M, (i - 1) % M
                v[j] = min(v[j], np.sqrt(v[p] ** 2 + 2 * A_ACC * ds))
            for i in range(2 * M, 0, -1):
                j, p = (i - 1) % M, i % M
                v[j] = min(v[j], np.sqrt(v[p] ** 2 + 2 * A_BRK * ds))
        self.vref = v
        self.tref = np.concatenate([[0.0], np.cumsum(ds / v)])  # reference time along s
        self.T_lap = self.tref[-1]
        # direction of the next corner within 60 m (for inside/outside choice)
        look = int(100 / ds)
        acc = np.zeros(M)
        for k in range(1, look):
            acc += np.roll(kap, -k) * np.exp(-3.0 * k / look)
        self.next_turn = np.sign(acc + 1e-9)
        self._occ = None

    # occupancy image for the BEV raster
    def occupancy(self, res=0.5):
        if self._occ is None:
            pad = HALF_W + 60
            x0, y0 = self.x.min() - pad, self.y.min() - pad
            nx = int((self.x.max() + pad - x0) / res) + 1
            ny = int((self.y.max() + pad - y0) / res) + 1
            img = np.zeros((nx, ny), dtype=np.uint8)
            nxv, nyv = -np.sin(self.heading), np.cos(self.heading)
            for dd in np.arange(-HALF_W, HALF_W + 1e-6, res * 0.7):
                fine = np.linspace(0, M, 4 * M, endpoint=False)
                xx = np.interp(fine, np.arange(M + 1), np.append(self.x + dd * nxv, self.x[0] + dd * nxv[0]))
                yy = np.interp(fine, np.arange(M + 1), np.append(self.y + dd * nyv, self.y[0] + dd * nyv[0]))
                img[((xx - x0) / res).astype(int), ((yy - y0) / res).astype(int)] = 1
            self._occ = (img, x0, y0, res)
        return self._occ


class TrackSet:
    """Stacks several tracks so that vectorised environments can index them."""

    def __init__(self, seeds):
        self.tracks = [Track(s) for s in seeds]
        self.seeds = list(seeds)
        for name in ["kappa", "dstar", "vref", "next_turn", "kbar", "x", "y", "heading"]:
            setattr(self, name, np.stack([getattr(t, name) for t in self.tracks]))
        self.tref = np.stack([t.tref for t in self.tracks])
        self.T_lap = np.array([t.T_lap for t in self.tracks])
        self.L = np.array([t.L for t in self.tracks])
        self.ds = np.array([t.ds for t in self.tracks])

    def idx(self, tid, s):
        return (np.floor(s / self.ds[tid]).astype(int)) % M

    def time_at(self, tid, s):
        """Reference time needed to reach arc length s (laps unrolled)."""
        lap = np.floor(s / self.L[tid])
        x = (s - lap * self.L[tid]) / self.ds[tid]
        i = np.floor(x).astype(int)
        f = x - i
        tr = self.tref[tid, i] * (1 - f) + self.tref[tid, np.minimum(i + 1, M)] * f
        return lap * self.T_lap[tid] + tr

    def at(self, name, tid, s):
        return getattr(self, name)[tid, self.idx(tid, s)]

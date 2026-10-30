"""Opponent families. theta = [u, b, tau, l, eps]."""
import numpy as np

# centre and jitter per family
FAMILIES = {
    # seen during training
    "quick-blocker":  dict(c=[0.996, 0.90, 2.0, 0.0, 0.01], j=[0.003, 0.08, 1.0, 0.15, 0.01]),
    "slow-blocker":   dict(c=[0.996, 0.85, 10.5, 0.0, 0.01], j=[0.003, 0.08, 1.5, 0.15, 0.01]),
    "inside-coverer": dict(c=[0.996, 0.15, 5.0, 0.9, 0.01], j=[0.003, 0.10, 2.0, 0.10, 0.01]),
    "wide-runner":    dict(c=[0.996, 0.15, 5.0, -0.9, 0.01], j=[0.003, 0.10, 2.0, 0.10, 0.01]),
    "line-keeper":    dict(c=[0.999, 0.05, 5.0, 0.0, 0.01], j=[0.003, 0.05, 2.0, 0.15, 0.01]),
    "error-prone":    dict(c=[0.999, 0.45, 5.0, 0.0, 0.10], j=[0.003, 0.10, 2.0, 0.20, 0.03]),
}
SEEN = list(FAMILIES.keys())
UNSEEN = ["blend", "adaptive", "mid-blocker-wide"]
LO = np.array([0.975, 0.0, 1.0, -1.0, 0.0])
HI = np.array([1.000, 1.0, 12.0, 1.0, 0.25])


def sample_family(name, n, rng):
    if name in FAMILIES:
        f = FAMILIES[name]
        th = np.array(f["c"]) + rng.uniform(-1, 1, (n, 5)) * np.array(f["j"])
    elif name == "blend":
        # convex mixture of two random seen families
        a = rng.integers(0, len(SEEN), n)
        b = (a + rng.integers(1, len(SEEN), n)) % len(SEEN)
        w = rng.uniform(0.3, 0.7, (n, 1))
        ta = np.stack([sample_family(SEEN[i], 1, rng)[0] for i in a])
        tb = np.stack([sample_family(SEEN[i], 1, rng)[0] for i in b])
        th = w * ta + (1 - w) * tb
    elif name == "adaptive":
        th = np.array([0.996, 0.15, 3.0, 0.0, 0.01]) + rng.uniform(-1, 1, (n, 5)) * np.array([0.003, 0.0, 1.0, 0.1, 0.01])
    elif name == "mid-blocker-wide":
        th = np.array([0.996, 0.60, 6.0, -0.6, 0.03]) + rng.uniform(-1, 1, (n, 5)) * np.array([0.003, 0.08, 1.5, 0.15, 0.02])
    else:
        raise KeyError(name)
    return np.clip(th, LO, HI)


def sample_mix(names, n, rng):
    fam = rng.integers(0, len(names), n)
    th = np.zeros((n, 5))
    for k, nm in enumerate(names):
        m = fam == k
        if m.any():
            th[m] = sample_family(nm, int(m.sum()), rng)
    return th, fam

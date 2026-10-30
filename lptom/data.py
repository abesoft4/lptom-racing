"""Interaction data collection with a uniform random logging policy."""
import numpy as np
from .sim import RaceEnv, OPTION_STEPS, RACE_STEPS, N_TACTICS, HALF_W, VLAT
from .opponents import sample_mix, SEEN, sample_family
from .language import make_message

SUB = 2            # evidence is computed at 5 Hz
WIN = 10           # evidence window (2 s)
RESP_H = 10        # response horizon in raw steps (1 s)
FEAT = 7
CF_SAMPLES = 3   # Monte Carlo futures per counterfactual label
CF_HORIZON = 40   # counterfactual horizon equals the commitment (4 s)
N_SUB = RACE_STEPS // SUB
DEC_EVERY = OPTION_STEPS // SUB


def step_features(env):
    st = env.st
    gap = env.gap()
    return np.stack([
        st["d_e"] / HALF_W, st["d_o"] / HALF_W, np.clip(gap, -40, 40) / 20.0,
        (st["v_o"] - st["v_e"]) / 4.0, st["vd_e"] / VLAT, st["vd_o"] / VLAT,
        ((gap < 0) & (gap > -25)).astype(float),
    ], axis=1).astype(np.float32)


def windows(F):
    """F [E, N, FEAT] -> sliding windows [E, N, WIN*FEAT] (zero padded at start)."""
    E, N, D = F.shape
    pad = np.concatenate([np.zeros((E, WIN - 1, D), F.dtype), F], axis=1)
    idx = np.arange(N)[:, None] + np.arange(WIN)[None, :]
    return pad[:, idx, :].reshape(E, N, WIN * D)


def responses(D_o, GAP):
    """Opponent response over the next second at every 5 Hz sample."""
    E, N = D_o.shape
    h = RESP_H // SUB
    y = np.zeros((E, N, 2), np.float32)
    valid = np.zeros((E, N), bool)
    y[:, :N - h, 0] = (D_o[:, h:] - D_o[:, :N - h]) / HALF_W * 2.0
    y[:, :N - h, 1] = (GAP[:, h:] - GAP[:, :N - h]) / 4.0
    valid[:, :N - h] = True
    return y, valid


def run_logging(ts, theta, tids, rng, counterfactual=True, switch=None, adaptive=None):
    E = len(tids)
    env = RaceEnv(ts, theta, tids, rng, switch=switch)
    if adaptive is not None:
        env.st["adaptive"][:] = adaptive
    K = RACE_STEPS // OPTION_STEPS
    F = np.zeros((E, N_SUB, FEAT), np.float32)
    D_o = np.zeros((E, N_SUB), np.float32)
    GAP = np.zeros((E, N_SUB), np.float32)
    TAC = np.zeros((E, N_SUB), np.int64)
    kin = np.zeros((E, K, 9), np.float32)
    ras = np.zeros((E, K, 2, 32, 16), np.uint8)
    tac = np.zeros((E, K), np.int64)
    ret = np.zeros((E, K), np.float32)
    ret_all = np.zeros((E, K, N_TACTICS), np.float32)
    alert = np.zeros((E, K), bool)
    for k in range(K):
        kin[:, k] = env.kin_features()
        alert[:, k] = env.st["alert"] > 0
        ras[:, k] = env.raster().astype(np.uint8)
        if counterfactual:
            # privileged training labels: outcome of every manoeuvre from this state
            # averaged over CF_SAMPLES resampled opponent-lapse futures
            base = env.copy_state()
            for m in range(CF_SAMPLES):
                lap = base["lapse_u"] if m == 0 else rng.uniform(size=base["lapse_u"].shape)
                for gg in range(N_TACTICS):
                    env.set_state(base)
                    env.st["lapse_u"] = lap
                    env.set_tactic(np.full(E, gg))
                    t0c, f0c = env.tgap(), env.st["fault_e"].copy()
                    for _ in range(CF_HORIZON):
                        env.step()
                    ret_all[:, k, gg] += (10.0 * (env.tgap() - t0c) - 2.0 * (env.st["fault_e"] - f0c)) / CF_SAMPLES
            env.set_state(base)
        g = rng.integers(0, N_TACTICS, E)
        tac[:, k] = g
        env.set_tactic(g)
        t0, f0 = env.tgap(), env.st["fault_e"].copy()
        for j in range(OPTION_STEPS):
            t = env.st["t"]
            if t % SUB == 0:
                i = t // SUB
                F[:, i] = step_features(env)
                D_o[:, i] = env.st["d_o"]
                GAP[:, i] = env.gap()
                TAC[:, i] = env.st["tactic"]
            env.step()
        ret[:, k] = 10.0 * (env.tgap() - t0) - 2.0 * (env.st["fault_e"] - f0)
    y, valid = responses(D_o, GAP)
    return dict(F=F, TAC=TAC, y=y, valid=valid, kin=kin, ras=ras, tac=tac, ret=ret, ret_all=ret_all, alert=alert,
                win=(env.gap() > 0), final_gap=env.tgap())


def collect(ts, n_eps, seed, families=SEEN, p_wrong=0.2, batch=200, counterfactual=True):
    rng = np.random.default_rng(seed)
    parts = []
    for b0 in range(0, n_eps, batch):
        n = min(batch, n_eps - b0)
        theta, fam = sample_mix(families, n, rng)
        tids = rng.integers(0, len(ts.tracks), n)
        d = run_logging(ts, theta, tids, rng, counterfactual=counterfactual)
        # spotter message: correct with prob 1-p_wrong, otherwise about another opponent
        wrong = rng.uniform(size=n) < p_wrong
        other, _ = sample_mix(families, n, rng)
        d["msg"] = [make_message(other[i] if wrong[i] else theta[i], rng, "train") for i in range(n)]
        d["msg_ok"] = ~wrong
        d["theta"], d["fam"] = theta, fam
        parts.append(d)
    out = {}
    for k in parts[0]:
        v = [p[k] for p in parts]
        out[k] = sum(v, []) if isinstance(v[0], list) else np.concatenate(v)
    out["W"] = windows(out["F"])
    return out

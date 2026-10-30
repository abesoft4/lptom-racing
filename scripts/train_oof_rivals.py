"""Train the two reinforcement-learned out-of-family defenders with PPO.

The defenders are part of the benchmark. They are trained once, on the
training tracks, against attackers that are independent of every evaluated
agent (uniformly random manoeuvres, the rule-based attacker and fixed-manoeuvre
attackers). Output benchmark/rl_rivals.pkl.

Usage: python scripts/train_oof_rivals.py [ITERS]
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from lptom import autograd as ag
from lptom.autograd import Tensor
from lptom.track import TrackSet
from lptom.sim import RaceEnv, RACE_STEPS, OPTION_STEPS, N_TACTICS
from lptom.oof import RivalPolicy, theta_oof, init_state, LIM, RL_EVERY, rival_features_delayed
from lptom.ppo import gae, clipped_surrogate, gauss_logp

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
E = 256
OUT = os.path.join(os.path.dirname(__file__), "..", "benchmark")
os.makedirs(OUT, exist_ok=True)
VARIANTS = {"a": dict(seed=901, lat_cost=0.0), "b": dict(seed=902, lat_cost=0.05)}


def rule_attack(env):
    gap = env.gap()
    rel = env.st["d_o"] * env.ref_side()
    behind = (gap < 0) & (gap > -25)
    return np.where(behind & (rel > 0.5), 2, np.where(behind, 1, 0))


def train(name, seed, lat_cost):
    rng = np.random.default_rng(seed)
    ts = TrackSet(range(6))
    pol = RivalPolicy(seed)
    opt = ag.Adam(pol.parameters(), lr=3e-4)
    t0 = time.time()
    for it in range(ITERS):
        theta = theta_oof(E, rng)
        env = RaceEnv(ts, theta, rng.integers(0, 6, E), rng)
        init_state(env, np.zeros(E, int), rng)
        env.ext_rival = lambda e: e.st["oof_tgt"]
        ego_kind = rng.integers(0, 3, E)          # 0 random, 1 rule, 2 fixed manoeuvre
        fixed = rng.integers(1, 5, E)
        X, U, LP, V, R, D = [], [], [], [], [], []
        prev_tg, prev_rf, prev_tgt = env.tgap(), np.zeros(E), env.st["d_o"].copy()
        std = float(np.exp(pol.logstd.data[0]))
        for t in range(RACE_STEPS):
            if t % OPTION_STEPS == 0:
                g = np.where(ego_kind == 0, rng.integers(0, N_TACTICS, E), np.where(ego_kind == 1, rule_attack(env), fixed))
                env.set_tactic(g)
            if t % RL_EVERY == 0:
                if t > 0:
                    tg = env.tgap()
                    rf = env.st["coll"] - env.st["fault_e"]
                    R.append(-10.0 * (tg - prev_tg) - 2.0 * (rf - prev_rf) - lat_cost * np.abs(env.st["oof_tgt"] - prev_tgt) / LIM)
                    prev_tg, prev_rf = tg, rf
                    D.append(np.zeros(E))
                x = rival_features_delayed(env)
                mu = pol.mean(x).data
                u = mu + std * rng.normal(size=E)
                X.append(x); U.append(u); V.append(pol.v(Tensor(x)).data[:, 0])
                LP.append((-0.5 * ((u - mu) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))
                prev_tgt = env.st["oof_tgt"].copy()
                env.st["oof_tgt"] = np.tanh(u) * LIM
            env.step()
        tg = env.tgap()
        rf = env.st["coll"] - env.st["fault_e"]
        R.append(-10.0 * (tg - prev_tg) - 2.0 * (rf - prev_rf) + 2.0 * (env.gap() < 0))
        D.append(np.ones(E))
        X, U, LP, V, R, D = map(np.array, (X, U, LP, V, R, D))
        adv, ret = gae(R, V, np.zeros(E), D)
        Tn = X.shape[0]
        X, U, LP, adv, ret = X.reshape(Tn * E, -1), U.reshape(-1), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(U)
        for ep in range(4):
            perm = rng.permutation(n)
            for b0 in range(0, n, 4096):
                ix = perm[b0:b0 + 4096]
                mean = pol.mean(X[ix])
                lp = gauss_logp(U[ix], mean, pol.logstd)
                loss = clipped_surrogate(lp, LP[ix], adv[ix])
                vloss = ag.square(pol.v(Tensor(X[ix]))[:, 0] - Tensor(ret[ix].astype(np.float32))).mean()
                ent = pol.logstd.sum()
                tot = loss + 0.5 * vloss - 1e-3 * ent
                opt.zero_grad(); tot.backward(); opt.step()
        if it % 10 == 0 or it == ITERS - 1:
            print(f"[{name}] it {it} ego-win {np.mean(env.gap() > 0):.3f} ret {R.sum(0).mean():+.3f} "
                  f"std {np.exp(pol.logstd.data[0]):.3f} {time.time() - t0:.0f}s", flush=True)
    return pol


if __name__ == "__main__":
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
    for name in names:
        cfg = VARIANTS[name]
        pol = train(name, cfg["seed"], cfg["lat_cost"])
        pickle.dump(pol.state_dict(), open(os.path.join(OUT, f"rl_rival_{name}.pkl"), "wb"))
    print("saved")

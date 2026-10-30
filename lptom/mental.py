"""Mental simulation of the rival (simulation-theoretic theory of mind).

The agent imagines the next option under each manoeuvre. Its imagination
uses the vehicle and interaction model of the simulator, a rival policy of
the known parametric family and rival styles sampled from the agent's belief.
Only quantities the ego can observe initialise the imagined state. Hidden
rival variables are estimated (alert timer) or reset (lapse state, adaptive
counter). Future lapses are sampled from the style's lapse rate.
"""
import numpy as np
from .sim import RaceEnv, N_TACTICS, THETA_LO, THETA_HI, ALERT_HOLD, THREAT

OBSERVED = ["tid", "s_e", "s_o", "v_e", "v_o", "d_e", "d_o", "vd_e", "vd_o", "hist_de",
            "side", "close_k", "tactic", "opt_step"]


def denorm(theta_n):
    return THETA_LO + (np.clip(theta_n, -1.0, 1.0) + 1.0) / 2.0 * (THETA_HI - THETA_LO)


def observable_state(env):
    st = env.st
    obs = {k: st[k].copy() for k in OBSERVED}
    gap = st["s_e"] - st["s_o"]
    # the defence trigger is a function of observed positions
    trig = (gap < 0) & (gap > -THREAT) & (np.abs(st["d_e"] - st["d_o"]) > 0.8)
    obs["alert"] = np.where(trig, ALERT_HOLD, 0.0)
    obs["t"] = st["t"]
    return obs


def mental_values(env, thetas, horizon, rng, model=None, rival_net=None, obs_noise=0.0, beta=0.0, reduce="mean", ext=None, resid=None):
    """thetas [E,S,5] raw styles. Returns imagined option values [E,G] averaged over S."""
    obs = observable_state(env)
    E, S, _ = thetas.shape
    if obs_noise > 0:
        # perception noise on the estimated rival state
        for k, sd in [("s_o", 0.5), ("d_o", 0.3), ("v_o", 0.3), ("vd_o", 0.3)]:
            obs[k] = obs[k] + obs_noise * sd * rng.normal(size=obs[k].shape)
    G = N_TACTICS
    rep = lambda a: np.repeat(np.repeat(a[:, None], S, 1)[:, :, None], G, 2).reshape((E * S * G,) + a.shape[1:])
    st = {k: rep(v) for k, v in obs.items() if k != "t"}
    n = E * S * G
    st["t"] = obs["t"]
    st["theta"] = np.repeat(thetas[:, :, None, :], G, 2).reshape(n, 5)
    st["lapse_u"] = rng.uniform(size=(n, env.st["lapse_u"].shape[1]))
    for k in ["lapse_left", "coll", "fault_e", "overtakes", "passed", "att_count"]:
        st[k] = np.zeros(n)
    st["adaptive"] = np.zeros(n, bool)
    st["sw_step"] = np.full(n, 10 ** 9)
    st["sw_theta"] = st["theta"].copy()
    sim = RaceEnv.__new__(RaceEnv)
    sim.T, sim.E, sim.st = env.T, n, st
    if model is not None:
        sim.model = model
    if rival_net is not None:
        sim.rival_net = rival_net
        if resid is not None:
            if resid.ndim == 3:            # one residual draw per imagined style sample
                sim.rival_resid = np.repeat(resid[:, :, None], G, 2).reshape(E * S * G, -1)
            else:
                sim.rival_resid = rep(resid)
    if ext is not None:
        for k, v in env.st.items():
            if k.startswith("oof_"):
                st[k] = rep(v)
        sim.ext_rival = ext
    g = np.tile(np.arange(G), E * S)
    sim.set_tactic(g)
    t0 = sim.tgap()
    for _ in range(horizon):
        sim.step()
    v = 10.0 * (sim.tgap() - t0) - 2.0 * sim.st["fault_e"]
    v = v.reshape(E, S, G)
    if reduce == "min":
        return v.min(1)
    # optimism bonus: manoeuvres whose imagined outcome differs across plausible styles are informative
    return v.mean(1) + beta * v.std(1)

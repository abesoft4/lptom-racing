"""Online agents and the evaluation loop.

All agents share the simulator, the manoeuvre set, the option length and the
evaluation protocol. They differ only in how they select a manoeuvre.
"""
import numpy as np
from .autograd import Tensor
from .sim import RaceEnv, OPTION_STEPS, RACE_STEPS, N_TACTICS, THETA_DIM, theta_norm, HALF_W
from .data import step_features, SUB, WIN, FEAT, RESP_H, CF_HORIZON
from .model import onehot, PRIOR_PREC

H_SUB = RESP_H // SUB
# perception noise per feature (s.d. at obs_noise=1). Order follows data.step_features and sim.kin_features.
NOISE_FEAT = np.array([0, 0.3 / HALF_W, 0.5 / 20.0, 0.3 / 4.0, 0, 0.3 / 3.5, 0], np.float32)
NOISE_KIN = np.array([0, 0.5 / 20.0, 0.3 / HALF_W, 0, 0.3 / HALF_W, 0.3 / 4.0, 0, 0.3 / 3.5, 0])
SBAR_K = 5


class BeliefState:
    """Diagonal Gaussian belief over theta in natural parameters."""

    def __init__(self, E, prior_mu=None, prior_prec=None):
        self.base_prec = np.full((E, THETA_DIM), PRIOR_PREC, np.float32)
        self.base_eta = np.zeros((E, THETA_DIM), np.float32)
        p = self.base_prec if prior_prec is None else np.asarray(prior_prec, np.float32)
        mu = np.zeros((E, THETA_DIM), np.float32) if prior_mu is None else np.asarray(prior_mu, np.float32)
        self.prec = p.copy()
        self.eta = p * mu

    def add(self, m, lam):
        self.prec = self.prec + lam
        self.eta = self.eta + lam * m

    def forget(self, rho, mask=None):
        """Exponential forgetting toward the population prior (rho=1 keeps all evidence)."""
        rho = np.asarray(rho, np.float32).reshape(-1, 1)
        if mask is not None:
            rho = np.where(mask[:, None], rho, 1.0)
        self.prec = rho * self.prec + (1 - rho) * self.base_prec
        self.eta = rho * self.eta + (1 - rho) * self.base_eta

    @property
    def mu(self):
        return self.eta / self.prec

    @property
    def sd(self):
        return 1.0 / np.sqrt(self.prec)


class MixtureBelief:
    """Bayesian online change-point inference over the learned evidence stream.

    Every hypothesis h is a run that started at time start_h from a prior and
    has accumulated all evidence since then. Its weight is the posterior
    probability of that run length. A new run starts from the population
    prior with hazard h_c. At the start of a session the language prior and the
    population prior compete as two hypotheses with weights w and 1-w.
    """

    def __init__(self, E, H=12, lang_mu=None, lang_prec=None, w_lang=0.8):
        self.E, self.H = E, H
        self.prec0 = np.full((E, THETA_DIM), PRIOR_PREC, np.float32)
        self.prec = np.tile(self.prec0[:, None], (1, H, 1))
        self.eta = np.zeros((E, H, THETA_DIM), np.float32)
        self.logw = np.full((E, H), -np.inf)
        self.start = np.zeros((E, H))
        self.t = 0
        self.logw[:, 0] = 0.0
        if lang_mu is not None:
            self.prec[:, 1] = lang_prec
            self.eta[:, 1] = lang_prec * lang_mu
            self.logw[:, 0] = np.log(1 - w_lang)
            self.logw[:, 1] = np.log(w_lang)

    @staticmethod
    def _ll(m, mu, v):
        return -0.5 * (((m - mu) ** 2) / v + np.log(v)).sum(-1)

    def update(self, m, lam, hazard, kappa=1.0):
        il = 1.0 / np.maximum(lam, 1e-6)
        mu_h = self.eta / self.prec
        ll_h = self._ll(m[:, None], mu_h, 1.0 / self.prec + il[:, None])
        ll_0 = self._ll(m, 0.0, 1.0 / self.prec0 + il)
        grow = self.logw + np.log(1 - hazard) + kappa * ll_h
        new = np.log(hazard) + kappa * ll_0                                  # weights are normalised
        prec = np.concatenate([self.prec + lam[:, None], (self.prec0 + lam)[:, None]], 1)
        eta = np.concatenate([self.eta + (lam * m)[:, None], (lam * m)[:, None]], 1)
        logw = np.concatenate([grow, new[:, None]], 1)
        start = np.concatenate([self.start, np.full((self.E, 1), self.t)], 1)
        keep = np.argsort(-logw, axis=1)[:, :self.H]                          # prune the weakest run
        ix = np.arange(self.E)[:, None]
        self.prec, self.eta, self.logw, self.start = prec[ix, keep], eta[ix, keep], logw[ix, keep], start[ix, keep]
        mx = self.logw.max(1, keepdims=True)
        self.logw = self.logw - (mx + np.log(np.exp(self.logw - mx).sum(1, keepdims=True)))
        self.t += 1

    @property
    def w(self):
        return np.exp(self.logw)

    def boundary(self, h_race):
        """A new race may start with a new style: open a population-prior run with mass h_race."""
        if h_race <= 0:
            return
        prec = np.concatenate([self.prec, self.prec0[:, None]], 1)
        eta = np.concatenate([self.eta, np.zeros((self.E, 1, THETA_DIM), np.float32)], 1)
        logw = np.concatenate([self.logw + np.log(1 - h_race), np.full((self.E, 1), np.log(h_race))], 1)
        start = np.concatenate([self.start, np.full((self.E, 1), self.t)], 1)
        keep = np.argsort(-logw, axis=1)[:, :self.H]
        ix = np.arange(self.E)[:, None]
        self.prec, self.eta, self.logw, self.start = prec[ix, keep], eta[ix, keep], logw[ix, keep], start[ix, keep]

    def recent_change(self, k=5):
        return (self.w * (self.start > self.t - k)).sum(1)

    @property
    def mu(self):
        return (self.w[..., None] * self.eta / self.prec).sum(1)

    @property
    def sd(self):
        mh = self.eta / self.prec
        m2 = (self.w[..., None] * (1.0 / self.prec + mh ** 2)).sum(1)
        return np.sqrt(np.maximum(m2 - self.mu ** 2, 1e-8))

    def sample(self, rng, S):
        w = self.w
        cum = np.cumsum(w, 1)
        u = rng.uniform(size=(self.E, S))
        h = np.minimum((u[..., None] > cum[:, None, :]).sum(-1), self.H - 1)
        ix = np.arange(self.E)[:, None]
        mh = (self.eta / self.prec)[ix, h]
        sh = 1.0 / np.sqrt(self.prec[ix, h])
        return mh + sh * rng.normal(size=mh.shape)


class Agent:
    """kind in {hold, random, rule, agnostic, oracle, bpr, lptom}.

    belief_mode for lptom: gated (surprise-gated forgetting), acc (pure
    accumulation), const (constant forgetting), window (sliding window).
    """

    def __init__(self, kind, q=None, tom=None, belief_mode="gated", use_lang=True, tau=None,
                 rho_const=0.98, rho_reset=0.1, window=25, n_samples=16, prototypes=None,
                 hazard=0.02, refractory=10, surprise_mode="nis", hazard_bcp=1e-3, kappa=1.0,
                 source="posterior", horizon=40, model=None, H=12, w_lang=0.8, obs_noise=0.0,
                 rival_net=None, beta=0.0, h_race=0.0, net=None, resid=False, resid_lam=1.0,
                 resid_rho=1.0, resid_reset=True, resid_mode="bma", resid_beta=4.0, resid_prior=0.5, resid_nu=4.0, seed=0):
        self.kind, self.q, self.tom = kind, q, tom
        self.source, self.horizon, self.model = source, horizon, model
        self.H, self.w_lang, self.obs_noise, self.rival_net, self.beta, self.h_race = H, w_lang, obs_noise, rival_net, beta, h_race
        self.hazard_bcp, self.kappa = hazard_bcp, kappa
        self.surprise_mode = surprise_mode
        self.belief_mode, self.use_lang = belief_mode, use_lang
        self.tau, self.rho_const, self.rho_reset = tau, rho_const, rho_reset
        self.window, self.S, self.proto = window, n_samples, prototypes
        self.hazard, self.refractory = hazard, refractory
        self.net = net                      # PPOEgo or LIAMNet for the literature baselines
        self.resid, self.resid_lam, self.resid_rho, self.resid_reset = resid, resid_lam, resid_rho, resid_reset
        self.resid_mode, self.resid_beta, self.resid_prior, self.resid_nu = resid_mode, resid_beta, resid_prior, resid_nu
        self.rng = np.random.default_rng(seed)


def residual_posterior(RA, Rb, agent):
    """Posterior of the residual rival model and the probability that the learned rival model is misspecified.

    RA = lam I + beta Phi'Phi and Rb = beta Phi'y. The log Bayes factor of the residual model against the
    null model (no residual) is (M/2) log lam - (1/2) log|RA| + (1/2) Rb' RA^-1 Rb.
    """
    M = RA.shape[-1]
    L = np.linalg.cholesky(RA)
    Linv = np.linalg.inv(L)
    cov_chol = np.transpose(Linv, (0, 2, 1))                     # RA^-1 = Linv' Linv
    m = np.einsum("eij,ej->ei", cov_chol, np.einsum("eij,ej->ei", Linv, Rb))
    logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
    log_bf = 0.5 * M * np.log(agent.resid_lam) - 0.5 * logdet + 0.5 * (Rb * m).sum(1)
    prior = np.log(agent.resid_prior / (1 - agent.resid_prior))
    p_mis = 1.0 / (1.0 + np.exp(-np.clip(log_bf + prior, -50, 50)))
    return (m, cov_chol), p_mis


def counterfactual_values(env, horizon=CF_HORIZON):
    """Exact simulated outcome of every manoeuvre from the current state (evaluation only)."""
    base = env.copy_state()
    q = np.zeros((env.E, N_TACTICS))
    for g in range(N_TACTICS):
        env.set_state(base)
        env.set_tactic(np.full(env.E, g))
        t0, f0 = env.tgap(), env.st["fault_e"].copy()
        for _ in range(horizon):
            env.step()
        q[:, g] = 10.0 * (env.tgap() - t0) - 2.0 * (env.st["fault_e"] - f0)
    env.set_state(base)
    return q


def run(ts, agent, theta, tids, rng, bow=None, switch=None, adaptive=None, record=False,
        regret=False, carry=None, ext=None, oof_kinds=None):
    """Run E races in parallel and return per-race metrics (and traces if requested).

    ext is an optional out-of-family rival policy with per-rival kinds oof_kinds."""
    from .baselines import new_hist, update_hist
    E = len(tids)
    env = RaceEnv(ts, theta, tids, rng, switch=switch)
    if adaptive is not None:
        env.st["adaptive"][:] = adaptive
    if ext is not None:
        from .oof import init_state
        init_state(env, oof_kinds, rng)
        env.ext_rival = ext
    use_resid = agent.kind == "mb" and agent.resid and agent.rival_net is not None
    if use_resid:
        from .sim import rival_features, VLAT as _VLAT
        from .rival import RESID_DIM
        if carry is not None and "resid" in carry:
            RA, Rb = carry["resid"]
        else:
            RA = np.tile(agent.resid_lam * np.eye(RESID_DIM), (E, 1, 1))
            Rb = np.zeros((E, RESID_DIM))
        agent.resid_post = None if carry is None else carry["resid_post"]
        agent.p_mis = np.zeros(E) if carry is None else carry["p_mis"]
    is_hist = agent.kind in ("ppo", "liam")
    if is_hist:
        hs = carry["hist"] if carry is not None else new_hist(E)
        if carry is not None:
            hs["race"] += 1
        agent.hs = hs
    T = agent.tom
    N = RACE_STEPS // SUB
    Fbuf = np.zeros((E, WIN, FEAT), np.float32)
    is_tom = agent.kind == "lptom" or (agent.kind == "mb" and agent.source == "posterior")
    is_bpr = agent.kind == "bpr" or (agent.kind == "mb" and agent.source == "bpr")
    need_belief = is_tom or is_bpr
    belief, logp, K = None, None, 0
    if is_tom:
        if carry is not None:
            belief = carry["belief"]
            if agent.belief_mode == "bocpd":
                belief.boundary(agent.h_race)
        elif agent.belief_mode == "bocpd":
            if agent.use_lang and bow is not None:
                lmu, lprec = T.lang_prior(bow)
                belief = MixtureBelief(E, agent.H, lmu.data, lprec.data, agent.w_lang)
            else:
                belief = MixtureBelief(E, agent.H)
        elif agent.use_lang and bow is not None:
            lmu, lprec = T.lang_prior(bow)
            belief = BeliefState(E, lmu.data, lprec.data)
        else:
            belief = BeliefState(E)
    if is_bpr:
        K = len(agent.proto)
        logp = carry["logp"] if carry is not None else np.full((E, K), -np.log(K))
    ev_hist, pred_buf = [], [None] * N
    sbar_hist = np.zeros((E, SBAR_K), np.float32)
    llr_hist = np.zeros((E, SBAR_K), np.float32)
    last_trig = np.full(E, -10 ** 6)
    D_o_hist, gap_hist = np.zeros((E, N), np.float32), np.zeros((E, N), np.float32)
    n_trig = np.zeros(E)
    first_trig = np.full(E, -1)
    surprise = np.zeros(E, np.float32)
    trig = np.zeros(E, bool)
    bo_prev = np.zeros(E, bool)
    reg = []
    tr = dict(mu=[], sd=[], surprise=[], trig=[], tactic=[], tgap=[]) if record else None
    for t in range(RACE_STEPS):
        if t % SUB == 0:
            i = t // SUB
            f = step_features(env)
            if agent.obs_noise > 0:
                f = f + agent.obs_noise * NOISE_FEAT * agent.rng.normal(size=f.shape).astype(np.float32)
            Fbuf = np.concatenate([Fbuf[:, 1:], f[:, None]], axis=1)
            W = Fbuf.reshape(E, WIN * FEAT)
            D_o_hist[:, i], gap_hist[:, i] = env.st["d_o"], env.gap()
            if is_hist:
                update_hist(hs, f, W, agent.net if agent.kind == "liam" else None)
                agent.W = W
            if need_belief:
                trig = np.zeros(E, bool)
                if i >= H_SUB and pred_buf[i - H_SUB] is not None:
                    pm, plv = pred_buf[i - H_SUB]
                    y = np.stack([(D_o_hist[:, i] - D_o_hist[:, i - H_SUB]) / HALF_W * 2.0,
                                  (gap_hist[:, i] - gap_hist[:, i - H_SUB]) / 4.0], axis=1)
                    if is_bpr:
                        ll = -0.5 * (((y[:, None] - pm) ** 2) * np.exp(-plv) + plv).sum(-1)
                        logp = logp + ll
                        logp -= logp.max(1, keepdims=True)
                        p = np.exp(logp)
                        p /= p.sum(1, keepdims=True)
                        p = (1 - agent.hazard) * p + agent.hazard / K
                        logp = np.log(p)
                    elif agent.surprise_mode == "resp":
                        # response surprise: Mahalanobis data-fit term of the ego-conditioned prediction
                        surprise = 0.5 * (((y - pm) ** 2) * np.exp(-plv)).sum(-1)
                tac_oh = onehot(env.st["tactic"])
                if is_tom:
                    m, lam = T.evidence(Tensor(W))
                    m, lam = m.data, lam.data
                    if agent.surprise_mode == "nis" and agent.belief_mode != "bocpd":
                        # normalised innovation: new evidence against the predictive belief
                        surprise = 0.5 * ((m - belief.mu) ** 2 / (1.0 / belief.prec + 1.0 / np.maximum(lam, 1e-6))).sum(-1)
                    sbar_hist = np.concatenate([sbar_hist[:, 1:], surprise[:, None]], axis=1)
                    sbar = sbar_hist.mean(1)
                    if agent.belief_mode == "gated" and agent.tau is not None and i >= SBAR_K:
                        trig = (sbar > agent.tau) & (i - last_trig > agent.refractory)
                        if trig.any():
                            belief.forget(np.full(E, agent.rho_reset), mask=trig)
                            last_trig = np.where(trig, i, last_trig)
                            first_trig = np.where(trig & (first_trig < 0), t, first_trig)
                            n_trig += trig
                    if agent.belief_mode == "bocpd":
                        belief.update(m, lam, agent.hazard_bcp, agent.kappa)
                        surprise = belief.recent_change(SBAR_K)
                        trig = (surprise > 0.5) & ~bo_prev
                        bo_prev = surprise > 0.5
                        n_trig += trig
                    if agent.belief_mode == "bcp":
                        # Bayesian change-point forgetting. Two hypotheses explain the new evidence.
                        # H0 the style is unchanged, m ~ N(mu, 1/Lambda + 1/lambda).
                        # H1 a new style was drawn from the population prior, m ~ N(0, 1/Lambda0 + 1/lambda).
                        il = 1.0 / np.maximum(lam, 1e-6)
                        v0 = 1.0 / belief.prec + il
                        v1 = 1.0 / belief.base_prec + il
                        ll0 = -0.5 * ((m - belief.mu) ** 2 / v0 + np.log(v0)).sum(-1)
                        ll1 = -0.5 * (m ** 2 / v1 + np.log(v1)).sum(-1)
                        llr_hist = np.concatenate([llr_hist[:, 1:], (agent.kappa * (ll1 - ll0))[:, None]], axis=1)
                        hk = min(agent.hazard_bcp * SBAR_K, 0.5)
                        post = 1.0 / (1.0 + np.exp(-(np.log(hk / (1 - hk)) + llr_hist.sum(1))))
                        surprise = post
                        belief.forget(1.0 - post)
                        trig = post > 0.5
                        llr_hist[trig] = 0.0
                        first_trig = np.where(trig & (first_trig < 0), t, first_trig)
                        n_trig += trig
                    if agent.belief_mode == "window":
                        ev_hist.append((m, lam))
                        if len(ev_hist) > agent.window:
                            om, ol = ev_hist.pop(0)
                            belief.prec = belief.prec - ol
                            belief.eta = belief.eta - ol * om
                    elif agent.belief_mode == "const":
                        belief.forget(np.full(E, agent.rho_const))
                    if agent.belief_mode != "bocpd":
                        belief.add(m, lam)
                    rm, rlv = T.response(W, tac_oh, belief.mu)
                    pred_buf[i] = (rm.data, rlv.data)
                else:
                    pms, plvs = [], []
                    for z in agent.proto:
                        rm, rlv = T.response(W, tac_oh, np.repeat(z[None], E, 0))
                        pms.append(rm.data)
                        plvs.append(rlv.data)
                    pred_buf[i] = (np.stack(pms, 1), np.stack(plvs, 1))
            if record and is_tom:
                tr["mu"].append(belief.mu.copy())
                tr["sd"].append(belief.sd.copy())
                tr["surprise"].append(surprise.copy())
                tr["trig"].append(trig.copy())
                tr["tgap"].append(env.tgap().copy())
        if use_resid and t % OPTION_STEPS == 0 and t > 0:
            agent.resid_post, agent.p_mis = residual_posterior(RA, Rb, agent)
        if use_resid and record and t % OPTION_STEPS == 0:
            tr.setdefault("p_mis", []).append(agent.p_mis.copy())
        if t % OPTION_STEPS == 0:
            g = choose(agent, env, belief, logp)
            if regret:
                qv = counterfactual_values(env)
                reg.append(qv.max(1) - qv[np.arange(E), g])
            env.set_tactic(g)
            if record:
                tr["tactic"].append(g.copy())
        if use_resid and t % SUB == 0:
            xr = rival_features(env)
            if agent.obs_noise > 0:
                xr = xr + agent.obs_noise * agent.rng.normal(size=xr.shape).astype(np.float32) * 0.05
            th_hat = np.clip(belief.mu, -1, 1).astype(np.float32)
            z_net, phi = agent.rival_net.logit_and_basis(xr, th_hat)
        env.step()
        if use_resid and t % SUB == 0:
            obs_vd = env.st["vd_o"] / _VLAT
            if agent.obs_noise > 0:
                obs_vd = obs_vd + agent.obs_noise * 0.3 / _VLAT * agent.rng.normal(size=E)
            phi = phi.astype(np.float64)
            if agent.resid_reset and agent.belief_mode == "bocpd":
                # the residual keeps its data with the posterior probability that no new run has just started
                rho = (1.0 - belief.recent_change(2))[:, None]          # mass on the run born at this step
            else:
                rho = np.full((E, 1), agent.resid_rho)
            if (rho < 1.0).any():
                RA = rho[..., None] * RA + (1 - rho[..., None]) * agent.resid_lam * np.eye(RESID_DIM)[None]
                Rb = rho * Rb
            r_t = np.arctanh(np.clip(obs_vd, -0.98, 0.98)) - z_net
            # Student-t likelihood by iteratively reweighted least squares: rare outliers such as lapses
            # are down-weighted, persistent model error is not
            wt = agent.resid_beta * (agent.resid_nu + 1) / (agent.resid_nu + agent.resid_beta * r_t ** 2)
            RA += wt[:, None, None] * phi[:, :, None] * phi[:, None, :]
            Rb += (wt * r_t)[:, None] * phi
    st = env.st
    out = dict(win=(env.gap() > 0).astype(float), tgap=env.tgap(), overtakes=st["overtakes"],
               coll=st["coll"], fault=st["fault_e"], n_trig=n_trig, first_trig=first_trig)
    if need_belief:
        out["carry"] = {"belief": belief, "logp": logp}
        if use_resid:
            out["carry"]["resid"] = (RA, Rb)
            out["carry"]["resid_post"] = agent.resid_post
            out["carry"]["p_mis"] = agent.p_mis
        out["p_mis"] = agent.p_mis.copy() if use_resid else None
    if is_hist:
        out["carry"] = {"hist": hs}
    if regret:
        out["regret"] = np.array(reg).T
    if record:
        out["trace"] = {k: np.array(v) for k, v in tr.items()}
    if is_tom:
        out["mu_end"] = belief.mu.copy()
        out["sd_end"] = belief.sd.copy()
    return out


def choose(agent, env, belief, logp):
    E, k = env.E, agent.kind
    if k == "random":
        return agent.rng.integers(0, N_TACTICS, E)
    if k == "hold":
        return np.zeros(E, int)
    if k == "rule":
        gap = env.gap()
        rel = env.st["d_o"] * env.ref_side()           # >0 when the rival sits on the inside
        behind = (gap < 0) & (gap > -25)
        g = np.where(behind & (rel > 0.5), 2, np.where(behind, 1, 0))
        return np.where((gap > 0) & (gap < 25), 5, g)
    if k == "mb":
        return mental_choice(agent, env, belief, logp)
    if k == "ppo":
        from .baselines import ppo_obs
        kin = env.kin_features()
        if agent.obs_noise > 0:
            kin = kin + agent.obs_noise * NOISE_KIN * agent.rng.normal(size=kin.shape)
        return agent.net.act(ppo_obs(kin.astype(np.float32), agent.W, agent.hs))
    if k == "liam":
        hs = agent.hs
        z = (hs["zs"] / np.maximum(hs["n"], 1)[:, None]).astype(np.float32)
        kin = env.kin_features()
        if agent.obs_noise > 0:
            kin = kin + agent.obs_noise * NOISE_KIN * agent.rng.normal(size=kin.shape)
        q = agent.net.q
        return q.qvalues(q.features(kin, env.raster()), Tensor(z)).data.argmax(1)
    Q = agent.q
    kin = env.kin_features()
    if agent.obs_noise > 0:
        kin = kin + agent.obs_noise * NOISE_KIN * agent.rng.normal(size=kin.shape)
    feat = Q.features(kin, env.raster())
    if k == "agnostic":
        return Q.qvalues(feat).data.argmax(1)
    if k == "oracle":
        th = theta_norm(env.st["theta"]).astype(np.float32)
        return Q.qvalues(feat, Tensor(th)).data.argmax(1)
    if k == "lptom":
        # Bayes decision: posterior-expected counterfactual value
        S = agent.S
        eps = agent.rng.normal(size=(S, E, THETA_DIM)).astype(np.float32)
        z = belief.mu[None] + belief.sd[None] * eps
        fe = np.repeat(feat.data[None], S, 0).reshape(S * E, -1)
        q = Q.qvalues(Tensor(fe), Tensor(z.reshape(S * E, THETA_DIM))).data.reshape(S, E, N_TACTICS)
        return q.mean(0).argmax(1)
    if k == "bpr":
        p = np.exp(logp)
        p /= p.sum(1, keepdims=True)
        q = 0.0
        for j, z in enumerate(agent.proto):
            q = q + p[:, j:j + 1] * Q.qvalues(feat, Tensor(np.repeat(z[None], E, 0).astype(np.float32))).data
        return q.argmax(1)
    raise KeyError(k)


def mental_choice(agent, env, belief, logp):
    """Simulation-theoretic decision: imagine every manoeuvre against rivals drawn from the belief."""
    from .mental import mental_values, denorm
    E, S = env.E, agent.S
    rng = agent.rng
    if agent.source == "true":
        th = np.repeat(env.st["theta"][:, None], S, 1)
    elif agent.source == "prior":
        th = denorm(rng.normal(0.0, 1.0 / np.sqrt(PRIOR_PREC), size=(E, S, THETA_DIM)))
    elif agent.source == "posterior" and hasattr(belief, "sample"):
        th = denorm(belief.sample(rng, S))
    elif agent.source == "posterior":
        th = denorm(belief.mu[:, None] + belief.sd[:, None] * rng.normal(size=(E, S, THETA_DIM)))
    elif agent.source == "maxmin":
        # game-theoretic planner: the rival is assumed to play the type that is worst for the ego
        th = denorm(np.repeat(np.asarray(agent.proto)[None], E, 0))
        return mental_values(env, th, agent.horizon, rng, agent.model, agent.rival_net, agent.obs_noise,
                             reduce="min").argmax(1)
    elif agent.source == "truepolicy":
        # privileged reference: imagines the rival with its true (out-of-family) policy and style
        th = np.repeat(env.st["theta"][:, None], S, 1)
        return mental_values(env, th, agent.horizon, rng, agent.model, None, agent.obs_noise,
                             ext=getattr(env, "ext_rival", None)).argmax(1)
    elif agent.source == "bpr":
        p = np.exp(logp); p /= p.sum(1, keepdims=True)
        idx = np.array([rng.choice(len(agent.proto), size=S, p=pi) for pi in p])
        th = denorm(np.asarray(agent.proto)[idx])
    else:
        raise KeyError(agent.source)
    resid = None
    if getattr(agent, "resid", False) and agent.rival_net is not None and getattr(agent, "resid_post", None) is not None:
        m, Lc = agent.resid_post
        if agent.resid_mode == "mean":
            resid = m
        else:
            # Bayesian model averaging by sampling. Each imagined rival uses the residual model with
            # probability p_mis and draws its weights from the residual posterior.
            eps = rng.normal(size=(E, S, m.shape[1]))
            w = m[:, None] + np.einsum("eij,esj->esi", Lc, eps)
            use = rng.uniform(size=(E, S)) < agent.p_mis[:, None]
            resid = w * use[..., None]
    return mental_values(env, th, agent.horizon, rng, agent.model, agent.rival_net, agent.obs_noise, agent.beta,
                         resid=resid).argmax(1)

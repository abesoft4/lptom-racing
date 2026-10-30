"""Full experimental protocol for one seed.

Usage: python scripts/run_experiments.py SEED [N_TRAIN_RACES] [EPOCHS] [N_EVAL_SESSIONS]
Writes results/seed_<SEED>.json and results/models_seed_<SEED>.pkl
"""
import sys, os, time, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from lptom.track import TrackSet
from lptom.data import collect
from lptom.model import TacticValueNet, ToMNet, train_q, train_tom
from lptom.language import Vocab, make_message
from lptom.sim import theta_norm, RACE_STEPS
from lptom.agents import Agent, run
from lptom.rival import RivalNet, collect_rival, train_rival
from lptom.opponents import sample_mix, SEEN, UNSEEN, sample_family, FAMILIES
from lptom.baselines import train_ppo_ego, LIAMNet, train_liam
from lptom.oof import OOFPolicy, RivalPolicy, OOF_KINDS, theta_oof

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N_EVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 400
R = 4                              # races per session against the same rival
TRAIN_TRACKS, TEST_TRACKS = range(6), range(100, 104)
COND_OFFSET = {"seen": 1, "unseen": 2, "switch": 3, "midswitch": 4, "misleading": 5, "robust": 1, "oof": 6}
PPO_ITERS = int(os.environ.get("LPTOM_PPO_ITERS", 300))
ALL_CONDS = ["seen", "unseen", "switch", "midswitch", "misleading", "oof", "robust"]
PART = os.environ.get("LPTOM_PART", "")          # "", "A", "B" or "C" (a seed may be split into jobs)
CONDS_RUN = os.environ.get("LPTOM_CONDS", ",".join(ALL_CONDS)).split(",")
BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmark")
OUT = os.environ.get("LPTOM_OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[seed {SEED}]", *a, flush=True)


def main():
    t0 = time.time()
    ts, tt = TrackSet(TRAIN_TRACKS), TrackSet(TEST_TRACKS)
    V = Vocab()
    MODELS = os.path.join(OUT, f"models_seed_{SEED}.pkl")
    if PART in ("", "A"):
        D = collect(ts, N_TRAIN, SEED)
        D["theta_n"] = theta_norm(D["theta"])
        log("collected", N_TRAIN, "races", round(time.time() - t0), "s")
        # the inference network uses its own, larger log without counterfactual labels
        D_tom = collect(ts, 2 * N_TRAIN, SEED + 5000, counterfactual=False)
        D_tom["theta_n"] = theta_norm(D_tom["theta"])
        tom = ToMNet(SEED, len(V))
        h_tom = train_tom(tom, D_tom, 2 * EPOCHS, SEED)
        del D_tom
        qT = TacticValueNet(SEED, zmode="theta")
        h_qT = train_q(qT, D, EPOCHS, SEED)
        qN = TacticValueNet(SEED, zmode="none")
        h_qN = train_q(qN, D, EPOCHS, SEED)
        Xr, THr, Yr = collect_rival(ts, 1500, SEED)
        rival = RivalNet(SEED)
        h_riv = train_rival(rival, Xr, THr, Yr, 12, SEED)
        Xv, THv, Yv = collect_rival(ts, 200, SEED + 999)
        rival_r2 = float(1 - ((rival(Xv, THv) - Yv) ** 2).mean() / Yv.var())
        # literature baselines: LIAM-style opponent modelling and a PPO racer
        liam = LIAMNet(SEED)
        train_liam(liam, D, EPOCHS, EPOCHS, SEED)
        ppo, h_ppo = train_ppo_ego(ts, SEED, PPO_ITERS, log=log)
        t_train = time.time() - t0
        log("trained", round(t_train), "s")
        # BPR library and game-theoretic types: mean style of every seen family in the training data
        proto = np.stack([D["theta_n"][D["fam"] == k].mean(0) for k in range(len(SEEN))]).astype(np.float32)
        meta = dict(t_train=t_train, rival_r2=rival_r2, proto=proto,
                    hist={"tom": h_tom, "qT": h_qT, "qN": h_qN, "rival": h_riv, "ppo": h_ppo})
        pickle.dump({"tom": tom.state_dict(), "qT": qT.state_dict(), "qN": qN.state_dict(), "rival": rival.state_dict(),
                     "liam": liam.state_dict(), "ppo": ppo.state_dict(), "meta": meta}, open(MODELS, "wb"))
    else:
        # second part of a split run: reuse the models trained by part A
        sd = pickle.load(open(MODELS, "rb"))
        tom = ToMNet(SEED, len(V)); tom.load_state_dict(sd["tom"])
        qT = TacticValueNet(SEED, zmode="theta"); qT.load_state_dict(sd["qT"])
        qN = TacticValueNet(SEED, zmode="none"); qN.load_state_dict(sd["qN"])
        rival = RivalNet(SEED); rival.load_state_dict(sd["rival"])
        liam = LIAMNet(SEED); liam.load_state_dict(sd["liam"])
        from lptom.baselines import PPOEgo
        ppo = PPOEgo(SEED); ppo.load_state_dict(sd["ppo"])
        meta = sd["meta"]
        proto, t_train, rival_r2 = meta["proto"], meta["t_train"], meta["rival_r2"]
        log("loaded models of part A")
    # out-of-family benchmark rivals (fixed artefacts trained once by scripts/train_oof_rivals.py)
    rl_nets = []
    for k in ["a", "b"]:
        pnet = RivalPolicy(0)
        pnet.load_state_dict(pickle.load(open(os.path.join(BENCH, f"rl_rival_{k}.pkl"), "rb")))
        rl_nets.append(pnet)
    oof_policy = OOFPolicy(rl_nets)

    tau, diag = None, {}
    if PART in ("", "A"):   # diagnostics are computed once per seed
        # surprise threshold: 95th percentile of the windowed innovation statistic on
        # stationary two-race sessions (training tracks, fresh rivals, no radio)
        rng = np.random.default_rng(50_000 + SEED)
        th, _ = sample_mix(SEEN, 300, rng)
        carry, vals = None, []
        for k in range(2):
            r = run(ts, Agent("lptom", qT, tom, belief_mode="acc", use_lang=False), th, rng.integers(0, 6, 300),
                    np.random.default_rng(k), record=True, carry=carry)
            carry = r["carry"]
            S = r["trace"]["surprise"]
            vals.append(np.stack([S[i - 4:i + 1].mean(0) for i in range(10, len(S))]))
        tau = float(np.quantile(np.concatenate(vals), 0.95))
        log("tau", round(tau, 3))

        # belief diagnostics on stationary seen rivals (test tracks)
        rng = np.random.default_rng(60_000 + SEED)
        th, fam = sample_mix(SEEN, 400, rng)
        tn = theta_norm(th)
        bow = V.bow([make_message(x, rng, "test") for x in th])
        r_nl = run(tt, Agent("lptom", qT, tom, belief_mode="acc", use_lang=False), th, rng.integers(0, 4, 400),
                   np.random.default_rng(1), record=True)
        r_l = run(tt, Agent("lptom", qT, tom, belief_mode="acc", use_lang=True), th, rng.integers(0, 4, 400),
                  np.random.default_rng(1), bow=bow, record=True)
        diag = {}
        for nm, rr in [("nolang", r_nl), ("lang", r_l)]:
            mu, sd = rr["trace"]["mu"], rr["trace"]["sd"]          # [N,E,5]
            r2 = 1 - ((mu - tn[None]) ** 2).mean(1) / tn.var(0)[None]
            zc = (((mu - tn[None]) / sd) ** 2).mean(1)
            diag[nm] = {"r2_curve": r2.tolist(), "zcal_curve": zc.tolist()}

    H, S16 = 3e-3, 16   # chosen on development seeds 0, 11 and 12
    RESID_LAM = 1.0     # prior precision of the residual rival model, chosen on development seeds
    MM = {"moderate": {"draft": 0.8, "attack": 1.2, "lift": 0.85, "vdef": 0.85, "pace": 1.002},
          "strong": {"draft": 0.6, "attack": 1.4, "lift": 0.95, "vdef": 0.7, "pace": 1.004}}

    def lptom(**kw):
        cfg = dict(tom=tom, source="posterior", belief_mode="bocpd", hazard_bcp=H, kappa=1.0, n_samples=S16,
                   rival_net=rival, resid=True, resid_lam=RESID_LAM, seed=SEED)
        cfg.update(kw)
        return Agent("mb", **cfg)

    def prior(**kw):
        cfg = dict(tom=tom, source="prior", n_samples=S16, rival_net=rival, seed=SEED)
        cfg.update(kw)
        return Agent("mb", **cfg)

    agents = {
        "random": Agent("random", seed=SEED),
        "rule": Agent("rule"),
        "agnosticQ": Agent("agnostic", qN),
        "ppo": Agent("ppo", net=ppo),
        "liam": Agent("liam", net=liam),
        "gt-maxmin": Agent("mb", source="maxmin", prototypes=proto, rival_net=rival, seed=SEED),
        "sim-prior": prior(),
        "sim-bpr": Agent("mb", tom=tom, source="bpr", prototypes=proto, n_samples=S16, rival_net=rival, seed=SEED),
        "lptom": lptom(),
        "lptom-structured": lptom(rival_net=None, resid=False),
        "lptom-nolang": lptom(use_lang=False),
        "lptom-noresid": lptom(resid=False),
        "lptom-acc": lptom(belief_mode="acc"),
        "lptom-const": lptom(belief_mode="const", rho_const=0.99),
        "lptom-single": lptom(belief_mode="bcp"),
        "lptom-amortised": Agent("lptom", qT, tom, belief_mode="bocpd", hazard_bcp=H, n_samples=S16, seed=SEED),
        "oracleQ": Agent("oracle", qT),
        "sim-oracle": Agent("mb", source="true", n_samples=4, rival_net=rival, seed=SEED),
    }
    OOF_AGENTS = ["random", "rule", "agnosticQ", "ppo", "liam", "gt-maxmin", "sim-prior", "sim-bpr", "lptom",
                  "lptom-structured", "lptom-noresid"]
    oof_agents = {nm: agents[nm] for nm in OOF_AGENTS if nm in agents}
    oof_agents["sim-truepolicy"] = Agent("mb", source="truepolicy", n_samples=4, seed=SEED)
    robust_agents = {}
    for lvl, mm in MM.items():
        robust_agents[f"sim-prior|mm-{lvl}"] = prior(model=mm)
        robust_agents[f"lptom|mm-{lvl}"] = lptom(model=mm)
        robust_agents[f"sim-oracle|mm-{lvl}"] = Agent("mb", source="true", n_samples=4, model=mm, rival_net=rival, seed=SEED)
    for lvl, z in [("noise1", 1.0), ("noise2", 2.0)]:
        robust_agents[f"agnosticQ|{lvl}"] = Agent("agnostic", qN, obs_noise=z, seed=SEED)
        robust_agents[f"ppo|{lvl}"] = Agent("ppo", net=ppo, obs_noise=z, seed=SEED)
        robust_agents[f"sim-prior|{lvl}"] = prior(obs_noise=z)
        robust_agents[f"lptom|{lvl}"] = lptom(obs_noise=z)
        robust_agents[f"sim-oracle|{lvl}"] = Agent("mb", source="true", n_samples=4, obs_noise=z, rival_net=rival, seed=SEED)
    results = {}
    for cond in CONDS_RUN:
        rng = np.random.default_rng(70_000 + 1000 * SEED + COND_OFFSET[cond])
        n = N_EVAL
        adaptive = np.zeros(n, bool)
        if cond == "unseen":
            fam_u = rng.integers(0, len(UNSEEN), n)
            tha = np.zeros((n, 5))
            for k, nm in enumerate(UNSEEN):
                m = fam_u == k
                tha[m] = sample_family(nm, int(m.sum()), rng)
            adaptive = fam_u == UNSEEN.index("adaptive")
            fam = fam_u
        elif cond == "oof":
            fam = rng.integers(0, len(OOF_KINDS), n)
            tha = theta_oof(n, rng)
        else:
            tha, fam = sample_mix(SEEN, n, rng)
        famb = (fam + rng.integers(1, len(SEEN), n)) % len(SEEN)
        thb = np.stack([sample_family(SEEN[i], 1, rng)[0] for i in famb])
        if cond == "misleading":
            other, _ = sample_mix(SEEN, n, rng)
            msgs = [make_message(x, rng, "test") for x in other]
        else:
            msgs = [make_message(x, rng, "test") for x in tha]
        bow = None if cond == "oof" else V.bow(msgs)
        tids = [rng.integers(0, len(tt.tracks), n) for _ in range(R)]
        sw_step = rng.integers(RACE_STEPS // 4, 3 * RACE_STEPS // 4, n)
        results[cond] = {}
        pool = robust_agents if cond == "robust" else (oof_agents if cond == "oof" else agents)
        ext_kw = dict(ext=oof_policy, oof_kinds=fam) if cond == "oof" else {}
        for nm, a in pool.items():
            carry, per = None, []
            for k in range(R):
                switch, thk = None, tha
                if cond == "switch" and k >= 2:
                    thk = thb
                if cond == "midswitch":
                    if k == 1:
                        switch = (sw_step, thb)
                    elif k >= 2:
                        thk = thb
                want_regret = (k == 0 and cond in ("seen", "misleading"))
                rr = run(tt, a, thk, tids[k], np.random.default_rng(80_000 + 10 * SEED + k), bow=bow,
                         switch=switch, adaptive=adaptive, carry=carry, regret=want_regret, **ext_kw)
                carry = rr.get("carry")
                d = {"win": rr["win"].tolist(), "tgap": rr["tgap"].tolist(), "overtakes": rr["overtakes"].tolist(),
                     "coll": rr["coll"].tolist(), "fault": rr["fault"].tolist(), "n_trig": rr["n_trig"].tolist(),
                     "first_trig": rr["first_trig"].tolist()}
                if want_regret:
                    d["regret"] = rr["regret"].tolist()
                per.append(d)
            results[cond][nm] = {"races": per, "fam": np.asarray(fam).tolist(), "sw_step": sw_step.tolist()}
            wins = [np.mean(p["win"]) for p in per]
            log(f"{cond:10s} {nm:15s} " + " ".join(f"{w:.3f}" for w in wins))
    # latency per decision (single environment, CPU)
    lat = {}
    for nm in ["agnosticQ", "sim-prior", "lptom", "lptom-structured", "sim-oracle"]:
        a = agents[nm]
        th1 = sample_mix(SEEN, 1, np.random.default_rng(0))[0]
        t1 = time.perf_counter()
        run(tt, a, th1, np.array([0]), np.random.default_rng(0))
        lat[nm] = (time.perf_counter() - t1) / RACE_STEPS * 1000.0
    out = {"seed": SEED, "tau": tau, "train_seconds": t_train, "part": PART,
           "hist": meta["hist"], "rival_r2": rival_r2,
           "params": {"tom": tom.n_params(), "q": qT.n_params(), "q_none": qN.n_params()},
           "latency_ms_per_step": lat, "diag": diag, "results": results}
    json.dump(out, open(os.path.join(OUT, f"seed_{SEED}{'_' + PART if PART else ''}.json"), "w"))
    log("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()

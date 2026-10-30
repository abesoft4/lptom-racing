"""Vectorised head-to-head racing simulator in the Frenet frame.

One ego car races one opponent for a fixed duration. All environments in a
batch advance in lock step. The simulator is deterministic given its state,
because opponent lapses are pre-sampled at reset. This allows exact
counterfactual roll-outs by copying the state.
"""
import numpy as np
from .track import HALF_W, A_ACC, A_BRK, V_TOP

DT = 0.1
OPTION_STEPS = 40          # one manoeuvre is committed for 4 s
RACE_STEPS = 320           # 32 s race (8 decisions)
VLAT = 3.5                 # max lateral speed [m/s]
VLAT_DEF = 3.5             # max lateral speed of a defensive move [m/s]
K_LAT = 1.6                # lateral tracking gain [1/s]
CAR_L, CAR_W = 4.6, 1.9
LANE = 0.7 * HALF_W
EGO_U = 0.99
PRIORITY = 0.0             # corner priority penalty (disabled in all reported runs)
THREAT = 12.0              # range at which the opponent starts defending [m]
ALERT_HOLD = 0.5           # seconds a defender stays alert after a move
PULL_OUT = 9.0             # distance at which a tow manoeuvre pulls out [m]
FEINT_STEPS = 14            # duration of the dummy move of a switchback
TIGHT = 0.0                # over-tight line penalty (disabled in all reported runs)
LIFT = 0.75               # speed factor after running into a covered line
ATTACK_BOOST = 0.035       # speed gain of a committed, unobstructed attack
A_DRAG = 1.2               # coasting deceleration [m/s^2]
HIST = 16                  # ring buffer for delayed mirroring
TACTICS = ["hold", "tow-in", "tow-out", "dive-in", "switchback", "cover"]
N_TACTICS = len(TACTICS)

# opponent parameter vector theta = [u, b, tau, l, eps]
#   u   pace utilisation of the reference speed
#   b   blocking gain toward the ego lateral position
#   tau reaction delay of the block in steps
#   l   lane preference (+1 covers inside, -1 runs wide)
#   eps lapse rate [1/s]
THETA_DIM = 5
THETA_LO = np.array([0.975, 0.0, 1.0, -1.0, 0.0])
THETA_HI = np.array([1.000, 1.0, 12.0, 1.0, 0.25])


def theta_norm(theta):
    return 2 * (theta - THETA_LO) / (THETA_HI - THETA_LO) - 1


def rival_features(env):
    """Observable features from which the rival's lateral action is learned."""
    st = env.st
    gap = env.gap()
    tid = st["tid"]
    lags = st["hist_de"][:, ::-1][:, 1:13] / HALF_W               # ego lateral path, 0.1 to 1.2 s ago
    return np.concatenate([np.stack([
        np.clip(gap, -40, 40) / 20.0, st["d_e"] / HALF_W, st["d_o"] / HALF_W,
        env.T.at("dstar", tid, st["s_o"]) / HALF_W, env.ref_side(), st["vd_e"] / VLAT, st["vd_o"] / VLAT,
        ((gap < 0) & (gap > -THREAT)).astype(float), (np.abs(st["d_e"] - st["d_o"]) > 0.8).astype(float),
        ((gap > 0) & (gap < 30)).astype(float)], 1), lags], 1).astype(np.float32)


RIVAL_FEAT = 22


def resid_basis(x):
    """Basis of the residual rival model. Features, features gated by the defence flag, and a bias."""
    return np.concatenate([x, x * x[:, 7:8], np.ones((len(x), 1), x.dtype)], 1)


class RaceEnv:
    def __init__(self, trackset, theta, tids, rng, start_gap=None, switch=None):
        """theta [E,5]. switch optional (step [E], theta_b [E,5]) for hidden policy change."""
        self.T = trackset
        E = len(tids)
        self.E = E
        st = {}
        st["tid"] = np.asarray(tids)
        L = trackset.L[st["tid"]]
        s0 = rng.uniform(0, 1, E) * L
        gap = rng.uniform(8, 14, E) if start_gap is None else np.full(E, float(start_gap))
        st["s_o"] = s0 + gap
        st["s_e"] = s0.copy()
        v0 = 0.95 * trackset.at("vref", st["tid"], s0)
        st["v_e"], st["v_o"] = v0.copy(), v0.copy()
        st["d_e"] = trackset.at("dstar", st["tid"], s0)
        st["d_o"] = trackset.at("dstar", st["tid"], st["s_o"])
        st["vd_e"], st["vd_o"] = np.zeros(E), np.zeros(E)
        st["hist_de"] = np.repeat(st["d_e"][:, None], HIST, 1)
        st["theta"] = np.array(theta, dtype=float)
        st["t"] = 0
        st["tactic"] = np.zeros(E, dtype=int)
        st["opt_step"] = np.zeros(E, dtype=int)
        st["close_k"] = np.zeros(E, dtype=int)
        st["side"] = np.ones(E)
        st["alert"] = np.zeros(E)
        st["lapse_left"] = np.zeros(E)
        # pre-sampled lapse onsets (Bernoulli per step with rate eps)
        u = rng.uniform(0, 1, (E, RACE_STEPS + 2 * OPTION_STEPS * 4))
        st["lapse_u"] = u
        st["coll"] = np.zeros(E)
        st["fault_e"] = np.zeros(E)
        st["overtakes"] = np.zeros(E)
        st["passed"] = np.zeros(E)
        st["att_count"] = np.zeros(E)   # used by the adaptive opponent
        st["adaptive"] = np.zeros(E, dtype=bool)
        if switch is not None:
            st["sw_step"], st["sw_theta"] = np.asarray(switch[0]), np.asarray(switch[1], dtype=float)
        else:
            st["sw_step"], st["sw_theta"] = np.full(E, 10 ** 9), st["theta"].copy()
        self.st = st

    # ------------------------------------------------------------------ helpers
    def copy_state(self):
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.st.items()}

    def set_state(self, st):
        self.st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}

    def gap(self):
        return self.st["s_e"] - self.st["s_o"]

    def tgap(self):
        """Time gap in seconds (positive when the ego car leads)."""
        tid = self.st["tid"]
        return self.T.time_at(tid, self.st["s_e"]) - self.T.time_at(tid, self.st["s_o"])

    def ref_side(self):
        """Inside of the corner where a pass would complete (shared by both drivers)."""
        st = self.st
        return self.T.at("next_turn", st["tid"], np.maximum(st["s_e"], st["s_o"]) + 40.0)

    def set_tactic(self, g, mask=None):
        st = self.st
        g = np.asarray(g)
        if mask is None:
            mask = np.ones(self.E, dtype=bool)
        st["tactic"] = np.where(mask, g, st["tactic"])
        st["opt_step"] = np.where(mask, 0, st["opt_step"])
        st["close_k"] = np.where(mask, 0, st["close_k"])
        nt = self.T.at("next_turn", st["tid"], np.maximum(st["s_e"], st["s_o"]) + 1.5 * st["v_o"])
        st["side"] = np.where(mask, nt, st["side"])

    # ------------------------------------------------------------------ policy
    def _ego_lane(self):
        """Lateral target of each racecraft manoeuvre."""
        st, T = self.st, self.T
        tid, s = st["tid"], st["s_e"]
        nt = self.ref_side()
        st["side"] = nt
        dstar = T.at("dstar", tid, s)
        g = st["tactic"]
        gap = self.gap()
        behind = gap < 0
        close = behind & (gap > -PULL_OUT)
        st["close_k"] = np.where(close, st["close_k"] + 1, 0)
        k = st["close_k"]
        tow = np.where(behind, st["d_o"], dstar)
        lane = dstar.copy()                                                     # 0 hold
        lane = np.where(g == 1, np.where(close, nt * LANE, tow), lane)          # 1 tow-inside
        lane = np.where(g == 2, np.where(close, -nt * LANE, tow), lane)         # 2 tow-outside
        lane = np.where(g == 3, nt * LANE, lane)                                # 3 dive-inside
        lane = np.where(g == 4, np.where(close, np.where(k < FEINT_STEPS, -nt * LANE, nt * LANE), tow), lane)  # 4 switchback
        lane = np.where(g == 5, np.where(behind, dstar, st["d_o"]), lane)       # 5 cover
        util = np.full(self.E, EGO_U)
        return lane, util

    def _opp_lane(self):
        st, T = self.st, self.T
        th = st["theta"]
        u, b, tau, l, eps = th.T
        tid, s = st["tid"], st["s_o"]
        nt = self.ref_side()
        dstar = T.at("dstar", tid, s)
        pref = (1 - np.abs(l)) * dstar + np.abs(l) * np.sign(l) * nt * LANE
        gap = self.gap()
        # the defender reacts once the attacker is close and leaves the slipstream lane
        trigger = (gap < 0) & (gap > -THREAT) & (np.abs(st["d_e"] - st["d_o"]) > 0.8)
        st["alert"] = np.where(trigger, ALERT_HOLD, np.maximum(st["alert"] - DT, 0.0))
        threat = (gap < 0) & (gap > -THREAT) & (st["alert"] > 0)
        idx = np.clip(np.round(tau).astype(int), 1, HIST - 1)
        d_del = st["hist_de"][np.arange(self.E), HIST - 1 - idx]
        b_eff = np.where(st["adaptive"], np.clip(0.15 + 0.12 * st["att_count"], 0, 0.95), b)
        lane_def = (1 - b_eff) * pref + b_eff * d_del
        lane_att = np.where(gap < 12, -np.sign(st["d_e"] + 1e-3) * 0.55 * HALF_W, st["d_e"])
        lane = np.where(threat, lane_def, np.where((gap > 0) & (gap < 30), lane_att, dstar))
        lapsing = st["lapse_left"] > 0
        lane = np.where(lapsing, -nt * 0.75 * HALF_W, lane)
        mm = getattr(self, "model", None)
        if mm is not None:
            u = u * mm.get("pace", 1.0)
        util = np.where(lapsing, u * 0.88, u)
        vdef = VLAT_DEF if mm is None else VLAT_DEF * mm.get("vdef", 1.0)
        self._opp_vlat = np.where(threat, vdef, VLAT)
        return lane, util

    # -------------------------------------------------------------- dynamics
    def step(self):
        st, T = self.st, self.T
        E = self.E
        t = st["t"]
        g_start = self.gap()
        # hidden opponent switch
        sw = st["sw_step"] == t
        if sw.any():
            st["theta"][sw] = st["sw_theta"][sw]
        # lapse onset
        rate = st["theta"][:, 4] * DT
        onset = (st["lapse_u"][:, min(t, st["lapse_u"].shape[1] - 1)] < rate) & (st["lapse_left"] <= 0)
        st["lapse_left"] = np.where(onset, 1.0, st["lapse_left"] - DT)

        lane_e, u_e = self._ego_lane()
        lane_o, u_o = self._opp_lane()
        ext = getattr(self, "ext_rival", None)
        if ext is not None:
            # out-of-family rival: an external policy sets the lateral target (NaN keeps the parametric one)
            lane_x = ext(self)
            use = ~np.isnan(lane_x) & (st["lapse_left"] <= 0)
            lane_o = np.where(use, np.nan_to_num(lane_x), lane_o)
            self._opp_vlat = np.where(use, VLAT, self._opp_vlat)
        prev_de, prev_do = st["d_e"].copy(), st["d_o"].copy()
        lim = HALF_W - 1.0
        # racing etiquette: once overlapped, neither car may steer into the other
        g0 = self.gap()
        overl = (np.abs(g0) < CAR_L - 0.3) & (np.abs(st["d_e"] - st["d_o"]) >= CAR_W)
        side = np.where(st["d_e"] >= st["d_o"], 1.0, -1.0)
        sep = CAR_W + 0.3
        lane_e = np.where(overl & (side > 0), np.maximum(lane_e, st["d_o"] + sep), lane_e)
        lane_e = np.where(overl & (side < 0), np.minimum(lane_e, st["d_o"] - sep), lane_e)
        lane_o = np.where(overl & (side > 0), np.minimum(lane_o, st["d_e"] - sep), lane_o)
        lane_o = np.where(overl & (side < 0), np.maximum(lane_o, st["d_e"] + sep), lane_o)
        st["vd_e"] = np.clip(K_LAT * (np.clip(lane_e, -lim, lim) - st["d_e"]), -VLAT, VLAT)
        st["vd_o"] = np.clip(K_LAT * (np.clip(lane_o, -lim, lim) - st["d_o"]), -self._opp_vlat, self._opp_vlat)
        net = getattr(self, "rival_net", None)
        if net is not None:
            # learned rival lateral policy replaces the parametric one (lapses keep their meaning)
            x = rival_features(self)
            res = getattr(self, "rival_resid", None)
            if res is None:
                vd = net(x, theta_norm(st["theta"]))
            else:
                # Bayesian last-layer correction of the learned rival model, learned online per rival
                z, phi = net.logit_and_basis(x, theta_norm(st["theta"]))
                vd = np.tanh(z + (phi * res).sum(1))
            vd = vd * VLAT
            toward = np.sign(st["d_e"] - st["d_o"]) * vd > 0
            vd = np.where(overl & toward, 0.0, np.clip(vd, -VLAT, VLAT))
            st["vd_o"] = np.where(st["lapse_left"] > 0, st["vd_o"], vd)
        st["d_e"] = np.clip(st["d_e"] + st["vd_e"] * DT, -lim, lim)
        st["d_o"] = np.clip(st["d_o"] + st["vd_o"] * DT, -lim, lim)

        # target speeds with aerodynamic and positional interaction
        gap = self.gap()
        dd = np.abs(st["d_e"] - st["d_o"])
        tid = st["tid"]
        kap_e = np.abs(T.at("kappa", tid, st["s_e"]))
        kap_o = np.abs(T.at("kappa", tid, st["s_o"]))
        vt_e = u_e * T.at("vref", tid, st["s_e"])
        vt_o = u_o * T.at("vref", tid, st["s_o"])
        straight_e = np.clip(1 - kap_e / 0.016, 0, 1)
        straight_o = np.clip(1 - kap_o / 0.016, 0, 1)
        # slipstream for the follower on straights, dirty air in corners
        fol = np.abs(gap)
        drafting = (fol > 3) & (fol < 28) & (dd < 1.8)
        mm = getattr(self, "model", None)
        boost = 0.10 * (1 - fol / 28) if mm is None else 0.10 * mm.get("draft", 1.0) * (1 - fol / 28)
        a_boost = ATTACK_BOOST if mm is None else ATTACK_BOOST * mm.get("attack", 1.0)
        lift = LIFT if mm is None else mm.get("lift", LIFT)
        vt_e = np.where(drafting & (gap < 0), vt_e * (1 + boost * straight_e - 0.008 * (1 - straight_e)), vt_e)
        vt_o = np.where(drafting & (gap > 0), vt_o * (1 + boost * straight_o - 0.008 * (1 - straight_o)), vt_o)
        # corner priority when overlapped: the inside car owns the corner
        over = (fol < 6.5) & (dd >= CAR_W)
        sgn = np.sign(T.at("kappa", tid, st["s_e"]))
        e_in = sgn * (st["d_e"] - st["d_o"]) > 0
        corner = 1 - straight_e
        # the outside car yields only if it is not clearly ahead at turn-in
        vt_e = np.where(over & ~e_in & (gap < 2.0), vt_e * (1 - PRIORITY * corner), vt_e)
        vt_o = np.where(over & e_in & (gap > -2.0), vt_o * (1 - PRIORITY * corner), vt_o)
        # running tighter than the racing line costs corner speed
        ds_e = T.at("dstar", tid, st["s_e"])
        ds_o = T.at("dstar", tid, st["s_o"])
        sk_o = np.sign(T.at("kappa", tid, st["s_o"]))
        vt_e = vt_e * (1 - TIGHT * corner * np.maximum(0, sgn * (st["d_e"] - ds_e)) / HALF_W)
        vt_o = vt_o * (1 - TIGHT * (1 - straight_o) * np.maximum(0, sk_o * (st["d_o"] - ds_o)) / HALF_W)
        # a committed attacker alongside or close behind brakes later into the corner
        feinting = (st["tactic"] == 4) & (st["close_k"] < FEINT_STEPS)
        att_e = np.isin(st["tactic"], [1, 2, 3, 4]) & ~feinting & (gap < 3.0) & (gap > -PULL_OUT - 1.0) & (dd >= 0.8 * CAR_W)
        m_e = np.where(att_e, 1 + a_boost, 1.0)
        att_o = (gap > 3.0) & (gap < PULL_OUT + 1.0) & (dd >= 0.8 * CAR_W)
        m_o = np.where(att_o, 1 + a_boost, 1.0)
        vt_e = np.minimum(vt_e, 1.1 * V_TOP)
        vt_o = np.minimum(vt_o, 1.1 * V_TOP)
        # brake hard only inside a braking zone, otherwise coast on aerodynamic drag
        bz_e = T.at("vref", tid, st["s_e"] + 1.2 * st["v_e"]) < 0.995 * T.at("vref", tid, st["s_e"])
        bz_o = T.at("vref", tid, st["s_o"] + 1.2 * st["v_o"]) < 0.995 * T.at("vref", tid, st["s_o"])
        dec_e = np.where(bz_e, A_BRK, A_DRAG)
        dec_o = np.where(bz_o, A_BRK, A_DRAG)
        st["v_e"] = st["v_e"] + np.clip(vt_e - st["v_e"], -dec_e * DT, A_ACC * DT)
        st["v_o"] = st["v_o"] + np.clip(vt_o - st["v_o"], -dec_o * DT, A_ACC * DT)
        # progress rate: an unobstructed committed attack takes a shorter, later-braking line
        st["s_e"] = st["s_e"] + st["v_e"] * DT * m_e
        st["s_o"] = st["s_o"] + st["v_o"] * DT * m_o

        # blocking: no driving through the car ahead in the same lane
        gap = self.gap()
        dd = np.abs(st["d_e"] - st["d_o"])
        same = dd < CAR_W
        # blocking applies only to a car that was fully behind; overlapping cars make contact instead
        eb = same & (gap < 0) & (gap > -CAR_L - 0.4) & (g_start <= -CAR_L)
        st["s_e"] = np.where(eb, st["s_o"] - CAR_L - 0.4, st["s_e"])
        # a blocked attack forces a lift: the attacker loses momentum
        lift_e = eb & (np.abs(st["vd_e"]) > 0.5)
        st["v_e"] = np.where(eb, np.minimum(st["v_e"], st["v_o"] * np.where(lift_e, lift, 1.0)), st["v_e"])
        ob = same & (gap > 0) & (gap < CAR_L + 0.4) & (g_start >= CAR_L)
        st["s_o"] = np.where(ob, st["s_e"] - CAR_L - 0.4, st["s_o"])
        lift_o = ob & (np.abs(st["vd_o"]) > 0.5)
        st["v_o"] = np.where(ob, np.minimum(st["v_o"], st["v_e"] * np.where(lift_o, lift, 1.0)), st["v_o"])
        # lateral contact while overlapped side by side
        gap = self.gap()
        dd = st["d_e"] - st["d_o"]
        contact = (np.abs(gap) < CAR_L) & (np.abs(dd) < CAR_W)
        if contact.any():
            close_e = -np.sign(dd) * (st["d_e"] - prev_de)
            close_o = np.sign(dd) * (st["d_o"] - prev_do)
            e_fault = close_e >= close_o
            st["coll"] += contact
            st["fault_e"] += contact & e_fault
            mid = 0.5 * (st["d_e"] + st["d_o"])
            sg = np.where(dd >= 0, 1.0, -1.0)
            st["d_e"] = np.where(contact, mid + sg * CAR_W / 2, st["d_e"])
            st["d_o"] = np.where(contact, mid - sg * CAR_W / 2, st["d_o"])
            st["v_e"] = np.where(contact & e_fault, st["v_e"] * 0.9, np.where(contact, st["v_e"] * 0.97, st["v_e"]))
            st["v_o"] = np.where(contact & ~e_fault, st["v_o"] * 0.9, np.where(contact, st["v_o"] * 0.97, st["v_o"]))
        new_gap = self.gap()
        st["overtakes"] += (g_start < 0) & (new_gap > 0)
        st["passed"] += (g_start > 0) & (new_gap < 0)
        # adaptive opponent counts close attacks
        st["att_count"] += (new_gap > -8) & (new_gap < 0) & (np.abs(st["d_e"] - st["d_o"]) > CAR_W) & (st["opt_step"] == 0)
        st["hist_de"] = np.concatenate([st["hist_de"][:, 1:], st["d_e"][:, None]], axis=1)
        st["opt_step"] = st["opt_step"] + 1
        st["t"] = t + 1

    # -------------------------------------------------------------- sensing
    def kin_features(self):
        """Ego-centric kinematic features used by every learned agent (no map data)."""
        st = self.st
        gap = self.gap()
        return np.stack([
            st["v_e"] / 40.0,
            np.clip(gap, -40, 40) / 20.0,
            (st["d_e"] - st["d_o"]) / HALF_W,
            st["d_e"] / HALF_W,
            st["d_o"] / HALF_W,
            (st["v_e"] - st["v_o"]) / 4.0,
            st["vd_e"] / VLAT,
            st["vd_o"] / VLAT,
            np.full(self.E, 1.0 - st["t"] / RACE_STEPS),
        ], axis=1)

    def opp_obs(self):
        """Per-step interaction features for opponent modelling."""
        st = self.st
        return np.stack([
            st["d_e"] / HALF_W,
            st["d_o"] / HALF_W,
            np.clip(self.gap(), -40, 40) / 20.0,
            (st["v_o"] - st["v_e"]) / 4.0,
            st["vd_e"] / VLAT,
        ], axis=1)

    def opp_action(self):
        st = self.st
        return np.stack([st["vd_o"] / VLAT, (st["v_o"] - st["v_e"]) / 4.0], axis=1)

    def raster(self):
        """Ego-centric bird's-eye raster [E,2,32,16]: drivable area and opponent."""
        st, T = self.st, self.T
        E = self.E
        out = np.zeros((E, 2, 32, 16), dtype=np.float32)
        fx = -4.0 + 2.0 * np.arange(32)
        fy = -15.0 + 2.0 * np.arange(16)
        GX, GY = np.meshgrid(fx, fy, indexing="ij")
        tid = st["tid"]
        i_e = T.idx(tid, st["s_e"])
        h = T.heading[tid, i_e]
        nx, ny = -np.sin(h), np.cos(h)
        px = T.x[tid, i_e] + st["d_e"] * nx
        py = T.y[tid, i_e] + st["d_e"] * ny
        for k in np.unique(tid):
            m = tid == k
            img, x0, y0, res = T.tracks[k].occupancy()
            c, s_ = np.cos(h[m])[:, None, None], np.sin(h[m])[:, None, None]
            wx = px[m][:, None, None] + GX * c - GY * s_
            wy = py[m][:, None, None] + GX * s_ + GY * c
            ix = np.clip(((wx - x0) / res).astype(int), 0, img.shape[0] - 1)
            iy = np.clip(((wy - y0) / res).astype(int), 0, img.shape[1] - 1)
            out[m, 0] = img[ix, iy]
        # opponent in the ego frame (Frenet approximation)
        rel_f = self.gap() * -1.0
        rel_l = st["d_o"] - st["d_e"]
        r = np.round((rel_f + 4.0) / 2.0).astype(int)
        c = np.round((rel_l + 15.0) / 2.0).astype(int)
        for dr in (-1, 0, 1):
            for dc in (0,):
                rr, cc = r + dr, c + dc
                ok = (rr >= 0) & (rr < 32) & (cc >= 0) & (cc < 16)
                out[np.where(ok)[0], 1, rr[ok], cc[ok]] = 1.0
        return out

"""Spotter radio messages generated from the opponent parameters.

A message names between one and three behavioural attributes. Training and
test messages use disjoint sentence frames, so the test frames are unseen
phrasings of the same content.
"""
import re
import numpy as np

ATTR = {
    "block_hi": ["defends hard", "covers every move", "blocks aggressively"],
    "block_lo": ["rarely defends", "does not fight back", "lets you through"],
    "block_mid": ["defends now and then", "sometimes covers"],
    "react_fast": ["reacts instantly", "is quick to respond"],
    "react_slow": ["reacts late", "is slow to respond"],
    "side_in": ["protects the inside", "covers the inside line"],
    "side_out": ["leaves the inside open", "runs wide into corners"],
    "errors": ["keeps making mistakes", "is locking up"],
    "pace_hi": ["is quick on the straights", "has good pace"],
}
FRAMES_TRAIN = [
    "car ahead {a}", "spotter here the car in front {a}", "copy that the leader {a}",
    "box info the rival {a}", "watch out he {a}", "radio check the opponent {a}",
]
FRAMES_TEST = [
    "heads up your rival {a}", "be aware the guy ahead {a}", "engineer says the leader {a}",
]


def attributes(theta):
    u, b, tau, l, eps = theta
    a = []
    a.append("block_hi" if b > 0.65 else ("block_lo" if b < 0.3 else "block_mid"))
    if b > 0.3:
        a.append("react_fast" if tau < 4 else ("react_slow" if tau > 8 else None))
    if l > 0.5:
        a.append("side_in")
    elif l < -0.5:
        a.append("side_out")
    if eps > 0.1:
        a.append("errors")
    if u > 0.9855:
        a.append("pace_hi")
    return [x for x in a if x is not None]


def make_message(theta, rng, split="train"):
    attrs = attributes(theta)
    k = rng.integers(1, min(3, len(attrs)) + 1)
    pick = rng.choice(len(attrs), size=k, replace=False)
    phrases = [ATTR[attrs[i]][rng.integers(len(ATTR[attrs[i]]))] for i in sorted(pick)]
    body = " and ".join(phrases)
    frames = FRAMES_TRAIN if split == "train" else FRAMES_TEST
    return frames[rng.integers(len(frames))].format(a=body)


def _tokens(s):
    return re.findall(r"[a-z]+", s.lower())


class Vocab:
    def __init__(self):
        words = set()
        for ps in ATTR.values():
            for p in ps:
                words.update(_tokens(p))
        for f in FRAMES_TRAIN:
            words.update(_tokens(f.replace("{a}", "")))
        self.words = sorted(words)
        self.index = {w: i for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def bow(self, msgs):
        out = np.zeros((len(msgs), len(self)), dtype=np.float32)
        for i, m in enumerate(msgs):
            if m is None:
                continue
            ids = [self.index[w] for w in _tokens(m) if w in self.index]
            for j in ids:
                out[i, j] += 1.0
            if ids:
                out[i] /= len(ids)
        return out

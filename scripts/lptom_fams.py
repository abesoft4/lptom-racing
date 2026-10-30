import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lptom.opponents import SEEN as SEEN_NAMES, UNSEEN as UNSEEN_NAMES  # noqa: F401
from lptom.oof import OOF_KINDS as _K  # noqa: E402
OOF_NAMES = [k.replace('-', ' ').replace('rl defender a', 'RL defender A').replace('rl defender b', 'RL defender B') for k in _K]

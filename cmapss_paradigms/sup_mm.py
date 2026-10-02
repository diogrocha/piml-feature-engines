"""Multi-mode LG with supervised versus unsupervised directions.
    python cmapss_paradigms/sup_mm.py FD003 1,2,3,4,5
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import json, os, sys, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import pipeline as P
from config import GBSA_AUG, ROLLING_WINDOW_CMAPSS as W
from engines import make_y, ci_score
from crossval import build_configs
from lg_multimode import supervised_dirs
from lg_comb_gbsa_pkg import eta_dir
SUB, FOLDS = sys.argv[1], [int(f) for f in sys.argv[2].split(',')]
OUT = os.environ.get('LG_OUT', 'results/sup_mm.json'); os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
d = P.load(SUB); Xb, Xs, ttf, cyc, u, ev = d['X_base'], d['X_stats'], d['ttf'], d['cyc'], d['units'], d['event']
sp = list(GroupKFold(5).split(Xb, groups=u))
for f in FOLDS:
    if f'fold{f}' in res.get(SUB, {}): continue
    tri, tei = sp[f-1]; Xt, Xe = build_configs(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], cyc[tri], cyc[tei], W)['Base']
    D = supervised_dirs(Xt, ttf[tri], 2)
    mt = np.column_stack([eta_dir(Xt, Xt, v, ttf[tri]) for v in D]); me = np.column_stack([eta_dir(Xt, Xe, v, ttf[tri]) for v in D])
    idx = np.random.RandomState(42).choice(len(tri), min(6000, len(tri)), replace=False)
    m = GradientBoostingSurvivalAnalysis(**GBSA_AUG).fit(np.hstack([Xt, mt])[idx], make_y(ev[tri][idx], ttf[tri][idx]))
    res.setdefault(SUB, {})[f'fold{f}'] = float(ci_score(m, np.hstack([Xe, me]), make_y(ev[tei], ttf[tei]))); json.dump(res, open(OUT, 'w'))
    print(SUB, f, res[SUB][f'fold{f}'], flush=True)

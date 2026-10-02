"""End-to-end inference cost per sample, feature map plus GBSA risk prediction.

    python cost/cost_e2e.py
Batch-amortised timings on FD001, fold 1; run on an otherwise idle machine.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import pipeline as P
from config import GBSA_AUG, LG_PARAMS, ROLLING_WINDOW_CMAPSS as W
from engines import LGPhysicsEngine, HIBuilder, WeibullEngine, ExpDegEngine, EntropyEngine, compute_stats, make_y
from crossval import build_configs
from lg_multimode import pca_dirs

d = P.load('FD001'); Xb, Xs, ttf, cyc, u, ev = d['X_base'], d['X_stats'], d['ttf'], d['cyc'], d['units'], d['event']
tri, tei = list(GroupKFold(5).split(Xb, groups=u))[0]
cfg = build_configs(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], cyc[tri], cyc[tei], W)
Xt, Xe = cfg['Base']
lg = LGPhysicsEngine(**LG_PARAMS); lg.calibrate(Xt, ttf[tri])
hi = HIBuilder(); hi.fit(Xt, ttf[tri]); wb = WeibullEngine(); wb.calibrate(ttf[tri])
ed = ExpDegEngine(window=W); ed.calibrate(Xt, ttf[tri])
en = EntropyEngine(window=W, second_feature='rolling_std'); en.calibrate(Xt, ttf[tri])
D = np.column_stack(pca_dirs(Xt, 2))
df = pd.DataFrame(np.random.RandomState(0).randn(3000, 15), columns=[f's{i}' for i in range(15)])
n = 3000; Xq = Xe[:n]


def best(fn, reps=5):
    b = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t0)
    return b


feat = {'Baseline': (lambda: None, n), 'LG': (lambda: lg.get_eta(Xq), n),
        'LG multi-mode': (lambda: np.clip(0.3 * (Xq @ D) + 0.1, 1e-3, 0.85), n),
        'HI': (lambda: hi.transform(Xq), n), 'Weibull': (lambda: wb.transform(cyc[tei][:n]), n),
        'Stats': (lambda: compute_stats(df, list(df.columns), w=W), n),
        'ExpDeg': (lambda: ed.transform(Xq[:1500]), 1500), 'Entropy': (lambda: en.transform(Xq[:300]), 300)}
arm = {'Baseline': 'Base', 'LG': None, 'LG multi-mode': None, 'HI': '+HI', 'Weibull': '+Weibull',
       'Stats': '+Stats', 'ExpDeg': '+ExpDeg', 'Entropy': '+Entropy'}
lgx = lambda X: np.hstack([X, lg.get_eta(X)[:, None]])
idx = np.random.RandomState(42).choice(len(tri), 1500, replace=False)
for nm, (fn, k) in feat.items():
    fu = 0.0 if nm == 'Baseline' else best(fn, 3 if k < n else 5) / k * 1e6
    xt, xe = (lgx(Xt), lgx(Xe)) if arm[nm] is None else cfg[arm[nm]]
    m = GradientBoostingSurvivalAnalysis(**GBSA_AUG).fit(xt[idx], make_y(ev[tri][idx], ttf[tri][idx]))
    mu = best(lambda: m.predict(xe[:n])) / n * 1e6
    print(f'{nm:<14} feature {fu:9.3f} us | GBSA {mu:6.3f} us | total {fu + mu:9.3f} us | feature share {100 * fu / (fu + mu):5.1f}%')

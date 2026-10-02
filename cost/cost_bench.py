"""Per-sample cost of each feature map.

    python cost/cost_bench.py
Run on an otherwise idle machine; timings near 0.01 us are at timer resolution.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import pipeline as P
from config import LG_PARAMS
from engines import LGPhysicsEngine, WeibullEngine, EntropyEngine, ExpDegEngine

d = P.load('FD001')
X = StandardScaler().fit_transform(d['X_base'])
ttf, cyc = d['ttf'], d['cyc']
lg = LGPhysicsEngine(**LG_PARAMS); lg.calibrate(X, ttf)
wb = WeibullEngine(); wb.calibrate(ttf)
ed = ExpDegEngine(window=P.WIN); ed.calibrate(X, ttf)
en = EntropyEngine(window=P.WIN, second_feature='rolling_std'); en.calibrate(X, ttf)


def bench(fn, n, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(n); best = min(best, time.perf_counter() - t0)
    return best / n * 1e6


cost = {'LG order parameter': bench(lambda n: lg.get_eta(X[:n]), 4000),
        'LG four descriptors': bench(lambda n: lg.transform(X[:n]), 4000),
        'Weibull': bench(lambda n: wb.transform(cyc[:n]), 4000),
        'ExpDeg': bench(lambda n: ed.transform(X[:n]), 2000),
        'Entropy': bench(lambda n: en.transform(X[:n]), 300, 1)}
for k, v in cost.items():
    print(f'{k:<22}{v:>10.3f} us/sample  ({v / cost["LG order parameter"]:,.0f}x)')

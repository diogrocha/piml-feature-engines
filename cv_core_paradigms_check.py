import os, json, time, numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from config import GBSA_AUG
from engines import make_y
OUT = '/mnt/user-data/outputs/results_cv_core.json'
z = np.load('/tmp/cv_core_FD003.npz')
d = json.load(open(OUT)) if os.path.exists(OUT) else {}
jobs = [(tag, f) for f in range(3) for tag in ('HI', 'LG', 'ED')]
t0 = time.time()
for tag, f in jobs:
    key = f'{tag}{f}'
    if key in d:
        continue
    if time.time() - t0 > 88:
        break
    Xtr, Xte = (z[f'{tag}{f}_tr'], z[f'{tag}{f}_te'])
    ytr = make_y(z[f'ev{f}_tr'], z[f'ttf{f}_tr'])
    yte = make_y(z[f'ev{f}_te'], z[f'ttf{f}_te'])
    rng = np.random.RandomState(42)
    n = len(Xtr)
    si = rng.choice(n, min(4000, n), replace=False)
    m = GradientBoostingSurvivalAnalysis(**GBSA_AUG).fit(Xtr[si], ytr[si])
    C = concordance_index_censored(yte['event'], yte['time'], m.predict(Xte))[0]
    d[key] = float(C)
    json.dump(d, open(OUT, 'w'), indent=2)
    print(f'{key}: C={C:.4f} ({time.time() - t0:.0f}s)')
"""Survival comparison for step 3 (criterion in fleet_recursive.py). Same base as paradigms.py (rolled window sensors + t)."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import json, os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
D = pd.read_csv('f234.csv').replace([np.inf, -np.inf], np.nan).dropna(subset=['d_rls', 'sigma_L2', 'T_full', 'T_L', 'sl_d', 'sl_s']).sort_values(['run', 't']).reset_index(drop=True)
SENS = ['rms', 'kurt', 'crest'] + [f'E{i}' for i in range(6)] + ['band_var', 'band_ac1']
XB = np.column_stack([D.groupby('run')[SENS].transform(lambda s: s.rolling(5, 1).mean()).values, D['t'].values])
A = {'+sigma_L2': ['sigma_L2'], '+T_full': ['T_full'], '+twin_full': ['d_rls', 'sl_d'], '+T_L': ['T_L'], '+twin_L': ['sigma_L2', 'sl_s'], 'Base': [], '+b_ls': ['b_ls'], '+b_rls': ['b_rls'], '+d_rls': ['d_rls'], '+full-twin': ['b_rls', 'zeta_rls', 'tau'],
     '+sigma_L': ['sigma_L'], '+Landau-twin': ['sigma_L', 'dw_L'], '+b_rls+sigma_L': ['b_rls', 'sigma_L'], 'oracle': ['b_frac']}
OUT = 'survival4.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
ARMS = sys.argv[1].split(','); ttf = D.ttf.values
for f, (tri, tei) in enumerate(GroupKFold(5).split(D, groups=D.run)):
    sc = StandardScaler().fit(XB[tri]); ytr = Surv.from_arrays(np.ones(len(tri), bool), ttf[tri])
    for a in ARMS:
        if len(res.get(a, [])) > f: continue
        Xt = np.hstack([sc.transform(XB[tri]), D[A[a]].values[tri]]); Xe = np.hstack([sc.transform(XB[tei]), D[A[a]].values[tei]])
        m = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.01, max_depth=3, random_state=42).fit(Xt, ytr)
        res.setdefault(a, []).append(float(concordance_index_censored(np.ones(len(tei), bool), ttf[tei], m.predict(Xe))[0])); json.dump(res, open(OUT, 'w'))
        print(a, f, round(res[a][-1], 4), flush=True)

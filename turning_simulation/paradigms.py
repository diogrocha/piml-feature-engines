"""All paradigms of the C-MAPSS study on the turning simulator (pre-registered 30 Sep 2026, before computation).
Base = window 'sensors' [rms, kurt, crest, E0..E5, band_var, band_ac1] rolled over 5 windows (mean) + t, standardised
on the training folds. Paradigms exactly as in the feature engines (Stats, HI, Weibull, Entropy, ExpDeg, LG, LG-4,
LG multi-mode, multi-mode + per-run rates, PCA twin, HI with per-run rate), plus: model-based physics estimator
(+b_ls: least-squares depth of cut from the regenerative equation; +d_hat: b_ls / b_lim(tau, zeta_ls)), its
non-physical twin (+AR-raw: AR(2) damping of the band signal), and the oracle (true distance b_frac).
GBSA 150 trees, lr 0.01, depth 3, GroupKFold 5 by run, all rows. Success for the physics estimator: +b_ls or +d_hat
beats Base and every other paradigm with paired 95% CI (t4) above zero."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import glob, json, os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from config import LG_PARAMS
from engines import LGPhysicsEngine, HIBuilder, WeibullEngine, ExpDegEngine, EntropyEngine, compute_stats
from lg_multimode import pca_dirs
from lg_comb_gbsa_pkg import eta_dir, unit_slope
D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('./f2_*.csv'))], ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna().sort_values(['run', 't']).reset_index(drop=True)
SENS = ['rms', 'kurt', 'crest'] + [f'E{i}' for i in range(6)] + ['band_var', 'band_ac1']
R = D.groupby('run')[SENS].transform(lambda s: s.rolling(5, 1).mean())
ST = pd.concat([compute_stats(g[SENS].reset_index(drop=True), SENS, w=5) for _, g in D.groupby('run', sort=False)], ignore_index=True)
XB = np.column_stack([R.values, D['t'].values]); XS = ST.values
u, t, ttf = D.run.values, D.t.values, D.ttf.values
OUT = './paradigms_sim.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
ARMS = sys.argv[1].split(',')
for f, (tri, tei) in enumerate(GroupKFold(5).split(D, groups=u)):
    todo = [a for a in ARMS if len(res.get(a, [])) <= f]
    if not todo: continue
    sc = StandardScaler().fit(XB[tri]); Xt, Xe = sc.transform(XB[tri]), sc.transform(XB[tei])
    ss = StandardScaler().fit(XS[tri]); St, Se = ss.transform(XS[tri]), ss.transform(XS[tei])
    sl = lambda A, uu, cc: np.column_stack([unit_slope(A[:, k], uu, cc) for k in range(A.shape[1])])
    lg = LGPhysicsEngine(**LG_PARAMS); lg.calibrate(Xt, ttf[tri]); et, ee = lg.get_eta(Xt), lg.get_eta(Xe)
    hi = HIBuilder(); hi.fit(Xt, ttf[tri]); Ht, He = np.asarray(hi.transform(Xt)), np.asarray(hi.transform(Xe))
    Dd = pca_dirs(Xt, 2); mt = np.column_stack([eta_dir(Xt, Xt, v, ttf[tri]) for v in Dd]); me = np.column_stack([eta_dir(Xt, Xe, v, ttf[tri]) for v in Dd])
    Pt, Pe = np.column_stack([Xt @ v for v in Dd]), np.column_stack([Xe @ v for v in Dd])
    col = lambda c: (D[c].values[tri][:, None], D[c].values[tei][:, None])
    def arm(a):
        if a == 'Base': return Xt, Xe
        if a == '+Stats': return np.hstack([Xt, St]), np.hstack([Xe, Se])
        if a == '+HI': return np.hstack([Xt, Ht]), np.hstack([Xe, He])
        if a == '+HI-fix': return np.hstack([Xt, Ht[:, :1], sl(Ht[:, :1], u[tri], t[tri])]), np.hstack([Xe, He[:, :1], sl(He[:, :1], u[tei], t[tei])])
        if a == '+Weibull': wb = WeibullEngine(); wb.calibrate(ttf[tri]); return np.hstack([Xt, wb.transform(t[tri])]), np.hstack([Xe, wb.transform(t[tei])])
        if a == '+Entropy': en = EntropyEngine(window=5, second_feature='rolling_std'); en.calibrate(Xt, ttf[tri]); return np.hstack([Xt, en.transform(Xt)]), np.hstack([Xe, en.transform(Xe)])
        if a == '+ExpDeg': ed = ExpDegEngine(window=5); ed.calibrate(Xt, ttf[tri]); return np.hstack([Xt, ed.transform(Xt)]), np.hstack([Xe, ed.transform(Xe)])
        if a == '+LG': return np.column_stack([Xt, et]), np.column_stack([Xe, ee])
        if a == '+LG-4': return np.hstack([Xt, lg.transform(Xt)]), np.hstack([Xe, lg.transform(Xe)])
        if a == '+LG-mm': return np.hstack([Xt, mt]), np.hstack([Xe, me])
        if a == 'LG-comb-mm': return np.hstack([Xt, mt, sl(mt, u[tri], t[tri])]), np.hstack([Xe, me, sl(me, u[tei], t[tei])])
        if a == 'PC-twin': return np.hstack([Xt, Pt, sl(Pt, u[tri], t[tri])]), np.hstack([Xe, Pe, sl(Pe, u[tei], t[tei])])
        if a == '+AR-raw': A, B = col('zeta_ar'); return np.hstack([Xt, A]), np.hstack([Xe, B])
        if a in ('+b_ls', '+d_hat', 'oracle'):
            c = {'+b_ls': 'b_ls', '+d_hat': 'd_hat', 'oracle': 'b_frac'}[a]; A, B = col(c); return np.hstack([Xt, A]), np.hstack([Xe, B])
        if a == '+twin-est': return np.hstack([Xt, D[['b_ls', 'zeta_ls', 'tau']].values[tri]]), np.hstack([Xe, D[['b_ls', 'zeta_ls', 'tau']].values[tei]])
    ytr = Surv.from_arrays(np.ones(len(tri), bool), ttf[tri]); ev = np.ones(len(tei), bool)
    for a in todo:
        A, B = arm(a); m = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.01, max_depth=3, random_state=42).fit(A, ytr)
        res.setdefault(a, []).append(float(concordance_index_censored(ev, ttf[tei], m.predict(B))[0])); json.dump(res, open(OUT, 'w'))
        print(a, f, round(res[a][-1], 4), flush=True)

"""Combined LG block under the GBSA learner.

Arms (all with the base features; same folds and 6000-row subsample as the main cross-validation):
  LG-comb : eta, eta on the first two principal directions, per-unit slope of eta
  LG+HI   : eta plus the health-index block (non-physical control)
The scalar LG and HI arms are those of cv_all.json.
    python cmapss_paradigms/lg_comb_gbsa.py FD001 1,2,3,4,5 LG-comb,LG+HI,LG-comb-mm,mm+HI,PC-twin,HI-fix
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import json, os, sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import pipeline as P
from config import GBSA_AUG, LG_PARAMS, ROLLING_WINDOW_CMAPSS as W
from engines import LGPhysicsEngine, make_y, ci_score
from crossval import build_configs
from lg_multimode import pca_dirs
from lg_comb_gbsa_pkg import eta_dir, unit_slope

if __name__ == '__main__':
    SUB, FOLDS, ARMS = sys.argv[1], [int(f) for f in sys.argv[2].split(',')], sys.argv[3].split(',')
    OUT = os.environ.get('LG_OUT', 'results/lg_comb_gbsa.json')
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d = P.load(SUB); Xb, Xs, ttf, cyc, u, ev = d['X_base'], d['X_stats'], d['ttf'], d['cyc'], d['units'], d['event']
    splits = list(GroupKFold(5).split(Xb, groups=u))
    for f in FOLDS:
        todo = [a for a in ARMS if a not in res.get(SUB, {}).get(f'fold{f}', {})]
        if not todo: continue
        tri, tei = splits[f - 1]
        cfg = build_configs(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], cyc[tri], cyc[tei], W)
        Xt, Xe = cfg['Base']; p = Xt.shape[1]; Ht, He = cfg['+HI'][0][:, p:], cfg['+HI'][1][:, p:]
        lg = LGPhysicsEngine(**LG_PARAMS); lg.calibrate(Xt, ttf[tri]); et, ee = lg.get_eta(Xt), lg.get_eta(Xe)
        D = pca_dirs(Xt, 2)
        mt = np.column_stack([eta_dir(Xt, Xt, v, ttf[tri]) for v in D]); me = np.column_stack([eta_dir(Xt, Xe, v, ttf[tri]) for v in D])
        slt, sle = unit_slope(et, u[tri], cyc[tri]), unit_slope(ee, u[tei], cyc[tei])
        smt = np.column_stack([unit_slope(mt[:, k], u[tri], cyc[tri]) for k in range(2)])
        sme = np.column_stack([unit_slope(me[:, k], u[tei], cyc[tei]) for k in range(2)])
        arms = {'LG-comb': (np.column_stack([Xt, et, mt, slt]), np.column_stack([Xe, ee, me, sle])),
                'LG+HI': (np.column_stack([Xt, et, Ht]), np.column_stack([Xe, ee, He])),
                'LG-comb-mm': (np.column_stack([Xt, mt, smt]), np.column_stack([Xe, me, sme])),
                'mm+HI': (np.column_stack([Xt, mt, Ht]), np.column_stack([Xe, me, He]))}
        Pt, Pe = np.column_stack([Xt @ v for v in D]), np.column_stack([Xe @ v for v in D])
        spt = np.column_stack([unit_slope(Pt[:, k], u[tri], cyc[tri]) for k in range(2)]); spe = np.column_stack([unit_slope(Pe[:, k], u[tei], cyc[tei]) for k in range(2)])
        arms['PC-twin'] = (np.column_stack([Xt, Pt, spt]), np.column_stack([Xe, Pe, spe]))
        arms['HI-fix'] = (np.column_stack([Xt, Ht[:, 0], unit_slope(Ht[:, 0], u[tri], cyc[tri])]),
                          np.column_stack([Xe, He[:, 0], unit_slope(He[:, 0], u[tei], cyc[tei])]))
        idx = np.random.RandomState(42).choice(len(tri), min(6000, len(tri)), replace=False)
        ytr = make_y(ev[tri][idx], ttf[tri][idx]); yte = make_y(ev[tei], ttf[tei])
        for a in todo:
            t0 = time.time(); m = GradientBoostingSurvivalAnalysis(**GBSA_AUG).fit(arms[a][0][idx], ytr)
            res.setdefault(SUB, {}).setdefault(f'fold{f}', {})[a] = float(ci_score(m, arms[a][1], yte))
            json.dump(res, open(OUT, 'w'), indent=1)
            print(f'{SUB} f{f} {a}: {res[SUB][f"fold{f}"][a]:.4f} ({time.time() - t0:.0f}s)', flush=True)

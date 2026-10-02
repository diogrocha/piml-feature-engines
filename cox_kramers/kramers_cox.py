"""Physics in the likelihood: Cox models with a Kramers-type term.

Hazard h(t|x) = h0(t) exp(beta'x + gamma * f(eta)), compared across the shape f:
  Cox-base   : base features only
  Cox+s      : base + the projection s (linear in eta)
  Cox+Kramers: base + Delta F(eta), the barrier of the LG potential (Arrhenius exponent)
  Cox+spline : base + a cubic spline in s (flexible, same information)
Same folds and subsample as the tree arms; ridge-stabilised Cox (alpha = 0.1).
    python cox_kramers/kramers_cox.py FD001 1,2,3,4,5
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import json, os, sys, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
import pipeline as P
from config import LG_PARAMS, ROLLING_WINDOW_CMAPSS as W
from engines import LGPhysicsEngine, make_y, ci_score
from crossval import build_configs


def ibs(m, xe, ytr, ttr, tte, ete):
    keep = tte < ttr.max()
    lo = max(np.percentile(ttr, 10), tte[keep].min() + 1e-6); hi = min(np.percentile(ttr, 90), tte[keep].max() - 1e-6)
    grid = np.linspace(lo, hi, 10)
    Pm = np.asarray([[f(x) for x in grid] for f in m.predict_survival_function(xe[keep])])
    return float(integrated_brier_score(ytr, make_y(ete[keep], tte[keep]), Pm, grid))


if __name__ == '__main__':
    SUB, FOLDS = sys.argv[1], [int(f) for f in sys.argv[2].split(',')]
    OUT = os.environ.get('LG_OUT', 'results/kramers_cox.json')
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d = P.load(SUB); Xb, Xs, ttf, cyc, u, ev = d['X_base'], d['X_stats'], d['ttf'], d['cyc'], d['units'], d['event']
    splits = list(GroupKFold(5).split(Xb, groups=u))
    for f in FOLDS:
        tri, tei = splits[f - 1]
        Xt, Xe = build_configs(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], cyc[tri], cyc[tei], W)['Base']
        lg = LGPhysicsEngine(**LG_PARAMS); lg.calibrate(Xt, ttf[tri])
        st, se = (Xt @ lg.weights)[:, None], (Xe @ lg.weights)[:, None]
        dFt, dFe = lg.transform(Xt)[:, [1]], lg.transform(Xe)[:, [1]]
        sp = SplineTransformer(n_knots=5, degree=3).fit(st)
        z = lambda A, B: (StandardScaler().fit(A).transform(A), StandardScaler().fit(A).transform(B))
        cols = {'Cox-base': (Xt, Xe), 'Cox+s': (np.hstack([Xt, st]), np.hstack([Xe, se])),
                'Cox+Kramers': (np.hstack([Xt, dFt]), np.hstack([Xe, dFe])),
                'Cox+spline': (np.hstack([Xt, sp.transform(st)]), np.hstack([Xe, sp.transform(se)]))}
        idx = np.random.RandomState(42).choice(len(tri), min(6000, len(tri)), replace=False)
        ytr = make_y(ev[tri][idx], ttf[tri][idx]); yte = make_y(ev[tei], ttf[tei])
        for a, (A, B) in cols.items():
            A, B = z(A[idx], B)
            m = CoxPHSurvivalAnalysis(alpha=0.1, ties='breslow', n_iter=200).fit(A, ytr)
            c = float(ci_score(m, B, yte)); b = ibs(m, B, ytr, ttf[tri][idx], ttf[tei], ev[tei])
            res.setdefault(SUB, {}).setdefault(f'fold{f}', {})[a] = {'C': c, 'IBS': b}
            print(f'{SUB} f{f} {a:<12} C={c:.4f} IBS={b:.4f}', flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)

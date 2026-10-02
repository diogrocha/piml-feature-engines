"""Robustness under alternative survival metrics.

Same protocol as the main cross-validation, under the same
protocol: grouped five-fold CV on FD001, budget-matched baseline, paired t
intervals. Metrics: C-index, integrated Brier score, mean time-dependent AUC and
the MAE of a restricted-mean-survival-time estimate of remaining useful life.

    python robustness/robustness.py --folds 1 2 3 4 5
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import argparse
import json
import os
import numpy as np
import pipeline as P
from config import GBSA_AUG
from engines import make_y, ci_score
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc

ARMS = ['Base-AUG', '+LG-eta', '+ExpDeg', '+HI']


def metrics(m, xe, ytr, ttr, tte, ete):
    c = float(ci_score(m, xe, make_y(ete, tte)))
    keep = tte < ttr.max()
    lo = max(np.percentile(ttr, 10), tte[keep].min() + 1e-6)
    hi = min(np.percentile(ttr, 90), tte[keep].max() - 1e-6, ttr.max() - 1e-6)
    grid = np.linspace(lo, hi, 10)
    yk = make_y(ete[keep], tte[keep])
    sf = m.predict_survival_function(xe[keep])
    Pm = np.asarray([[f(x) for x in grid] for f in sf])
    ibs = float(integrated_brier_score(ytr, yk, Pm, grid))
    _, auc = cumulative_dynamic_auc(ytr, yk, m.predict(xe[keep]), grid)
    # RMST up to tau = largest training time: expected remaining life under the predicted survival
    S = m.predict_survival_function(xe, return_array=True)
    t = m.unique_times_
    tau = ttr.max()
    tt = np.concatenate([[0.0], t[t <= tau]])
    SS = np.concatenate([np.ones((len(xe), 1)), S[:, t <= tau]], axis=1)
    rmst = np.sum(SS[:, :-1] * np.diff(tt), axis=1)
    mae = float(np.mean(np.abs(rmst - np.minimum(tte, tau))))
    return {'C': c, 'IBS': ibs, 'AUC': float(auc), 'RUL_MAE': mae}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--out', default='results/robustness_FD001.json')
    ap.add_argument('--nfast', type=int, default=6000)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    d = P.load('FD001')
    from lg_field import physical_groups
    groups, gnames = physical_groups(d['base_cols'])
    for fold in a.folds:
        todo = [x for x in a.arms if x not in res.get(f'fold{fold}', {})]
        if not todo:
            continue
        cfg, tri, tei = P.fold_design(d, fold, 5, todo, groups, gnames)
        idx = np.random.RandomState(42).choice(len(tri), min(a.nfast, len(tri)), replace=False)
        ttr, etr = d['ttf'][tri][idx], d['event'][tri][idx]
        ytr = make_y(etr, ttr)
        for arm in todo:
            m = GradientBoostingSurvivalAnalysis(**dict(GBSA_AUG)).fit(cfg[arm][0][idx], ytr)
            res.setdefault(f'fold{fold}', {})[arm] = metrics(m, cfg[arm][1], ytr, ttr, d['ttf'][tei], d['event'][tei])
            json.dump(res, open(a.out, 'w'), indent=1)
            r = res[f'fold{fold}'][arm]
            print(f'fold{fold} {arm:<9} C={r["C"]:.4f} IBS={r["IBS"]:.4f} AUC={r["AUC"]:.4f} MAE={r["RUL_MAE"]:.1f}', flush=True)

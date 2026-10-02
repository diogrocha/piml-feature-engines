"""Azure and NASA battery with a budget-matched baseline.

Expected from the age-only form of the Weibull block: on Azure, +Weibull equals Base-AUG (operating hours are
base columns); on the batteries, +Weibull equals +Cycle (base plus the raw cycle
count). This script checks both and quantifies the tree-budget effect.

    python azure_run.py prep                      # once, builds /tmp/azure_cache
    python azure_batteries/other_datasets_matched.py azure
    python azure_batteries/other_datasets_matched.py battery --data /tmp/NASA_bat/dataset
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import argparse
import json
import os
import numpy as np
from scipy.stats import t as tdist
import pipeline as P  # noqa: F401  (sets sys.path)
from config import GBSA_AUG, GBSA_BASE
from engines import make_y, ci_score
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

TREES = int(os.environ.get('LG_TREES', 0)) or None   # test hook only


def fit_c(arm, xt, ytr, xe, yte):
    kw = dict(GBSA_BASE) if arm == 'Base' else dict(GBSA_AUG)
    if TREES:
        kw['n_estimators'] = TREES
    return float(ci_score(GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr), xe, yte))


def azure(cache, out, comps=None, arms=None):
    import azure_run as AR
    arms = arms or ['Base', 'Base-AUG', '+Weibull', '+HI', '+LG']
    res = json.load(open(out)) if os.path.exists(out) else {}
    for comp in (comps or AR.COMPONENTS):
        f = os.path.join(cache, comp + '.npz')
        if not os.path.exists(f):
            print(f'  {comp}: no cache, run `python azure_run.py prep` first'); continue
        d = np.load(f)
        mats = AR._cfg_matrices(d)
        mats['Base-AUG'] = mats['Base']
        ytr, yte = make_y(d['ev_tr'], d['ttf_tr']), make_y(d['ev_te'], d['ttf_te'])
        for a in arms:
            if a in res.get(comp, {}):
                continue
            res.setdefault(comp, {})[a] = fit_c(a, mats[a][0], ytr, mats[a][1], yte)
            json.dump(res, open(out, 'w'), indent=1)
            print(f'  {comp} {a:<9} C={res[comp][a]:.4f}', flush=True)
    done = [c for c in res if all(a in res[c] for a in arms)]
    if done:
        print('\n  mean over components: ' + '  '.join(
            f'{a}={np.mean([res[c][a] for c in done]):.4f}' for a in arms))
        print('  +Weibull minus Base-AUG per component (expected 0): ' + ', '.join(
            f'{res[c]["+Weibull"] - res[c]["Base-AUG"]:+.1e}' for c in done))


def battery(data, out):
    import battery_data as BD
    import battery_run as BR
    from sklearn.model_selection import GroupKFold
    d = BD.load(data)
    Xb, Xs, ttf, ev, oh, u = d['X_base'], d['X_stats'], d['ttf'], d['event'], d['op_hours'], d['units']
    arms = ['Base', 'Base-AUG', '+Stats', '+HI', '+Weibull', '+Cycle', '+Entropy', '+ExpDeg', '+LG']
    sc = {a: [] for a in arms}
    for tri, tei in GroupKFold(n_splits=len(np.unique(u))).split(Xb, groups=u):
        if ev[tei].sum() == 0:
            continue
        c = BR.build(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], oh[tri], oh[tei])
        Xbt, Xbe = c['Base']
        c['Base-AUG'] = c['Base']
        c['+Cycle'] = (np.hstack([Xbt, oh[tri][:, None]]), np.hstack([Xbe, oh[tei][:, None]]))
        ytr, yte = make_y(ev[tri], ttf[tri]), make_y(ev[tei], ttf[tei])
        for a in arms:
            if a in c: sc[a].append(fit_c(a, c[a][0], ytr, c[a][1], yte))
        print(f'  held out {np.unique(u[tei])[0]}: ' + '  '.join(f'{a}={sc[a][-1]:.4f}' for a in arms), flush=True)
    n = len(sc['Base'])
    tc = tdist.ppf(0.975, n - 1) if n > 1 else np.nan
    def ci(a, b):
        dd = np.array(sc[a]) - np.array(sc[b]); se = dd.std(ddof=1) / np.sqrt(n)
        return f'{dd.mean():+.4f} [{dd.mean() - tc * se:+.4f}, {dd.mean() + tc * se:+.4f}]'
    print('\n  means: ' + '  '.join(f'{a}={np.mean(sc[a]):.4f}' for a in arms))
    arms = [a for a in arms if sc[a]]
    for a, b in [('Base-AUG', 'Base'), ('+Weibull', 'Base-AUG'), ('+Weibull', '+Cycle'), ('+LG', 'Base-AUG'), ('+Stats', 'Base-AUG'), ('+HI', 'Base-AUG'), ('+Entropy', 'Base-AUG'), ('+ExpDeg', 'Base-AUG'), ('+ExpDeg', '+LG')]:
        print(f'  {a} minus {b}: {ci(a, b)}')
    json.dump(sc, open(out, 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['azure', 'battery'])
    ap.add_argument('--cache', default='/tmp/azure_cache')
    ap.add_argument('--data', default='/tmp/NASA_bat/dataset')
    ap.add_argument('--out', default=None)
    ap.add_argument('--comps', nargs='+', default=None)
    ap.add_argument('--arms', nargs='+', default=None)
    a = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    out = a.out or f'results/{a.dataset}_matched.json'
    azure(a.cache, out, a.comps, a.arms) if a.dataset == 'azure' else battery(a.data, out)

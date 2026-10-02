"""Shared evaluation protocol for the C-MAPSS experiments.

All C-MAPSS experiments use this one protocol:
grouped 5-fold cross-validation by engine unit, identical folds for every arm,
paired per-fold contrasts, 95% intervals from Student's t with (folds - 1) d.f.
Results are checkpointed after every (subset, fold), so an interrupted run
resumes where it stopped.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import json
import os
import time
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import t as tdist
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# the analysis scripts live in their own folders and reuse the engines in src/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
DATA_DIR = os.environ.get('LG_DATA', os.path.join(_ROOT, 'data', 'CMAPSSData'))

import cmapss_data as DL
from config import GBSA_AUG, GBSA_BASE, LG_PARAMS, ROLLING_WINDOW_CMAPSS as WIN
from crossval import build_configs
from engines import LGPhysicsEngine, make_y, ci_score
from lg_field import LGFieldEngine, physical_groups
from lg_multimode import desc, pca_dirs

MAIN_ARMS = ['Base', 'Base-AUG', '+LG', '+LG-eta', '+LG-mm2']
PARADIGM_ARMS = ['+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg']
EXTENSION_ARMS = ['+LG-field-rob', '+LG-hyb']
META = {'FD001': (1, 1), 'FD002': (6, 1), 'FD003': (1, 2), 'FD004': (6, 2)}


def mm_features(Xt, Xall, ttf_tr, dirs, with_base=True):
    """One order parameter per direction. Fixes the repository defect that
    stacked the training matrix for the test set."""
    cols = [desc(d, Xt, Xall, ttf_tr)[:, 3] for d in dirs]
    block = np.column_stack(cols)
    return np.hstack([Xall, block]) if with_base else block


def load(sub):
    return DL.load(DATA_DIR, subset=sub,
                   condition_normalize=sub in ('FD002', 'FD004'), verbose=False)


def fold_design(d, fold, n_folds, arms, groups, gnames):
    """All arms for one fold, built on the same scaled matrices."""
    Xb, Xs, ttf = d['X_base'], d['X_stats'], d['ttf']
    cyc, units = d['cyc'], d['units']
    tri, tei = list(GroupKFold(n_splits=n_folds).split(Xb, groups=units))[fold - 1]
    cfg = build_configs(Xb[tri], Xb[tei], Xs[tri], Xs[tei],
                        ttf[tri], cyc[tri], cyc[tei], WIN)
    Xt, Xe = cfg['Base']
    cfg['Base-AUG'] = cfg['Base']
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xt, ttf[tri])
    et, ee = lg.get_eta(Xt).reshape(-1, 1), lg.get_eta(Xe).reshape(-1, 1)
    cfg['+LG-eta'] = (np.hstack([Xt, et]), np.hstack([Xe, ee]))
    if '+LG-field-rob' in arms:
        fr = LGFieldEngine(groups, gnames, calib='robust', **LG_PARAMS)
        fr.calibrate(Xt, ttf[tri])
        cfg['+LG-field-rob'] = (np.hstack([Xt, fr.transform_field(Xt)]),
                                np.hstack([Xe, fr.transform_field(Xe)]))
    if '+LG-mm2' in arms or '+LG-hyb' in arms:
        dirs = pca_dirs(Xt, 2)
        cfg['+LG-mm2'] = (mm_features(Xt, Xt, ttf[tri], dirs),
                          mm_features(Xt, Xe, ttf[tri], dirs))
        cfg['+LG-hyb'] = (np.hstack([Xt, et, mm_features(Xt, Xt, ttf[tri], dirs, False)]),
                          np.hstack([Xe, ee, mm_features(Xt, Xe, ttf[tri], dirs, False)]))
    return cfg, tri, tei


def fit_eval(arm, xt, xe, ev_tr, ttr, ev_te, tte, trees_override=None, n_times=10):
    params = dict(GBSA_BASE) if arm == 'Base' else dict(GBSA_AUG)
    if trees_override:
        params['n_estimators'] = trees_override
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import integrated_brier_score
    t0 = time.time()
    m = GradientBoostingSurvivalAnalysis(**params).fit(xt, make_y(ev_tr, ttr))
    c = float(ci_score(m, xe, make_y(ev_te, tte)))
    try:
        keep = tte < ttr.max()
        lo = max(np.percentile(ttr, 10), tte[keep].min() + 1e-06)
        hi = min(np.percentile(ttr, 90), tte[keep].max() - 1e-06, ttr.max() - 1e-06)
        grid = np.linspace(lo, hi, n_times)
        P = np.asarray([[f(x) for x in grid] for f in m.predict_survival_function(xe[keep])])
        ibs = float(integrated_brier_score(make_y(ev_tr, ttr),
                                           make_y(np.ones(keep.sum(), bool), tte[keep]),
                                           P, grid))
    except Exception:
        ibs = float('nan')
    return arm, c, ibs, time.time() - t0


def run(plan, results_path, n_folds=5, nfast=6000, n_jobs=2, seed=42,
        trees_override=None, only_folds=None):
    """plan: {subset: [arms]}. Skips anything already in the checkpoint."""
    res = json.load(open(results_path)) if os.path.exists(results_path) else {}
    for sub, arms in plan.items():
        d = load(sub)
        groups, gnames = physical_groups(d['base_cols'])
        ttf, ev = d['ttf'], d['event']
        for fold in (only_folds or range(1, n_folds + 1)):
            key = f'fold{fold}'
            done = res.get(sub, {}).get(key, {})
            todo = [a for a in arms if a not in done]
            if not todo:
                continue
            t0 = time.time()
            cfg, tri, tei = fold_design(d, fold, n_folds, todo, groups, gnames)
            rng = np.random.RandomState(seed)
            sub_idx = (rng.choice(len(tri), nfast, replace=False)
                       if nfast and len(tri) > nfast else np.arange(len(tri)))
            ttr, evtr = ttf[tri][sub_idx], ev[tri][sub_idx]
            out = Parallel(n_jobs=n_jobs)(
                delayed(fit_eval)(a, cfg[a][0][sub_idx], cfg[a][1], evtr, ttr,
                                  ev[tei], ttf[tei], trees_override)
                for a in todo)
            for a, c, i, _ in out:
                res.setdefault(sub, {}).setdefault(key, {})[a] = {'C': c, 'IBS': i}
            json.dump(res, open(results_path, 'w'), indent=1)
            print(f'{sub} {key}: {len(todo)} arms in {time.time() - t0:.0f}s  '
                  + '  '.join(f'{a}={c:.4f}' for a, c, _, _ in out), flush=True)
    return res


def series(res, sub, arm, metric='C'):
    folds = sorted(res.get(sub, {}), key=lambda k: int(k[4:]))
    v = [res[sub][f][arm][metric] for f in folds if arm in res[sub][f]]
    return np.asarray(v, dtype=float)


def paired(res, sub, a, b, metric='C'):
    x, y = series(res, sub, a, metric), series(res, sub, b, metric)
    n = min(len(x), len(y))
    if n < 2:
        return (np.nan, np.nan, np.nan, n)
    dd = x[:n] - y[:n]
    tc = tdist.ppf(0.975, n - 1)
    se = dd.std(ddof=1) / np.sqrt(n)
    return (dd.mean(), dd.mean() - tc * se, dd.mean() + tc * se, n)


def fmt(ci, digits=4):
    m, lo, hi, n = ci
    if np.isnan(m):
        return 'n/d'
    sig = '' if lo <= 0 <= hi else ' *'
    return f'{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{sig}'

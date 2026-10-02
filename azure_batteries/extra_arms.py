"""Extra arms on Azure and the NASA batteries: +LG multi-mode, and on the batteries also
+LG multi-mode + rates, +HI per-unit rate and +PCA twin (not definable on Azure: one machine per component,
split at random in time, so no per-unit rate exists). Same learner and partitions as other_datasets_matched.py.
    python azure_batteries/extra_arms.py battery --data data/nasa_battery
    python azure_batteries/extra_arms.py azure  --cache /tmp/azure_cache    (after azure_run.py prep)
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import json, os, sys, argparse, numpy as np, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
from sklearn.model_selection import GroupKFold
from other_datasets_matched import fit_c
from engines import make_y
from lg_multimode import pca_dirs
from lg_comb_gbsa_pkg import eta_dir, unit_slope
ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['battery', 'azure']); ap.add_argument('--data', default='data/nasa_battery')
ap.add_argument('--cache', default='/tmp/azure_cache'); ap.add_argument('--out', default=os.path.join(HERE, '..', 'results', 'extra_arms.json')); args = ap.parse_args()
OUT = args.out; os.makedirs(os.path.dirname(OUT), exist_ok=True); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
which = args.which
if which == 'battery':
    import battery_data as BD, battery_run as BR
    d = BD.load(args.data, verbose=False)
    Xb, Xs, ttf, ev, oh, u = d['X_base'], d['X_stats'], d['ttf'], d['event'], d['op_hours'], d['units']
    sc = {}
    for tri, tei in GroupKFold(n_splits=len(np.unique(u))).split(Xb, groups=u):
        if ev[tei].sum() == 0: continue
        c = BR.build(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], oh[tri], oh[tei]); Xt, Xe = c['Base']; p = Xt.shape[1]
        D = pca_dirs(Xt, 2); mt = np.column_stack([eta_dir(Xt, Xt, v, ttf[tri]) for v in D]); me = np.column_stack([eta_dir(Xt, Xe, v, ttf[tri]) for v in D])
        Pt, Pe = np.column_stack([Xt @ v for v in D]), np.column_stack([Xe @ v for v in D])
        sl = lambda A, uu, cc: np.column_stack([unit_slope(A[:, k], uu, cc) for k in range(A.shape[1])])
        Ht, He = c['+HI'][0][:, p:p + 1], c['+HI'][1][:, p:p + 1]
        arms = {'+LG-mm': (np.hstack([Xt, mt]), np.hstack([Xe, me])),
                'LG-comb-mm': (np.hstack([Xt, mt, sl(mt, u[tri], oh[tri])]), np.hstack([Xe, me, sl(me, u[tei], oh[tei])])),
                'HI-fix': (np.hstack([Xt, Ht, sl(Ht, u[tri], oh[tri])]), np.hstack([Xe, He, sl(He, u[tei], oh[tei])])),
                'PC-twin': (np.hstack([Xt, Pt, sl(Pt, u[tri], oh[tri])]), np.hstack([Xe, Pe, sl(Pe, u[tei], oh[tei])]))}
        ytr, yte = make_y(ev[tri], ttf[tri]), make_y(ev[tei], ttf[tei])
        for a, (A, B) in arms.items(): sc.setdefault(a, []).append(fit_c(a, A, ytr, B, yte))
    res['battery'] = sc; print({a: round(np.mean(v), 4) for a, v in sc.items()})
else:
    for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
        if comp in res.get('azure', {}): continue
        z = np.load(os.path.join(args.cache, f'{comp}.npz')); Xt, Xe, tt = z['Xbt'], z['Xbe'], z['ttf_tr']
        D = pca_dirs(Xt, 2); mt = np.column_stack([eta_dir(Xt, Xt, v, tt) for v in D]); me = np.column_stack([eta_dir(Xt, Xe, v, tt) for v in D])
        res.setdefault('azure', {})[comp] = fit_c('+LG-mm', np.hstack([Xt, mt]), make_y(z['ev_tr'], tt), np.hstack([Xe, me]), make_y(z['ev_te'], z['ttf_te']))
        json.dump(res, open(OUT, 'w'), indent=1); print(comp, round(res['azure'][comp], 4), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

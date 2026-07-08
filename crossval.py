import argparse
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from config import GBSA_BASE, GBSA_AUG, LG_PARAMS
from engines import LGPhysicsEngine, WeibullEngine, EntropyEngine, ExpDegEngine, HIBuilder, make_y, ci_score
warnings.filterwarnings('ignore')
NAMES = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']

def build_configs(Xb_tr, Xb_te, Xs_tr, Xs_te, ttf_tr, cyc_tr, cyc_te, w):
    sb = StandardScaler()
    Xbt = sb.fit_transform(Xb_tr)
    Xbe = sb.transform(Xb_te)
    ss = StandardScaler()
    Xst = ss.fit_transform(Xs_tr)
    Xse = ss.transform(Xs_te)
    hi = HIBuilder()
    hi.fit(Xbt, ttf_tr)
    hit, hie = (hi.transform(Xbt), hi.transform(Xbe))
    wb = WeibullEngine()
    wb.calibrate(ttf_tr)
    wbt, wbe = (wb.transform(cyc_tr), wb.transform(cyc_te))
    en = EntropyEngine(window=w, second_feature='rolling_std')
    en.calibrate(Xbt, ttf_tr)
    ent, ene = (en.transform(Xbt), en.transform(Xbe))
    ed = ExpDegEngine(window=w)
    ed.calibrate(Xbt, ttf_tr)
    edt, ede = (ed.transform(Xbt), ed.transform(Xbe))
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xbt, ttf_tr)
    lgt, lge = (lg.transform(Xbt), lg.transform(Xbe))
    return {'Base': (Xbt, Xbe), '+Stats': (np.hstack([Xbt, Xst]), np.hstack([Xbe, Xse])), '+HI': (np.hstack([Xbt, hit]), np.hstack([Xbe, hie])), '+Weibull': (np.hstack([Xbt, wbt]), np.hstack([Xbe, wbe])), '+Entropy': (np.hstack([Xbt, ent]), np.hstack([Xbe, ene])), '+ExpDeg': (np.hstack([Xbt, edt]), np.hstack([Xbe, ede])), '+LG': (np.hstack([Xbt, lgt]), np.hstack([Xbe, lge]))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cmapss', 'xjtu'], required=True)
    ap.add_argument('--data', default=None)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    if args.dataset == 'cmapss':
        import cmapss_data as DL
        from config import ROLLING_WINDOW_CMAPSS as W
    else:
        import xjtu_data as DL
        from config import ROLLING_WINDOW_XJTU as W
    d = DL.load(args.data)
    assert d is not None, 'data not found — pass --data PATH'
    Xb, Xs, ttf, ev, cyc, units = (d['X_base'], d['X_stats'], d['ttf'], d['event'], d['cyc'], d['units'])
    n_groups = len(np.unique(units))
    folds = min(args.folds, n_groups)
    gkf = GroupKFold(n_splits=folds)
    scores = {n: [] for n in NAMES}
    print(f'{args.dataset.upper()} — {folds}-fold grouped CV over {n_groups} units')
    for k, (tri, tei) in enumerate(gkf.split(Xb, groups=units), 1):
        cfgs = build_configs(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], cyc[tri], cyc[tei], W)
        ytr = make_y(ev[tri], ttf[tri])
        yte = make_y(ev[tei], ttf[tei])
        for nm, (xt, xe) in cfgs.items():
            kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
            m = GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr)
            scores[nm].append(float(ci_score(m, xe, yte)))
        print(f'  fold {k}/{folds} done')
    print(f'\n  {'paradigm':<10} {'mean':>7} {'std':>7}   (test C-index)')
    for nm in NAMES:
        s = np.array(scores[nm])
        print(f'  {nm:<10} {s.mean():>7.4f} {s.std():>7.4f}')
if __name__ == '__main__':
    main()
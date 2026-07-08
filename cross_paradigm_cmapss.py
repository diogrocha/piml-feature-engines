import json
import argparse
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from config import GBSA_BASE, GBSA_AUG, LG_PARAMS, ROLLING_WINDOW_CMAPSS, EVAL_TIME_QUANTILES, N_EVAL_TIMES
from engines import LGPhysicsEngine, WeibullEngine, EntropyEngine, ExpDegEngine, HIBuilder, make_y, ci_score
import cmapss_data
from optimize import optimize_lg
warnings.filterwarnings('ignore')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--out', default='results_cmapss.json')
    ap.add_argument('--optimize', action='store_true', help='re-optimise LG via Optuna instead of using config.LG_PARAMS')
    args = ap.parse_args()
    data = cmapss_data.load(args.data)
    if data is None:
        print('ERROR: train_FD001.txt not found. Pass --data PATH.')
        return
    print('=' * 78)
    print('C-MAPSS FD001 — SURVIVAL CROSS-PARADIGM')
    print(f'LG: {('Optuna' if args.optimize else 'fixed config.LG_PARAMS')}')
    print('=' * 78)
    tr, te = (data['tr'], data['te'])
    w = ROLLING_WINDOW_CMAPSS
    Xb_tr, Xb_te = (data['X_base'][tr], data['X_base'][te])
    Xs_tr, Xs_te = (data['X_stats'][tr], data['X_stats'][te])
    ttf_tr, ttf_te = (data['ttf'][tr], data['ttf'][te])
    ev_tr, ev_te = (data['event'][tr], data['event'][te])
    cyc_tr, cyc_te = (data['cyc'][tr], data['cyc'][te])
    sb = StandardScaler()
    Xbt = sb.fit_transform(Xb_tr)
    Xbe = sb.transform(Xb_te)
    ss = StandardScaler()
    Xst = ss.fit_transform(Xs_tr)
    Xse = ss.transform(Xs_te)
    ytr = make_y(ev_tr, ttf_tr)
    yte = make_y(ev_te, ttf_te)
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
    if args.optimize:
        bp = optimize_lg(Xbt, ttf_tr, ev_tr)
        lg = LGPhysicsEngine(c=bp['c'], kB_T_eff=bp['kT'], eta_h_frac=bp['eta_h_frac'], eta_f_frac=bp['eta_f_frac'])
    else:
        lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xbt, ttf_tr)
    lgt, lge = (lg.transform(Xbt), lg.transform(Xbe))
    cfgs = {'Base': (Xbt, Xbe), '+Stats': (np.hstack([Xbt, Xst]), np.hstack([Xbe, Xse])), '+HI': (np.hstack([Xbt, hit]), np.hstack([Xbe, hie])), '+Weibull': (np.hstack([Xbt, wbt]), np.hstack([Xbe, wbe])), '+Entropy': (np.hstack([Xbt, ent]), np.hstack([Xbe, ene])), '+ExpDeg': (np.hstack([Xbt, edt]), np.hstack([Xbe, ede])), '+LG': (np.hstack([Xbt, lgt]), np.hstack([Xbe, lge]))}
    lo, hival = EVAL_TIME_QUANTILES
    eval_times = np.linspace(np.percentile(ttf_te, lo), np.percentile(ttf_te, hival), N_EVAL_TIMES)
    results = {}
    for nm, (xt, xe) in cfgs.items():
        kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
        try:
            m = GradientBoostingSurvivalAnalysis(**kw)
            m.fit(xt, ytr)
            c_te = float(ci_score(m, xe, yte))
            c_tr = float(ci_score(m, xt, ytr))
            _, auc = cumulative_dynamic_auc(ytr, yte, m.predict(xe), eval_times)
            sf = m.predict_survival_function(xe)
            preds = np.row_stack([fn(eval_times) for fn in sf])
            ibs = float(integrated_brier_score(ytr, yte, preds, eval_times))
            results[nm] = {'c': c_te, 'c_train': c_tr, 'gap': c_tr - c_te, 'auc': float(auc), 'ibs': ibs, 'n_feat': int(xt.shape[1])}
            print(f'  {nm:<11} C_test={c_te:.4f}  C_train={c_tr:.4f}  gap={c_tr - c_te:+.4f}')
        except Exception as e:
            results[nm] = {'c': 0.0, 'auc': 0.0, 'ibs': 9.0, 'n_feat': int(xt.shape[1]), 'error': str(e)}
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {args.out}')
if __name__ == '__main__':
    main()
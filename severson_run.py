import json, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from config import GBSA_BASE, GBSA_AUG, LG_PARAMS
from engines import LGPhysicsEngine, WeibullEngine, EntropyEngine, ExpDegEngine, HIBuilder, make_y, ci_score
import severson_data as DL
W = DL.ROLLING_WINDOW
NAMES = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']

def build(Xb_tr, Xb_te, Xs_tr, Xs_te, ttf_tr, oh_tr, oh_te):
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
    wbt, wbe = (wb.transform(oh_tr), wb.transform(oh_te))
    en = EntropyEngine(window=W, second_feature='rolling_std')
    en.calibrate(Xbt, ttf_tr)
    ent, ene = (en.transform(Xbt), en.transform(Xbe))
    ed = ExpDegEngine(window=W)
    ed.calibrate(Xbt, ttf_tr)
    edt, ede = (ed.transform(Xbt), ed.transform(Xbe))
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xbt, ttf_tr)
    lgt, lge = (lg.transform(Xbt), lg.transform(Xbe))
    P = lambda a, b: (np.hstack([Xbt, a]), np.hstack([Xbe, b]))
    return {'Base': (Xbt, Xbe), '+Stats': P(Xst, Xse), '+HI': P(hit, hie), '+Weibull': P(wbt, wbe), '+Entropy': P(ent, ene), '+ExpDeg': P(edt, ede), '+LG': P(lgt, lge)}

def single_split(d):
    tr, te = (d['tr'], d['te'])
    cfgs = build(d['X_base'][tr], d['X_base'][te], d['X_stats'][tr], d['X_stats'][te], d['ttf'][tr], d['op_hours'][tr], d['op_hours'][te])
    ytr = make_y(d['event'][tr], d['ttf'][tr])
    yte = make_y(d['event'][te], d['ttf'][te])
    print('\n=== SINGLE SPLIT (41 train cells / 83 test cells) ===')
    out = {}
    for nm in NAMES:
        xt, xe = cfgs[nm]
        kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
        m = GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr)
        c_te = float(ci_score(m, xe, yte))
        c_tr = float(ci_score(m, xt, ytr))
        out[nm] = {'test': c_te, 'train': c_tr, 'gap': c_tr - c_te, 'n_feat': int(xt.shape[1])}
        print(f'  {nm:<10} C_test={c_te:.4f}  C_train={c_tr:.4f}  gap={c_tr - c_te:+.4f}')
    return out

def grouped_cv(d, folds=5):
    Xb, Xs, ttf, ev, oh, u = (d['X_base'], d['X_stats'], d['ttf'], d['event'], d['op_hours'], d['units'])
    gkf = GroupKFold(n_splits=folds)
    sc = {n: [] for n in NAMES}
    print(f'\n=== {folds}-FOLD GROUPED CV (by cell) ===')
    for tri, tei in gkf.split(Xb, groups=u):
        cfgs = build(Xb[tri], Xb[tei], Xs[tri], Xs[tei], ttf[tri], oh[tri], oh[tei])
        ytr = make_y(ev[tri], ttf[tri])
        yte = make_y(ev[tei], ttf[tei])
        for nm in NAMES:
            xt, xe = cfgs[nm]
            kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
            m = GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr)
            sc[nm].append(float(ci_score(m, xe, yte)))
    res = {}
    print(f'  {'paradigm':<10}{'mean':>8}{'std':>8}')
    for nm in NAMES:
        s = np.array(sc[nm])
        res[nm] = {'mean': float(s.mean()), 'std': float(s.std())}
        print(f'  {nm:<10}{s.mean():>8.4f}{s.std():>8.4f}')
    return res
if __name__ == '__main__':
    d = DL.load('/tmp/severson/data', cycle_step=5)
    r = {'single': single_split(d), 'cv': grouped_cv(d, folds=5)}
    json.dump(r, open('/mnt/user-data/outputs/results_severson.json', 'w'), indent=2)
    print('\nsaved -> results_severson.json')
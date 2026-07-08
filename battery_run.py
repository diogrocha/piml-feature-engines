import json, warnings, numpy as np
warnings.filterwarnings('ignore')
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from config import GBSA_BASE, GBSA_AUG, LG_PARAMS
from engines import LGPhysicsEngine, WeibullEngine, EntropyEngine, ExpDegEngine, HIBuilder, make_y, ci_score
import battery_data as DL
W = battery_data_W = DL.ROLLING_WINDOW
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
    print('\n=== SINGLE SPLIT (train B0005/6/7, test B0018) ===')
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

def grouped_cv(d):
    Xb, Xs, ttf, ev, oh, u = (d['X_base'], d['X_stats'], d['ttf'], d['event'], d['op_hours'], d['units'])
    gkf = GroupKFold(n_splits=len(np.unique(u)))
    sc = {n: [] for n in NAMES}
    print('\n=== LEAVE-ONE-BATTERY-OUT CV ===')
    for tri, tei in gkf.split(Xb, groups=u):
        held = np.unique(u[tei])[0]
        if ev[tei].sum() == 0:
            print(f'  (fold holding out {held}: all censored, no C-index — skipped)')
            continue
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

def csd_test(d):
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(d['X_base'], d['ttf'])
    sig = d['X_base'] @ lg.weights
    print('\n=== CRITICAL SLOWING DOWN (LG signal lag-1 autocorrelation -> EOL) ===')
    win, taus = (10, [])
    for b in DL.BATTERIES:
        m = d['units'] == b
        if m.sum() < win + 6:
            continue
        s = sig[m][np.argsort(d['cyc'][m])]
        ac = []
        for i in range(win, len(s)):
            wseg = s[i - win:i]
            ac.append(np.corrcoef(wseg[:-1], wseg[1:])[0, 1] if np.std(wseg) > 1e-09 else np.nan)
        ac = np.array(ac)
        ok = np.isfinite(ac)
        if ok.sum() > 5:
            tau, _ = kendalltau(np.arange(len(ac))[ok], ac[ok])
            taus.append(tau)
            print(f'  {b}: Kendall tau(autocorr, cycle) = {tau:+.3f}  ({('rising' if tau > 0 else 'falling')})')
    if taus:
        print(f'  mean tau = {np.mean(taus):+.3f}  (positive => critical slowing down before EOL)')
    return {'mean_tau': float(np.mean(taus)) if taus else None}
if __name__ == '__main__':
    d = DL.load('/tmp/NASA_bat/dataset')
    assert d is not None
    r = {'single': single_split(d), 'cv': grouped_cv(d), 'csd': csd_test(d)}
    json.dump(r, open('/mnt/user-data/outputs/results_battery.json', 'w'), indent=2)
    print('\nsaved -> results_battery.json')
import os, sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from config import GBSA_BASE, GBSA_AUG, LG_PARAMS, ROLLING_WINDOW_CMAPSS as Wc
from engines import LGPhysicsEngine, WeibullEngine, EntropyEngine, ExpDegEngine, HIBuilder, make_y, ci_score
import cmapss_data
CACHE = '/tmp/cmapss_cache'
os.makedirs(CACHE, exist_ok=True)
OUT = os.environ.get('LG_OUT') or ('/mnt/user-data/outputs' if os.path.isdir('/mnt/user-data/outputs') else os.getcwd())
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, 'results_cmapss_fd.json')
NAMES = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']

def prep(subset):
    d = cmapss_data.load('CMAPSSData', subset=subset)
    tr, te = (d['tr'], d['te'])
    sb = StandardScaler()
    Xbt = sb.fit_transform(d['X_base'][tr])
    Xbe = sb.transform(d['X_base'][te])
    ss = StandardScaler()
    Xst = ss.fit_transform(d['X_stats'][tr])
    Xse = ss.transform(d['X_stats'][te])
    ttf_tr = d['ttf'][tr]
    cyc_tr, cyc_te = (d['cyc'][tr], d['cyc'][te])
    hi = HIBuilder()
    hi.fit(Xbt, ttf_tr)
    hit, hie = (hi.transform(Xbt), hi.transform(Xbe))
    wb = WeibullEngine()
    wb.calibrate(ttf_tr)
    wbt, wbe = (wb.transform(cyc_tr), wb.transform(cyc_te))
    en = EntropyEngine(window=Wc, second_feature='rolling_std')
    en.calibrate(Xbt, ttf_tr)
    ent, ene = (en.transform(Xbt), en.transform(Xbe))
    ed = ExpDegEngine(window=Wc)
    ed.calibrate(Xbt, ttf_tr)
    edt, ede = (ed.transform(Xbt), ed.transform(Xbe))
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xbt, ttf_tr)
    lgt, lge = (lg.transform(Xbt), lg.transform(Xbe))
    np.savez_compressed(os.path.join(CACHE, subset + '.npz'), Xbt=Xbt, Xbe=Xbe, Xst=Xst, Xse=Xse, hit=hit, hie=hie, wbt=wbt, wbe=wbe, ent=ent, ene=ene, edt=edt, ede=ede, lgt=lgt, lge=lge, ev_tr=d['event'][tr], ev_te=d['event'][te], ttf_tr=ttf_tr, ttf_te=d['ttf'][te], units_te=d['units'][te])
    print(f'  {subset}: cached, train={tr.sum()} test={te.sum()}, base={Xbt.shape[1]} feats')

def _mats(z):
    Xbt, Xbe = (z['Xbt'], z['Xbe'])
    P = lambda a, b: (np.hstack([Xbt, a]), np.hstack([Xbe, b]))
    return {'Base': (Xbt, Xbe), '+Stats': P(z['Xst'], z['Xse']), '+HI': P(z['hit'], z['hie']), '+Weibull': P(z['wbt'], z['wbe']), '+Entropy': P(z['ent'], z['ene']), '+ExpDeg': P(z['edt'], z['ede']), '+LG': P(z['lgt'], z['lge'])}

def fit(subset, budget=250):
    z = np.load(os.path.join(CACHE, subset + '.npz'))
    ytr = make_y(z['ev_tr'], z['ttf_tr'])
    yte = make_y(z['ev_te'], z['ttf_te'])
    mats = _mats(z)
    nfast = int(os.environ.get('CMAPSS_FAST', '0'))
    sub_idx = None
    if nfast and len(ytr) > nfast:
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(len(ytr), nfast, replace=False)
        ytr = ytr[sub_idx]
    tag = subset + ('_fast' if nfast else '')
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    res.setdefault(tag, {})
    t0 = time.time()
    fit_order = ['Base', '+LG', '+ExpDeg', '+HI', '+Weibull', '+Entropy', '+Stats']
    for nm in fit_order:
        if nm in res[tag]:
            continue
        if time.time() - t0 > budget:
            print('  [budget] stop; re-run to continue')
            break
        xt, xe = mats[nm]
        if sub_idx is not None:
            xt = xt[sub_idx]
        kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
        ts = time.time()
        m = GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr)
        c_te = float(ci_score(m, xe, yte))
        c_tr = float(ci_score(m, xt, ytr))
        res[tag][nm] = {'test': c_te, 'train': c_tr, 'gap': c_tr - c_te, 'n_feat': int(xt.shape[1])}
        json.dump(res, open(RES, 'w'), indent=2)
        print(f'  {tag} {nm:<9} C_test={c_te:.4f} C_train={c_tr:.4f} gap={c_tr - c_te:+.4f} ({time.time() - ts:.0f}s)')
    print(f'  {tag}: {len([n for n in fit_order if n in res[tag]])}/7 done')

def report(subset):
    res = json.load(open(RES))[subset]
    print(f'\n=== C-MAPSS {subset} — per-paradigm ===')
    for nm in NAMES:
        if nm in res:
            r = res[nm]
            print(f'  {nm:<10} C_test={r['test']:.4f}  C_train={r['train']:.4f}  gap={r['gap']:+.4f}')
if __name__ == '__main__':
    cmd, subset = (sys.argv[1], sys.argv[2])
    {'prep': prep, 'fit': fit, 'report': report}[cmd](subset)
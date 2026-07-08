import os, sys, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from config import SEED, GBSA_BASE, GBSA_AUG, LG_PARAMS
from engines import LGPhysicsEngine, WeibullEngine, EntropyEngine, ExpDegEngine, HIBuilder, make_y, ci_score
import azure_data
DATA = os.environ.get('AZURE_DATA', '/tmp/azure_in')
CACHE = '/tmp/azure_cache'
RESULTS = '/mnt/user-data/outputs/results_azure.json'
COMPONENTS = ['comp1', 'comp2', 'comp3', 'comp4']
CONFIGS = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']
os.makedirs(CACHE, exist_ok=True)

def _split(event):
    np.random.seed(SEED)
    fi, ci = (np.where(event)[0], np.where(~event)[0])
    np.random.shuffle(fi)
    np.random.shuffle(ci)
    sf, sc = (max(1, int(0.8 * len(fi))), int(0.8 * len(ci)))
    tr = np.concatenate([fi[:sf], ci[:sc]])
    te = np.concatenate([fi[sf:], ci[sc:]])
    np.random.shuffle(tr)
    np.random.shuffle(te)
    return (tr, te)

def prep(comp):
    sub = azure_data.build_component_subset(DATA, comp)
    if sub is None:
        print(f'  {comp}: skipped (insufficient failures)')
        return
    ev = sub['event']
    tr, te = _split(ev)
    sb = StandardScaler()
    Xbt = sb.fit_transform(sub['X_base'][tr])
    Xbe = sb.transform(sub['X_base'][te])
    ss = StandardScaler()
    Xst = ss.fit_transform(sub['X_stats'][tr])
    Xse = ss.transform(sub['X_stats'][te])
    ttf_tr = sub['ttf'][tr]
    oh_tr, oh_te = (sub['op_hours'][tr], sub['op_hours'][te])
    hi = HIBuilder()
    hi.fit(Xbt, ttf_tr)
    hit, hie = (hi.transform(Xbt), hi.transform(Xbe))
    wb = WeibullEngine()
    wb.calibrate(ttf_tr)
    wbt, wbe = (wb.transform(oh_tr), wb.transform(oh_te))
    en = EntropyEngine(window=24, second_feature='sample_entropy')
    en.calibrate(Xbt, ttf_tr)
    ent, ene = (en.transform(Xbt), en.transform(Xbe))
    ed = ExpDegEngine(window=48)
    ed.calibrate(Xbt, ttf_tr)
    edt, ede = (ed.transform(Xbt), ed.transform(Xbe))
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xbt, ttf_tr)
    lgt, lge = (lg.transform(Xbt), lg.transform(Xbe))
    np.savez_compressed(os.path.join(CACHE, comp + '.npz'), Xbt=Xbt, Xbe=Xbe, Xst=Xst, Xse=Xse, hit=hit, hie=hie, wbt=wbt, wbe=wbe, ent=ent, ene=ene, edt=edt, ede=ede, lgt=lgt, lge=lge, ev_tr=ev[tr], ev_te=ev[te], ttf_tr=ttf_tr, ttf_te=sub['ttf'][te], machine=sub['machine'])
    print(f'  {comp}: cached (machine #{sub['machine']}, train={len(tr)} test={len(te)}, events tr={int(ev[tr].sum())}/te={int(ev[te].sum())})')

def _cfg_matrices(d):
    Xbt, Xbe = (d['Xbt'], d['Xbe'])
    pair = lambda a, b: (np.hstack([Xbt, a]), np.hstack([Xbe, b]))
    return {'Base': (Xbt, Xbe), '+Stats': pair(d['Xst'], d['Xse']), '+HI': pair(d['hit'], d['hie']), '+Weibull': pair(d['wbt'], d['wbe']), '+Entropy': pair(d['ent'], d['ene']), '+ExpDeg': pair(d['edt'], d['ede']), '+LG': pair(d['lgt'], d['lge'])}

def _load_results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return {}

def fit(comp, budget=235):
    d = np.load(os.path.join(CACHE, comp + '.npz'))
    ytr = make_y(d['ev_tr'], d['ttf_tr'])
    yte = make_y(d['ev_te'], d['ttf_te'])
    mats = _cfg_matrices(d)
    res = _load_results()
    res.setdefault(comp, {})
    t0 = time.time()
    for nm in CONFIGS:
        if nm in res[comp]:
            continue
        if time.time() - t0 > budget:
            print(f'  [budget] stopping before {nm}; re-run to continue.')
            break
        xt, xe = mats[nm]
        kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
        ts = time.time()
        m = GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr)
        c_te = float(ci_score(m, xe, yte))
        c_tr = float(ci_score(m, xt, ytr))
        res[comp][nm] = {'test': c_te, 'train': c_tr, 'gap': c_tr - c_te, 'n_feat': int(xt.shape[1])}
        json.dump(res, open(RESULTS, 'w'), indent=2)
        print(f'  {comp} {nm:<9} C_test={c_te:.4f} C_train={c_tr:.4f} ({time.time() - ts:.0f}s)')
    done = [c for c in CONFIGS if c in res[comp]]
    print(f'  {comp}: {len(done)}/{len(CONFIGS)} configs done')

def report():
    res = _load_results()
    comps = [c for c in COMPONENTS if c in res]
    print('\n=== PER COMPONENT (test C-index) ===')
    hdr = '  config     ' + ''.join((f'{c:>10}' for c in comps)) + f'{'avg':>10}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    avg = {}
    for nm in CONFIGS:
        vals = [res[c][nm]['test'] for c in comps if nm in res[c]]
        avg[nm] = float(np.mean(vals)) if vals else 0.0
        row = '  %-10s' % nm + ''.join((f'{res[c].get(nm, {}).get('test', 0):>10.4f}' for c in comps))
        print(row + f'{avg[nm]:>10.4f}')
    print('\n=== AVERAGED (paper headline) ===')
    for nm in CONFIGS:
        print(f'  {nm:<10} {avg[nm]:.4f}')
if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if cmd == 'prep':
        targets = sys.argv[2:] or COMPONENTS
        for c in targets:
            prep(c)
    elif cmd == 'fit':
        fit(sys.argv[2])
    elif cmd == 'report':
        report()
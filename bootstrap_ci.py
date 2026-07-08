import os, sys, json, numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from config import GBSA_BASE, GBSA_AUG
from engines import make_y
CACHE = '/tmp/cmapss_cache'
PRED = '/tmp/cmapss_cache/preds_{}.npz'
PREDTR = '/tmp/cmapss_cache/predstr_{}.npz'
NAMES = ['Base', '+LG', '+ExpDeg', '+HI', '+Weibull', '+Entropy', '+Stats']

def cidx(event, time, risk):
    return concordance_index_censored(event.astype(bool), time, risk)[0]

def boot_ci(preds, time, event, units, n_boot=300, seed=42):
    rng = np.random.RandomState(seed)
    groups = np.unique(units)
    gidx = {g: np.where(units == g)[0] for g in groups}
    point = {k: cidx(event, time, r) for k, r in preds.items()}
    samples = {k: [] for k in preds}
    pairs = [('+LG', '+ExpDeg')] + [(a, 'Base') for a in NAMES if a not in ('Base', '+LG', '+ExpDeg')] + [('+LG', 'Base'), ('+ExpDeg', 'Base')]
    dsamp = {p: [] for p in pairs}
    for _ in range(n_boot):
        gsel = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([gidx[g] for g in gsel])
        ev_b, t_b = (event[idx], time[idx])
        cb = {}
        for k, r in preds.items():
            c = cidx(ev_b, t_b, r[idx])
            cb[k] = c
            samples[k].append(c)
        for a, b in pairs:
            dsamp[a, b].append(cb[a] - cb[b])

    def ci(v):
        v = np.array(v)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = {'point': point, 'ci': {k: ci(v) for k, v in samples.items()}, 'diff': {f'{a}-{b}': {'point': point[a] - point[b], 'ci': ci(v), 'sig': not ci(v)[0] <= 0 <= ci(v)[1]} for (a, b), v in dsamp.items()}}
    return out

def fit_preds(subset, budget=250, nfast=6000):
    z = np.load(os.path.join(CACHE, subset + '.npz'))
    ytr = make_y(z['ev_tr'], z['ttf_tr'])
    Xbt, Xbe = (z['Xbt'], z['Xbe'])
    P = lambda a, b: (np.hstack([Xbt, a]), np.hstack([Xbe, b]))
    mats = {'Base': (Xbt, Xbe), '+Stats': P(z['Xst'], z['Xse']), '+HI': P(z['hit'], z['hie']), '+Weibull': P(z['wbt'], z['wbe']), '+Entropy': P(z['ent'], z['ene']), '+ExpDeg': P(z['edt'], z['ede']), '+LG': P(z['lgt'], z['lge'])}
    sub = None
    if nfast and len(ytr) > nfast:
        sub = np.random.RandomState(42).choice(len(ytr), nfast, replace=False)
        ytr = ytr[sub]
    pf = PRED.format(subset)
    pftr = PREDTR.format(subset)
    store = dict(np.load(pf)) if os.path.exists(pf) else {}
    store_tr = dict(np.load(pftr)) if os.path.exists(pftr) else {}
    import time as _t
    t0 = _t.time()
    for nm in NAMES:
        if nm in store:
            continue
        if _t.time() - t0 > budget:
            print('  [budget] re-run to continue')
            break
        xt, xe = mats[nm]
        if sub is not None:
            xt = xt[sub]
        kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
        m = GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr)
        store[nm] = m.predict(xe).astype(float)
        store_tr[nm] = m.predict(xt).astype(float)
        store_tr['_ev_tr'] = z['ev_tr'][sub] if sub is not None else z['ev_tr']
        store_tr['_tt_tr'] = (z['ttf_tr'][sub] if sub is not None else z['ttf_tr']).astype(float)
        np.savez(pf, **store)
        np.savez(pftr, **store_tr)
        print(f'  pred {nm} ok ({len(store)}/7)')
    return len(store)

def full_report(subset, n_boot=2000):
    z = np.load(os.path.join(CACHE, subset + '.npz'))
    test = dict(np.load(PRED.format(subset)))
    train = dict(np.load(PREDTR.format(subset)))
    ev_te, t_te = (z['ev_te'], z['ttf_te'].astype(float))
    ev_tr, t_tr = (train['_ev_tr'], train['_tt_tr'].astype(float))
    preds = {k: test[k] for k in NAMES if k in test}
    print(f'=== C-MAPSS {subset} — paradigm table (full settings) ===')
    table = {}
    for nm in NAMES:
        if nm not in preds:
            continue
        c_te = cidx(ev_te, t_te, test[nm])
        c_tr = cidx(ev_tr, t_tr, train[nm])
        table[nm] = {'test': float(c_te), 'train': float(c_tr), 'gap': float(c_tr - c_te)}
        print(f'  {nm:<10} test={c_te:.4f}  train={c_tr:.4f}  gap={c_tr - c_te:+.4f}')
    out = boot_ci(preds, t_te, ev_te, z['units_te'], n_boot=n_boot)
    print(f'\n=== bootstrap 95% CI ({n_boot} resamples over {len(np.unique(z['units_te']))} test engines) ===')
    for nm in NAMES:
        if nm in out['ci']:
            lo, hi = out['ci'][nm]
            print(f'  {nm:<10} {out['point'][nm]:.4f}  [{lo:.4f}, {hi:.4f}]')
    print('  --- paired differences (sig = CI excludes 0) ---')
    for k, d in out['diff'].items():
        lo, hi = d['ci']
        print(f'  {k:<18} {d['point']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {('SIG' if d['sig'] else 'n.s.')}')
    _out = os.environ.get('LG_OUT') or ('/mnt/user-data/outputs' if os.path.isdir('/mnt/user-data/outputs') else os.getcwd())
    os.makedirs(_out, exist_ok=True)
    json.dump({'table': table, 'bootstrap': out}, open(os.path.join(_out, f'results_full_{subset}.json'), 'w'), indent=2)
    print(f'\nsaved -> {os.path.join(_out, f'results_full_{subset}.json')}')
    return (table, out)

def run_ci(subset):
    z = np.load(os.path.join(CACHE, subset + '.npz'))
    pf = PRED.format(subset)
    store = dict(np.load(pf))
    preds = {k: store[k] for k in NAMES if k in store}
    out = boot_ci(preds, z['ttf_te'].astype(float), z['ev_te'], z['units_te'])
    _out = os.environ.get('LG_OUT') or ('/mnt/user-data/outputs' if os.path.isdir('/mnt/user-data/outputs') else os.getcwd())
    os.makedirs(_out, exist_ok=True)
    json.dump(out, open(os.path.join(_out, f'results_bootstrap_{subset}.json'), 'w'), indent=2)
    print(f'\n=== {subset}: C-index 95% CI (cluster bootstrap over test engines) ===')
    for nm in NAMES:
        if nm in out['ci']:
            p = out['point'][nm]
            lo, hi = out['ci'][nm]
            print(f'  {nm:<10} {p:.4f}  [{lo:.4f}, {hi:.4f}]')
    print('  --- paired differences (95% CI; sig = excludes 0) ---')
    for k, d in out['diff'].items():
        lo, hi = d['ci']
        print(f'  {k:<18} {d['point']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {('SIG' if d['sig'] else 'n.s.')}')
if __name__ == '__main__':
    cmd = sys.argv[1]
    subset = sys.argv[2] if len(sys.argv) > 2 else 'FD003'
    if cmd == 'fit':
        fit_preds(subset)
    elif cmd == 'ci':
        run_ci(subset)
import os, json, time, numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored
from config import GBSA_AUG, GBSA_BASE
from engines import make_y
OUT = '/mnt/user-data/outputs/results_validation.json'
CACHE = '/tmp/cmapss_cache/FD001.npz'
NSUB = 6000
SEED = 42

def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}

def save(d):
    json.dump(d, open(OUT, 'w'), indent=2)

def grids(ytr, yte):
    tmax = min(ytr['time'].max(), yte['time'].max())
    times = np.percentile(yte['time'][yte['event']], np.arange(10, 91, 10))
    times = np.unique(np.clip(times, 1, tmax - 1))
    fine = np.linspace(0, tmax, 60)
    return (times, fine)

def metrics_for(m, xe, ytr, yte, times, fine):
    risk = m.predict(xe)
    C = concordance_index_censored(yte['event'], yte['time'], risk)[0]
    sf = m.predict_survival_function(xe)
    prob = np.vstack([fn(times) for fn in sf])
    ibs = integrated_brier_score(ytr, yte, prob, times)
    _, mauc = cumulative_dynamic_auc(ytr, yte, risk, times)
    Sg = np.vstack([fn(fine) for fn in sf])
    rmst = np.sum((Sg[:, 1:] + Sg[:, :-1]) / 2.0 * np.diff(fine), axis=1)
    rul_mae = float(np.mean(np.abs(rmst - yte['time'])))
    return dict(C=float(C), IBS=float(ibs), meanAUC=float(mauc), RUL_MAE=rul_mae)

def main():
    z = np.load(CACHE)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(z['ttf_tr']), NSUB, replace=False)
    ytr = make_y(z['ev_tr'][idx], z['ttf_tr'][idx])
    yte = make_y(z['ev_te'], z['ttf_te'])
    times, fine = grids(ytr, yte)
    Xbt, Xbe = (z['Xbt'], z['Xbe'])
    blocks = {'+Stats': (z['Xst'], z['Xse']), '+HI': (z['hit'], z['hie']), '+Weibull': (z['wbt'], z['wbe']), '+Entropy': (z['ent'], z['ene']), '+ExpDeg': (z['edt'], z['ede']), '+LG': (z['lgt'], z['lge'])}

    def aug(bt, be):
        return (np.hstack([Xbt, bt])[idx], np.hstack([Xbe, be]))
    mats = {'Base': (Xbt[idx], Xbe)}
    for k, (bt, be) in blocks.items():
        mats[k] = aug(bt, be)
    d = load()
    d.setdefault('metrics', {})
    d.setdefault('ablation', {})
    d.setdefault('bootstrap', {})
    risks = {}
    order = ['Base', '+HI', '+ExpDeg', '+LG']
    for nm in order:
        xt, xe = mats[nm]
        if nm not in d['metrics']:
            kw = dict(GBSA_BASE) if nm == 'Base' else dict(GBSA_AUG)
            t0 = time.time()
            m = GradientBoostingSurvivalAnalysis(**kw).fit(xt, ytr)
            d['metrics'][nm] = metrics_for(m, xe, ytr, yte, times, fine)
            d['metrics'][nm]['fit_s'] = round(time.time() - t0)
            save(d)
            print(f'[metrics] {nm:<9} {d['metrics'][nm]}', flush=True)
            if nm in ('+LG', '+ExpDeg'):
                np.save(f'/tmp/risk_{nm.strip('+')}.npy', m.predict(xe))
        if nm in ('+LG', '+ExpDeg') and os.path.exists(f'/tmp/risk_{nm.strip('+')}.npy'):
            risks[nm] = np.load(f'/tmp/risk_{nm.strip('+')}.npy')
    lgt, lge = (z['lgt'], z['lge'])
    cols = ['R', 'dF', 'Gamma', 'kappa']
    for j, cn in enumerate(cols):
        name = f'+LG drop {cn}'
        if name not in d['ablation']:
            keep = [c for c in range(lgt.shape[1]) if c != j]
            xt = np.hstack([Xbt, lgt[:, keep]])[idx]
            xe = np.hstack([Xbe, lge[:, keep]])
            m = GradientBoostingSurvivalAnalysis(**GBSA_AUG).fit(xt, ytr)
            d['ablation'][name] = {'C': float(concordance_index_censored(yte['event'], yte['time'], m.predict(xe))[0])}
            save(d)
            print(f'[ablation] {name:<16} {d['ablation'][name]}', flush=True)
    if 'shuffled physics' not in d['ablation']:
        perm = rng.permutation(len(idx))
        xt = np.hstack([Xbt[idx], lgt[idx][perm]])
        xe = np.hstack([Xbe, lge])
        m = GradientBoostingSurvivalAnalysis(**GBSA_AUG).fit(xt, ytr)
        d['ablation']['shuffled physics'] = {'C': float(concordance_index_censored(yte['event'], yte['time'], m.predict(xe))[0])}
        save(d)
        print(f'[ablation] shuffled physics {d['ablation']['shuffled physics']}', flush=True)
    if 'diff_mean' not in d['bootstrap'] and {'+LG', '+ExpDeg'} <= set(risks):
        units = z['units_te']
        uq = np.unique(units)
        B = 1500
        uidx = {u: np.where(units == u)[0] for u in uq}
        rL, rE = (risks['+LG'], risks['+ExpDeg'])
        ev = yte['event']
        tm = yte['time']
        bf = '/tmp/boot_diffs.npy'
        diffs = list(np.load(bf)) if os.path.exists(bf) else []
        rs = np.random.RandomState(7 + len(diffs))
        t0 = time.time()
        while len(diffs) < B and time.time() - t0 < 90:
            samp = rs.choice(uq, len(uq), replace=True)
            mask = np.concatenate([uidx[u] for u in samp])
            cL = concordance_index_censored(ev[mask], tm[mask], rL[mask])[0]
            cE = concordance_index_censored(ev[mask], tm[mask], rE[mask])[0]
            diffs.append(cL - cE)
            if len(diffs) % 200 == 0:
                np.save(bf, np.array(diffs))
        np.save(bf, np.array(diffs))
        print(f'[bootstrap] {len(diffs)}/{B} resamples', flush=True)
        if len(diffs) >= B:
            a = np.array(diffs)
            d['bootstrap'] = {'diff_mean': float(a.mean()), 'lo': float(np.percentile(a, 2.5)), 'hi': float(np.percentile(a, 97.5)), 'B': B}
            save(d)
            print(f'[bootstrap] +LG - +ExpDeg: {d['bootstrap']}', flush=True)
    print('DONE', flush=True)
if __name__ == '__main__':
    main()
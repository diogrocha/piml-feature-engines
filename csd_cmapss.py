import os, sys, json, warnings, numpy as np
warnings.filterwarnings('ignore')
from scipy.stats import kendalltau
from config import LG_PARAMS
from engines import LGPhysicsEngine
import cmapss_data
RES = '/mnt/user-data/outputs/results_cmapss_csd.json'
MULTI = {'FD002', 'FD004'}
SUBSETS = ['FD001', 'FD003', 'FD002', 'FD004']

def csd(subset, win=20):
    d = cmapss_data.load('CMAPSSData', subset=subset, condition_normalize=subset in MULTI, verbose=False)
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(d['X_base'], d['ttf'])
    sig = d['X_base'] @ lg.weights
    taus, rising = ([], 0)
    for u in np.unique(d['units']):
        m = d['units'] == u
        s = sig[m][np.argsort(d['cyc'][m])]
        if len(s) < win + 10:
            continue
        ac = [np.corrcoef(s[i - win:i][:-1], s[i - win:i][1:])[0, 1] if np.std(s[i - win:i]) > 1e-09 else np.nan for i in range(win, len(s))]
        ac = np.array(ac)
        ok = np.isfinite(ac)
        if ok.sum() > 8:
            tau = kendalltau(np.arange(len(ac))[ok], ac[ok])[0]
            taus.append(tau)
            rising += tau > 0
    return {'mean_tau': float(np.mean(taus)), 'frac_rising': float(rising / len(taus)), 'n_units': len(taus), 'stationary': subset not in MULTI}

def main():
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    import time
    t0 = time.time()
    for s in SUBSETS:
        if s in res:
            continue
        if time.time() - t0 > 250:
            print('  [budget] stop; re-run to continue')
            break
        res[s] = csd(s)
        json.dump(res, open(RES, 'w'), indent=2)
        r = res[s]
        print(f'  {s}: mean tau={r['mean_tau']:+.3f}, rising {r['frac_rising'] * 100:.0f}% ({('stationary' if r['stationary'] else 'multi-condition')})')
    if all((s in res for s in SUBSETS)):
        print('  all subsets done.')
if __name__ == '__main__':
    main()
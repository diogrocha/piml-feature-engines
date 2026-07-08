import argparse
import warnings
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from config import LG_PARAMS, ROLLING_WINDOW_CMAPSS
from engines import LGPhysicsEngine
import cmapss_data
warnings.filterwarnings('ignore')

def rolling_var_ar1(sig, w):
    n = len(sig)
    var = np.full(n, np.nan)
    ar1 = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - w + 1)
        win = sig[lo:i + 1]
        if len(win) > 3 and np.std(win) > 1e-10:
            var[i] = np.var(win)
            x, y = (win[:-1], win[1:])
            if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                ar1[i] = np.corrcoef(x, y)[0, 1]
    return (var, ar1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--window', type=int, default=ROLLING_WINDOW_CMAPSS)
    args = ap.parse_args()
    data = cmapss_data.load(args.data)
    if data is None:
        print('ERROR: C-MAPSS data not found. Pass --data PATH.')
        return
    tr = data['tr']
    sb = StandardScaler()
    Xtr = sb.fit_transform(data['X_base'][tr])
    Xall = sb.transform(data['X_base'])
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xtr, data['ttf'][tr])
    sig = Xall @ lg.weights
    R = lg.transform(Xall)[:, 0]
    units, cyc, ttf = (data['units'], data['cyc'], data['ttf'])
    print('=' * 66)
    print('CRITICAL-SLOWING-DOWN OBSERVABLES (exploratory)')
    print('=' * 66)
    tau_var, tau_ar1 = ([], [])
    var_all = np.full(len(sig), np.nan)
    ar1_all = np.full(len(sig), np.nan)
    for u in np.unique(units):
        m = units == u
        order = np.argsort(cyc[m])
        idx = np.where(m)[0][order]
        v, a = rolling_var_ar1(sig[idx], args.window)
        var_all[idx], ar1_all[idx] = (v, a)
        t = cyc[m][order]
        ok = ~np.isnan(v)
        if ok.sum() > 10:
            tau_var.append(kendalltau(t[ok], v[ok]).correlation)
        ok = ~np.isnan(a)
        if ok.sum() > 10:
            tau_ar1.append(kendalltau(t[ok], a[ok]).correlation)
    tau_var, tau_ar1 = (np.array(tau_var), np.array(tau_ar1))
    print(f'\nWithin-trajectory trend over {len(tau_var)} engines (Kendall tau vs cycle; >0 => rises toward failure):')
    print(f'  variance : mean tau = {np.nanmean(tau_var):+.3f}  median {np.nanmedian(tau_var):+.3f}  positive in {(tau_var > 0).mean() * 100:.0f}% of engines')
    print(f'  AR(1)    : mean tau = {np.nanmean(tau_ar1):+.3f}  median {np.nanmedian(tau_ar1):+.3f}  positive in {(tau_ar1 > 0).mean() * 100:.0f}% of engines')
    ok = ~np.isnan(var_all) & ~np.isnan(ar1_all)
    print('\nOverlap with the instantaneous LG descriptor R (near 0 => complementary information):')
    print(f'  Spearman(variance, R) = {spearmanr(var_all[ok], R[ok]).correlation:+.3f}')
    print(f'  Spearman(AR(1),    R) = {spearmanr(ar1_all[ok], R[ok]).correlation:+.3f}')
    print('\nReading: a strong, consistent rise in lag-1 autocorrelation toward')
    print('failure is the critical-slowing-down signature predicted by the model')
    print('as eta -> 0. Variance is a weaker indicator here (the base features are')
    print('already rolling aggregates). AR(1) partly overlaps R; variance is more')
    print('independent -- a candidate for breaking the four-feature collinearity.')
if __name__ == '__main__':
    main()
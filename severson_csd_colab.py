import argparse, pickle, os, glob
import numpy as np
from scipy.stats import kendalltau
EOL_FRAC = 0.8
WIN = 30

def _ar1_var_trends(cap, eol_idx, win=WIN):
    c = np.asarray(cap, float)[:eol_idx + 1]
    if len(c) < 2 * win:
        return None
    k = win
    trend = np.convolve(c, np.ones(k) / k, mode='same')
    resid = c - trend
    ar1, var, idx = ([], [], [])
    for i in range(win, len(resid)):
        w = resid[i - win:i]
        if np.std(w) > 1e-12:
            ar1.append(np.corrcoef(w[:-1], w[1:])[0, 1])
            var.append(np.var(w))
            idx.append(i)
    if len(idx) < 8:
        return None
    idx = np.asarray(idx, float)
    tau_ar1 = kendalltau(idx, ar1)[0]
    tau_var = kendalltau(idx, var)[0]
    return (tau_ar1, tau_var)

def load_capacities(args):
    caps = {}
    if args.pkl:
        with open(args.pkl, 'rb') as f:
            bat = pickle.load(f)
        for cell, d in bat.items():
            qd = np.asarray(d['summary']['QD'], float).ravel()
            qd = qd[np.isfinite(qd) & (qd > 0)]
            if len(qd) > 60:
                caps[cell] = qd
    elif args.capdir:
        for f in sorted(glob.glob(os.path.join(args.capdir, '*.csv'))):
            qd = np.loadtxt(f, delimiter=',').ravel()
            qd = qd[np.isfinite(qd) & (qd > 0)]
            if len(qd) > 60:
                caps[os.path.basename(f)] = qd
    return caps

def run(caps):
    tau_a, tau_v, n = ([], [], 0)
    for cell, qd in caps.items():
        nominal = np.median(qd[:5])
        eol = np.where(qd <= EOL_FRAC * nominal)[0]
        eol_idx = int(eol[0]) if eol.size else len(qd) - 1
        r = _ar1_var_trends(qd, eol_idx)
        if r is None:
            continue
        ta, tv = r
        if np.isfinite(ta):
            tau_a.append(ta)
            tau_v.append(tv)
            n += 1
    if not tau_a:
        print('No usable cells.')
        return
    tau_a, tau_v = (np.array(tau_a), np.array(tau_v))
    print(f'Cells analysed: {n}')
    print(f'lag-1 autocorrelation rises toward EOL:  mean tau = {tau_a.mean():+.3f}, {100 * np.mean(tau_a > 0):.0f}% of cells positive')
    print(f'variance rises toward EOL:               mean tau = {tau_v.mean():+.3f}, {100 * np.mean(tau_v > 0):.0f}% of cells positive')
    print('Positive values support critical slowing down before the knee.')

def _self_test():
    rng = np.random.RandomState(0)
    n = 600
    t = np.arange(n)
    cap = 1.1 - 5e-05 * t - 0.2 / (1 + np.exp(-(t - 480) / 25.0))
    eps = np.zeros(n)
    for i in range(1, n):
        phi = 0.2 + 0.7 * (i / n) ** 3
        eps[i] = phi * eps[i - 1] + 0.004 * rng.randn()
    caps = {'synthetic_knee': cap + eps}
    print('=== SELF-TEST (synthetic knee, CSD present) ===')
    run(caps)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=None)
    ap.add_argument('--capdir', default=None)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest or (not args.pkl and (not args.capdir)):
        _self_test()
    else:
        run(load_capacities(args))
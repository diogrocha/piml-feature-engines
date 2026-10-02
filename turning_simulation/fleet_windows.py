"""Fleet v2 (pre-registered 30 Sep 2026): same runs as fleet.py (same seeds), richer per-window features.
Base 'sensors': rms, kurtosis, crest, energy in 6 log-spaced frequency bands, elapsed time t (the cycle-counter analogue).
Model-based physics estimator: over each window, least squares on the regenerative model
    x'' + 2 zeta x' + x = -b h,  h = x(t) - x(t - tau)   (tau known from spindle speed)
gives (zeta_ls, b_ls); the model then gives the stability limit b_lim(tau, zeta_ls), and the estimated distance
to the Hopf boundary d_hat = b_ls / b_lim(tau, zeta_ls). Non-physical twin: (b_ls, zeta_ls, tau) given separately.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import numpy as np, pandas as pd, sys, os
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis
from scipy.interpolate import RegularGridInterpolator
from turning import b_lim, simulate
DT, WIN, STEP = 0.02, 300.0, 50.0
TAB = 'blim_table.npz'
if not os.path.exists(TAB):
    tg, zg = np.linspace(11, 26, 31), np.linspace(0.002, 0.06, 30)
    np.savez(TAB, tg=tg, zg=zg, B=np.array([[b_lim(t, z) for z in zg] for t in tg]))
T_ = np.load(TAB); BL = RegularGridInterpolator((T_['tg'], T_['zg']), T_['B'], bounds_error=False, fill_value=None)
BANDS = np.geomspace(0.05, 3.0, 7) / (2 * np.pi)

def ar2(y):
    Y = np.column_stack([y[1:-1], y[:-2]]); a = np.linalg.lstsq(Y, y[2:], rcond=None)[0]
    r = np.roots([1, -a[0], -a[1]]); z = r[np.argmax(np.abs(np.angle(r)))]; s = np.log(max(abs(z), 1e-12)); th = abs(np.angle(z))
    return -s / np.sqrt(s * s + th * th)

def run(seed):
    rng = np.random.default_rng(seed); tau = rng.uniform(12, 25); z = 0.02; bl = b_lim(tau, z)
    f0, T = rng.uniform(0.3, 0.6), rng.uniform(3000, 6000); n = int(T / DT)
    b = bl * np.linspace(f0, 1.3, n); tstar = T * (1 - f0) / (1.3 - f0)
    y = simulate(tau, z, b, dt=DT, s_f=rng.uniform(0.01, 0.03), a_r=rng.uniform(0.005, 0.02), rng=rng)
    y = np.nan_to_num(y, nan=0.0); d = int(round(tau / DT))
    ys = sosfiltfilt(butter(4, 3.0 / (2 * np.pi), fs=1 / DT, output='sos'), y)          # remove measurement noise
    v = np.gradient(ys, DT); a = np.gradient(v, DT)
    sos_b = butter(4, [0.7 / (2 * np.pi), 1.4 / (2 * np.pi)], btype='band', fs=1 / DT, output='sos')
    rows = []; w = int(WIN / DT); st = int(STEP / DT)
    for k0 in range(d, n - w, st):
        t = (k0 + w) * DT
        if t >= tstar: break
        sl = slice(k0, k0 + w); seg = y[sl]; h = ys[sl] - ys[k0 - d:k0 - d + w]
        M = np.column_stack([v[sl], h]); rhs = a[sl] + ys[sl]
        c2z, bls = np.linalg.lstsq(M, -rhs, rcond=None)[0]; zls = c2z / 2
        blim_hat = float(BL([[tau, np.clip(zls, 0.002, 0.06)]])[0])
        bp = sosfiltfilt(sos_b, seg)[::15]
        f, P = welch(seg - seg.mean(), fs=1 / DT, nperseg=4096); E = [np.log(P[(f >= BANDS[i]) & (f < BANDS[i + 1])].sum() + 1e-20) for i in range(6)]
        rows.append(dict(run=seed, t=t, ttf=tstar - t, tau=tau, b_frac=float(b[k0 + w]) / bl, rms=np.std(seg), kurt=kurtosis(seg),
                         crest=np.max(np.abs(seg - seg.mean())) / (np.std(seg) + 1e-12), **{f'E{i}': E[i] for i in range(6)},
                         band_var=np.var(bp), band_ac1=np.corrcoef(bp[:-1], bp[1:])[0, 1], zeta_ar=ar2(bp),
                         b_ls=bls, zeta_ls=zls, d_hat=bls / blim_hat))
    return rows

if __name__ == '__main__':
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = []
    for s in range(lo, hi): out += run(s)
    pd.DataFrame(out).to_csv(f'f2_{lo}_{hi}.csv', index=False); print(lo, hi, len(out))

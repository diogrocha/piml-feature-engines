"""Step 4 (pre-registered 30 Sep 2026, before computation).
(1) Reduced Landau estimator, demodulated at the data's own mode frequency: at each window end t, over the trailing
    400 time units (causal), omega_hat = spectral peak of the band signal; complex amplitude by demodulation at
    omega_hat and causal low-pass; sigma_L2 = log|c| / L from least squares A(t) = c A(t - L), L = 10.
    Criterion: within-run Spearman with the true distance >= 0.6, survival gain over Base with CI above zero, and at
    least half the gain of the full-model distance d_rls (+0.0079), i.e. >= +0.004.
(2) Time to transition from the physics: T_full = (1 - d_rls) / slope(d_rls), T_L = -sigma_L2 / slope(sigma_L2),
    slopes over the trailing 10 windows (500 time units) of the same run; controls: the parts given separately.
    Criterion: T_full beats Base and its control (d_rls, slope) with paired 95% CI (t4) above zero.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import numpy as np, pandas as pd, sys
from scipy.signal import butter, sosfilt, welch
from turning import b_lim, simulate
DT = 0.02

def run(seed):
    rng = np.random.default_rng(seed); tau = rng.uniform(12, 25); z = 0.02; bl = b_lim(tau, z)
    f0, T = rng.uniform(0.3, 0.6), rng.uniform(3000, 6000); n = int(T / DT)
    b = bl * np.linspace(f0, 1.3, n); tstar = T * (1 - f0) / (1.3 - f0)
    y = np.nan_to_num(simulate(tau, z, b, dt=DT, s_f=rng.uniform(0.01, 0.03), a_r=rng.uniform(0.005, 0.02), rng=rng), nan=0.0)
    xb = sosfilt(butter(2, [0.7 / (2 * np.pi), 1.4 / (2 * np.pi)], btype='band', fs=1 / DT, output='sos'), y)
    lp = butter(2, 0.25 / (2 * np.pi), fs=1 / DT, output='sos'); rows = []; W = int(400 / DT); m = int(10.0 / DT)
    for tw in np.arange(300.0, tstar, 50.0):
        k = int(tw / DT); seg = xb[max(0, k - W):k]; tt = np.arange(max(0, k - W), k) * DT
        f, P = welch(seg, fs=1 / DT, nperseg=min(len(seg), 8192)); w_hat = 2 * np.pi * f[np.argmax(P)]
        A = sosfilt(lp, seg * np.exp(-1j * w_hat * tt))[int(40 / DT):]                  # drop filter transient
        if len(A) <= m + 10: continue
        c = np.vdot(A[:-m], A[m:]) / np.vdot(A[:-m], A[:-m])
        rows.append(dict(run=seed, t=tw, w_hat=w_hat, sigma_L2=float(np.log(abs(c) + 1e-12) / 10.0)))
    return rows

if __name__ == '__main__':
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = []
    for s in range(lo, hi): out += run(s)
    pd.DataFrame(out).to_csv(f'f4_{lo}_{hi}.csv', index=False); print(lo, hi, len(out))

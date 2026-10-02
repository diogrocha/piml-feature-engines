"""Step 3 (pre-registered 30 Sep 2026): recursive, causal state estimators tracking the approach to the Hopf point.
Same runs as fleet2 (same seeds). All filters are causal (no future samples), estimates read at the fleet2 window ends.
 (a) Full model: recursive least squares with forgetting on  x'' + x = -2 zeta x' - b h,  h = x(t) - x(t - tau);
     parameters (zeta, b) tracked along the run; d_rls = b_rls / b_lim(tau, zeta_rls).
 (b) Reduced Landau (Stuart-Landau) model: complex amplitude A of the mode by causal demodulation at omega0 = 1,
     A(t) = exp[(sigma + i dw) L] A(t - L), L = 10, tracked by complex RLS with forgetting; sigma_L -> 0 at the Hopf point.
 Forgetting horizon 400 time units for both. Criterion: the physics filters beat Base with paired 95% CI (t4) above zero,
 and the reduced Landau filter reaches at least half of the full-model gain over Base.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import numpy as np, pandas as pd, sys
from scipy.signal import butter, sosfilt
from scipy.interpolate import RegularGridInterpolator
from turning import b_lim, simulate
DT, DEC, H = 0.02, 5, 400.0
T_ = np.load('blim_table.npz'); BL = RegularGridInterpolator((T_['tg'], T_['zg']), T_['B'], bounds_error=False, fill_value=None)

def rls(Phi, y, lam, p0=100.0):
    n, k = Phi.shape; th = np.zeros(k, dtype=Phi.dtype); P = np.eye(k) * p0; out = np.zeros((n, k), dtype=Phi.dtype)
    for i in range(n):
        ph = Phi[i]; Pph = P @ ph.conj(); g = Pph / (lam + ph @ Pph); th = th + g * (y[i] - ph @ th)
        P = (P - np.outer(g, ph @ P)) / lam; out[i] = th
    return out

def run(seed):
    rng = np.random.default_rng(seed); tau = rng.uniform(12, 25); z = 0.02; bl = b_lim(tau, z)
    f0, T = rng.uniform(0.3, 0.6), rng.uniform(3000, 6000); n = int(T / DT)
    b = bl * np.linspace(f0, 1.3, n); tstar = T * (1 - f0) / (1.3 - f0)
    y = np.nan_to_num(simulate(tau, z, b, dt=DT, s_f=rng.uniform(0.01, 0.03), a_r=rng.uniform(0.005, 0.02), rng=rng), nan=0.0)
    d = int(round(tau / DT)); ys = sosfilt(butter(2, 3.0 / (2 * np.pi), fs=1 / DT, output='sos'), y)    # causal
    v = np.gradient(ys, DT); a = np.gradient(v, DT); h = np.r_[np.zeros(d), ys[d:] - ys[:-d]]
    idx = np.arange(d, n, DEC); dt2 = DT * DEC; lam = 1 - dt2 / H
    th = rls(np.column_stack([v[idx], h[idx]]), -(a[idx] + ys[idx]), lam)                  # [2 zeta, b]
    xb = sosfilt(butter(2, [0.7 / (2 * np.pi), 1.4 / (2 * np.pi)], btype='band', fs=1 / DT, output='sos'), y)
    tt = np.arange(n) * DT; A = sosfilt(butter(2, 0.25 / (2 * np.pi), fs=1 / DT, output='sos'), xb * np.exp(-1j * tt))[::DEC]
    m = int(10.0 / dt2)                                          # lag of 10 time units: >> filter time, << relaxation time
    c = rls(A[:-m, None].astype(complex), A[m:].astype(complex), lam)[:, 0]   # A(t) = c A(t - 10), c = e^{(sigma + i dw) 10}
    sL = np.r_[np.full(m, np.nan + 0j), np.log(np.abs(c) + 1e-12) / 10.0 + 1j * np.angle(c) / 10.0]
    rows = []
    for tw in np.arange(300.0, tstar, 50.0):
        k = int(tw / DT); j = np.searchsorted(idx, k); j2 = min(k // DEC, len(sL) - 1)
        if j >= len(th): break
        zr, br = th[j, 0] / 2, th[j, 1]
        rows.append(dict(run=seed, t=tw, b_rls=br, zeta_rls=zr, d_rls=br / float(BL([[tau, np.clip(zr, 0.002, 0.06)]])[0]),
                         sigma_L=float(sL[j2].real), dw_L=float(sL[j2].imag)))
    return rows

if __name__ == '__main__':
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = []
    for s in range(lo, hi): out += run(s)
    pd.DataFrame(out).to_csv(f'f3_{lo}_{hi}.csv', index=False); print(lo, hi, len(out))

"""Regenerative turning with a slowly ramped depth of cut (dimensionless 1-DOF model).
x'' + 2 zeta x' + x = -b [ h + e2 h^2 + e3 h^3 ] + s_f xi(t) + a_r cos(2 pi t / tau),  h = x(t) - x(t - tau)
Time in units of 1/omega_n. b is the dimensionless depth of cut (k_c b / k). The nonlinear cutting law makes the
Hopf bifurcation subcritical for e3 < 0 (Stepan). Spindle runout gives forced vibration at the spindle frequency.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import numpy as np
from scipy.optimize import fsolve

def b_lim(tau, zeta):
    """smallest b > 0 on the stability boundary for delay tau: -w^2 + 2i zeta w + 1 + b(1 - e^{-i w tau}) = 0."""
    best = np.inf
    for w0 in np.linspace(0.9, 1.6, 60):
        def f(v):
            w, b = v; z = -w * w + 2j * zeta * w + 1 + b * (1 - np.exp(-1j * w * tau)); return [z.real, z.imag]
        (w, b), info, ier, _ = fsolve(f, [w0, 0.05], full_output=True)
        if ier == 1 and b > 1e-6 and w > 0 and abs(np.array(f([w, b]))).max() < 1e-9: best = min(best, b)
    return best

def simulate(tau, zeta, b_path, dt=0.02, s_f=0.02, a_r=0.01, e2=0.0, e3=-2.0, s_m=0.002, rng=None):
    """Euler-Maruyama with a delay buffer. b_path: array of b(t) at each step. Returns measured displacement."""
    rng = rng or np.random.default_rng(); n = len(b_path); d = int(round(tau / dt))
    x = np.zeros(n + d); v = 0.0; out = np.empty(n)
    for k in range(n):
        i = k + d; h = x[i - 1] - x[i - 1 - d] if i - 1 - d >= 0 else 0.0
        F = -b_path[k] * (h + e2 * h * h + e3 * h ** 3) + a_r * np.cos(2 * np.pi * k * dt / tau)
        a = -2 * zeta * v - x[i - 1] + F
        v += a * dt + s_f * np.sqrt(dt) * rng.standard_normal(); x[i] = x[i - 1] + v * dt
        out[k] = x[i] + s_m * rng.standard_normal()
        if abs(x[i]) > 5: out[k:] = np.nan; break
    return out

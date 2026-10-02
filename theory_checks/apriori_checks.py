"""A priori checks of the Landau-Ginzburg feature map.

Part 1 needs no data: collapse of the four descriptors, the reduced form and identifiability, and the
Kramers regime. Part 2 uses one fold of FD001 to verify the consequence of Proposition 1:
[x, s] and [x, eta] give the same concordance, and the potential's parameters
act only through clip saturation.

    python theory_checks/apriori_checks.py            # both parts
    python theory_checks/apriori_checks.py --no-data  # part 1 only
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import argparse
import numpy as np
from scipy.stats import spearmanr
import pipeline as P
from config import LG_PARAMS

b, c, kB = LG_PARAMS['b'], LG_PARAMS['c'], LG_PARAMS['kB_T_eff']
SPIN = 9 * c ** 2 / (32 * b)


def F(psi, eta, b=b, c=c):
    return eta * psi ** 2 - c * np.abs(psi) ** 3 + b * psi ** 4


def barrier(eta, b=b, c=c):
    disc = np.maximum(9 * c ** 2 - 32 * b * eta, 0.0)
    psi_b = (3 * c - np.sqrt(disc)) / (8 * b)
    return np.maximum(F(psi_b, eta, b, c) - F(0.0, eta, b, c), 0.0)


def part1():
    print('=== Collapse of the four descriptors onto eta ===')
    for tag, lo, hi in [('full clip range', 0.001, SPIN * 0.999),
                        ('calibrated range', LG_PARAMS['eta_f_frac'] * SPIN, LG_PARAMS['eta_h_frac'] * SPIN)]:
        eta = np.linspace(lo, hi, 4000)
        B = barrier(eta)
        R = B / kB
        D = np.column_stack([R, B, np.exp(-R), 2 * eta])
        rho = [spearmanr(D[:, j], eta).statistic for j in range(4)]
        print(f'  [{tag}] Spearman with eta {np.round(rho, 6)}  '
              f'max|barrier - kB*R| = {np.abs(B - kB * R).max():.1e}  '
              f'exp(-R) in [{np.exp(-R).min():.4f}, {np.exp(-R).max():.4f}]  R max {R.max():.4f}')
    print('\n=== Reduced form and identifiability ===')
    rng = np.random.RandomState(0)
    bb, cc = rng.uniform(0.5, 2.0, 20000), rng.uniform(1.0, 2.5, 20000)
    eh = rng.uniform(0.05, 0.99, 20000)
    ee = eh * 9 * cc ** 2 / (32 * bb)
    root = np.sqrt(np.maximum(9 * cc ** 2 - 32 * bb * ee, 0))
    ratio = (3 * cc - root) / (3 * cc + root)
    pred = (1 - np.sqrt(1 - eh)) / (1 + np.sqrt(1 - eh))
    print(f'  max|psi_b/psi_min - f(eta_hat)| = {np.abs(ratio - pred).max():.1e}')
    eh = np.linspace(0.05, 0.95, 200)
    sets = [(1.0, 1.749, 1.431), (2.0, 1.749 * 2 ** 0.75, 1.431)]
    expo = [barrier(eh * 9 * c2 ** 2 / (32 * b2), b2, c2) / k2 for b2, c2, k2 in sets]
    kap = [c2 ** 4 / (b2 ** 3 * k2) for b2, c2, k2 in sets]
    print(f'  kappa {np.round(kap, 6)}: max|exponent difference| = {np.abs(expo[0] - expo[1]).max():.1e}')
    rmax = (barrier(np.linspace(0.001, SPIN * 0.999, 5000)) / kB).max()
    print(f'\n=== Kramers regime ===\n  max R = {rmax:.4f}; Kramers requires ~5 '
          f'({5 / rmax:.0f}x above); kB_T needed at peak barrier: {rmax * kB / 5:.4f} (calibrated {kB})')


def part2(nfast):
    from sklearn.model_selection import GroupKFold
    from engines import LGPhysicsEngine
    from crossval import build_configs
    print('\n=== Proposition 1: the tree sees the projection, not the potential (FD001, fold 1) ===')
    d = P.load('FD001')
    tri, tei = list(GroupKFold(n_splits=5).split(d['X_base'], groups=d['units']))[0]
    Xt, Xe = build_configs(d['X_base'][tri], d['X_base'][tei], d['X_stats'][tri], d['X_stats'][tei],
                           d['ttf'][tri], d['cyc'][tri], d['cyc'][tei], P.WIN)['Base']
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xt, d['ttf'][tri])
    alt = dict(LG_PARAMS, b=2.3, c=0.9, kB_T_eff=0.2, eta_h_frac=0.7, eta_f_frac=0.1)
    lg2 = LGPhysicsEngine(**alt)
    lg2.calibrate(Xt, d['ttf'][tri])
    sat = lambda e, L: float(((e <= 0.0011) | (e >= L.spinodal * 0.9989)).mean())
    arms = {'[x, s] raw projection': (Xt @ lg.weights, Xe @ lg.weights),
            '[x, eta] calibrated': (lg.get_eta(Xt), lg.get_eta(Xe)),
            '[x, R] altered parameters': (lg2.transform(Xt)[:, 0], lg2.transform(Xe)[:, 0])}
    idx = np.random.RandomState(42).choice(len(tri), min(nfast, len(tri)), replace=False)
    for nm, (a, q) in arms.items():
        _, cval, _, _ = P.fit_eval('+LG', np.hstack([Xt, a[:, None]])[idx], np.hstack([Xe, q[:, None]]),
                                   d['event'][tri][idx], d['ttf'][tri][idx], d['event'][tei], d['ttf'][tei])
        print(f'  {nm:<28} C = {cval:.4f}')
    print(f'  saturation: calibrated {sat(lg.get_eta(Xt), lg):.1%}, altered {sat(lg2.get_eta(Xt), lg2):.1%}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-data', action='store_true')
    ap.add_argument('--nfast', type=int, default=6000)
    a = ap.parse_args()
    part1()
    if not a.no_data:
        part2(a.nfast)

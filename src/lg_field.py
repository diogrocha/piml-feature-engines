"""Landau-Ginzburg feature engine with a vector order parameter and
trajectory functionals.

Fixes the rank-1 degeneracy of the original four-descriptor block: the
Jacobian of the feature map is no longer an outer product, so the block
cannot be collapsed onto a single scalar.

Two independent extensions:

  (A) field   : one order parameter per sensor family, eta_1 .. eta_m,
                each with its own supervised projection. Only ONE
                descriptor per family is emitted (the reduced barrier R),
                because within a family the four original descriptors are
                strictly monotone functions of each other.

  (B) csd     : causal trajectory functionals of the aggregate eta(t)
                per unit - lag-1 autocorrelation, rolling standard
                deviation and local slope. These are functionals of a
                window, not of the instantaneous eta, so they enlarge the
                generated sigma-algebra. They are the critical-slowing-down
                signature that the Landau-Ginzburg reading predicts.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import numpy as np

from engines import LGPhysicsEngine, _corr_weights

FAMILIES = {
    'temperature': ['T2', 'T24', 'T30', 'T50'],
    'pressure': ['P2', 'P15', 'P30', 'Ps30', 'epr'],
    'speed': ['Nf', 'Nc', 'NRf', 'NRc', 'Nf_dmd', 'PCNfR_dmd'],
    'flow': ['W31', 'W32', 'phi', 'BPR', 'farB', 'htBleed'],
}


def physical_groups(base_cols):
    """Map base feature columns onto sensor families.

    base_cols are names such as 'T24_mean', 'T24_std', 'cycle'. Columns that
    belong to no family (e.g. 'cycle') are left out of every group.
    """
    groups, names = ([], [])
    for fam, sensors in FAMILIES.items():
        idx = [j for j, col in enumerate(base_cols)
               if col.rsplit('_', 1)[0] in sensors]
        if idx:
            groups.append(np.asarray(idx, dtype=int))
            names.append(fam)
    return groups, names


def correlation_groups(X, m=4, seed=42):
    """Fallback partition when column names are unavailable."""
    from sklearn.cluster import KMeans
    C = np.corrcoef(X.T)
    C = np.nan_to_num(C)
    lbl = KMeans(n_clusters=m, n_init=10, random_state=seed).fit_predict(np.abs(C))
    return [np.where(lbl == k)[0] for k in range(m) if (lbl == k).sum() > 0], \
           [f'cluster{k}' for k in range(m) if (lbl == k).sum() > 0]


class LGFieldEngine(LGPhysicsEngine):
    """Vector order parameter: one eta per group, calibrated independently."""

    def __init__(self, groups, group_names=None, calib='affine',
                 calib_pct=(1.0, 99.0), **kw):
        """calib: 'affine'   - original two-point calibration (saturates);
                  'robust'   - map an empirical percentile range of the
                               projection onto [eta_f, eta_h], so that the clip
                               is essentially never active;
                  'quantile' - map the training ECDF of the projection onto
                               [eta_f, eta_h]; saturation is zero by
                               construction, at the cost of a nonlinear
                               (but strictly monotone) calibration."""
        super().__init__(**kw)
        self.groups = [np.asarray(g, dtype=int) for g in groups]
        self.group_names = group_names or [f'g{k}' for k in range(len(groups))]
        self.calib = calib
        self.calib_pct = calib_pct
        self.w_ = None
        self.off_ = None
        self.scl_ = None
        self.knots_ = None

    def calibrate(self, X, ttf):
        self.w_, self.off_, self.scl_, self.knots_ = ([], [], [], [])
        eta_h = self.spinodal * self.eta_h_frac
        eta_f = self.spinodal * self.eta_f_frac
        hi, lo = (ttf > np.percentile(ttf, 75), ttf < np.percentile(ttf, 25))
        for g in self.groups:
            w = _corr_weights(X[:, g], ttf)
            sig = X[:, g] @ w
            s_h = np.median(sig[hi]) if hi.sum() else np.percentile(sig, 25)
            s_f = np.median(sig[lo]) if lo.sum() else np.percentile(sig, 75)
            sgn = 1.0 if s_h >= s_f else -1.0
            knots = None
            if self.calib == 'quantile':
                q = np.linspace(0.0, 100.0, 256)
                knots = (np.percentile(sig, q), q / 100.0, sgn)
                scl = off = 0.0
            elif self.calib == 'robust':
                p_lo, p_hi = np.percentile(sig, self.calib_pct)
                if abs(p_hi - p_lo) > 1e-08:
                    # healthy end of the projection maps to eta_h
                    a, z = (p_lo, p_hi) if sgn > 0 else (p_hi, p_lo)
                    scl = (eta_h - eta_f) / (z - a)
                    off = eta_f - scl * a
                else:
                    scl, off = (0.0, self.spinodal * 0.5)
            elif abs(s_f - s_h) > 1e-08:
                scl = (eta_h - eta_f) / (s_h - s_f)
                off = eta_h - scl * s_h
            else:
                scl, off = (0.0, self.spinodal * 0.5)
            self.w_.append(w)
            self.off_.append(off)
            self.scl_.append(scl)
            self.knots_.append(knots)
        return self

    def _raw_eta(self, X, k):
        g, w = (self.groups[k], self.w_[k])
        sig = X[:, g] @ w
        if self.calib == 'quantile':
            xs, ys, sgn = self.knots_[k]
            u = np.interp(sig, xs, ys)
            if sgn < 0:
                u = 1.0 - u
            eta_h = self.spinodal * self.eta_h_frac
            eta_f = self.spinodal * self.eta_f_frac
            return eta_f + (eta_h - eta_f) * u
        return self.off_[k] + self.scl_[k] * sig

    def eta_field(self, X):
        """(n, m) matrix of per-family order parameters."""
        cols = [np.clip(self._raw_eta(X, k), 0.001, self.spinodal * 0.999)
                for k in range(len(self.groups))]
        return np.column_stack(cols)

    def transform_field(self, X):
        """(n, m) reduced barriers R(eta_k), one per family."""
        E = self.eta_field(X)
        R = np.empty_like(E)
        for k in range(E.shape[1]):
            R[:, k] = [self.compute_R(e) for e in E[:, k]]
        return R

    def saturation_fraction(self, X):
        """Share of samples hitting the clip, per family."""
        cols = []
        for k in range(len(self.groups)):
            e = self._raw_eta(X, k)
            cols.append(((e <= 0.001) | (e >= self.spinodal * 0.999)).mean())
        return np.asarray(cols)


def csd_features(eta, units, cyc, win=30):
    """Causal critical-slowing-down functionals of eta(t), per unit.

    Returns (n, 3): lag-1 autocorrelation, rolling std, local slope of eta,
    each computed on the trailing window only (no future information).
    """
    eta = np.asarray(eta, dtype=float).ravel()
    out = np.zeros((len(eta), 3))
    t = np.arange(win, dtype=float)
    t_c = t - t.mean()
    denom = (t_c ** 2).sum()
    for u in np.unique(units):
        m = np.where(units == u)[0]
        order = m[np.argsort(cyc[m])]
        e = eta[order]
        for i in range(len(e)):
            lo = max(0, i - win + 1)
            wnd = e[lo:i + 1]
            if len(wnd) >= 5 and np.std(wnd) > 1e-12:
                a, b = (wnd[:-1], wnd[1:])
                sa, sb = (np.std(a), np.std(b))
                ac1 = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)) \
                    if sa > 1e-12 and sb > 1e-12 else 0.0
                sd = float(np.std(wnd))
                if len(wnd) == win:
                    slope = float((t_c * (wnd - wnd.mean())).sum() / denom)
                else:
                    tt = np.arange(len(wnd), dtype=float)
                    tc = tt - tt.mean()
                    dn = (tc ** 2).sum()
                    slope = float((tc * (wnd - wnd.mean())).sum() / dn) if dn > 0 else 0.0
            else:
                ac1, sd, slope = (0.0, 0.0, 0.0)
            out[order[i]] = (ac1, sd, slope)
    return out


def jacobian_rank(feature_fn, X, n_probe=64, eps=1e-05, seed=42):
    """Numerical rank of the Jacobian of a feature block, averaged over probes.

    rank 1 means the block factorises through a single scalar and can be
    collapsed without loss. rank > 1 means it cannot.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n_probe, len(X)), replace=False)
    ranks = []
    for i in idx:
        x0 = X[i:i + 1]
        f0 = feature_fn(x0)
        J = np.zeros((f0.shape[1], X.shape[1]))
        for j in range(X.shape[1]):
            xp = x0.copy()
            xp[0, j] += eps
            J[:, j] = ((feature_fn(xp) - f0) / eps).ravel()
        ranks.append(np.linalg.matrix_rank(J, tol=1e-06))
    return float(np.mean(ranks)), int(np.max(ranks))

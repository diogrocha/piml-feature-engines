import sys, numpy as np
from sklearn.decomposition import PCA
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from config import LG_PARAMS, GBSA_AUG
from engines import LGPhysicsEngine, _corr_weights, make_y
import cmapss_data

def desc(d, Xtr, Xall, ttf_tr):
    e = LGPhysicsEngine(**LG_PARAMS)
    d = d / (np.linalg.norm(d) + 1e-12)
    e.weights = d
    s = Xtr @ d
    spin = e.spinodal
    eh, ef = (spin * e.eta_h_frac, spin * e.eta_f_frac)
    hi, lo = (ttf_tr > np.percentile(ttf_tr, 75), ttf_tr < np.percentile(ttf_tr, 25))
    sh, sf = (np.median(s[hi]), np.median(s[lo]))
    if abs(sh - sf) > 1e-09:
        e.eta_scale = (eh - ef) / (sh - sf)
        e.eta_offset = eh - e.eta_scale * sh
    else:
        e.eta_scale, e.eta_offset = (0.0, spin * 0.5)
    return e.transform(Xall)

def pca_dirs(Xtr, K):
    return [c for c in PCA(K).fit(Xtr).components_]

def deflation_dirs(Xtr, ttf_tr, K):
    dirs = []
    for _ in range(K):
        Xr = Xtr - sum(((Xtr @ dd)[:, None] * dd[None, :] for dd in dirs)) if dirs else Xtr
        w = _corr_weights(Xr, ttf_tr)
        d = w.copy()
        for dd in dirs:
            d = d - d @ dd * dd
        n = np.linalg.norm(d)
        if n < 1e-09:
            break
        dirs.append(d / n)
    return dirs

def supervised_dirs(Xtr, ttf_tr, K):
    from numpy.linalg import eigh
    Xc = Xtr - Xtr.mean(0)
    Cxx = np.cov(Xc.T) + 1e-06 * np.eye(Xc.shape[1])
    wv, wc = eigh(Cxx)
    W = wc @ np.diag(1 / np.sqrt(np.maximum(wv, 1e-09))) @ wc.T
    Xw = Xc @ W
    Y = np.column_stack([ttf_tr ** p for p in range(1, K + 1)])
    Yc = (Y - Y.mean(0)) / (Y.std(0) + 1e-09)
    U, S, _ = np.linalg.svd(Xw.T @ Yc / len(Xw), full_matrices=False)
    return [W @ U[:, k] for k in range(K)]

def multimode_features(Xtr, Xall, ttf_tr, dirs):
    return np.hstack([Xtr] + [desc(d, Xtr, Xall, ttf_tr) for d in dirs])

def fit_cindex(Xtr, Xte, ytr, yte, sub=None):
    if sub is not None:
        rng = np.random.RandomState(42)
        si = rng.choice(len(Xtr), sub, replace=False)
        Xtr, ytr = (Xtr[si], ytr[si])
    m = GradientBoostingSurvivalAnalysis(**GBSA_AUG).fit(Xtr, ytr)
    return concordance_index_censored(yte['event'], yte['time'], m.predict(Xte))[0]
if __name__ == '__main__':
    full = '--full' in sys.argv
    sub = None if full else 6000
    d = cmapss_data.load(subset='FD003', verbose=False)
    Xb, ttf, ev, tr, te = (d['X_base'], d['ttf'], d['event'], d['tr'], d['te'])
    mu, sd = (Xb[tr].mean(0), Xb[tr].std(0) + 1e-09)
    Xbz = (Xb - mu) / sd
    Xtr, Xte = (Xbz[tr], Xbz[te])
    ttf_tr = ttf[tr]
    ytr, yte = (make_y(ev[tr], ttf[tr]), make_y(ev[te], ttf[te]))
    lg = LGPhysicsEngine(**LG_PARAMS)
    lg.calibrate(Xtr, ttf_tr)
    scalar_tr = np.hstack([Xtr, lg.transform(Xtr)])
    scalar_te = np.hstack([Xte, lg.transform(Xte)])
    print(f'scalar LG      C={fit_cindex(scalar_tr, scalar_te, ytr, yte, sub):.4f}')
    for K in (2, 3):
        feat_tr = multimode_features(Xtr, Xtr, ttf_tr, pca_dirs(Xtr, K))
        feat_te = multimode_features(Xtr, Xte, ttf_tr, pca_dirs(Xtr, K))
        print(f'multi-LG PCA K={K}         C={fit_cindex(feat_tr, feat_te, ytr, yte, sub):.4f}')
    for K in (2, 3):
        sd = supervised_dirs(Xtr, ttf_tr, K)
        feat_tr = multimode_features(Xtr, Xtr, ttf_tr, sd)
        feat_te = multimode_features(Xtr, Xte, ttf_tr, sd)
        print(f'multi-LG supervised K={K}  C={fit_cindex(feat_tr, feat_te, ytr, yte, sub):.4f}')
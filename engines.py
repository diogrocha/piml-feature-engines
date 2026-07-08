import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.decomposition import PCA
from sksurv.metrics import concordance_index_censored

def make_y(event, ttf):
    return np.array(list(zip(event, ttf)), dtype=[('event', 'bool'), ('time', 'float')])

def ci_score(model, X, y):
    return concordance_index_censored(y['event'], y['time'], model.predict(X))[0]

def _corr_weights(X, ttf):
    nf = X.shape[1]
    corrs = np.array([abs(np.corrcoef(X[:, j], ttf)[0, 1]) if np.std(X[:, j]) > 1e-10 else 0.0 for j in range(nf)])
    corrs = np.nan_to_num(corrs)
    s = corrs.sum()
    return corrs / s if s > 0 else np.ones(nf) / nf

class LGPhysicsEngine:

    def __init__(self, b=1.0, c=1.5, kB_T_eff=0.5, eta_h_frac=0.85, eta_f_frac=0.05):
        self.b, self.c, self.kB_T_eff = (b, c, kB_T_eff)
        self.eta_h_frac, self.eta_f_frac = (eta_h_frac, eta_f_frac)
        self.weights = self.eta_offset = self.eta_scale = None

    def free_energy(self, psi, eta):
        return eta * psi ** 2 - self.c * np.abs(psi) ** 3 + self.b * psi ** 4

    @property
    def spinodal(self):
        return 9 * self.c ** 2 / (32 * self.b)

    def compute_R(self, eta):
        disc = 9 * self.c ** 2 - 32 * self.b * eta
        if eta > 0 and disc > 0:
            psi_b = (3 * self.c - np.sqrt(disc)) / (8 * self.b)
            dF = self.free_energy(psi_b, eta) - self.free_energy(0, eta)
            return max(0, dF) / self.kB_T_eff
        return 0.0

    def calibrate(self, X, ttf):
        self.weights = _corr_weights(X, ttf)
        sig = X @ self.weights
        eta_h = self.spinodal * self.eta_h_frac
        eta_f = self.spinodal * self.eta_f_frac
        hi = ttf > np.percentile(ttf, 75)
        lo = ttf < np.percentile(ttf, 25)
        s_h = np.median(sig[hi]) if hi.sum() > 0 else np.percentile(sig, 25)
        s_f = np.median(sig[lo]) if lo.sum() > 0 else np.percentile(sig, 75)
        if abs(s_f - s_h) > 1e-08:
            self.eta_scale = (eta_h - eta_f) / (s_h - s_f)
            self.eta_offset = eta_h - self.eta_scale * s_h
        else:
            self.eta_scale, self.eta_offset = (0.0, self.spinodal * 0.5)

    def transform(self, X):
        sig = X @ self.weights
        n = len(sig)
        out = np.zeros((n, 4))
        for i in range(n):
            eta = np.clip(self.eta_offset + self.eta_scale * sig[i], 0.001, self.spinodal * 0.999)
            R = self.compute_R(eta)
            disc = 9 * self.c ** 2 - 32 * self.b * eta
            barrier = 0.0
            if disc > 0 and eta > 0:
                psi_b = (3 * self.c - np.sqrt(disc)) / (8 * self.b)
                barrier = max(0, self.free_energy(psi_b, eta) - self.free_energy(0, eta))
            out[i] = [R, barrier, np.exp(-R) if R > 0 else 1.0, 2 * eta]
        return out

    def get_eta(self, X):
        sig = X @ self.weights
        return np.clip(self.eta_offset + self.eta_scale * sig, 0.001, self.spinodal * 0.999)

class WeibullEngine:

    def __init__(self):
        self.beta, self.lam = (2.0, 200.0)

    def calibrate(self, ttf):
        tp = ttf[ttf > 0]
        if len(tp) > 10:
            try:
                sh, _, sc = weibull_min.fit(tp, floc=0)
                self.beta, self.lam = (max(0.5, min(sh, 10.0)), max(1.0, sc))
            except Exception:
                self.beta, self.lam = (2.0, float(np.median(tp)))

    def transform(self, cycles):
        t = np.maximum(cycles, 0.1)
        rel = np.clip(np.exp(-(t / self.lam) ** self.beta), 0, 1)
        haz = np.clip(self.beta / self.lam * (t / self.lam) ** (self.beta - 1), 0, 100)
        return np.column_stack([rel, haz])

class ExpDegEngine:

    def __init__(self, window=30):
        self.window = window
        self.weights = None

    def calibrate(self, X, ttf):
        self.weights = _corr_weights(X, ttf)

    def transform(self, X):
        sig = X @ self.weights
        n = len(sig)
        alpha, level = (np.zeros(n), np.zeros(n))
        for i in range(n):
            s = max(0, i - self.window + 1)
            w = sig[s:i + 1]
            level[i] = w[-1]
            if len(w) > 2:
                d = np.abs(w) + 1e-10
                t = np.arange(len(w), dtype=float)
                try:
                    if np.std(t) > 0 and np.std(np.log(d)) > 0:
                        alpha[i] = np.polyfit(t, np.log(d), 1)[0]
                except Exception:
                    pass
        for arr in (alpha, level):
            sd = np.std(arr)
            if sd > 1e-10:
                arr /= sd
        return np.column_stack([alpha, level])

class EntropyEngine:

    def __init__(self, window=30, n_bins=10, second_feature='rolling_std'):
        self.window, self.n_bins = (window, n_bins)
        self.second_feature = second_feature
        self.weights = None

    def calibrate(self, X, ttf):
        self.weights = _corr_weights(X, ttf)

    def _shannon(self, x):
        if len(x) < 2 or np.std(x) < 1e-10:
            return 0.0
        counts, _ = np.histogram(x, bins=self.n_bins)
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _sample_entropy(self, x, m=2, r_frac=0.2):
        if len(x) < m + 1:
            return 0.0
        r = r_frac * np.std(x)
        if r < 1e-10:
            return 0.0
        n = len(x)

        def count_matches(tl):
            count = 0
            for i in range(n - tl):
                for j in range(i + 1, n - tl):
                    if np.max(np.abs(x[i:i + tl] - x[j:j + tl])) < r:
                        count += 1
            return count
        A = count_matches(m + 1)
        B = count_matches(m)
        if B == 0:
            return 0.0
        return -np.log(A / B) if A > 0 else 0.0

    def transform(self, X):
        sig = X @ self.weights
        n = len(sig)
        sh = np.zeros(n)
        feat2 = np.zeros(n)
        for i in range(n):
            s = max(0, i - self.window + 1)
            w = sig[s:i + 1]
            sh[i] = self._shannon(w)
            if self.second_feature == 'sample_entropy' and len(w) > 6:
                feat2[i] = self._sample_entropy(w[:min(50, len(w))])
        if self.second_feature == 'rolling_std':
            feat2 = pd.Series(sig).rolling(self.window, min_periods=1).std().fillna(0).to_numpy(copy=True)
        sh = np.asarray(sh, dtype=float).copy()
        feat2 = np.asarray(feat2, dtype=float).copy()
        sd_sh, sd_f2 = (np.std(sh), np.std(feat2))
        if sd_sh > 1e-10:
            sh = sh / sd_sh
        if sd_f2 > 1e-10:
            feat2 = feat2 / sd_f2
        return np.column_stack([sh, feat2])

class HIBuilder:

    def __init__(self):
        self.pca = PCA(n_components=1)
        self.sign = 1.0

    def fit(self, X, ttf):
        self.pca.fit(X)
        hi = self.pca.transform(X).ravel()
        self.sign = -1.0 if np.corrcoef(hi, ttf)[0, 1] > 0 else 1.0

    def transform(self, X):
        hi = self.pca.transform(X).ravel() * self.sign
        return np.column_stack([hi, np.gradient(hi)])

def compute_stats(df, sensors, w=30):
    st = pd.DataFrame(index=df.index)
    for s in sensors:
        if s not in df.columns:
            continue
        c = df[s]
        st[f'{s}_skew'] = c.rolling(w, min_periods=1).skew().fillna(0)
        st[f'{s}_kurt'] = c.rolling(w, min_periods=1).kurt().fillna(0)
        mn = c.rolling(w, min_periods=1).min().fillna(c.median())
        mx = c.rolling(w, min_periods=1).max().fillna(c.median())
        st[f'{s}_min'], st[f'{s}_max'] = (mn, mx)
        st[f'{s}_roc'] = c.diff().fillna(0)
        st[f'{s}_range'] = (mx - mn).fillna(0)
    return st
"""The three unsupervised order-parameter channels on C-MAPSS FD003.
Principal directions of the standardised base features (cycle counter excluded); each channel in units of its own
standard deviation, oriented to rise towards failure; per-engine mean in 10-cycle bins of remaining life, then fleet
median and interquartile band.  python mode_channels/plot_mode_channels.py  -> results/fig_mode_loadings.pdf"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import os, numpy as np, matplotlib, warnings; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pipeline as P
d = P.load('FD003'); X = StandardScaler().fit_transform(d['X_base'][:, :-1]); ttf, u = d['ttf'], d['units']
pca = PCA(3).fit(X); S = X @ pca.components_.T; S = (S - S.mean(0)) / S.std(0)
cols = d['base_cols'][:-1]
for k in range(3): print(f'channel {k + 1}:', [cols[i] for i in np.argsort(-np.abs(pca.components_[k]))[:5]])
for k in range(3):
    if S[ttf < np.percentile(ttf, 15), k].mean() < S[ttf > np.percentile(ttf, 70), k].mean(): S[:, k] = -S[:, k]
bins = np.arange(0, 191, 10); ctr = (bins[:-1] + bins[1:]) / 2; E = np.full((len(np.unique(u)), len(ctr), 3), np.nan)
for i, un in enumerate(np.unique(u)):
    for j, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        w = (u == un) & (ttf >= b0) & (ttf < b1)
        if w.sum() >= 3: E[i, j] = S[w].mean(0)
E -= np.nanmedian(E[:, ctr > 160, :], axis=(0, 1))
med, q1, q3 = np.nanmedian(E, 0), np.nanpercentile(E, 25, 0), np.nanpercentile(E, 75, 0); ok = np.sum(np.isfinite(E[:, :, 0]), 0) >= 15
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6})
fig, ax = plt.subplots(figsize=(3.5, 2.5)); ax.axvspan(0, 20, color='#eceef2', lw=0, zorder=0); ax.grid(axis='y', color='#e5e7eb', lw=0.5, zorder=0)
spec = [(0, '#2c2a7a', '-', 2.0, 'core / compressor'), (1, '#6f6cc4', '-', 1.3, 'bypass and bleed'), (2, '#9a98e0', (0, (4, 1.5)), 1.3, 'sensor variability')]
for k, c, ls, lw, lab in spec:
    ax.fill_between(ctr[ok], q1[ok, k], q3[ok, k], color=c, alpha=0.09, lw=0, zorder=1); ax.plot(ctr[ok], med[ok, k], color=c, ls=ls, lw=lw, zorder=3)
ys = {k: med[ok, k][0] for k, *_ in spec}; order = sorted(ys, key=lambda k: -ys[k]); pos = {}
for i, k in enumerate(order): pos[k] = ys[k] if i == 0 else min(ys[k], pos[order[i - 1]] - 0.38)
for k, c, ls, lw, lab in spec: ax.annotate(lab, (ctr[ok][0], med[ok, k][0]), xytext=(9, pos[k]), textcoords=('offset points', 'data'), va='center', ha='left', fontsize=6.3, color=c, annotation_clip=False)
ax.set_xlim(ctr[ok].max() + 2, -2); ax.set_xlabel(r'remaining useful life (cycles) $\rightarrow$ failure', fontsize=7); ax.set_ylabel('order parameter (s.d. units)', fontsize=7)
top = np.nanmax(q3[ok]) * 1.05; ax.set_ylim(-0.6, top); ax.text(10, top * 0.97, 'failure\nzone', ha='center', va='top', fontsize=6, color='#6b7280'); ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.subplots_adjust(right=0.76); os.makedirs('results', exist_ok=True); fig.savefig('results/fig_mode_loadings.pdf'); print('results/fig_mode_loadings.pdf')

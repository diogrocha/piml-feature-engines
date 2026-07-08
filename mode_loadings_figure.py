import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cmapss_data
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.5})
INDIGO, GRAY, STEEL, BLACK, ZONE_C = ('#332288', '#999999', '#4477AA', '#222222', '#DDDDDD')
d = cmapss_data.load(subset='FD003', verbose=False)
z = np.load('/tmp/cmapss_cache/FD003.npz')
Xbt = z['Xbt']
Tf = z['ttf_tr']
dirs = PCA(3).fit(Xbt).components_
S = np.zeros((len(Xbt), 3))
for k in range(3):
    s = Xbt @ dirs[k]
    s = (s - s.mean()) / (s.std() + 1e-09)
    if s[Tf < np.percentile(Tf, 15)].mean() < s[Tf > np.percentile(Tf, 70)].mean():
        s = -s
    S[:, k] = s
bins = np.arange(0, 195, 10)
ctr = (bins[:-1] + bins[1:]) / 2
M = np.full((len(ctr), 3), np.nan)
for j, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
    m = (Tf >= b0) & (Tf < b1)
    if m.sum() >= 20:
        M[j] = S[m].mean(0)
base = np.nanmean(M[-3:], axis=0)
M = M - base
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ZONE = 22
ax.axvspan(0, ZONE, color=ZONE_C, alpha=0.85, lw=0)
ax.text(ZONE / 2, 0.15, 'FAILURE\nZONE', ha='center', va='bottom', fontsize=6.5, color='#555555', weight='bold')
ax.plot(ctr, M[:, 1], color=GRAY, lw=1.2, label='bypass / fan-flow')
ax.plot(ctr, M[:, 2], color=BLACK, lw=1.0, ls='--', alpha=0.9, label='fluctuation (late, CSD)')
ax.plot(ctr, M[:, 0], color=INDIGO, lw=2.4, label='core / compressor drift (dominant)')
ax.set_xlim(ctr.max(), 0)
ax.set_ylim(-0.2, 2.8)
ax.set_xlabel('remaining useful life (cycles)  $\\rightarrow$ failure', fontsize=8)
ax.set_ylabel('degradation signal (fleet mean)', fontsize=8)
ax.tick_params(labelsize=7)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(fontsize=6.2, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 0.99))
plt.tight_layout()
import os
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_mode_loadings.pdf')
plt.close()
print('figura simples (zona de falha) guardada')
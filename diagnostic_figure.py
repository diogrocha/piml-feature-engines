import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import origin_style
import cmapss_data
from engines import LGPhysicsEngine, ExpDegEngine, HIBuilder
from config import LG_PARAMS
d = cmapss_data.load(subset='FD001', verbose=False)
Xb, ttf, cyc, units, tr, te = (d['X_base'], d['ttf'], d['cyc'], d['units'], d['tr'], d['te'])
base_cols = d['base_cols']
lg = LGPhysicsEngine(**LG_PARAMS)
lg.calibrate(Xb[tr], ttf[tr])
ed = ExpDegEngine()
ed.calibrate(Xb[tr], ttf[tr])
hi = HIBuilder()
hi.fit(Xb[tr], ttf[tr])
target = 87
m = units == target
o = np.argsort(cyc[m])
Xe = Xb[m][o]
cyce = cyc[m][o]
n = len(cyce)
x = np.arange(1, n + 1)
print(f'target test engine {int(target)}: {n} cycles')
R, dF, Gam, kap = lg.transform(Xe).T
eta = lg.get_eta(Xe)
spin = lg.spinodal
HIv = hi.transform(Xe)
alpha, level = ed.transform(Xe).T
HItr = hi.transform(Xb[tr])
thr = float(np.percentile(HItr[ttf[tr] <= 30], 50))
w = lg.weights
agg = {}
for name, wi in zip(base_cols, w):
    s = name.replace('_mean', '').replace('_std', '')
    agg[s] = agg.get(s, 0.0) + abs(wi)
rank = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:8]
INK = '#1A1A1A'
BLUE = '#1F4E79'
GREY = '#8A8A8A'
RED = '#8A6D1B'
GHEAL = '#DCEBF5'
GBUF = '#ECECEC'
GCRIT = '#EFE3C5'
Rtr = lg.transform(Xb[tr])[:, 0]
Rmax = float(np.percentile(Rtr, 95))

def zone(r):
    return 0 if r >= 0.6 * Rmax else 1 if r >= 0.2 * Rmax else 2
zc = np.array([zone(r) for r in R])
bands = [GHEAL, GBUF, GCRIT]
znames = ['HEALTHY', 'BUFFER', 'CRITICAL']
zcol = [BLUE, GREY, RED]
fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.0), dpi=300)

def tag(ax, t):
    ax.set_title(t, fontsize=8.6, color=INK, loc='left')

def shade_zones(ax):
    prev = 0
    for i in range(1, n + 1):
        if i == n or zc[i] != zc[i - 1]:
            ax.axvspan(prev + 0.5, i + 0.5, color=bands[zc[i - 1]], lw=0, zorder=0)
            prev = i
ax = axs[0, 0]
ax.plot(x, HIv, color=BLUE, lw=1.8)
ax.axhline(thr, color='#444444', lw=1.2, ls='--')
ax.text(1, thr, ' alarm', color='#444444', fontsize=7, va='bottom')
tag(ax, '(a) HI: scalar health index')
ax.set_ylabel('HI', fontsize=8)
ax = axs[0, 1]
ax.plot(x, level, color=BLUE, lw=1.8)
ax2 = ax.twinx()
ax2.plot(x, alpha, color=GREY, lw=1.3)
ax.set_ylabel('level $D(t)$', fontsize=8, color=BLUE)
ax2.set_ylabel('growth $\\alpha$', fontsize=8, color=GREY)
ax2.tick_params(labelsize=7)
tag(ax, '(b) ExpDeg: level and growth')
ax = axs[0, 2]
shade_zones(ax)
ax.plot(x, R, color=INK, lw=1.9, zorder=3)
ax.set_ylim(0, R.max() * 1.18)
for zi in sorted(set(zc)):
    xs = x[zc == zi]
    ax.text(xs.mean(), R.max() * 1.1, znames[zi], color=zcol[zi], fontsize=6.3, ha='center', va='center')
tag(ax, '(c) LG: robustness $R(t)$ + zones')
ax.set_ylabel('R', fontsize=8)
ax = axs[1, 0]
ax.plot(x, Gam, color=BLUE, lw=1.9)
ax.set_ylim(max(0, Gam.min() - 0.01), 1.003)
ax.text(2, Gam.min(), ' bounded in $(0,1]$', color=GREY, fontsize=6.5, va='bottom')
tag(ax, '(d) LG: transition risk $\\Gamma=e^{-R}$')
ax.set_ylabel('$\\Gamma$', fontsize=8)
ax = axs[1, 1]
ax.plot(x, eta, color=BLUE, lw=1.9)
ax.axhline(spin, color=GREY, ls=':', lw=1.0)
ax.text(1, spin, ' $\\eta_s$', color=GREY, fontsize=7, va='top')
tag(ax, '(e) LG: state $\\eta(t)$ vs spinodal')
ax.set_ylabel('$\\eta$', fontsize=8)
ax = axs[1, 2]
labs = [k for k, _ in rank][::-1]
vals = [v for _, v in rank][::-1]
ax.barh(range(len(labs)), vals, color=BLUE, height=0.7)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels(labs, fontsize=6.8)
tag(ax, '(f) Sensor correlation ranking')
ax.set_xlabel('weight (|corr with TTF|)', fontsize=7.5)
for ax in [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]]:
    ax.set_xlabel('cycle', fontsize=8)
    ax.tick_params(labelsize=7)
plt.tight_layout()
origin_style.bar(axs[1, 2], 'y')
plt.savefig('/mnt/user-data/outputs/fig6.pdf', bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig6.png', dpi=300, bbox_inches='tight')
print('ok')
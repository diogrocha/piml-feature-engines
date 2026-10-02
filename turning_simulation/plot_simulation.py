"""Bar chart of the simulation results from paradigms_sim.json, survival3.json and survival4.json (outputs of paradigms.py, estimators_recursive.py, estimators_landau.py) ."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import t
tc = t.ppf(.975, 4); L = lambda f: {a: np.array(v) for a, v in json.load(open(f)).items()}
P, S3, S4 = L('paradigms_sim.json'), L('survival3.json'), L('survival4.json')
def d(src, a): x = src[a] - src['Base']; h = tc * x.std(ddof=1) / np.sqrt(5) if x.std() > 0 else 0; return x.mean(), bool(h and abs(x.mean()) > h)
G, LG, MM, MMR, HIF, TW, FULL = '#b9c3cf', '#2c2a7a', '#6f6cc4', '#9a98e0', '#4c8a4c', '#8fb98f', '#4a5d73'
rows = [('+Stats', *d(P, '+Stats'), G), ('+HI', *d(P, '+HI'), G), ('+Entropy', *d(P, '+Entropy'), G), ('+ExpDeg', *d(P, '+ExpDeg'), G), ('+Weibull', *d(P, '+Weibull'), G),
        ('+HI, per-unit rate', *d(P, '+HI-fix'), HIF), ('+PCA twin', *d(P, 'PC-twin'), TW), ('+AR(2) damping', *d(P, '+AR-raw'), TW),
        ('+LG', *d(P, '+LG'), LG), ('+LG, 4 descriptors', *d(P, '+LG-4'), LG), ('+LG multi-mode', *d(P, '+LG-mm'), MM), ('+LG multi-mode + rates', *d(P, 'LG-comb-mm'), MMR),
        ('+reduced Landau, $\\hat\\sigma$', *d(S3, '+sigma_L'), MMR), ('+reduced Landau, own freq.', *d(S4, '+sigma_L2'), MMR), ('+physics time to transition', *d(S4, '+T_full'), G),
        ('+full model, windowed $\\hat b$', *d(P, '+b_ls'), FULL), ('+full model, recursive $\\hat b$', *d(S3, '+b_rls'), FULL), ('+full model, distance $\\hat d$', *d(S3, '+d_rls'), FULL),
        ('+full model, $\\hat b,\\hat\\zeta,\\tau$', *d(S3, '+full-twin'), FULL)]
orc = d(P, 'oracle')[0]
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(3.5, 4.2)); y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=.72); ax.axvline(0, color='k', lw=.6)
for yi, r in zip(y, rows):
    v = 0.0 if abs(r[1]) < 5e-5 else r[1]; ax.text((v if v > 0 else 0) + 0.0003, yi, f"{v:+.4f}{'*' if r[2] else ''}" if v else '0.0000', va='center', ha='left', fontsize=5.8)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.3); ax.set_xlim(-0.004, 0.018)
ax.set_xlabel(r'$\Delta C$ vs matched baseline (5 grouped folds)', fontsize=6.8)
ax.text(0.0175, y[-1] - 1.2, f'oracle (true distance): {orc:+.3f}', ha='right', fontsize=6.2, style='italic'); ax.set_ylim(y[-1] - 1.8, y[0] + 0.6)
fig.tight_layout(); fig.savefig('fig_turning_sim.pdf'); print('fig_turning_sim.pdf')
for k, r in zip([x[0] for x in rows], rows): print(f'{k:<34}{r[1]:+.4f}{"*" if r[2] else ""}')
print(f'oracle {orc:+.4f}')

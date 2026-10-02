import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; os.makedirs('results', exist_ok=True)
import origin_style
from matplotlib.patches import FancyArrowPatch
b, c = (1.0, 1.6)
eta_s = 9 * c ** 2 / (32 * b)

def F(p, eta):
    return eta * p ** 2 - c * np.abs(p) ** 3 + b * p ** 4
INK = '#1A1A1A'
BLUE = '#1F4E79'
GREY = '#8A8A8A'
fracs = [0.85, 0.5, 0.1]
stage = ['Healthy', 'Degrading', 'Near failure']
psi = np.linspace(-3.0, 3.0, 800)
fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.1), dpi=300, sharey=True)
plt.subplots_adjust(top=0.74, bottom=0.17, left=0.06, right=0.99, wspace=0.1)
for ax, fr, st in zip(axs, fracs, stage):
    eta = fr * eta_s
    ax.plot(psi, F(psi, eta), color=BLUE, lw=1.7)
    ax.set_title(st, fontsize=11, color=INK, fontweight='bold', pad=14)
    ax.text(0.5, 0.97, f'$\\eta={fr:.2f}\\,\\eta_s$', transform=ax.transAxes, ha='center', va='top', fontsize=8.5, color=GREY)
    ax.set_xlabel('order parameter  $\\psi$', fontsize=8.5)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.8, 2.0)
    ax.tick_params(labelsize=7)
axs[0].set_ylabel('free energy  $F(\\psi,\\eta)$', fontsize=9)
fig.add_artist(FancyArrowPatch((0.13, 0.93), (0.88, 0.93), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=18, color=INK, lw=1.4))
fig.text(0.5, 0.955, 'Degradation: stability margin $\\eta$ decreases', ha='center', fontsize=9.5, color=INK)
plt.savefig('results/fig_landscape.pdf', bbox_inches='tight')
plt.savefig('results/fig_landscape.png', dpi=300, bbox_inches='tight')
print('ok')
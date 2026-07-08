import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})
INK = '#1F2A36'
NEUT = '#9AA6B2'
NEUT2 = '#6E7B89'
ACC = '#2E6CA4'
ACCF = '#D9E6F2'
TINT = '#EFF4F9'
fig, ax = plt.subplots(figsize=(9.3, 2.8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 124)
ax.set_ylim(0, 30)
CY = 18.5
xs = [13, 37, 61, 85, 109]

def arrow(x1, x2, c=NEUT2):
    ax.add_patch(FancyArrowPatch((x1, CY), (x2, CY), arrowstyle='-|>', mutation_scale=11, lw=1.3, color=c, zorder=2, shrinkA=0, shrinkB=0))

def label(cx, t, c=INK, bold=False):
    ax.text(cx, 7.6, t, ha='center', va='center', fontsize=8.6, color=c, weight='bold' if bold else 'normal')
ax.add_patch(FancyBboxPatch((xs[2] - 9.5, 10.5), 19, 15.5, boxstyle='round,pad=0.3,rounding_size=1.4', facecolor=TINT, edgecolor='none', zorder=0))
cx = xs[0]
xx = np.linspace(cx - 5.5, cx + 5.5, 80)
for k, yy in enumerate([CY + 3.4, CY, CY - 3.4]):
    ax.plot(xx, yy + 0.95 * np.sin((xx - cx) * 1.5 + 1.2 * k), color=NEUT2, lw=1.2, zorder=3, solid_capstyle='round')
label(cx, 'Sensor signals')
cx = xs[1]
for i in range(3):
    for j in range(3):
        ax.add_patch(Rectangle((cx - 3.3 + i * 2.3, CY - 3.3 + j * 2.3), 1.7, 1.7, facecolor=NEUT if (i + j) % 2 else 'white', edgecolor=NEUT2, lw=0.8, zorder=3))
label(cx, 'Features')
cx = xs[2]
cw, ch = (12.5, 3.0)
for n, (dx, dy) in enumerate([(-2.4, -3.6), (-1.2, -1.8), (0.0, 0.0)]):
    ax.add_patch(FancyBboxPatch((cx - cw / 2 + dx, CY - ch / 2 + dy - 1.0), cw, ch, boxstyle='round,pad=0.2,rounding_size=0.7', facecolor='white', edgecolor=NEUT2, lw=1.0, zorder=3 + n))
ax.add_patch(FancyBboxPatch((cx - cw / 2 + 1.0, CY + 2.6), cw, ch + 0.4, boxstyle='round,pad=0.2,rounding_size=0.7', facecolor=ACCF, edgecolor=ACC, lw=1.7, zorder=7))
ax.annotate('Landau–Ginzburg', xy=(cx + cw / 2 + 1.0, CY + 4.3), xytext=(cx + 10.5, CY + 7.2), fontsize=7.6, color=ACC, style='italic', ha='left', va='center', zorder=8, arrowprops=dict(arrowstyle='-', color=ACC, lw=0.8))
label(cx, 'Representations', c=INK)
cx = xs[3]

def tree(ox):
    pts = {'r': (ox, CY + 3.4), 'a': (ox - 2.0, CY + 0.2), 'b': (ox + 2.0, CY + 0.2), 'c': (ox - 3.0, CY - 3.0), 'd': (ox - 1.0, CY - 3.0), 'e': (ox + 1.0, CY - 3.0), 'f': (ox + 3.0, CY - 3.0)}
    for u, v in [('r', 'a'), ('r', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'e'), ('b', 'f')]:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=NEUT2, lw=0.9, zorder=3)
    for k, (px, py) in pts.items():
        ax.add_patch(Circle((px, py), 0.62, facecolor='white', edgecolor=NEUT2, lw=0.9, zorder=4))
tree(cx)
label(cx, 'Shared model')
cx = xs[4]
base = CY - 3.6
for k, h in enumerate([4.2, 6.6, 5.2]):
    ax.add_patch(Rectangle((cx - 3.6 + k * 3.0, base), 2.0, h, facecolor=NEUT if k != 1 else ACC, edgecolor='none', zorder=3))
ax.plot([cx - 4.4, cx + 4.4], [base, base], color=NEUT2, lw=1.0, zorder=4)
label(cx, 'Evaluation')
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + 7.0, b - 7.0)
plt.savefig('/mnt/user-data/outputs/fig_pipeline.pdf', bbox_inches='tight')
plt.savefig('/tmp/fig_pipeline_preview.png', dpi=160, bbox_inches='tight')
print('gerado')
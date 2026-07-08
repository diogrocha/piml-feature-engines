import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.5})
INDIGO, STEEL, LGSC, GRAYL, GRAY, BLACK = ('#332288', '#4477AA', '#7A6FB0', '#DDDDDD', '#999999', '#222222')
N = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG', '+LG\n(multi-mode)']
v = [0.822, 0.826, 0.835, 0.833, 0.83, 0.827, 0.827, 0.834]
col = [GRAYL, GRAYL, STEEL, GRAY, GRAY, GRAY, LGSC, INDIGO]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
b = ax.bar(range(len(N)), v, color=col, edgecolor=BLACK, linewidth=0.5, width=0.7)
ax.set_xticks(range(len(N)))
ax.set_xticklabels(N, rotation=30, ha='right', fontsize=7.5)
ax.set_ylabel('test C-index (FD003)', fontsize=8.5)
ax.set_ylim(0.815, 0.84)
ax.tick_params(axis='y', labelsize=7.5)
ax.set_title('C-MAPSS FD003: C-index by feature paradigm (shared GBSA)', fontsize=8.5)
for i, val in enumerate(v):
    ax.text(i, val + 0.0006, f'{val:.3f}', ha='center', fontsize=6.6, color=BLACK)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.grid(axis='y', color='#EEEEEE', lw=0.6, zorder=0)
plt.tight_layout()
import os
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_fd003_bars.pdf')
plt.savefig('/mnt/user-data/outputs/fig_fd003_bars.png', dpi=200)
print('barras FD003 guardadas')
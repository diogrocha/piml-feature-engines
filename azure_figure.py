import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import origin_style
R = json.load(open('/mnt/user-data/outputs/results_azure.json'))
order = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']
comps = [c for c in ['comp1', 'comp2', 'comp3', 'comp4'] if c in R]
avg = {nm: float(np.mean([R[c][nm]['test'] for c in comps if nm in R[c]])) for nm in order}
colors = {'+HI': '#117733', '+LG': '#332288'}
cs = [colors.get(nm, '#DDDDDD' if nm in ('Base', '+Stats') else '#88CCEE') for nm in order]
vals = [avg[nm] for nm in order]
fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=150)
x = np.arange(len(order))
ax.bar(x, vals, color=cs, edgecolor='#222', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=8, rotation=15)
ax.set_ylabel('Mean test C-index (4 components)')
ax.set_ylim(0.82, 0.87)
for i, v in enumerate(vals):
    ax.text(i, v + 0.0008, f'{v:.4f}', ha='center', va='bottom', fontsize=6.5)
ax.set_title('Azure PdM — feature-engineering convergence')
ax.axhline(avg['Base'], ls='--', lw=0.7, color='#888', zorder=0)
plt.tight_layout()
origin_style.bar(ax, 'x')
plt.savefig('/mnt/user-data/outputs/fig_azure.png', dpi=150, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig_azure.pdf', bbox_inches='tight')
print('saved fig_azure.png')
print('\nAveraged test C-index:')
for nm in order:
    print(f'  {nm:<10} {avg[nm]:.4f}')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import origin_style
R = json.load(open('/mnt/user-data/outputs/results_battery.json'))['cv']
order = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']
means = [R[n]['mean'] for n in order]
stds = [R[n]['std'] for n in order]
colors = {'+Weibull': '#44AA99', '+LG': '#332288'}
cs = [colors.get(n, '#DDDDDD' if n in ('Base', '+Stats') else '#88CCEE') for n in order]
fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=150)
x = np.arange(len(order))
ax.bar(x, means, yerr=stds, capsize=3, color=cs, edgecolor='#222', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=8, rotation=15)
ax.set_ylabel('Test C-index (leave-one-battery-out)')
ax.set_ylim(0.88, 0.97)
for i, (mu, sd) in enumerate(zip(means, stds)):
    ax.text(i, mu + sd + 0.001, f'{mu:.3f}', ha='center', va='bottom', fontsize=6.5)
ax.set_title('NASA battery — survival by paradigm (mean ± std)')
origin_style.bar(ax, 'x')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig_battery.png', dpi=150, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig_battery.pdf', bbox_inches='tight')
print('saved fig_battery.png')
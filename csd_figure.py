import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import origin_style
R = json.load(open('/mnt/user-data/outputs/results_cmapss_csd.json'))
order = ['FD001', 'FD003', 'FD002', 'FD004']
labels = ['FD001\n1 cond\n1 fault', 'FD003\n1 cond\n2 faults', 'FD002\n6 cond\n1 fault', 'FD004\n6 cond\n2 faults']
tau = [R[s]['mean_tau'] for s in order]
cs = ['#332288' if R[s]['stationary'] else '#BBBBBB' for s in order]
fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=150)
x = np.arange(len(order))
ax.bar(x, tau, color=cs, edgecolor='#222', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel('mean Kendall $\\tau$ (LG autocorr. vs cycle)')
ax.axhline(0, color='#888', lw=0.6)
for i, s in enumerate(order):
    ax.text(i, tau[i] + 0.012, f'{tau[i]:+.3f}\n{R[s]['frac_rising'] * 100:.0f}% rising', ha='center', va='bottom', fontsize=6.5)
ax.set_ylim(-0.02, 0.68)
ax.set_title('LG critical slowing down: strong only under stationary operation')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#332288', label='stationary'), Patch(color='#BBBBBB', label='multi-condition')], fontsize=7, loc='upper right')
origin_style.bar(ax, 'x')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig_cmapss_csd.png', dpi=150, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/fig_cmapss_csd.pdf', bbox_inches='tight')
print('saved fig_cmapss_csd.png')
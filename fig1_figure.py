import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import origin_style
O = '/mnt/user-data/outputs'
r = json.load(open(O + '/results_cmapss.json'))
N = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']
v = [r[n]['c'] for n in N]
lead = max(range(len(N)), key=lambda i: v[i])
COL = {'Base': '#DDDDDD', '+Stats': '#DDDDDD', '+HI': '#4477AA', '+Weibull': '#999999', '+Entropy': '#999999', '+ExpDeg': '#7A6FB0', '+LG': '#332288'}
cs = [COL[n] for n in N]
fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=300)
ax.bar(range(len(N)), v, color=cs, edgecolor='#222', linewidth=0.5)
ax.set_xticks(range(len(N)))
ax.set_xticklabels(N, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('test C-index')
ax.set_ylim(0.79, 0.84)
ax.set_title('C-MAPSS FD001: C-index by feature paradigm (shared GBSA)')
for i, val in enumerate(v):
    ax.text(i, val + 0.0008, f'{val:.3f}', ha='center', fontsize=7)
plt.tight_layout()
origin_style.bar(ax, 'x')
plt.savefig(O + '/fig1_cmapss_reproduced.png', dpi=300, bbox_inches='tight')
plt.savefig(O + '/fig1_cmapss_reproduced.pdf', bbox_inches='tight')
print('fig1 png+pdf ok')
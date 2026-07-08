import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
INDIGO, STEEL, CYAN, TEAL, SAND, GRAY_L = ('#332288', '#4477AA', '#88CCEE', '#44AA99', '#DDCC77', '#DDDDDD')
BLACK = '#222222'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], 'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.4, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.15})

def _load(path):
    if not os.path.exists(path):
        print(f'  MISSING: {path} — run the matching cross_paradigm_*.py first.')
        return None
    with open(path) as f:
        return json.load(f)

def fig1_cmapss(results, out='figures/fig1_cmapss.png'):
    order = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']
    labels, vals = ([], [])
    for nm in order:
        if nm in results:
            labels.append(f'{nm}\n({results[nm].get('n_feat', '?')})')
            vals.append(results[nm]['c'])
    if not vals:
        return
    colors = []
    for nm in order:
        if nm not in results:
            continue
        colors.append({'+ExpDeg': SAND, '+LG': INDIGO, '+Stats+LG': TEAL}.get(nm, GRAY_L if nm in ('Base', '+Stats') else CYAN))
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.65, color=colors, edgecolor=BLACK, linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel('Test C-index')
    lo = min(vals) - 0.005
    ax.set_ylim(max(0, lo), max(vals) + 0.006)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0004, f'{v:.4f}', ha='center', va='bottom', fontsize=5, color=BLACK)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f'  wrote {out}')

def fig_xjtu(results, out='figures/fig_xjtu.png'):
    fig1_cmapss(results, out=out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmapss', default='results_cmapss.json')
    ap.add_argument('--xjtu', default='results_xjtu.json')
    args = ap.parse_args()
    print('Generating figures from computed results...')
    rc = _load(args.cmapss)
    if rc:
        fig1_cmapss(rc)
    rx = _load(args.xjtu)
    if rx:
        fig_xjtu(rx)
if __name__ == '__main__':
    main()
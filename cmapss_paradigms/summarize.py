"""Summary tables and figures from the cross-validation results.

    python cmapss_paradigms/summarize.py --res results/cv_all.json
Writes results/summary.md, tables.tex and fig_*.pdf.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import argparse
import json
import os
import numpy as np
import pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument('--res', default='results/cv_all.json')
ap.add_argument('--outdir', default='results')
a = ap.parse_args()
res = json.load(open(a.res))
S = [s for s in ['FD001', 'FD003', 'FD002', 'FD004'] if s in res]
L = []
w = L.append


def tex(ci):
    m, lo, hi, n = ci
    if np.isnan(m):
        return 'n/d'
    star = '' if lo <= 0 <= hi else '^{*}'
    return f'${m:+.4f}$ $[{lo:+.4f},{hi:+.4f}]{star}$'


w('# Results\n')
w('## Order parameter minus budget-matched baseline\n')
w('| subset | dC [95% CI] | dIBS [95% CI] |\n|---|---|---|')
for s in S:
    w(f'| {s} | {P.fmt(P.paired(res, s, "+LG-eta", "Base-AUG"))} | {P.fmt(P.paired(res, s, "+LG-eta", "Base-AUG", "IBS"))} |')
w('\n## Collapse: four descriptors minus eta\n')
for s in S:
    w(f'- {s}: C {P.fmt(P.paired(res, s, "+LG", "+LG-eta"), 5)}; IBS {P.fmt(P.paired(res, s, "+LG", "+LG-eta", "IBS"), 5)}')
w('\n## Tree budget: 150 minus 100 trees\n')
for s in S:
    w(f'- {s}: C {P.fmt(P.paired(res, s, "Base-AUG", "Base"))}; IBS {P.fmt(P.paired(res, s, "Base-AUG", "Base", "IBS"))}')
w('\n## Weibull minus matched baseline (expected zero: age-only block)\n')
for s in S:
    dd = P.series(res, s, '+Weibull') - P.series(res, s, 'Base-AUG')
    w(f'- {s}: max |dC| per fold = {np.abs(dd).max():.1e}')
w('\n## LG minus paradigm, C\n')
w('| paradigm | ' + ' | '.join(S) + ' |\n|---|' + '---|' * len(S))
for p in ['+ExpDeg', '+HI', '+Entropy', '+Stats']:
    w(f'| {p} | ' + ' | '.join(P.fmt(P.paired(res, s, '+LG-eta', p)) for s in S) + ' |')
w('\n## Extensions minus scalar, C (supplementary)\n')
for arm in ['+LG-mm2', '+LG-hyb', '+LG-field-rob']:
    w(f'- {arm}: ' + '; '.join(f'{s} {P.fmt(P.paired(res, s, arm, "+LG-eta"))}' for s in S if arm in res[s].get('fold1', {})))
w('- multi-mode minus HI: ' + '; '.join(f'{s} {P.fmt(P.paired(res, s, "+LG-mm2", "+HI"))}' for s in S))
w('\n## Mean concordance by arm\n')
arms = ['+LG-eta', '+LG-mm2', '+ExpDeg', '+HI', '+Entropy', '+Stats', 'Base-AUG', 'Base']
w('| arm | ' + ' | '.join(S) + ' |\n|---|' + '---|' * len(S))
for arm in arms:
    w(f'| {arm} | ' + ' | '.join(f'{np.nanmean(P.series(res, s, arm)):.4f}' for s in S) + ' |')
os.makedirs(a.outdir, exist_ok=True)
open(os.path.join(a.outdir, 'summary.md'), 'w').write('\n'.join(L) + '\n')
T = ['% LG minus baseline', *[f'{s} & {tex(P.paired(res, s, "+LG-eta", "Base-AUG"))} & {tex(P.paired(res, s, "+LG-eta", "Base-AUG", "IBS"))} \\\\' for s in S],
     '% LG minus paradigms', *[f'{p[1:]} & ' + ' & '.join(tex(P.paired(res, s, '+LG-eta', p)) for s in S) + ' \\\\' for p in ['+ExpDeg', '+HI', '+Entropy', '+Stats']]]
open(os.path.join(a.outdir, 'tables.tex'), 'w').write('\n'.join(T) + '\n')
print('\n'.join(L))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for k, (met, tt) in enumerate([('C', '$\\Delta C$ (higher is better)'), ('IBS', '$\\Delta$IBS (lower is better)')]):
    for i, s in enumerate(S):
        m, lo, hi, _ = P.paired(res, s, '+LG-eta', 'Base-AUG', met)
        if np.isnan(m):
            continue
        ax[k].errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt='o', capsize=3, ms=5,
                       color='#2c2a7a' if not lo <= 0 <= hi else '#9a9a9a')
    ax[k].axhline(0, color='k', lw=0.7)
    ax[k].set_xticks(range(len(S)))
    ax[k].set_xticklabels(S)
    ax[k].set_title(tt, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(a.outdir, 'fig_effect.pdf'))
names = {'+LG-eta': 'LG scalar', '+LG-mm2': 'LG multi-mode', '+ExpDeg': 'ExpDeg', '+HI': 'HI',
         '+Entropy': 'Entropy', '+Stats': 'Stats', 'Base-AUG': 'Baseline (matched)'}
cols = {'+LG-eta': '#2c2a7a', '+LG-mm2': '#6f6cc4', 'Base-AUG': '#000000'}
fig, ax = plt.subplots(1, len(S), figsize=(1.85 * len(S), 2.9), squeeze=False)
for j, s in enumerate(S):
    arms7 = [x for x in names if not np.isnan(np.nanmean(P.series(res, s, x)))]
    order = sorted(arms7, key=lambda x: np.nanmean(P.series(res, s, x)))
    for i, x in enumerate(order):
        ax[0, j].barh(i, np.nanmean(P.series(res, s, x)) - 0.82, left=0.82, color=cols.get(x, '#b9c3cf'), height=0.7)
    ax[0, j].set_yticks(range(len(order)))
    ax[0, j].set_yticklabels([names[x] for x in order], fontsize=7)
    ax[0, j].set_xlim(0.82, 0.86)
    ax[0, j].set_title(s, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(a.outdir, 'fig_paradigms.pdf'))
print('\nwrote', a.outdir)

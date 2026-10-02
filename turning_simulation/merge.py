"""Merge the three fleet passes into the tables used by estimators_recursive.py and estimators_landau.py.
f2_*.csv (windows) + f3_*.csv (recursive estimators) -> f23.csv; + f4_*.csv (reduced Landau at own frequency,
per-run slopes over the trailing 10 windows, physics time to transition) -> f234.csv. Causal: slopes use past windows only."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import glob, numpy as np, pandas as pd
rd = lambda p: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(p))], ignore_index=True).sort_values('t')
f2, f3, f4 = rd('f2_*.csv'), rd('f3_*.csv'), rd('f4_*.csv')
m = pd.merge_asof(f2, f3, on='t', by='run', direction='backward', tolerance=60.0); m.to_csv('f23.csv', index=False)
m = pd.merge_asof(m.sort_values('t'), f4, on='t', by='run', direction='backward', tolerance=60.0).sort_values(['run', 't']).reset_index(drop=True)
def slope(s, w=10):
    x = np.arange(len(s)); out = np.full(len(s), np.nan)
    for i in range(len(s)):
        j = max(0, i - w + 1)
        if i - j >= 3: out[i] = np.polyfit(x[j:i + 1], s.values[j:i + 1], 1)[0]
    return out
m['sl_d'] = m.groupby('run')['d_rls'].transform(slope); m['sl_s'] = m.groupby('run')['sigma_L2'].transform(slope)
m['T_full'] = np.log(np.clip((1 - m.d_rls) / np.clip(m.sl_d, 1e-6, None), 1e-3, 1e6))
m['T_L'] = np.log(np.clip(-m.sigma_L2 / np.clip(m.sl_s, 1e-7, None), 1e-3, 1e6))
m.to_csv('f234.csv', index=False); print('f23.csv e f234.csv:', len(m), 'janelas')

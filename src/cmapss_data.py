import os
import numpy as np
import pandas as pd
from config import ROLLING_WINDOW_CMAPSS
from engines import compute_stats
SENSOR_NAMES = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
OP_NAMES = ['op1', 'op2', 'op3']

def find_cmapss_dir(base='.'):
    needed = 'train_FD001.txt'
    for d in ['CMAPSSData', 'data/CMAPSSData', 'data', '.', '..', '/content/CMAPSSData', '/content/data', '/content']:
        if os.path.exists(os.path.join(d, needed)):
            return d
    for root, _, files in os.walk(base):
        if needed in files:
            return root
    return None

def load(data_dir=None, subset='FD001', condition_normalize=False, verbose=True):
    data_dir = data_dir or find_cmapss_dir()
    if not data_dir:
        return None
    cols = ['unit', 'cycle'] + OP_NAMES + SENSOR_NAMES
    df = pd.read_csv(os.path.join(data_dir, f'train_{subset}.txt'), sep='\\s+', header=None, names=cols)
    mx = df.groupby('unit')['cycle'].max()
    df['ttf'] = df.apply(lambda r: mx[r['unit']] - r['cycle'], axis=1)
    if condition_normalize:
        from sklearn.cluster import KMeans
        lbl = KMeans(n_clusters=6, n_init=10, random_state=42).fit_predict(df[OP_NAMES].values)
        df['cond'] = lbl
        for s in SENSOR_NAMES:
            g = df.groupby('cond')[s]
            df[s] = ((df[s] - g.transform('mean')) / (g.transform('std') + 1e-09)).fillna(0)
    useful = df[SENSOR_NAMES].std()[lambda s: s > 0.001].index.tolist()
    w = ROLLING_WINDOW_CMAPSS
    bfn = []
    for s in useful:
        df[f'{s}_mean'] = df.groupby('unit')[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'{s}_std'] = df.groupby('unit')[s].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        bfn.extend([f'{s}_mean', f'{s}_std'])
    bfn.append('cycle')
    sdf = pd.concat([compute_stats(df[df['unit'] == u].copy(), useful, w=w) for u in sorted(df['unit'].unique())])
    scols = list(sdf.columns)
    Xb = df[bfn].values.astype(float)
    Xs = sdf[scols].values.astype(float)
    ttf = df['ttf'].values.astype(float) + 1
    cyc = df['cycle'].values.astype(float)
    units = df['unit'].values
    event = np.ones(len(df), dtype=bool)
    v = np.all(np.isfinite(Xb), 1) & np.all(np.isfinite(Xs), 1) & (ttf > 0)
    Xb, Xs, ttf, cyc, units, event = (Xb[v], Xs[v], ttf[v], cyc[v], units[v], event[v])
    engines = sorted(df['unit'].unique())
    n_tr = int(round(0.8 * len(engines)))
    tr = np.isin(units, engines[:n_tr])
    te = np.isin(units, engines[n_tr:])
    if verbose:
        print(f'  C-MAPSS {subset}: {len(Xb)} samples, {len(useful)} sensors, base={Xb.shape[1]} feats, train={tr.sum()} / test={te.sum()} ({n_tr} train / {len(engines) - n_tr} test engines)')
    return {'X_base': Xb, 'X_stats': Xs, 'ttf': ttf, 'event': event, 'cyc': cyc, 'units': units, 'tr': tr, 'te': te, 'useful': useful, 'base_cols': bfn, 'stat_cols': scols}
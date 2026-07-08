import os
import numpy as np
import pandas as pd
from engines import compute_stats
SENSORS = ['volt', 'rotate', 'pressure', 'vibration']
COMPONENTS = ['comp1', 'comp2', 'comp3', 'comp4']
ROLLING_WINDOW = 24
PAPER_MACHINES = {'comp1': 5, 'comp2': 81, 'comp3': 16, 'comp4': 37}

def find_azure_dir(base='.'):
    needed = 'PdM_telemetry.csv'
    for d in ['data/azure', 'data', '.', '..', '/content/data', '/content/data/azure', '/content', '/tmp/azure_in']:
        if os.path.exists(os.path.join(d, needed)):
            return d
    for root, _, files in os.walk(base):
        if needed in files:
            return root
    return None

def _load(data_dir):
    tel = pd.read_csv(os.path.join(data_dir, 'PdM_telemetry.csv'))
    mnt = pd.read_csv(os.path.join(data_dir, 'PdM_maint.csv'))
    fail = pd.read_csv(os.path.join(data_dir, 'PdM_failures.csv'))
    for df in (tel, mnt, fail):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    return (tel, mnt, fail)

def build_component_subset(data_dir, comp, verbose=True):
    tel, mnt, fail = _load(data_dir)
    fc = fail[fail['failure'] == comp].groupby('machineID').size().sort_values(ascending=False)
    if len(fc) == 0 or fc.iloc[0] < 2:
        return None
    mid = int(fc.index[0])
    if verbose:
        flag = '' if mid == PAPER_MACHINES.get(comp) else f'  (paper used #{PAPER_MACHINES.get(comp)})'
        print(f'  {comp}: machine #{mid} ({int(fc.iloc[0])} failures){flag}')
    tm = tel[tel['machineID'] == mid].copy().sort_values('datetime').reset_index(drop=True)
    mm, fm = (mnt[mnt['machineID'] == mid], fail[fail['machineID'] == mid])
    reps = {}
    for c in COMPONENTS:
        cm = mm[mm['comp'] == c]['datetime'].tolist()
        cf = fm[fm['failure'] == c]['datetime'].tolist()
        reps[c] = sorted(set(cm + cf))
    dt_arr = tm['datetime'].to_numpy()
    n = len(tm)
    for c in COMPONENTS:
        evts = np.array(reps[c], dtype='datetime64[ns]')
        cnt = np.zeros(n, dtype=int)
        hrs = np.zeros(n, dtype=float)
        k, ei = (0, 0)
        for ri in range(n):
            while ei < len(evts) and evts[ei] <= dt_arr[ri]:
                k = 0
                ei += 1
            cnt[ri] = k
            hrs[ri] = float(k)
            k += 1
        tm[f'{c}_cycle'] = cnt
        tm[f'{c}_hours'] = hrs
    for s in SENSORS:
        tm[f'{s}_mean'] = tm[s].rolling(ROLLING_WINDOW, min_periods=1).mean()
        tm[f'{s}_std'] = tm[s].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    sdf = compute_stats(tm, SENSORS, w=ROLLING_WINDOW)
    base_cols = [f'{s}_{t}' for s in SENSORS for t in ('mean', 'std')] + [f'{c}_{t}' for c in COMPONENTS for t in ('cycle', 'hours')]
    stat_cols = list(sdf.columns)
    cev = np.array(reps[comp], dtype='datetime64[ns]')
    ftimes = set(fm[fm['failure'] == comp]['datetime'].tolist())
    dts = tm['datetime'].values
    bd = tm[base_cols].values.astype(float)
    sd = sdf[stat_cols].values.astype(float)
    oh = tm[f'{comp}_hours'].values.astype(float)
    mdt = pd.Timestamp(dts.max())
    xb, xs, ttfs, evs, ohs = ([], [], [], [], [])
    for ri in range(n):
        dt = pd.Timestamp(dts[ri])
        nxt, isf = (None, False)
        nxt_arr = cev[cev > dts[ri]]
        if nxt_arr.size:
            nxt = pd.Timestamp(nxt_arr[0])
            isf = nxt in ftimes
        if nxt is not None:
            th = max(1.0, (nxt - dt).total_seconds() / 3600.0)
        else:
            th = (mdt - dt).total_seconds() / 3600.0 + 1
            isf = False
        if th > 0:
            xb.append(bd[ri])
            xs.append(sd[ri])
            ttfs.append(th)
            evs.append(isf)
            ohs.append(oh[ri])
    Xb, Xs = (np.array(xb), np.array(xs))
    ttf, event, oph = (np.array(ttfs), np.array(evs), np.array(ohs))
    v = (ttf > 0) & np.all(np.isfinite(Xb), 1) & np.all(np.isfinite(Xs), 1)
    Xb, Xs, ttf, event, oph = (Xb[v], Xs[v], ttf[v], event[v], oph[v])
    n_fail = int(event.sum())
    if verbose:
        print(f'    {len(Xb)} samples, {n_fail} failures, base={Xb.shape[1]}, stats={Xs.shape[1]}')
    if n_fail < 3:
        return None
    return {'component': comp, 'machine': mid, 'X_base': Xb, 'X_stats': Xs, 'ttf': ttf, 'event': event, 'op_hours': oph, 'base_cols': base_cols, 'stat_cols': stat_cols, 'sensors': SENSORS}
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from engines import compute_stats
RATED_AH = 2.0
EOL_AH = 1.4
ROLLING_WINDOW = 5
BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']
DEFAULT_TEST = {'B0018'}
FEAT_NAMES = ['cap', 'v_mean', 'v_min', 't_mean', 't_max', 'dur', 'i_mean', 't_to_3v']

def find_battery_dir(base='.'):
    for d in ['data/battery', 'dataset', 'NASA_bat/dataset', '/tmp/NASA_bat/dataset', '.', '/content/dataset', '/content']:
        if all((os.path.exists(os.path.join(d, b + '.mat')) for b in BATTERIES)):
            return d
    for root, _, files in os.walk(base):
        if all((b + '.mat' in files for b in BATTERIES)):
            return root
    return None

def _cycle_features(d):
    v = np.asarray(d['Voltage_measured']).ravel()
    i = np.asarray(d['Current_measured']).ravel()
    t = np.asarray(d['Temperature_measured']).ravel()
    tm = np.asarray(d['Time']).ravel()
    cap = float(np.asarray(d['Capacity']).ravel()[0])
    below = np.where(v < 3.0)[0]
    t_to_3v = float(tm[below[0]]) if below.size else float(tm[-1])
    return ([cap, float(v.mean()), float(v.min()), float(t.mean()), float(t.max()), float(tm.max()), float(np.abs(i).mean()), t_to_3v], cap)

def _discharge_cycles(matfile, name):
    m = loadmat(matfile)
    cyc = m[name][0, 0]['cycle'][0]
    feats, caps = ([], [])
    for c in cyc:
        if c['type'][0] == 'discharge':
            try:
                f, cap = _cycle_features(c['data'][0, 0])
                if np.isfinite(cap) and cap > 0:
                    feats.append(f)
                    caps.append(cap)
            except Exception:
                pass
    return (np.asarray(feats, float), np.asarray(caps, float))

def load(data_dir=None, test_bearings=None, verbose=True):
    data_dir = data_dir or find_battery_dir()
    if not data_dir:
        return None
    test_set = set(test_bearings) if test_bearings else DEFAULT_TEST
    rows, ttf, cyc, units, event = ([], [], [], [], [])
    for name in BATTERIES:
        mf = os.path.join(data_dir, name + '.mat')
        if not os.path.exists(mf):
            continue
        F, caps = _discharge_cycles(mf, name)
        n = len(caps)
        if n < 10:
            continue
        cap_s = pd.Series(caps).rolling(3, min_periods=1, center=True).mean().to_numpy()
        eol_idx = np.where(cap_s <= EOL_AH)[0]
        reaches = eol_idx.size > 0
        E = int(eol_idx[0]) if reaches else n - 1
        for i in range(E + 1 if reaches else n):
            rows.append(F[i])
            ttf.append(float(E - i))
            cyc.append(float(i + 1))
            units.append(name)
            event.append(bool(reaches))
        if verbose:
            tag = f'EOL@{E}' if reaches else 'censored (no EOL)'
            print(f'  {name}: {n} cycles, cap {caps[0]:.3f}->{caps[-1]:.3f} Ah, {tag}')
    if not rows:
        return None
    Xb = np.asarray(rows, float)
    ttf = np.asarray(ttf, float)
    cyc = np.asarray(cyc, float)
    units = np.asarray(units)
    event = np.asarray(event, bool)
    w = ROLLING_WINDOW
    base = pd.DataFrame(Xb, columns=FEAT_NAMES)
    base['unit'] = units
    stat_cols = []
    for c in FEAT_NAMES:
        base[f'{c}_rm'] = base.groupby('unit')[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
        base[f'{c}_rs'] = base.groupby('unit')[c].transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
        stat_cols += [f'{c}_rm', f'{c}_rs']
    Xs = base[stat_cols].to_numpy(float)
    v = np.all(np.isfinite(Xb), 1) & np.all(np.isfinite(Xs), 1) & (ttf >= 0)
    Xb, Xs, ttf, cyc, units, event = (Xb[v], Xs[v], ttf[v], cyc[v], units[v], event[v])
    te = np.isin(units, list(test_set))
    tr = ~te
    if verbose:
        print(f'  NASA-battery: {len(Xb)} samples, base={Xb.shape[1]} feats, events={int(event.sum())}/{len(event)} (censored={int((~event).sum())}), train={tr.sum()} / test={te.sum()} ({len(np.unique(units[tr]))} train / {len(np.unique(units[te]))} test batteries)')
    return {'X_base': Xb, 'X_stats': Xs, 'ttf': ttf + 1, 'event': event, 'cyc': cyc, 'op_hours': cyc, 'units': units, 'tr': tr, 'te': te, 'base_cols': FEAT_NAMES, 'stat_cols': stat_cols, 'useful': FEAT_NAMES}
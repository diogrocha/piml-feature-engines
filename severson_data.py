import os, re, glob
import numpy as np
import pandas as pd
from engines import compute_stats
SPLITS = ['train', 'test1', 'test2']
FEAT_NAMES = ['cap', 'qd_mean', 'qd_min', 'qd_lowV', 'dq_mean', 'dq_logvar', 'dq_min']
ROLLING_WINDOW = 5

def _idx(fn):
    return int(re.findall('cell(\\d+)', fn)[0])

def load(data_dir, cycle_step=5, verbose=True):
    rows, ttf, cyc, units, event, splitcol = ([], [], [], [], [], [])
    for split in SPLITS:
        clf = os.path.join(data_dir, 'cycle_lives', f'{split}_cycle_lives.csv')
        if not os.path.exists(clf):
            continue
        cl = np.atleast_1d(np.loadtxt(clf))
        files = sorted(glob.glob(os.path.join(data_dir, split, 'cell*.csv')), key=lambda p: _idx(os.path.basename(p)))
        for k, f in enumerate(files):
            Q = np.loadtxt(f, delimiter=',')
            if Q.ndim == 1:
                Q = Q[:, None]
            ncyc = Q.shape[1]
            if k >= len(cl):
                break
            life = float(cl[k])
            ref = Q[:, 0]
            name = f'{split}_c{_idx(os.path.basename(f))}'
            for j in range(0, ncyc, cycle_step):
                q = Q[:, j]
                cap = np.nanmax(q)
                if not np.isfinite(cap) or cap <= 0:
                    continue
                dq = q - ref
                feats = [cap, float(np.nanmean(q)), float(np.nanmin(q)), float(q[100]), float(np.nanmean(dq)), float(np.log10(np.nanvar(dq) + 1e-12)), float(np.nanmin(dq))]
                cynum = j + 2
                t = life - cynum
                if t <= 0:
                    continue
                rows.append(feats)
                ttf.append(t)
                cyc.append(float(cynum))
                units.append(name)
                event.append(True)
                splitcol.append(split)
    if not rows:
        return None
    Xb = np.asarray(rows, float)
    ttf = np.asarray(ttf, float)
    cyc = np.asarray(cyc, float)
    units = np.asarray(units)
    event = np.asarray(event, bool)
    splitcol = np.asarray(splitcol)
    w = ROLLING_WINDOW
    base = pd.DataFrame(Xb, columns=FEAT_NAMES)
    base['unit'] = units
    stat_cols = []
    for c in FEAT_NAMES:
        base[f'{c}_rm'] = base.groupby('unit')[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
        base[f'{c}_rs'] = base.groupby('unit')[c].transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
        stat_cols += [f'{c}_rm', f'{c}_rs']
    Xs = base[stat_cols].to_numpy(float)
    v = np.all(np.isfinite(Xb), 1) & np.all(np.isfinite(Xs), 1) & (ttf > 0)
    Xb, Xs, ttf, cyc, units, event, splitcol = (Xb[v], Xs[v], ttf[v], cyc[v], units[v], event[v], splitcol[v])
    tr = splitcol == 'train'
    te = ~tr
    if verbose:
        print(f'  Severson: {len(Xb)} samples from {len(np.unique(units))} cells, base={Xb.shape[1]} feats, train={tr.sum()} ({len(np.unique(units[tr]))} cells) / test={te.sum()} ({len(np.unique(units[te]))} cells), cycle_step={cycle_step}')
    return {'X_base': Xb, 'X_stats': Xs, 'ttf': ttf, 'event': event, 'cyc': cyc, 'op_hours': cyc, 'units': units, 'tr': tr, 'te': te, 'base_cols': FEAT_NAMES, 'stat_cols': stat_cols, 'useful': FEAT_NAMES}
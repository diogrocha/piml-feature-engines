import os
import re
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
from config import ROLLING_WINDOW_XJTU
FS = 25600.0
CONDITIONS = {'35Hz12kN': 1, '37.5Hz11kN': 2, '40Hz10kN': 3}
DEFAULT_TEST = {'Bearing1_4', 'Bearing1_5', 'Bearing2_4', 'Bearing2_5', 'Bearing3_4', 'Bearing3_5'}

def _time_features(x):
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    absmean = np.mean(np.abs(x)) + 1e-12
    peak = np.max(np.abs(x))
    sq = np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-12
    return [rms, np.std(x), kurtosis(x, fisher=True), skew(x), peak, np.max(x) - np.min(x), peak / rms, peak / absmean, rms / absmean, peak / sq]

def _freq_features(x, fs=FS):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = mag ** 2
    tot = psd.sum() + 1e-12
    edges = np.linspace(0, fs / 2, 6)
    bands = [psd[(freqs >= edges[i]) & (freqs < edges[i + 1])].sum() / tot for i in range(5)]
    s_kurt = float(kurtosis(mag, fisher=True))
    p = psd / tot
    p = p[p > 0]
    s_ent = float(-np.sum(p * np.log(p)) / np.log(len(p) + 1e-12)) if len(p) else 0.0
    env = np.abs(hilbert(x))
    env = env - env.mean()
    emag = np.abs(np.fft.rfft(env * win))
    epsd = emag ** 2
    etot = epsd.sum() + 1e-12
    eedges = [0, 250, 500, 1000]
    ebands = [epsd[(freqs >= eedges[i]) & (freqs < eedges[i + 1])].sum() / etot for i in range(3)]
    return bands + [s_kurt, s_ent] + ebands
_TNAMES = ['rms', 'std', 'kurt', 'skew', 'peak', 'p2p', 'crest', 'impulse', 'shape', 'clearance']
_FNAMES = ['sb0', 'sb1', 'sb2', 'sb3', 'sb4', 'skurt', 'sent', 'eb0', 'eb1', 'eb2']
FEAT_NAMES = [f'H_{n}' for n in _TNAMES] + [f'V_{n}' for n in _TNAMES] + [f'H_{n}' for n in _FNAMES] + [f'V_{n}' for n in _FNAMES]

def find_xjtu_dir(base='.'):
    for d in ['XJTU-SY', 'XJTU-SY_Bearing_Datasets', 'data/XJTU-SY', 'data', '.', '/content/XJTU-SY', '/content', '/content/xjtu_extracted']:
        if any((os.path.isdir(os.path.join(d, c)) for c in CONDITIONS)):
            return d
    for root, dirs, _ in os.walk(base):
        if any((c in dirs for c in CONDITIONS)):
            return root
    return None

def _csv_order(fn):
    m = re.match('(\\d+)\\.csv$', fn)
    return int(m.group(1)) if m else 1 << 30

def load(data_dir=None, test_bearings=None, max_snapshots=None, verbose=True):
    data_dir = data_dir or find_xjtu_dir()
    if not data_dir:
        return None
    test_bearings = set(test_bearings) if test_bearings else DEFAULT_TEST
    rows, ttf, cyc, units = ([], [], [], [])
    for cdir in CONDITIONS:
        cpath = os.path.join(data_dir, cdir)
        if not os.path.isdir(cpath):
            continue
        for bearing in sorted(os.listdir(cpath)):
            bpath = os.path.join(cpath, bearing)
            if not os.path.isdir(bpath):
                continue
            files = sorted([f for f in os.listdir(bpath) if f.endswith('.csv')], key=_csv_order)
            if max_snapshots:
                files = files[:max_snapshots]
            n = len(files)
            if n < 5:
                continue
            for i, f in enumerate(files):
                df = pd.read_csv(os.path.join(bpath, f))
                h = df.iloc[:, 0].to_numpy(float)
                v = df.iloc[:, 1].to_numpy(float)
                rows.append(_time_features(h) + _time_features(v) + _freq_features(h) + _freq_features(v))
                ttf.append(n - i)
                cyc.append(i + 1)
                units.append(bearing)
            if verbose:
                print(f'  {bearing}: {n} snapshots')
    if not rows:
        return None
    Xb = np.asarray(rows, float)
    ttf = np.asarray(ttf, float)
    cyc = np.asarray(cyc, float)
    units = np.asarray(units)
    w = ROLLING_WINDOW_XJTU
    base = pd.DataFrame(Xb, columns=FEAT_NAMES)
    base['unit'] = units
    stat_cols = []
    for c in FEAT_NAMES:
        base[f'{c}_rm'] = base.groupby('unit')[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
        base[f'{c}_rs'] = base.groupby('unit')[c].transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
        stat_cols += [f'{c}_rm', f'{c}_rs']
    Xs = base[stat_cols].to_numpy(float)
    event = np.ones(len(Xb), dtype=bool)
    v = np.all(np.isfinite(Xb), 1) & np.all(np.isfinite(Xs), 1) & (ttf > 0)
    Xb, Xs, ttf, cyc, units, event = (Xb[v], Xs[v], ttf[v], cyc[v], units[v], event[v])
    te = np.isin(units, list(test_bearings))
    tr = ~te
    if verbose:
        print(f'  XJTU-SY: {len(Xb)} snapshots, base={Xb.shape[1]} feats (20 time + 20 freq), train={tr.sum()} / test={te.sum()} ({len(np.unique(units[tr]))} train / {len(np.unique(units[te]))} test bearings)')
    return {'X_base': Xb, 'X_stats': Xs, 'ttf': ttf + 1, 'event': event, 'cyc': cyc, 'units': units, 'tr': tr, 'te': te, 'useful': FEAT_NAMES, 'base_cols': FEAT_NAMES, 'stat_cols': stat_cols}
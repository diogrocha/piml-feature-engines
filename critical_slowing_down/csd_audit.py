"""Audit of the critical-slowing-down test with a memoryless null model.

    python critical_slowing_down/csd_audit.py FD001 FD003
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.stats import kendalltau
from scipy.ndimage import gaussian_filter1d
import pipeline as P  # sets sys.path and DATA_DIR
from config import LG_PARAMS
from engines import LGPhysicsEngine
import cmapss_data
WIN=20; RNG=np.random.RandomState(0)
cols=['unit','cycle','op1','op2','op3']+[f's{i}' for i in range(1,22)]

def ac1_series(s, detrend=False):
    out=[]
    for i in range(WIN, len(s)):
        w=s[i-WIN:i].astype(float)
        if detrend:
            t=np.arange(WIN); w=w-np.polyval(np.polyfit(t,w,1),t)
        out.append(np.corrcoef(w[:-1],w[1:])[0,1] if np.std(w)>1e-9 else np.nan)
    return np.array(out)

def tau_of(series_by_unit, detrend=False):
    taus=[]
    for s in series_by_unit:
        if len(s)<WIN+10: continue
        ac=ac1_series(s,detrend); ok=np.isfinite(ac)
        if ok.sum()>8: taus.append(kendalltau(np.arange(len(ac))[ok],ac[ok])[0])
    taus=np.array(taus); return taus.mean(), (taus>0).mean()

def run(sub):
    d=cmapss_data.load(P.DATA_DIR,subset=sub,verbose=False)
    lg=LGPhysicsEngine(**LG_PARAMS); lg.calibrate(d['X_base'],d['ttf'])
    sig=d['X_base']@lg.weights
    A=[sig[d['units']==u][np.argsort(d['cyc'][d['units']==u])] for u in np.unique(d['units'])]
    df=pd.read_csv(f'{P.DATA_DIR}/train_{sub}.txt',sep=r'\s+',header=None,names=cols)
    df['ttf']=df['unit'].map(df.groupby('unit')['cycle'].max())-df['cycle']+1
    sens=[c for c in cols if c.startswith('s') and df[c].std()>1e-3]
    S=(df[sens]-df[sens].mean())/df[sens].std()
    w=np.abs([np.corrcoef(S[c],df['ttf'])[0,1] for c in sens]); w/=w.sum()
    df['z']=S.values@w
    R,Rd,N=[],[],[]
    for u,g in df.sort_values(['unit','cycle']).groupby('unit'):
        z=g['z'].values; R.append(z)
        trend=gaussian_filter1d(z, sigma=max(3,0.1*len(z)), mode='nearest')
        Rd.append(z-trend)                                   # Dakos et al. (2012): detrended residuals
        res=(z-trend)[RNG.permutation(len(z))]               # null: same trend, memoryless residuals
        N.append(pd.Series(trend+res).rolling(30,min_periods=1).mean().values)
    rows=[('A  rolling (30-cycle rolling mean, 20-cycle window)', tau_of(A)),
          ('B  rolling, linear trend removed in window', tau_of(A,True)),
          ('C  raw signal, no rolling mean', tau_of(R)),
          ('D  raw signal, Gaussian-detrended residuals', tau_of(Rd)),
          ('N  NULL: own trend + permuted residuals, procedure A', tau_of(N))]
    print(f'\n=== {sub} ===')
    for nm,(t,f) in rows: print(f'  {nm:<56} tau={t:+.3f}  rising {f*100:4.0f}%')
for s in sys.argv[1:]: run(s)

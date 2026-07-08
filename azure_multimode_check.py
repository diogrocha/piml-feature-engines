import os, json, time, sys, numpy as np, joblib
from sklearn.decomposition import PCA
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from config import LG_PARAMS
from engines import LGPhysicsEngine, make_y
OUT = '/mnt/user-data/outputs/results_azure_full.json'
STEP = 75
TARGET = 150

def desc(dd, Xf, Xa, Tf):
    e = LGPhysicsEngine(**LG_PARAMS)
    dd = dd / (np.linalg.norm(dd) + 1e-12)
    e.weights = dd
    s = Xf @ dd
    spin = e.spinodal
    eh = spin * e.eta_h_frac
    ef = spin * e.eta_f_frac
    hi = Tf > np.percentile(Tf, 75)
    lo = Tf < np.percentile(Tf, 25)
    sh = np.median(s[hi])
    sf = np.median(s[lo])
    if abs(sh - sf) > 1e-09:
        e.eta_scale = (eh - ef) / (sh - sf)
        e.eta_offset = eh - e.eta_scale * sh
    else:
        e.eta_scale = 0.0
        e.eta_offset = spin * 0.5
    return e.transform(Xa)
comp = sys.argv[1]
MP = f'/tmp/azmm_{comp}.joblib'
d = np.load(f'/tmp/azure_cache/{comp}.npz')
Xbt, Xbe, Tf = (d['Xbt'], d['Xbe'], d['ttf_tr'])
ytr = make_y(d['ev_tr'], d['ttf_tr'])
yte = make_y(d['ev_te'], d['ttf_te'])
dirs = [c for c in PCA(3).fit(Xbt).components_]
mm_tr = np.hstack([Xbt] + [desc(dd, Xbt, Xbt, Tf) for dd in dirs])
mm_te = np.hstack([Xbe] + [desc(dd, Xbt, Xbe, Tf) for dd in dirs])
t0 = time.time()
if not os.path.exists(MP):
    m = GradientBoostingSurvivalAnalysis(n_estimators=STEP, learning_rate=0.01, max_depth=3, random_state=42, warm_start=True).fit(mm_tr, ytr)
else:
    m = joblib.load(MP)
    m.set_params(n_estimators=min(m.n_estimators + STEP, TARGET))
    m.fit(mm_tr, ytr)
joblib.dump(m, MP)
print(f'{comp} árvores={m.n_estimators}/{TARGET} ({time.time() - t0:.0f}s)')
if m.n_estimators >= TARGET:
    C = concordance_index_censored(yte['event'], yte['time'], m.predict(mm_te))[0]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault(comp, {})
    res[comp]['+multiPCA'] = float(C)
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'{comp} +multiPCA FULL C={C:.4f}')
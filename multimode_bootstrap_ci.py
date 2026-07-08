import numpy as np, joblib, time, os
from sksurv.metrics import concordance_index_censored
F = '/tmp/mm_boot.npy'
m = joblib.load('/tmp/multiLG_model.joblib')
z = np.load('/tmp/lgortho_FD003.npz')
zc = np.load('/tmp/cmapss_cache/FD003.npz')
ev = z['ev_te'] > 0
ttf = z['ttf_te']
units = zc['units_te']
risk = m.predict(z['PCA3_te'])
uniq = np.unique(units)
idx_by = {u: np.where(units == u)[0] for u in uniq}
Cs = list(np.load(F)) if os.path.exists(F) else []
rng = np.random.RandomState(1000 + len(Cs))
t0 = time.time()
while time.time() - t0 < 92 and len(Cs) < 2000:
    samp = rng.choice(uniq, len(uniq), replace=True)
    idx = np.concatenate([idx_by[u] for u in samp])
    Cs.append(concordance_index_censored(ev[idx], ttf[idx], risk[idx])[0])
np.save(F, np.array(Cs))
lo, hi = np.percentile(Cs, [2.5, 97.5])
print(f'n_boot={len(Cs)}  IC 95% [{lo:.3f}, {hi:.3f}]', flush=True)
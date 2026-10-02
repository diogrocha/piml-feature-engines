import warnings
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from config import SEED, OPTUNA_RANGES, OPTUNA_N_TRIALS, OPTUNA_N_SPLITS, OPTUNA_PROXY_GBSA
from engines import LGPhysicsEngine, make_y, ci_score
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_lg(X_tr, ttf_tr, ev_tr, n_trials=OPTUNA_N_TRIALS, verbose=True):
    if verbose:
        print(f'      Optimising LG via Optuna ({n_trials} trials)...')
    rng = OPTUNA_RANGES

    def objective(trial):
        c = trial.suggest_float('c', *rng['c'])
        kT = trial.suggest_float('kT', *rng['kB_T_eff'])
        ehf = trial.suggest_float('eta_h_frac', *rng['eta_h_frac'])
        eff = trial.suggest_float('eta_f_frac', 0.01, ehf - rng['eta_f_min_gap'])
        kf = KFold(n_splits=OPTUNA_N_SPLITS, shuffle=True, random_state=SEED)
        cis = []
        if ev_tr.sum() < OPTUNA_N_SPLITS:
            return 0.0
        for ti, vi in kf.split(X_tr):
            if not ev_tr[vi].any():
                continue
            try:
                lg = LGPhysicsEngine(c=c, kB_T_eff=kT, eta_h_frac=ehf, eta_f_frac=eff)
                lg.calibrate(X_tr[ti], ttf_tr[ti])
                pt, pv = (lg.transform(X_tr[ti]), lg.transform(X_tr[vi]))
                if np.any(np.std(pt, axis=0) < 1e-08):
                    continue
                yt, yv = (make_y(ev_tr[ti], ttf_tr[ti]), make_y(ev_tr[vi], ttf_tr[vi]))
                m = GradientBoostingSurvivalAnalysis(**OPTUNA_PROXY_GBSA)
                m.fit(pt, yt)
                cis.append(ci_score(m, pv, yv))
            except Exception:
                continue
        return float(np.mean(cis)) if cis else 0.0
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    bp = study.best_params
    if verbose:
        print(f'        best CV={study.best_value:.3f} | c={bp['c']:.3f}, kT={bp['kT']:.3f}, eh={bp['eta_h_frac']:.3f}, ef={bp['eta_f_frac']:.3f}')
    return bp
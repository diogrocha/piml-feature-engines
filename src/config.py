SEED = 42
LG_PARAMS = dict(b=1.0, c=1.749, kB_T_eff=1.431, eta_h_frac=0.856, eta_f_frac=0.487)
ROLLING_WINDOW_CMAPSS = 30
ROLLING_WINDOW_AZURE = 24
ROLLING_WINDOW_XJTU = 10
GBSA_BASE = dict(n_estimators=100, learning_rate=0.01, random_state=SEED)
GBSA_AUG = dict(n_estimators=150, learning_rate=0.01, max_depth=3, random_state=SEED)
RUL_CAP = 125
EVAL_TIME_QUANTILES = (5, 90)
N_EVAL_TIMES = 20
OPTUNA_RANGES = dict(c=(1.0, 3.0), kB_T_eff=(0.1, 1.5), eta_h_frac=(0.6, 0.95), eta_f_min_gap=0.05)
OPTUNA_N_TRIALS = 30
OPTUNA_N_SPLITS = 3
OPTUNA_PROXY_GBSA = dict(n_estimators=10, learning_rate=0.1, max_depth=2, random_state=SEED)
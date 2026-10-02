# Turning simulation

Regenerative turning with a slow ramp of the depth of cut through a known Hopf boundary. No external data.
Run from this folder, in order:

    python -c "import fleet_windows"      # builds the stability-limit table blim_table.npz once
    python fleet_windows.py 0 80          # windows, base 'sensors', windowed least-squares estimator   -> f2_0_80.csv
    python fleet_recursive.py 0 80        # recursive full-model and reduced-Landau estimators           -> f3_0_80.csv
    python fleet_landau.py 0 80           # reduced Landau at the mode's own frequency                   -> f4_0_80.csv
    python merge.py                       # -> f23.csv, f234.csv
    python paradigms.py Base,+Stats,+HI,+HI-fix,+Weibull,+Entropy,+ExpDeg,+LG,+LG-4,+LG-mm,LG-comb-mm,PC-twin,+AR-raw,+b_ls,+d_hat,+twin-est,oracle
    python estimators_recursive.py Base,+b_ls,+b_rls,+d_rls,+full-twin,+sigma_L,+Landau-twin,+b_rls+sigma_L,oracle
    python estimators_landau.py Base,+d_rls,+sigma_L2,+T_full,+twin_full,+T_L,+twin_L,oracle
    python plot_simulation.py             # -> fig_turning_sim.pdf

The fleet scripts accept seed ranges and can run in parallel (e.g. 0 20, 20 40, 40 60, 60 80); every script
resumes. `turning.py` holds the model and the exact stability limit; the evaluation criteria are in the header
of each script.

#!/usr/bin/env bash
# Runs every experiment.
#   bash run_all.sh            # full run, about 7-8 h on 2 CPUs
#   bash run_all.sh --smoke    # 1 fold, reduced, a few minutes
set -e
cd "$(dirname "$0")"
SMOKE=0; for a in "$@"; do [ "$a" = "--smoke" ] && SMOKE=1; done
FOLDS=1,2,3,4,5; [ $SMOKE -eq 1 ] && FOLDS=1
mkdir -p results
bash get_data.sh
# theory checks
if [ $SMOKE -eq 1 ]; then python theory_checks/apriori_checks.py --nfast 600; else python theory_checks/apriori_checks.py; fi
python theory_checks/landscape_figure.py
# C-MAPSS, all feature arms
if [ $SMOKE -eq 1 ]; then python cmapss_paradigms/run_cv.py --smoke --out results/smoke.json
else python cmapss_paradigms/run_cv.py && python cmapss_paradigms/summarize.py; fi
for s in FD001 FD003 FD002 FD004; do python cmapss_paradigms/lg_comb_gbsa.py $s $FOLDS LG-comb-mm,PC-twin,HI-fix; done
python cmapss_paradigms/sup_mm.py FD003 $FOLDS
# Azure and NASA batteries (see DATA.md)
if [ -d data/azure ]; then
  python src/azure_run.py prep comp1 comp2 comp3 comp4
  python azure_batteries/other_datasets_matched.py azure --arms Base Base-AUG +Stats +HI +Weibull +Entropy +ExpDeg +LG
  python azure_batteries/extra_arms.py azure --cache /tmp/azure_cache
fi
if [ -d data/nasa_battery ]; then
  python azure_batteries/other_datasets_matched.py battery --data data/nasa_battery
  python azure_batteries/extra_arms.py battery --data data/nasa_battery
fi
# order-parameter channels and critical-slowing-down test
python mode_channels/plot_mode_channels.py
python critical_slowing_down/csd_audit.py FD001 FD003
# turning simulation (no data needed)
if [ $SMOKE -eq 0 ]; then
  ( cd turning_simulation && python -c "import fleet_windows" && for p in fleet_windows fleet_recursive fleet_landau; do python $p.py 0 80; done && python merge.py \
    && python paradigms.py Base,+Stats,+HI,+HI-fix,+Weibull,+Entropy,+ExpDeg,+LG,+LG-4,+LG-mm,LG-comb-mm,PC-twin,+AR-raw,+b_ls,+d_hat,+twin-est,oracle \
    && python estimators_recursive.py Base,+b_ls,+b_rls,+d_rls,+full-twin,+sigma_L,+Landau-twin,+b_rls+sigma_L,oracle \
    && python estimators_landau.py Base,+d_rls,+sigma_L2,+T_full,+twin_full,+T_L,+twin_L,oracle && python plot_simulation.py )
fi
# robustness, case study, cost, Cox with a Kramers term
[ $SMOKE -eq 0 ] && python robustness/robustness.py
python case_study/case_study.py
python cost/cost_e2e.py
for s in FD001 FD003 FD002 FD004; do python cox_kramers/kramers_cox.py $s $FOLDS; done
echo "All done. Outputs in results/"

"""Grouped five-fold cross-validation for every feature arm.

Checkpointed after each (subset, fold): re-running resumes where it stopped.

    python cmapss_paradigms/run_cv.py                       # full plan, all four subsets
    python cmapss_paradigms/run_cv.py --subsets FD001 --jobs 4
    python cmapss_paradigms/run_cv.py --smoke               # 1 fold, 600 rows, 10 trees
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import argparse
import os
import pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument('--subsets', nargs='+', default=['FD001', 'FD003', 'FD002', 'FD004'])
ap.add_argument('--out', default='results/cv_all.json')
ap.add_argument('--nfast', type=int, default=6000, help='training rows per fold; 0 = all')
ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
ap.add_argument('--smoke', action='store_true')
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

plan = {s: P.MAIN_ARMS + P.PARADIGM_ARMS + (P.EXTENSION_ARMS if s in ('FD001', 'FD003') else [])
        for s in a.subsets}
P.run(plan, a.out, n_folds=5,
      nfast=600 if a.smoke else (a.nfast or None),
      n_jobs=a.jobs,
      trees_override=10 if a.smoke else None,
      only_folds=[1] if a.smoke else None)
print('done ->', a.out)

import os
import json
import argparse
PAPER_CMAPSS = {'+LG': 0.832, '+ExpDeg': 0.832, '_tie': ('+LG', '+ExpDeg')}
TOL_PASS, TOL_CHECK = (0.005, 0.015)
ORDER = ['Base', '+Stats', '+HI', '+Weibull', '+Entropy', '+ExpDeg', '+LG']

def _scalar(entry):
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for k in ('c', 'test'):
            if k in entry:
                return float(entry[k])
    return None

def _status(d):
    a = abs(d)
    return 'PASS ' if a <= TOL_PASS else 'CHECK' if a <= TOL_CHECK else 'FAIL '

def check_cmapss(path):
    if not os.path.exists(path):
        print(f'  {path} missing — run cross_paradigm_cmapss.py first.')
        return
    res = json.load(open(path))
    print(f'\nC-MAPSS FD001  (vs {path}, paper anchors)')
    print(f'  {'config':<10} {'paper':>7} {'got':>7} {'delta':>8}  status')
    for nm, exp in PAPER_CMAPSS.items():
        if nm.startswith('_'):
            continue
        got = _scalar(res.get(nm))
        if got is None:
            print(f'  {nm:<10} {exp:>7.3f}   (absent)')
            continue
        d = got - exp
        print(f'  {nm:<10} {exp:>7.3f} {got:>7.3f} {d:>+8.3f}  {_status(d)}')
    a, b = PAPER_CMAPSS['_tie']
    ga, gb = (_scalar(res.get(a)), _scalar(res.get(b)))
    if ga is not None and gb is not None:
        d = ga - gb
        print(f'  tie {a} vs {b}: delta={d:+.4f}  {_status(d)}  (paper: indistinguishable)')

def report_ranking(path, name):
    if not os.path.exists(path):
        print(f'  {path} missing — run the matching cross_paradigm_*.py first.')
        return
    res = json.load(open(path))
    rows = [(nm, _scalar(res.get(nm))) for nm in ORDER if _scalar(res.get(nm)) is not None]
    rows_sorted = sorted(rows, key=lambda r: -r[1])
    print(f'\n{name}  (vs {path}, ranking — no published anchors)')
    for nm, c in rows_sorted:
        print(f'  {nm:<10} C={c:.4f}')
    lg, ed = (dict(rows).get('+LG'), dict(rows).get('+ExpDeg'))
    if lg is not None and ed is not None:
        top2 = {r[0] for r in rows_sorted[:2]}
        lead = 'PASS ' if {'+LG', '+ExpDeg'} <= top2 else 'CHECK'
        print(f'  LG vs ExpDeg gap = {lg - ed:+.4f}')
        print(f'  degradation-assumption paradigms lead (top 2): {lead}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmapss', default='results_cmapss.json')
    ap.add_argument('--xjtu', default='results_xjtu.json')
    args = ap.parse_args()
    print('=' * 60)
    print('VERIFICATION')
    print(f'  C-MAPSS: PASS <= {TOL_PASS}  CHECK <= {TOL_CHECK}  else FAIL')
    print('=' * 60)
    check_cmapss(args.cmapss)
    report_ranking(args.xjtu, 'XJTU-SY bearings')
    print('\nNote: XJTU-SY has no published anchors here; the check is the')
    print('paradigm ranking and whether LG/ExpDeg lead, as on C-MAPSS.')
if __name__ == '__main__':
    main()
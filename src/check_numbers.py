"""Recompute every number Sections IV and V print, from the data alone.

Two of these went stale once. On 2026-07-17 the boundary was measured over a
pool of six front ends; three learned restorers were added later, and the
Holm family grew from 288 tests to 432. The counts that carry a number in the
text were updated (19 -> 17, lens blur 3 -> 2). Two figures that carry no
front end's name were not: the best-front-end gain on shadow (2.4 -> 2.9) and
on decolorization (8.3 -> 8.4). A search for "Zero-DCE" or "PromptIR" could
never have found them, because neither name appears in that sentence. Only
recomputation finds this class of error, which is what this script is for.

    python check_numbers.py ZA_deep_vs_injection.json merged_per_image.csv dcp_cure.csv

Any argument may be omitted; the checks that need it are then skipped and
reported as skipped rather than silently passed.
"""
import json
import sys

ALPHA = 0.05
SEV_BOUNDARY = [2, 3, 4, 5]          # severity 1 fails the positive control

FREE = ['gamma', 'clahe', 'stretch', 'dcp']
LEARNED = ['adair', 'cidnet', 'zero_dce', 'ffa_net', 'promptir']
POOL = FREE + LEARNED                 # the nine tested in Section IV

CH = {1: 'Decolorization', 2: 'LensBlur', 3: 'CodecError', 4: 'Darkening',
      5: 'DirtyLens', 6: 'Exposure', 7: 'GaussianBlur', 8: 'Noise',
      9: 'Rain', 10: 'Shadow', 11: 'Snow', 12: 'Haze'}

# ---- what the manuscript prints ------------------------------------------
IV_CELLS, IV_TESTS, IV_TOTAL = 48, 432, 17
IV_CLEARED = {'Haze': 4, 'Darkening': 3, 'Rain': 3, 'LensBlur': 2, 'Snow': 2,
              'Noise': 1, 'DirtyLens': 1, 'Exposure': 1, 'Decolorization': 0,
              'GaussianBlur': 0, 'CodecError': 0, 'Shadow': 0}
IV_GAINS = {'CodecError': (1.3, 1.8), 'Shadow': (2.8, 2.9),
            'Decolorization': (8.1, 8.4), 'GaussianBlur': (12.7, 14.7)}

TABLE_II = {'dcp': 60.72, 'vb': 58.23, 'adair': 57.78, 'va_rule': 56.65,
            'cidnet': 56.31, 'ffa_net': 55.91, 'promptir': 55.24,
            'passthrough': 54.49, 'gamma': 52.57, 'zero_dce': 50.93,
            'stretch': 50.41, 'clahe': 46.41}
V_ORACLE = {'o4': 66.42, 'o5': 70.55, 'o10': 73.06}
V_DERIVED = {'o5 - dcp': 9.84, 'o4 - va_rule': 9.77, 'o5 - o4': 4.14,
             'o10 - o5': 2.51, 'o10 - vb': 14.83}
V_HEADROOM = {'GaussianBlur': 25.1, 'LensBlur': 24.9, 'Rain': 17.6,
              'Noise': 15.9, 'Decolorization': 15.8, 'CodecError': 15.1,
              'Snow': 14.0, 'Haze': 12.4, 'Shadow': 10.6, 'Darkening': 10.0,
              'DirtyLens': 8.5, 'Exposure': 8.2}
V_ZERO_SHARE = 5.55                   # of the 14.83, from the four that clear nothing

args = sys.argv[1:]
za = next((a for a in args if a.endswith('.json')), None)
csvs = [a for a in args if a.endswith('.csv')]
merged = next((a for a in csvs if 'dcp' not in a.lower()), None)
dcp = next((a for a in csvs if 'dcp' in a.lower()), None)

fail = skip = 0


def check(label, got, want):
    """Compare at the precision the manuscript prints, not at some other one.

    An earlier version rounded to two places and compared against values the
    paper gives to one, so it reported four failures on data that was right.
    A check that cries wolf gets switched off, which is worse than no check.
    """
    global fail
    text = repr(float(want))
    dp = len(text.split('.')[1].rstrip('0')) or 1
    bad = round(got, dp) != round(float(want), dp)
    fail += bad
    print('  %-34s %9.3f %9.*f%s'
          % (label, got, dp, want, '   <-- differs' if bad else ''))


# ==========================================================================
# Section IV, from ZA
# ==========================================================================
print('SECTION IV  (needs ZA_deep_vs_injection.json)')
if not za:
    print('  skipped, no json given'); skip += 1
else:
    D = json.load(open(za))['deep_vs_injection']
    cells = [k for k, v in D.items() if v.get('validated')]
    tests = [(o['p'], k, op) for k in cells for op, o in D[k]['ops'].items()]
    print('  %-34s %9d %9d%s' % ('validated cells', len(cells), IV_CELLS,
                                 '' if len(cells) == IV_CELLS else '   <-- differs'))
    print('  %-34s %9d %9d%s' % ('tests in the Holm family', len(tests), IV_TESTS,
                                 '' if len(tests) == IV_TESTS else '   <-- differs'))
    fail += (len(cells) != IV_CELLS) + (len(tests) != IV_TESTS)

    tests.sort(key=lambda t: t[0])
    m = len(tests)
    survivors = set()
    for i, (p, k, op) in enumerate(tests):
        if p > ALPHA / (m - i):
            break
        survivors.add((k, op))

    cleared = {}
    for k, op in survivors:
        if D[k]['ops'][op]['vs_oracle'] > 0:
            cleared.setdefault(k.rsplit('_sev', 1)[0], set()).add(k)
    for ch in IV_CLEARED:
        got, want = len(cleared.get(ch, ())), IV_CLEARED[ch]
        bad = got != want
        fail += bad
        print('  %-34s %9d %9d%s' % ('cleared, ' + ch, got, want,
                                     '   <-- differs' if bad else ''))
    total = sum(len(v) for v in cleared.values())
    print('  %-34s %9d %9d%s' % ('cleared, total', total, IV_TOTAL,
                                 '' if total == IV_TOTAL else '   <-- differs'))
    fail += total != IV_TOTAL

    for ch, (p_orc, p_best) in IV_GAINS.items():
        cs = [D['%s_sev%d' % (ch, s)] for s in SEV_BOUNDARY]
        check('oracle gain, ' + ch,
              sum(c['oracle_acc'] - c['raw'] for c in cs) / len(cs), p_orc)
        check('best front end, ' + ch,
              sum(max(c['ops'][o]['acc'] for o in POOL) - c['raw']
                  for c in cs) / len(cs), p_best)

# ==========================================================================
# Table II and Section V, from the per-image files
# ==========================================================================
print('\nTABLE II AND SECTION V  (needs merged_per_image.csv and dcp_cure.csv)')
if not (merged and dcp):
    print('  skipped, need both csv files'); skip += 1
else:
    import numpy as np
    import pandas as pd
    df = pd.read_csv(merged).merge(
        pd.read_csv(dcp)[['filename', 'occ', 'pred_dcp']], on=['filename', 'occ'])
    deg = df[df.ch >= 1].copy()
    # selector V-B is the rule with the dark channel prior on its CLAHE branch
    deg['pred_vb'] = np.where(
        deg.rule_branch == 'clahe', deg.pred_dcp,
        np.select([deg.rule_branch == 'gamma', deg.rule_branch == 'stretch'],
                  [deg.pred_gamma, deg.pred_stretch], deg.pred_passthrough))

    def cellwise(hit):
        """Degraded average: mean over the sixty cells, not over the crops."""
        return deg.assign(_h=hit).groupby(['ch', 'sev'])['_h'].mean()

    def acc(col):
        return cellwise((deg[col] == deg['true']).values).mean() * 100

    def oracle(names):
        cols = ['pred_' + n for n in names]
        hit = np.logical_or.reduce([(deg[c] == deg['true']).values for c in cols])
        return cellwise(hit).mean() * 100

    for name, want in TABLE_II.items():
        check('degraded average, ' + name, acc('pred_' + name), want)

    BRANCHES = ['passthrough', 'gamma', 'clahe', 'stretch']
    o = {'o4': oracle(BRANCHES),
         'o5': oracle(BRANCHES + ['dcp']),
         'o10': oracle(BRANCHES + ['dcp'] + LEARNED)}
    for k, want in V_ORACLE.items():
        check('operator-selection oracle, ' + k, o[k], want)
    o['dcp'], o['vb'], o['va_rule'] = acc('pred_dcp'), acc('pred_vb'), acc('pred_va_rule')
    for expr, want in V_DERIVED.items():
        a, b = expr.split(' - ')
        check(expr, o[a] - o[b], want)

    hit10 = np.logical_or.reduce(
        [(deg['pred_' + n] == deg['true']).values
         for n in BRANCHES + ['dcp'] + LEARNED])
    gap = (cellwise(hit10) - cellwise((deg.pred_vb == deg['true']).values)) * 100
    per = gap.groupby(level=0).mean()
    for ch_id, v in per.items():
        check('headroom over V-B, ' + CH[ch_id], v, V_HEADROOM[CH[ch_id]])
    zero = sum(per[i] for i in CH if CH[i] in
               ('Decolorization', 'GaussianBlur', 'CodecError', 'Shadow')) / 12
    check('of which the four that clear nothing', zero, V_ZERO_SHARE)

print('\n%s%s' % ('ALL CHECKS PASSED' if not fail else '%d CHECK(S) FAILED' % fail,
                  '' if not skip else '   (%d block(s) skipped)' % skip))
sys.exit(1 if fail else 0)

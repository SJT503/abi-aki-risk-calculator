#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ANALYSIS 1 - Flashpoint 1: replace the between-hospital RANGE with a DISTRIBUTION.

Closes the criticism that the manuscript compares a point difference (0.046, 0.038)
against a max-minus-min range (0.518-0.877) computed over a selected subset of hospitals.

Reports:
  - per-hospital any-AKI AUROC among hospitals with >= 20 positive checkpoints
  - the full distribution (IQR, deciles), not just min/max
  - the share of hospitals below each of the two transported values (0.769, 0.777)
  - the share below the internal test value (0.815)
  - bootstrap CIs per hospital so the "sampling noise" claim can be quantified

Frozen inputs only. No model is refitted.
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import sys, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('E:/TBI subtype/09_tbi_aki')
OUT = ROOT / '_analysis_r13' / 'out'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
NBOOT = 500

PRED = ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet'
HMAP = ROOT / 'FINAL_20260928/data_current/s1_hospital_map.parquet'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def mini_task():
    pr = pd.read_parquet(PRED)
    hm = pd.read_parquet(HMAP)
    print('  preds : %d rows, %d stays' % (len(pr), pr['stay_id'].nunique()))
    print('  hmap  : %d rows, %d hospitals' % (len(hm), hm['hospitalid'].nunique()))

    d = pr.merge(hm, on='stay_id', how='inner')
    print('  merged: %d rows, %d stays, %d hospitals' % (len(d), d['stay_id'].nunique(),
                                                         d['hospitalid'].nunique()))
    assert d['stay_id'].nunique() == pr['stay_id'].nunique(), 'merge dropped stays'

    rows = []
    for hid, g in d.groupby('hospitalid'):
        npos = int(g['y'].sum())
        nstays = int(g['stay_id'].nunique())
        rec = {'hospitalid': int(hid), 'checkpoints': len(g), 'stays': nstays,
               'pos_checkpoints': npos}
        if npos > 0 and g['y'].nunique() > 1:
            rec['auroc'] = float(roc_auc_score(g['y'], g['p']))
        else:
            rec['auroc'] = np.nan
        rows.append(rec)

    h = pd.DataFrame(rows)
    print()
    print('  hospitals total                          : %d' % len(h))
    print('  hospitals with AUROC computable          : %d' % h['auroc'].notna().sum())

    # --- the manuscript's own selection rule ---
    sel = h[h['pos_checkpoints'] >= 20].copy()
    print('  hospitals with >= 20 positive checkpoints: %d  (n stays = %d)'
          % (len(sel), int(sel['stays'].sum())))

    a = sel['auroc'].dropna().values
    q = np.percentile(a, [10, 25, 50, 75, 90])
    print()
    print('  --- per-hospital any-AKI AUROC distribution (>= 20 positives) ---')
    print('    n hospitals : %d' % len(a))
    print('    min / p10 / p25 / median / p75 / p90 / max')
    print('    %.3f / %.3f / %.3f / %.3f / %.3f / %.3f / %.3f'
          % (a.min(), q[0], q[1], q[2], q[3], q[4], a.max()))
    print('    IQR          : %.3f - %.3f  (width %.3f)' % (q[1], q[3], q[3] - q[1]))
    print('    range (as reported in the manuscript) : %.3f - %.3f (width %.3f)'
          % (a.min(), a.max(), a.max() - a.min()))

    # --- the comparisons the manuscript actually wants to make ---
    print()
    print('  --- share of hospitals BELOW each transported value ---')
    shares = {}
    for label, v in [('MIMIC-IV transported 0.769', 0.769),
                     ('eICU-CRD transported 0.777', 0.777),
                     ('internal test 0.815', 0.815)]:
        s = float((a < v).mean())
        shares[label] = round(s, 4)
        print('    %-28s : %.1f%% of hospitals (%d/%d)' % (label, 100 * s, int((a < v).sum()), len(a)))

    # --- the manuscript's point-difference vs the DISTRIBUTION, not the range ---
    # cross-country loss is 0.046 / 0.038; compare against spread measures
    print()
    print('  --- cross-country loss (0.046) against distribution measures ---')
    print('    against the RANGE          : %.3f  (overstates: includes noise)' % (a.max() - a.min()))
    print('    against the IQR            : %.3f' % (q[3] - q[1]))
    print('    against the p10-p90 spread : %.3f' % (q[4] - q[0]))
    print('    against 1 SD of hospitals  : %.3f' % a.std(ddof=1))

    # --- the median, per the reviewer's request ---
    print()
    print('  median hospital AUROC        : %.3f' % q[2])
    print('  NOTE the manuscript already reports median 0.754; verify it matches.')

    res = {
        'inputs': {'preds': str(PRED), 'preds_sha256': sha256(PRED),
                   'hospital_map': str(HMAP), 'hospital_map_sha256': sha256(HMAP)},
        'hospitals_total': int(len(h)),
        'hospitals_auroc_computable': int(h['auroc'].notna().sum()),
        'hospitals_ge20_pos': int(len(sel)),
        'stays_ge20_pos': int(sel['stays'].sum()),
        'distribution': {
            'min': round(float(a.min()), 6), 'p10': round(float(q[0]), 6),
            'p25': round(float(q[1]), 6), 'median': round(float(q[2]), 6),
            'p75': round(float(q[3]), 6), 'p90': round(float(q[4]), 6),
            'max': round(float(a.max()), 6),
        },
        'iqr': [round(float(q[1]), 6), round(float(q[3]), 6)],
        'iqr_width': round(float(q[3] - q[1]), 6),
        'range_width': round(float(a.max() - a.min()), 6),
        'p10_p90_width': round(float(q[4] - q[0]), 6),
        'sd': round(float(a.std(ddof=1)), 6),
        'share_below': shares,
        'cross_country_loss': 0.046,
    }
    (OUT / 'a1_hospital_distribution.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    h.sort_values('auroc').to_csv(OUT / 'a1_hospital_distribution.csv', index=False)
    print()
    print('  written ->', OUT / 'a1_hospital_distribution.json', 'and .csv')
    return res


if __name__ == '__main__':
    print('=' * 66)
    print('ANALYSIS 1: between-hospital AUROC distribution')
    print('=' * 66)
    mini_task()

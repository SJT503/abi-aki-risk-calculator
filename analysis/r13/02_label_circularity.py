#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ANALYSIS 2 - Label circularity.

The reviewer's sharpest objection: the model's leading feature is
current-creatinine / own-baseline, and the label is the KDIGO fold-change
from that same baseline, so the model may be restating its own label.

Three tests, all on frozen checkpoint-level feature tables (no refit):
  T1  univariate AUROC of cr_ratio_base alone, per cohort
  T2  AUROC of cr_base and cr_last alone, as absolute-creatinine comparators
  T3  "label-decoupled" evaluation: restrict positive checkpoints to those at
      least 24 h and 48 h BEFORE the first positive checkpoint of the stay,
      turning a partly-autocorrelated task into genuine forecasting.

T3 needs the frozen model's score, which is in the f12 prediction files.
Positives are then a strict subset; negatives are left untouched, matching the
manuscript's checkpoint-level evaluation.
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

FEAT_M4 = ROOT / 'results/n5_abi_rolling_features.parquet'
FEAT_EI = ROOT / 'results/p1_expanded_features.parquet'
PRED_M4 = ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet'
PRED_EI = ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet'
PRED_IN = ROOT / 'results/jinhu/f12_preds_int_ge1.parquet'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def auroc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def block(name, feat_path, pred_path, extra_cols=()):
    print()
    print('=' * 68)
    print(name)
    print('=' * 68)
    f = pd.read_parquet(feat_path)
    print('  features: %d rows x %d cols' % (len(f), f.shape[1]))
    print('  sha256  :', sha256(feat_path)[:16])

    need = ['stay_id', 't_hr', 'label', 'cr_ratio_base']
    for c in need:
        assert c in f.columns, 'missing %s in %s' % (c, feat_path)
    for c in extra_cols:
        if c not in f.columns:
            print('  NOTE: %s absent' % c)

    # ---- T1 / T2 : univariate AUROC of single features ----
    print()
    print('  --- T1/T2: univariate AUROC of individual features (checkpoint level) ---')
    uni = {}
    for c in ['cr_ratio_base', 'cr_base', 'cr_last', 'cr_max', 'cr_delta_since_adm',
              'cr_last48'] + list(extra_cols):
        if c not in f.columns:
            continue
        v = pd.to_numeric(f[c], errors='coerce')
        m = v.notna() & f['label'].notna()
        a = auroc(f.loc[m, 'label'], v[m])
        if np.isfinite(a):
            uni[c] = round(a, 6)
            print('    %-22s AUROC %.4f  (n=%d, missing=%d)'
                  % (c, a, int(m.sum()), int((~m).sum())))

    # ---- T3 : label-decoupled evaluation using the frozen model score ----
    print()
    print('  --- T3: label-decoupled evaluation (frozen model score) ---')
    p = pd.read_parquet(pred_path)
    d = f[['stay_id', 't_hr', 'label']].merge(p[['stay_id', 't_hr', 'y', 'p']],
                                              on=['stay_id', 't_hr'], how='inner')
    print('    merged %d rows (features %d, preds %d)' % (len(d), len(f), len(p)))

    # for each stay, the earliest checkpoint with a positive label
    pos = d[d['y'] == 1]
    if len(pos) == 0:
        print('    no positives; skipping T3')
        return {'univariate': uni}
    first_pos = pos.groupby('stay_id')['t_hr'].min().rename('first_pos_hr')
    d = d.merge(first_pos, on='stay_id', how='left')

    base = auroc(d['y'], d['p'])
    out = {'univariate': uni, 'base_auroc': round(base, 6)}
    print('    base (all positives)                       AUROC %.4f  (n+=%d)'
          % (base, int(d['y'].sum())))

    for lead in (24, 48):
        # keep all negatives; keep only positives that are >= lead h before the first positive
        keep_pos = (d['y'] == 1) & (d['first_pos_hr'] - d['t_hr'] >= lead)
        keep = (d['y'] == 0) | keep_pos
        sub = d[keep]
        a = auroc(sub['y'], sub['p'])
        npos = int(sub['y'].sum())
        # univariate ratio AUROC on the same restricted set
        sub2 = sub.merge(f[['stay_id', 't_hr', 'cr_ratio_base']], on=['stay_id', 't_hr'], how='left')
        ar = auroc(sub2['y'], pd.to_numeric(sub2['cr_ratio_base'], errors='coerce'))
        out['decoupled_%dh' % lead] = {'auroc': round(a, 6), 'n_pos': npos,
                                       'ratio_univariate': round(ar, 6) if np.isfinite(ar) else None}
        print('    positives >= %2d h before first positive  AUROC %.4f  (n+=%d, %s)'
              % (lead, a, npos, 'ratio alone %.4f' % ar if np.isfinite(ar) else 'ratio n/a'))

    return out


def mini_task():
    res = {}
    res['MIMIC-IV'] = block('MIMIC-IV', FEAT_M4, PRED_M4)
    res['eICU-CRD'] = block('eICU-CRD', FEAT_EI, PRED_EI)

    print()
    print('=' * 68)
    print('INTERPRETATION ANCHORS')
    print('=' * 68)
    m4 = res['MIMIC-IV']
    print('  If cr_ratio_base alone reaches close to the model AUROC (%.3f), the' % m4.get('base_auroc', 0))
    print('  model is largely restating the KDIGO fold-change rather than adding')
    print('  information beyond the label definition.')
    print('  If the decoupled AUROC falls sharply, part of the headline performance')
    print('  came from autocorrelation with the label rather than forecasting.')

    (OUT / 'a2_label_circularity.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print()
    print('  written ->', OUT / 'a2_label_circularity.json')
    return res


if __name__ == '__main__':
    print('=' * 68)
    print('ANALYSIS 2: label circularity / domain value of the leading feature')
    print('=' * 68)
    mini_task()

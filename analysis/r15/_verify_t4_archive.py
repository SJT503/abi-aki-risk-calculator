#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT VERIFICATION T4: frozen model archive.

1) sha256(frozen_models.pkl) == manifest.json sha256; byte size matches.
2) Load pkl; rebuild MIMIC-IV ge1 label machinery MYSELF from
   n2_m4_abi_cr_serial.parquet (cr 0.1-30, 6-h bins, per-bin max,
   forward 9-bin max incl current, fold vs cr_base clip 0.1, rise>=0.3,
   fold>=1.5, stage1/2/3, NaN->0) on the n5 feature grid; score with the
   bundled model+selected_features; AUROC must equal 0.7694 (+/- 0.0005).
"""
import hashlib
import json
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
R = 'E:/TBI subtype/09_tbi_aki'
ART = f'{R}/_analysis_r15/model_objects'

# ---------- 1) manifest ----------
man = json.load(open(f'{ART}/manifest.json', encoding='utf-8'))
raw = open(f'{ART}/frozen_models.pkl', 'rb').read()
sha = hashlib.sha256(raw).hexdigest()
print('sha256 recomputed:', sha)
print('sha256 manifest  :', man['sha256'],
      '->', 'MATCH' if sha == man['sha256'] else '*** MISMATCH ***')
print('bytes: file', len(raw), 'manifest', man['bytes'],
      '->', 'MATCH' if len(raw) == man['bytes'] else '*** MISMATCH ***')

# ---------- 2) load bundle ----------
with open(f'{ART}/frozen_models.pkl', 'rb') as fh:
    b = pickle.load(fh)
print('bundle keys:', sorted(b.keys()))
SEL = b['selected_features']
print('n selected features:', len(SEL), '(manifest', man['n_features'], ')',
      'MATCH' if len(SEL) == man['n_features'] else 'MISMATCH')
print('recal layer in bundle:', b.get('recalibration_layer'))
print('params:', b['params'])
print('canary aurocs in bundle:', b['canary_aurocs'])

# ---------- label machinery (my own rebuild) ----------
m5 = pd.read_parquet(f'{R}/results/n5_abi_rolling_features.parquet')
cr = pd.read_parquet(f'{R}/results/n2_m4_abi_cr_serial.parquet')
cr = cr[cr.valuenum.between(0.1, 30)].copy()
cr['t_hr'] = (cr.hr // 6) * 6
binmax = cr.groupby(['stay_id', 't_hr']).valuenum.max().reset_index()
grid = m5[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(
    binmax, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
print('grid rows:', len(grid))
# forward max over current + next 8 bins (9 bins), running max via 9 shifts
fwd = pd.Series(np.nan, index=grid.index)
for s in range(9):
    fwd = pd.concat([fwd.rename('c'),
                     grid.groupby('stay_id').valuenum.shift(-s).rename('n')],
                    axis=1).max(axis=1)
grid['f'] = fwd
fold = grid.f / grid.cr_base.clip(lower=0.1)
rise = grid.f - grid.cr_base
aki = (fold >= 1.5) | (rise >= 0.3)
stage = pd.Series(0, index=grid.index)
stage[aki.fillna(False)] = 1
stage[fold >= 2.0] = 2
stage[(fold >= 3.0) | ((grid.f >= 4.0) & aki)] = 3
stage[grid.f.isna()] = 0
grid['ge1'] = (stage >= 1).astype(int)
grid['ge2'] = (stage >= 2).astype(int)
grid['ge3'] = (stage == 3).astype(int)
print('label positives ge1: rows %d / %d, stays %d' %
      (grid.ge1.sum(), len(grid), grid.loc[grid.ge1 == 1, 'stay_id'].nunique()))

# ---------- score ----------
X = m5[['stay_id', 't_hr'] + SEL].merge(
    grid[['stay_id', 't_hr', 'ge1', 'ge2', 'ge3']], on=['stay_id', 't_hr'], how='left')
print('scoring rows:', len(X), 'ge1 NaN:', int(X.ge1.isna().sum()))
for t in (1, 2, 3):
    p = b['models'][t].predict_proba(X[SEL].values)[:, 1]
    auc = roc_auc_score(X[f'ge{t}'], p)
    tag = ' (target)' if t == 1 else ''
    print('ge%d AUROC = %.4f%s  claim %s' % (t, auc, tag, b['canary_aurocs'][f'ge{t}']['M4']),
          '->', 'MATCH' if abs(auc - b['canary_aurocs'][f'ge{t}']['M4']) < 0.0005 else '*** CHECK ***')
auc1 = roc_auc_score(X.ge1, b['models'][1].predict_proba(X[SEL].values)[:, 1])
print('T4 VERDICT: AUROC %.4f vs 0.7694 ->' % auc1,
      'MATCH' if abs(auc1 - 0.7694) <= 0.0005 else '*** MISMATCH ***')

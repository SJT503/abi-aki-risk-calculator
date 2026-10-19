#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""r16-1  Leave-hospital-out silent-mode recalibration simulation (eICU-CRD).

Answers the blind reviewer's must-fix #2: the transported two-parameter layer
is no better than prevalence externally — does the silent-mode path (refit the
layer on other hospitals, deploy locally) actually fix calibration?

Canary: applying the frozen development layer (a=0.005067, b=0.268467) to all
of eICU-CRD must reproduce the frozen recalibration slope 0.881 before any
leave-one-out number is trusted.
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit

sys.stdout.reconfigure(encoding='utf-8')
R = Path('E:/TBI subtype/09_tbi_aki')
OUT = R / '_analysis_r16' / 'out'
OUT.mkdir(parents=True, exist_ok=True)
A1, B1 = 0.005067, 0.268467

pe = pd.read_parquet(R / 'results/jinhu/f12_preds_eICU_ge1.parquet')
hm = pd.read_parquet(R / 'FINAL_20260928/data_current/s1_hospital_map.parquet')
d = pe.merge(hm, on='stay_id', how='inner')
d['lg'] = logit(np.clip(d['p'], 1e-9, 1 - 1e-9))
d['p_tr'] = expit(A1 + B1 * d['lg'])
print('rows %d | hospitals %d' % (len(d), d.hospitalid.nunique()))


def slope_ece(y, p):
    """Calibration slope (logistic of y on logit(p)) and 10-bin ECE
    (unweighted mean over deciles of |bin mean p - bin mean y|)."""
    x = logit(np.clip(p, 1e-9, 1 - 1e-9)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    sl = float(lr.coef_[0][0])
    q = pd.qcut(pd.Series(p), 10, duplicates='drop')
    g = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)}).groupby(q, observed=True).mean()
    ece = float(np.abs(g['p'] - g['y']).mean())
    return sl, ece


# ---------------- canary: transported slope over all of eICU ----------------
sl_all, ece_all = slope_ece(d['y'].values, d['p_tr'].values)
print('canary transported slope %.3f (frozen log: 0.881), ECE %.4f' % (sl_all, ece_all))
assert abs(sl_all - 0.881) < 0.02, 'canary failed'

# ---------------- leave-hospital-out ----------------
hosp = d.groupby('hospitalid').agg(n=('y', 'size'), npos=('y', 'sum'))
elig = hosp[hosp.npos >= 20].index.tolist()
print('hospitals with >=20 positives: %d' % len(elig))

rows = []
for h in elig:
    te = d[d.hospitalid == h]
    tr = d[d.hospitalid != h]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(
        tr['lg'].values.reshape(-1, 1), tr['y'].values)
    a_h, b_h = float(lr.intercept_[0]), float(lr.coef_[0][0])
    p_lo = expit(a_h + b_h * te['lg'].values)
    sl_tr, ece_tr = slope_ece(te['y'].values, te['p_tr'].values)
    sl_lo, ece_lo = slope_ece(te['y'].values, p_lo)
    rows.append({'hospitalid': h, 'n': int(len(te)), 'npos': int(te['y'].sum()),
                 'slope_transported': round(sl_tr, 3),
                 'slope_local': round(sl_lo, 3),
                 'ece_transported': round(ece_tr, 4),
                 'ece_local': round(ece_lo, 4),
                 'b_h': round(b_h, 3)})

t = pd.DataFrame(rows)
print()
print('slope transported : median %.3f (IQR %.3f-%.3f)' % (
    t.slope_transported.median(), t.slope_transported.quantile(.25),
    t.slope_transported.quantile(.75)))
print('slope local-refit : median %.3f (IQR %.3f-%.3f)' % (
    t.slope_local.median(), t.slope_local.quantile(.25),
    t.slope_local.quantile(.75)))
print('hospitals with local slope in [0.8, 1.25]: %d/%d' % (
    ((t.slope_local >= 0.8) & (t.slope_local <= 1.25)).sum(), len(t)))
print('same for transported: %d/%d' % (
    ((t.slope_transported >= 0.8) & (t.slope_transported <= 1.25)).sum(), len(t)))
print('ECE transported median %.4f -> local %.4f' % (
    t.ece_transported.median(), t.ece_local.median()))

res = {'canary_slope_all_eicu': round(sl_all, 3),
       'n_hospitals': len(t),
       'slope_transported': {'median': round(float(t.slope_transported.median()), 3),
                             'iqr': [round(float(t.slope_transported.quantile(.25)), 3),
                                     round(float(t.slope_transported.quantile(.75)), 3)]},
       'slope_local': {'median': round(float(t.slope_local.median()), 3),
                       'iqr': [round(float(t.slope_local.quantile(.25)), 3),
                               round(float(t.slope_local.quantile(.75)), 3)]},
       'in_0.8_1.25': {'local': int(((t.slope_local >= 0.8) & (t.slope_local <= 1.25)).sum()),
                       'transported': int(((t.slope_transported >= 0.8) & (t.slope_transported <= 1.25)).sum())},
       'ece_median': {'transported': round(float(t.ece_transported.median()), 4),
                      'local': round(float(t.ece_local.median()), 4)}}
(OUT / 'lho_recalibration.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
t.to_csv(OUT / 'lho_recalibration_per_hospital.csv', index=False)
print()
print('written ->', OUT / 'lho_recalibration.json')

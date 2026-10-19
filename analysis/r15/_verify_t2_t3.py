#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT VERIFICATION T2 (SMD excluded vs included) and T3 (comparator).

T2: SMD on age and male sex, included = stays in prediction files,
    excluded = rest of cohort files. |SMD| claims:
    MIMIC age 0.100, male -0.023 ; eICU age 0.029, male 0.019 ; all <= 0.10.
    (sign convention: included - excluded)
T3: cr_ratio_base >= 1.5 at checkpoint strictly before anchor, on the
    RECALIBRATED captured sets (239 / 260). Claims ~0.4% MIMIC (1 of 239), 0% eICU.
"""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
R = 'E:/TBI subtype/09_tbi_aki'
A, B = 0.005067, 0.268467

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1-1e-12)
    return np.log(p/(1-p))

# ============================= T2 =============================
print('='*80); print('T2  SMD excluded vs included'); print('='*80)
pm = pd.read_parquet(f'{R}/results/jinhu/f12_preds_M4_ge1.parquet',
                     columns=['stay_id'])
pe = pd.read_parquet(f'{R}/results/jinhu/f12_preds_eICU_ge1.parquet',
                     columns=['stay_id'])

def smd_cont(x1, x2):
    x1 = pd.to_numeric(x1, errors='coerce').dropna()
    x2 = pd.to_numeric(x2, errors='coerce').dropna()
    sp = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return (x1.mean() - x2.mean()) / sp

def smd_bin(p1, p2):
    sp = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / sp

# --- MIMIC ---
n1 = pd.read_parquet(f'{R}/results/n1_m4_abi_cohort.parquet')
print('MIMIC cohort total rows:', len(n1), '(claim 10,723); unique stays:',
      n1.stay_id.nunique())
inc = n1.stay_id.isin(set(pm.stay_id))
g1, g0 = n1[inc], n1[~inc]
s_age = smd_cont(g1.anchor_age, g0.anchor_age)
m1 = (g1.gender == 'M').mean(); m0 = (g0.gender == 'M').mean()
s_male = smd_bin(m1, m0)
print(f'MIMIC  inc n={len(g1)} exc n={len(g0)}')
print(f'  age  {g1.anchor_age.mean():.2f} vs {g0.anchor_age.mean():.2f}  SMD={s_age:+.4f}  (claim 0.100, |.|<=0.10)',
      'MATCH' if abs(abs(round(s_age,3)) - 0.100) < 0.0005 else '*** CHECK ***')
print(f'  male {100*m1:.2f}% vs {100*m0:.2f}%  SMD={s_male:+.4f}  (claim -0.023)',
      'MATCH' if abs(round(s_male,3) - (-0.023)) < 0.0005 else '*** CHECK ***')

# --- eICU ---
e1 = pd.read_parquet(f'{R}/results/n8_1_eicu_abi_cohort.parquet')
print('eICU cohort total rows:', len(e1), '(claim 13,344); unique stays:',
      e1.stay_id.nunique())
print('  gender raw values:', e1.gender.value_counts(dropna=False).to_dict())
print('  age raw sample:', sorted(set(e1.age.astype(str)))[:8], '...')
age_n = pd.to_numeric(e1.age.astype(str).str.replace('> 89', '89', regex=False),
                      errors='coerce')
e1['age_n'] = age_n
ince = e1.stay_id.isin(set(pe.stay_id))
h1, h0 = e1[ince], e1[~ince]
s_age_e = smd_cont(h1.age_n, h0.age_n)
m1e = (h1.gender.str.strip().str.lower() == 'male').mean()
m0e = (h0.gender.str.strip().str.lower() == 'male').mean()
s_male_e = smd_bin(m1e, m0e)
print(f'eICU  inc n={len(h1)} exc n={len(h0)}  age NaN: inc {int(h1.age_n.isna().sum())} exc {int(h0.age_n.isna().sum())}')
print(f'  age  {h1.age_n.mean():.2f} vs {h0.age_n.mean():.2f}  SMD={s_age_e:+.4f}  (claim 0.029)',
      'MATCH' if abs(round(s_age_e,3) - 0.029) < 0.0005 else '*** CHECK ***')
print(f'  male {100*m1e:.2f}% vs {100*m0e:.2f}%  SMD={s_male_e:+.4f}  (claim 0.019)',
      'MATCH' if abs(round(s_male_e,3) - 0.019) < 0.0005 else '*** CHECK ***')
allv = [s_age, s_male, s_age_e, s_male_e]
print('  all |SMD| <= 0.10 :', all(abs(v) <= 0.10 for v in allv),
      ' (max |SMD| = %.4f)' % max(abs(v) for v in allv))

# ============================= T3 =============================
print(); print('='*80); print('T3  creatinine-ratio comparator on recalibrated captured sets'); print('='*80)
for name, pf, ffeat, claim in [
        ('MIMIC-IV', 'f12_preds_M4_ge1.parquet', 'n5_abi_rolling_features.parquet', 0.4),
        ('eICU-CRD', 'f12_preds_eICU_ge1.parquet', 'p1_expanded_features.parquet', 0.0)]:
    pred = pd.read_parquet(f'{R}/results/jinhu/{pf}')
    pred['p_rec'] = 1/(1+np.exp(-(A + B*logit(pred['p'].values))))
    pred = pred.sort_values(['stay_id','t_hr'], kind='mergesort')
    anchor = pred[pred.y == 1].groupby('stay_id')['t_hr'].max().rename('anchor')
    pred = pred.merge(anchor, on='stay_id', how='left')
    ab = pred[(pred.p_rec >= 0.15) & (pred.t_hr < pred.anchor)]
    capids = set(ab.stay_id.unique())
    f = pd.read_parquet(f'{R}/results/{ffeat}', columns=['stay_id','t_hr','cr_ratio_base'])
    r = pd.to_numeric(f.cr_ratio_base, errors='coerce')
    f = f.assign(r=r)
    fired = set(f[(f.r >= 1.5) & (f.t_hr < f.stay_id.map(anchor))].stay_id.unique())
    nfired = len(capids & fired)
    pct = 100*nfired/len(capids)
    print(f'{name}: captured={len(capids)}  cr-rule fired before anchor on {nfired} '
          f'({pct:.2f}%)   claim ~{claim}% ->',
          'MATCH' if abs(pct - claim) < 0.051 else '*** CHECK ***')
    # extra: also on RAW-scale capture for cross-check vs surveillance_comparator.json
    ab_raw = pred[(pred.p >= 0.15) & (pred.t_hr < pred.anchor)]
    capraw = set(ab_raw.stay_id.unique())
    nfr = len(capraw & fired)
    print(f'   [cross-check raw-scale capture n={len(capraw)}: cr fired {nfr} '
          f'({100*nfr/len(capraw):.1f}%)]')

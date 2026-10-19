#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""r15 ANALYSES for the PI's priority list (2026-10-19).

A2  Excluded-vs-included baseline comparison (SMD on age and sex) for both
    external cohorts — answers the selection-bias question the benchmark
    answers with 'all variables SMD < 0.2'.

A5  Surveillance comparator for the 18 h lead time: at the same six-hour
    cadence, when would a creatinine-triggered alert (current-to-baseline
    ratio reaching the Stage 1 fold-change of 1.5) have fired on the same
    captured event stays? Quantifies how much of the model's lead is model
    value rather than measurement cadence.

Canaries first: the frozen prediction files must reproduce the manuscript
AUROCs before anything downstream is trusted.
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
R = Path('E:/TBI subtype/09_tbi_aki')
OUT = R / '_analysis_r15' / 'out'
OUT.mkdir(parents=True, exist_ok=True)

print('=' * 74)
print('r15-1  SMD (excluded vs included) + creatinine-triggered comparator')
print('=' * 74)

# ---------------- canary ----------------
pm = pd.read_parquet(R / 'results/jinhu/f12_preds_M4_ge1.parquet')
pe = pd.read_parquet(R / 'results/jinhu/f12_preds_eICU_ge1.parquet')
a_m = roc_auc_score(pm['y'], pm['p'])
a_e = roc_auc_score(pe['y'], pe['p'])
print('canary M4 %.4f eICU %.4f' % (a_m, a_e))
assert abs(a_m - 0.7694) < 0.001 and abs(a_e - 0.7773) < 0.001, 'canary failed'

# ---------------- A2: SMD ----------------
def smd_cont(x1, x2):
    x1 = pd.to_numeric(x1, errors='coerce').dropna()
    x2 = pd.to_numeric(x2, errors='coerce').dropna()
    pooled = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return (x1.mean() - x2.mean()) / pooled if pooled else np.nan


def smd_bin(p1, p2):
    p1, p2 = float(p1), float(p2)
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / pooled if pooled else np.nan


res_smd = {}
print()
print('--- A2  excluded vs included (age, male sex) ---')

n1 = pd.read_parquet(R / 'results/n1_m4_abi_cohort.parquet')
inc_m = set(pm['stay_id'])
n1['inc'] = n1['stay_id'].isin(inc_m)
g1, g0 = n1[n1['inc']], n1[~n1['inc']]
smd_age_m = smd_cont(g1['anchor_age'], g0['anchor_age'])
m1 = (g1['gender'] == 'M').mean()
m0 = (g0['gender'] == 'M').mean()
smd_male_m = smd_bin(m1, m0)
res_smd['MIMIC-IV'] = {
    'n_included': int(len(g1)), 'n_excluded': int(len(g0)),
    'age_included': round(float(g1['anchor_age'].mean()), 1),
    'age_excluded': round(float(g0['anchor_age'].mean()), 1),
    'smd_age': round(float(smd_age_m), 3),
    'male_included': round(float(m1), 3), 'male_excluded': round(float(m0), 3),
    'smd_male': round(float(smd_male_m), 3),
}
print('MIMIC-IV  included %d vs excluded %d | age %.1f vs %.1f (SMD %.3f) | male %.1f%% vs %.1f%% (SMD %.3f)'
      % (len(g1), len(g0), g1['anchor_age'].mean(), g0['anchor_age'].mean(), smd_age_m,
         100 * m1, 100 * m0, smd_male_m))

e1 = pd.read_parquet(R / 'results/n8_1_eicu_abi_cohort.parquet',
                     columns=['stay_id', 'gender', 'age'])
inc_e = set(pe['stay_id'])
e1['inc'] = e1['stay_id'].isin(inc_e)
e1['age_n'] = pd.to_numeric(e1['age'].replace('> 89', '89'), errors='coerce')
h1, h0 = e1[e1['inc']], e1[~e1['inc']]
smd_age_e = smd_cont(h1['age_n'], h0['age_n'])
m1e = (h1['gender'].str.strip().str.lower() == 'male').mean()
m0e = (h0['gender'].str.strip().str.lower() == 'male').mean()
smd_male_e = smd_bin(m1e, m0e)
res_smd['eICU-CRD'] = {
    'n_included': int(len(h1)), 'n_excluded': int(len(h0)),
    'age_included': round(float(h1['age_n'].mean()), 1),
    'age_excluded': round(float(h0['age_n'].mean()), 1),
    'smd_age': round(float(smd_age_e), 3),
    'male_included': round(float(m1e), 3), 'male_excluded': round(float(m0e), 3),
    'smd_male': round(float(smd_male_e), 3),
    'age_missing_note': 'eICU age is a string field; "> 89" mapped to 89',
}
print('eICU-CRD  included %d vs excluded %d | age %.1f vs %.1f (SMD %.3f) | male %.1f%% vs %.1f%% (SMD %.3f)'
      % (len(h1), len(h0), h1['age_n'].mean(), h0['age_n'].mean(), smd_age_e,
         100 * m1e, 100 * m0e, smd_male_e))

# ---------------- A5: creatinine-triggered comparator ----------------
print()
print('--- A5  creatinine-triggered alert vs the model, same cadence ---')

THR = 1.5
comp = {}
feat = {'MIMIC-IV': R / 'results/n5_abi_rolling_features.parquet',
        'eICU-CRD': R / 'results/p1_expanded_features.parquet'}
preds = {'MIMIC-IV': pm, 'eICU-CRD': pe}

for name in preds:
    f = pd.read_parquet(feat[name], columns=['stay_id', 't_hr', 'cr_ratio_base'])
    d = preds[name].merge(f, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
    r = pd.to_numeric(d['cr_ratio_base'], errors='coerce')
    rows = []
    for sid, g in d.groupby('stay_id', sort=False):
        pos = g[g['y'] == 1]
        if len(pos) == 0:
            continue
        anchor = float(pos['t_hr'].max())
        # model alert: p >= 0.15 strictly before anchor (definition v2)
        ma = g[(g['p'] >= 0.15) & (g['t_hr'] < anchor)]
        model_t = float(ma['t_hr'].min()) if len(ma) else np.nan
        captured = bool(len(ma))
        # creatinine-triggered alert: ratio >= 1.5 strictly before anchor
        gg = g.assign(r=pd.to_numeric(g['cr_ratio_base'], errors='coerce'))
        ca = gg[(gg['r'] >= THR) & (gg['t_hr'] < anchor)]
        cr_t = float(ca['t_hr'].min()) if len(ca) else np.nan
        rows.append({'stay_id': sid, 'anchor': anchor,
                     'captured': captured, 'model_t': model_t, 'cr_t': cr_t})
    t = pd.DataFrame(rows)
    cap = t[t['captured']]
    model_lead = cap['anchor'] - cap['model_t']
    cr_fired = cap[cap['cr_t'].notna()]
    cr_lead = cr_fired['anchor'] - cr_fired['cr_t']
    both = cap[cap['cr_t'].notna() & cap['model_t'].notna()]
    earlier = (both['model_t'] < both['cr_t']).mean()
    never_cr = cap['cr_t'].isna().mean()
    gain = (both['cr_t'] - both['model_t'])
    comp[name] = {
        'n_event_stays': int(len(t)), 'n_captured': int(len(cap)),
        'model_lead_median_h': round(float(model_lead.median()), 1),
        'cr_alert_fired_pct': round(100 * float(cr_fired.shape[0] / len(cap)), 1),
        'cr_lead_median_h': round(float(cr_lead.median()), 1) if len(cr_lead) else None,
        'model_earlier_than_cr_pct': round(100 * float(earlier), 1),
        'cr_never_fired_pct': round(100 * float(never_cr), 1),
        'model_lead_gain_median_h': round(float(gain.median()), 1) if len(gain) else None,
        'model_lead_gain_p25_h': round(float(gain.quantile(.25)), 1) if len(gain) else None,
        'model_lead_gain_p75_h': round(float(gain.quantile(.75)), 1) if len(gain) else None,
    }
    print('%s: captured %d | model lead %.1f h | cr-alert fired on %.1f%% of them (lead %.1f h)'
          % (name, len(cap), model_lead.median(), comp[name]['cr_alert_fired_pct'],
             comp[name]['cr_lead_median_h'] or float('nan')))
    print('        model fired earlier than cr-alert in %.1f%%; cr-alert never fired on %.1f%%; median gain %.1f h'
          % (comp[name]['model_earlier_than_cr_pct'], comp[name]['cr_never_fired_pct'],
             comp[name]['model_lead_gain_median_h'] or float('nan')))

(OUT / 'smd_excluded_included.json').write_text(json.dumps(res_smd, indent=2), encoding='utf-8')
(OUT / 'surveillance_comparator.json').write_text(json.dumps(comp, indent=2), encoding='utf-8')
print()
print('written ->', OUT / 'smd_excluded_included.json')
print('written ->', OUT / 'surveillance_comparator.json')

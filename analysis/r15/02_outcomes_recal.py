#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""r15-2  Outcome gradient and comparator on the RECALIBRATED scale.

The r13 outcome analysis grouped stays with raw p >= 0.15 (107 captured in
MIMIC-IV), but the manuscript's deployment metrics use 0.15 on the
recalibrated scale (239 captured). This script recomputes capture groups,
mortality/LOS and the surveillance comparator on the recalibrated scale.
Canaries: captured counts must reproduce the manuscript's 239 and 260.
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
R = Path('E:/TBI subtype/09_tbi_aki')
OUT = R / '_analysis_r15' / 'out'
A1, B1 = 0.005067, 0.268467   # Methods: logistic recalibration layer


def recal(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return 1 / (1 + np.exp(-(A1 + B1 * np.log(p / (1 - p)))))


DATA = Path('E:/TBI subtype/data')

print('=' * 74)
print('r15-2  outcomes + comparator on the recalibrated scale')
print('=' * 74)

res = {}
for name, pf, ffeat, kind in [
    ('MIMIC-IV', 'f12_preds_M4_ge1.parquet', 'n5_abi_rolling_features.parquet', 'mimic'),
    ('eICU-CRD', 'f12_preds_eICU_ge1.parquet', 'p1_expanded_features.parquet', 'eicu'),
]:
    print()
    print('===', name, '===')
    pred = pd.read_parquet(R / ('results/jinhu/' + pf))
    pred['p_rec'] = recal(pred['p'])
    # ---- capture on the recalibrated scale (definition v2) ----
    rows = []
    for sid, g in pred.groupby('stay_id', sort=False):
        g = g.sort_values('t_hr')
        pos = g[g['y'] == 1]
        ev = len(pos) > 0
        anchor = float(pos['t_hr'].max()) if ev else np.nan
        alerted = g['p_rec'] >= 0.15
        cap = bool(ev and (alerted & (g['t_hr'] < anchor)).any())
        model_t = float(g.loc[alerted, 't_hr'].min()) if alerted.any() else np.nan
        rows.append({'stay_id': sid, 'event': ev, 'captured': cap,
                     'anchor': anchor, 'model_t': model_t,
                     'max_p_rec': float(g['p_rec'].max())})
    st = pd.DataFrame(rows)
    n_cap = int((st.event & st.captured).sum())
    print('canary captured = %d (manuscript %s)' % (
        n_cap, '239' if kind == 'mimic' else '260'))
    expect = 239 if kind == 'mimic' else 260
    assert n_cap == expect, 'capture canary failed: %d != %d' % (n_cap, expect)
    capst = st[st.event & st.captured]
    lead = capst['anchor'] - capst['model_t']
    print('model lead median %.1f h (IQR %.0f-%.0f)' % (
        lead.median(), lead.quantile(.25), lead.quantile(.75)))

    # ---- join outcomes ----
    if kind == 'mimic':
        icu = pd.read_csv(DATA / 'mimic-iv-3.1/icu/icustays.csv.gz',
                          usecols=['stay_id', 'hadm_id', 'los'])
        adm = pd.read_csv(DATA / 'mimic-iv-3.1/hosp/admissions.csv.gz',
                          usecols=['hadm_id', 'hospital_expire_flag'])
        oc = st.merge(icu, on='stay_id').merge(adm, on='hadm_id')
        dv, col = 'hospital_expire_flag', 'death_pct'
    else:
        ep = pd.read_csv(DATA / 'eicu-crd-2.0/patient.csv.gz',
                         usecols=['patientunitstayid', 'unitdischargestatus',
                                  'unitdischargeoffset'], low_memory=False)
        ep = ep.rename(columns={'patientunitstayid': 'stay_id'})
        oc = st.merge(ep, on='stay_id')
        oc['hospital_expire_flag'] = (
            oc['unitdischargestatus'].astype(str).str.lower()
            .str.startswith('expired')).astype(float)
        oc['los'] = oc['unitdischargeoffset'] / 1440.0
    grp = {}
    for label, mask in [('captured', oc.event & oc.captured),
                        ('missed', oc.event & ~oc.captured),
                        ('eventfree', ~oc.event)]:
        g = oc[mask]
        grp[label] = {'n': int(len(g)),
                      'death_pct': round(100 * float(g['hospital_expire_flag'].mean()), 1),
                      'los_med': round(float(g['los'].median()), 2)}
        print('  %-10s n=%-5d death %.1f%%  LOS %.2f d'
              % (label, len(g), grp[label]['death_pct'], grp[label]['los_med']))
    # quartiles of stay-level max recalibrated probability
    oc['band'] = pd.qcut(oc['max_p_rec'], 4,
                         labels=['Q1', 'Q2', 'Q3', 'Q4'])
    q = oc.groupby('band', observed=True)['hospital_expire_flag'] \
          .agg(['size', 'mean']).round(4)
    print('  quartile death%:', [round(100 * v, 2) for v in q['mean']])

    # ---- comparator: creatinine-trigger at same cadence ----
    f = pd.read_parquet(R / ('results/' + ffeat),
                        columns=['stay_id', 't_hr', 'cr_ratio_base'])
    d = pred.merge(f, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
    d['r'] = pd.to_numeric(d['cr_ratio_base'], errors='coerce')
    cr_rows = []
    for sid, g in d.groupby('stay_id', sort=False):
        pos = g[g['y'] == 1]
        if len(pos) == 0:
            continue
        anchor = float(pos['t_hr'].max())
        ca = g[(g['r'] >= 1.5) & (g['t_hr'] < anchor)]
        cr_rows.append({'stay_id': sid,
                        'cr_fired': bool(len(ca)),
                        'cr_t': float(ca['t_hr'].min()) if len(ca) else np.nan})
    cr = pd.DataFrame(cr_rows).set_index('stay_id')
    capidx = capst.set_index('stay_id').index
    crcap = cr.loc[cr.index.intersection(capidx)]
    fired_pct = round(100 * float(crcap['cr_fired'].mean()), 1)
    print('  creatinine-trigger fired before anchor on %.1f%% of captured stays' % fired_pct)

    res[name] = {
        'n_captured': n_cap,
        'model_lead_median_h': round(float(lead.median()), 1),
        'model_lead_iqr': [round(float(lead.quantile(.25)), 1),
                           round(float(lead.quantile(.75)), 1)],
        'groups': grp,
        'quartile_death_pct': [round(100 * float(v), 2) for v in q['mean']],
        'cr_trigger_fired_pct_of_captured': fired_pct,
        'layer': {'a': A1, 'b': B1},
    }

(OUT / 'outcomes_recalibrated.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print()
print('written ->', OUT / 'outcomes_recalibrated.json')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT VERIFICATION T1: outcome groups on the recalibrated scale.

My own implementation from the task spec (not copying r15 scripts):
  p_rec = sigmoid(0.005067 + 0.268467*logit(p))
  capture def v2: anchor = stay's final positive checkpoint (max t_hr with y==1);
                  captured iff any checkpoint with p_rec>=0.15 and t_hr strictly < anchor.
Outcomes joined from raw MIMIC-IV / eICU csv.gz.
"""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
R = 'E:/TBI subtype/09_tbi_aki'
DATA = 'E:/TBI subtype/data'
A, B = 0.005067, 0.268467

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1-1e-12)
    return np.log(p/(1-p))

def report(name, claim_cap, claim_groups, claim_quart):
    print('='*80); print('T1', name); print('='*80)
    pf = 'f12_preds_M4_ge1.parquet' if 'MIMIC' in name else 'f12_preds_eICU_ge1.parquet'
    pred = pd.read_parquet(f'{R}/results/jinhu/{pf}')
    pred['p_rec'] = 1/(1+np.exp(-(A + B*logit(pred['p'].values))))

    # ---- capture, definition v2 (vectorized, independent of r15 loop style) ----
    pred = pred.sort_values(['stay_id','t_hr'], kind='mergesort')
    pos = pred[pred.y == 1]
    anchor = pos.groupby('stay_id')['t_hr'].max().rename('anchor')
    pred = pred.merge(anchor, on='stay_id', how='left')
    event = pred.groupby('stay_id')['y'].max().rename('event').astype(bool)
    alert_before = pred[(pred.p_rec >= 0.15) & (pred.t_hr < pred.anchor)]
    cap = alert_before.groupby('stay_id').size().reindex(event.index, fill_value=0) > 0
    maxp = pred.groupby('stay_id')['p_rec'].max().rename('max_p_rec')
    st = pd.DataFrame({'event': event, 'captured': cap & event, 'max_p_rec': maxp})
    st.index.name = 'stay_id'

    n_ev = int(st.event.sum()); n_cap = int(st.captured.sum())
    print(f'event stays={n_ev}  captured={n_cap}   CLAIM {claim_cap}  ->',
          'MATCH' if n_cap == claim_cap else '*** MISMATCH ***')

    # ---- outcomes ----
    if 'MIMIC' in name:
        icu = pd.read_csv(f'{DATA}/mimic-iv-3.1/icu/icustays.csv.gz',
                          usecols=['stay_id','hadm_id','los'])
        adm = pd.read_csv(f'{DATA}/mimic-iv-3.1/hosp/admissions.csv.gz',
                          usecols=['hadm_id','hospital_expire_flag'])
        oc = st.reset_index().merge(icu, on='stay_id', how='left').merge(adm, on='hadm_id', how='left')
        print('  join misses (los/death NaN):', int(oc.los.isna().sum()), int(oc.hospital_expire_flag.isna().sum()))
    else:
        ep = pd.read_csv(f'{DATA}/eicu-crd-2.0/patient.csv.gz',
                         usecols=['patientunitstayid','unitdischargestatus','unitdischargeoffset'])
        ep = ep.rename(columns={'patientunitstayid':'stay_id'})
        oc = st.reset_index().merge(ep, on='stay_id', how='left')
        print('  join misses:', int(oc.unitdischargestatus.isna().sum()))
        oc['hospital_expire_flag'] = oc.unitdischargestatus.str.strip().str.lower().eq('expired').astype(float)
        oc['los'] = oc.unitdischargeoffset/1440.0

    ok = True
    for gname, mask in [('captured', oc.event & oc.captured),
                        ('missed',    oc.event & ~oc.captured),
                        ('eventfree', ~oc.event)]:
        g = oc[mask]
        n = len(g); mort = 100*g.hospital_expire_flag.mean(); los = g.los.median()
        cn, cm, cl = claim_groups[gname]
        m_n = (n == cn); m_m = abs(mort-cm) < 0.051; m_l = abs(los-cl) < 0.0051
        ok &= m_n and m_m and m_l
        print(f'  {gname:9s} n={n:5d} ({cn}) mort={mort:5.2f}% ({cm}) los={los:6.3f} ({cl}) '
              f'-> {"MATCH" if m_n and m_m and m_l else "*** MISMATCH ***"}')

    # ---- quartiles of stay-level max p_rec: BOTH interpretations ----
    qbins = pd.qcut(oc.max_p_rec, 4)
    qd = oc.groupby(qbins, observed=True).hospital_expire_flag.mean()*100
    print('  death% by max_p_rec quartile:', [round(v,2) for v in qd.values],
          ' CLAIM', claim_quart,
          '->', 'MATCH' if all(abs(a-b) < 0.011 for a, b in zip(qd.values, claim_quart)) else '*** MISMATCH ***')
    qb = oc.max_p_rec.quantile([0, .25, .5, .75, 1.0])*100
    print('  (alt reading) max_p_rec quartile boundaries %:',
          [round(v,2) for v in qb.values])
    nq = oc.groupby(qbins, observed=True).size()
    print('  quartile n:', list(nq.values))
    print('T1', name, 'OVERALL:', 'MATCH' if ok else 'CHECK ABOVE')

report('MIMIC-IV', 239,
       {'captured': (239, 38.9, 6.47), 'missed': (492, 21.7, 5.45), 'eventfree': (6711, 9.8, 3.40)},
       [2.9, 6.13, 12.85, 24.29])
report('eICU-CRD', 260,
       {'captured': (260, 27.7, 6.91), 'missed': (487, 23.6, 5.21), 'eventfree': (8324, 6.0, 2.89)},
       [1.98, 4.63, 8.16, 15.48])

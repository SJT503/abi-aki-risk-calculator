#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ANALYSIS 5 - Clinical outcome anchor (the benchmarks' second evidence pillar).

The manuscript currently has no clinical outcome anywhere: the Introduction stakes
the clinical claim on mortality and the Results never return to it. Every benchmark
carries an outcome (Tomasev dialysis, Zhang mortality/LOS, Cao staged incidence).

This script attaches hard outcomes to the frozen stays and asks whether the stays the
model captures are the stays that die or stay longer.

Outcomes:
  MIMIC-IV  : hospital mortality (admissions.hospital_expire_flag),
              ICU LOS (icustays.los), hospital LOS (dischtime - admittime)
  eICU-CRD  : unit and hospital mortality (unitdischargestatus / hospitaldischargestatus),
              unit LOS (unitdischargeoffset), hospital LOS (hospitaldischargeoffset)

Two questions:
  E1 outcome gradient by CAPTURE STATUS at the working threshold 0.15
     (captured event stays vs missed event stays vs event-free stays)
  E2 outcome gradient across RISK BANDS (quantiles of the stay-level max probability)
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import sys, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('E:/TBI subtype/09_tbi_aki')
DATA = Path('E:/TBI subtype/data')
OUT = ROOT / '_analysis_r13' / 'out'
OUT.mkdir(parents=True, exist_ok=True)
THR = 0.15


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def capture_flags(pred, thr=THR):
    """Frozen definition version 2: anchor = final positive checkpoint of the stay;
    a stay is captured when an alert fires at a checkpoint EARLIER than the anchor."""
    rows = []
    for sid, g in pred.groupby('stay_id', sort=False):
        g = g.sort_values('t_hr')
        pos = g[g['y'] == 1]
        ev = len(pos) > 0
        anchor = float(pos['t_hr'].max()) if ev else np.nan
        alerted = g['p'] >= thr
        cap = bool(ev and (alerted & (g['t_hr'] < anchor)).any())
        rows.append({'stay_id': sid, 'event': ev, 'captured': cap,
                     'max_p': float(g['p'].max()), 'anchor_hr': anchor,
                     'first_alert_hr': float(g.loc[alerted, 't_hr'].min()) if alerted.any() else np.nan})
    return pd.DataFrame(rows)


def rep(series, label, scale=1.0, unit=''):
    a = series.dropna() * scale
    if len(a) == 0:
        return 'n/a'
    return '%s n=%-5d %s' % (label, len(a), (
        'median %-7.2f' % a.median() if unit == 'cont' else '%.1f%% (%d/%d)' % (
            100 * a.mean(), int(a.sum()), len(a))))


def mimic():
    print()
    print('=' * 72)
    print('MIMIC-IV outcomes')
    print('=' * 72)
    pred = pd.read_parquet(ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet')
    print('  preds: %d rows / %d stays' % (len(pred), pred['stay_id'].nunique()))

    icu = pd.read_csv(DATA / 'mimic-iv-3.1/icu/icustays.csv.gz',
                      usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])
    adm = pd.read_csv(DATA / 'mimic-iv-3.1/hosp/admissions.csv.gz',
                      usecols=['hadm_id', 'admittime', 'dischtime', 'hospital_expire_flag'])
    print('  icustays %d | admissions %d' % (len(icu), len(adm)))

    cap = capture_flags(pred)
    d = cap.merge(icu, on='stay_id', how='left').merge(adm, on='hadm_id', how='left')
    miss = int(d['subject_id'].isna().sum())
    print('  stays without icustays match : %d' % miss)
    d['hosp_days'] = (pd.to_datetime(d['dischtime']) - pd.to_datetime(d['admittime'])).dt.total_seconds() / 86400

    ev = d['event']
    print()
    print('  --- E1: outcome by capture status at threshold %.2f ---' % THR)
    grp = {
        'captured event stays': d[ev & d['captured']],
        'missed event stays  ': d[ev & ~d['captured']],
        'event-free stays    ': d[~ev],
    }
    for k, g in grp.items():
        print('    %s : %s | %s'
              % (k,
                 rep(g['hospital_expire_flag'], 'in-hosp death', unit='prop'),
                 rep(g['los'], 'ICU LOS d', unit='cont')))

    print()
    print('  --- E2: outcome by stay-level risk band ---')
    d['band'] = pd.qcut(d['max_p'], 4, labels=['Q1 lowest', 'Q2', 'Q3', 'Q4 highest'])
    band_tab = d.groupby('band', observed=True).agg(
        n=('stay_id', 'size'),
        deaths=('hospital_expire_flag', 'sum'),
        death_pct=('hospital_expire_flag', lambda s: 100 * s.mean()),
        icu_los_med=('los', 'median'),
        hosp_days_med=('hosp_days', 'median')).round(2)
    print(band_tab.to_string())

    return {'n_stays': int(len(d)), 'n_event': int(ev.sum()),
            'n_captured': int((ev & d['captured']).sum()),
            'e1': {k: {'n': int(len(g)),
                       'death_pct': round(float(100 * g['hospital_expire_flag'].mean()), 2),
                       'icu_los_median': round(float(g['los'].median()), 3)}
                   for k, g in grp.items()},
            'e2': band_tab.reset_index().to_dict('records'),
            'sha': sha256(ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet')[:16]}


def eicu():
    print()
    print('=' * 72)
    print('eICU-CRD outcomes')
    print('=' * 72)
    pred = pd.read_parquet(ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet')
    print('  preds: %d rows / %d stays' % (len(pred), pred['stay_id'].nunique()))

    ep = pd.read_csv(DATA / 'eicu-crd-2.0/patient.csv.gz',
                     usecols=['patientunitstayid', 'unitdischargeoffset',
                              'unitdischargestatus', 'hospitaldischargestatus',
                              'hospitaldischargeoffset', 'unittype'], low_memory=False)
    print('  eICU patient %d' % len(ep))

    cap = capture_flags(pred)
    ep = ep.rename(columns={'patientunitstayid': 'stay_id'})
    d = cap.merge(ep, on='stay_id', how='left')
    print('  stays without patient match : %d' % int(d['unitdischargeoffset'].isna().sum()))
    d['unit_los_d'] = d['unitdischargeoffset'] / 1440.0
    d['hosp_los_d'] = d['hospitaldischargeoffset'] / 1440.0
    # eICU status: 'Expired' means died
    d['died_unit'] = (d['unitdischargestatus'].astype(str).str.lower().str.startswith('expired')).astype(float)
    d['died_hosp'] = (d['hospitaldischargestatus'].astype(str).str.lower().str.startswith('expired')).astype(float)

    ev = d['event']
    print()
    print('  --- E1: outcome by capture status at threshold %.2f ---' % THR)
    grp = {
        'captured event stays': d[ev & d['captured']],
        'missed event stays  ': d[ev & ~d['captured']],
        'event-free stays    ': d[~ev],
    }
    for k, g in grp.items():
        print('    %s : %s | %s | %s'
              % (k,
                 rep(g['died_unit'], 'unit death', unit='prop'),
                 rep(g['died_hosp'], 'hosp death', unit='prop'),
                 rep(g['unit_los_d'], 'unit LOS d', unit='cont')))

    print()
    print('  --- E2: outcome by stay-level risk band ---')
    d['band'] = pd.qcut(d['max_p'], 4, labels=['Q1 lowest', 'Q2', 'Q3', 'Q4 highest'])
    band_tab = d.groupby('band', observed=True).agg(
        n=('stay_id', 'size'),
        unit_death_pct=('died_unit', lambda s: 100 * s.mean()),
        hosp_death_pct=('died_hosp', lambda s: 100 * s.mean()),
        unit_los_med=('unit_los_d', 'median')).round(2)
    print(band_tab.to_string())

    print()
    print('  --- by eICU unit type (the ABI-relevant subgroup) ---')
    ut = d.groupby('unittype', observed=True).agg(
        n=('stay_id', 'size'),
        death_pct=('died_unit', lambda s: 100 * s.mean()),
        los_med=('unit_los_d', 'median')).round(2).sort_values('n', ascending=False).head(8)
    print(ut.to_string())

    return {'n_stays': int(len(d)), 'n_event': int(ev.sum()),
            'n_captured': int((ev & d['captured']).sum()),
            'e1': {k: {'n': int(len(g)),
                       'unit_death_pct': round(float(100 * g['died_unit'].mean()), 2),
                       'hosp_death_pct': round(float(100 * g['died_hosp'].mean()), 2),
                       'unit_los_median': round(float(g['unit_los_d'].median()), 3)}
                   for k, g in grp.items()},
            'e2': band_tab.reset_index().to_dict('records'),
            'unittype': ut.reset_index().to_dict('records'),
            'sha': sha256(ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet')[:16]}


def mini_task():
    res = {'MIMIC-IV': mimic(), 'eICU-CRD': eicu()}
    print()
    print('=' * 72)
    print('SUMMARY - does capture track outcome?')
    print('=' * 72)
    for nm, r in res.items():
        e = r['e1']
        ck = 'unit_death_pct' if nm.startswith('eICU') else 'death_pct'
        lk = 'unit_los_median' if nm.startswith('eICU') else 'icu_los_median'
        print('  %-9s captured %.1f%% death / %.1f d | missed %.1f%% death / %.1f d | free %.1f%% death / %.1f d'
              % (nm, e['captured event stays'][ck], e['captured event stays'][lk],
                 e['missed event stays  '][ck], e['missed event stays  '][lk],
                 e['event-free stays    '][ck], e['event-free stays    '][lk]))
    (OUT / 'a5_outcomes.json').write_text(json.dumps(res, indent=2, default=str), encoding='utf-8')
    print()
    print('  written ->', OUT / 'a5_outcomes.json')


if __name__ == '__main__':
    print('=' * 72)
    print('ANALYSIS 5: clinical outcome anchor')
    print('=' * 72)
    mini_task()

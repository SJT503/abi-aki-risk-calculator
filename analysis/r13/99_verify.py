# -*- coding: utf-8 -*-
"""Independent re-verification of the r13 analysis outputs.

Re-derives each headline number from the raw frozen inputs by a DIFFERENT route
than the analysis script, so a coding slip in the analysis cannot survive.
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('E:/TBI subtype/09_tbi_aki')
OUT = ROOT / '_analysis_r13' / 'out'
DATA = Path('E:/TBI subtype/data')

ok = []


def check(name, got, want, tol=0.0005):
    good = abs(float(got) - float(want)) <= tol
    ok.append((name, good, got, want))
    print('  %-52s got %-10s want %-10s %s' % (name, got, want, 'PASS' if good else '** FAIL **'))


print('=' * 74)
print('VERIFICATION of r13 outputs (independent routes)')
print('=' * 74)

# ---------- V1: AUROC reproduced from a fresh read ----------
print()
print('V1  frozen AUROCs')
p = pd.read_parquet(ROOT / 'results/jinhu/f12_preds_M4_ge1.parquet')
check('MIMIC-IV any-AKI AUROC (frozen p)', round(roc_auc_score(p['y'], p['p']), 4), 0.7694)
e = pd.read_parquet(ROOT / 'results/jinhu/f12_preds_eICU_ge1.parquet')
check('eICU-CRD any-AKI AUROC (frozen p)', round(roc_auc_score(e['y'], e['p']), 4), 0.7773)

# ---------- V2: hospital distribution, recomputed ----------
print()
print('V2  hospital distribution')
hm = pd.read_parquet(ROOT / 'FINAL_20260928/data_current/s1_hospital_map.parquet')
d = e.merge(hm, on='stay_id', how='inner')
# alternative route: ranking-based AUROC per hospital via numpy (Mann-Whitney)
rows = []
for hid, g in d.groupby('hospitalid'):
    npos = int(g['y'].sum())
    if npos == 0 or g['y'].nunique() < 2 or npos < 20:
        continue
    r = pd.Series(g['p']).rank().values
    pos = g['y'].values == 1
    a = (r[pos].sum() - npos * (npos + 1) / 2) / (npos * (len(g) - npos))
    rows.append(a)
a = np.array(rows)
q = np.percentile(a, [25, 50, 75])
check('hospitals with >=20 positives', len(a), 50, tol=0)
check('median hospital AUROC', round(float(q[1]), 3), 0.754, tol=0.001)
check('IQR low', round(float(q[0]), 3), 0.708, tol=0.001)
check('IQR high', round(float(q[2]), 3), 0.778, tol=0.001)
check('share below 0.769', round(float((a < 0.769).mean()), 3), 0.680, tol=0.001)

# ---------- V3: label circularity, recomputed ----------
print()
print('V3  label circularity')
f = pd.read_parquet(ROOT / 'results/n5_abi_rolling_features.parquet',
                    columns=['stay_id', 't_hr', 'label', 'cr_ratio_base', 'cr_delta_since_adm'])
g = pd.read_parquet(ROOT / 'results/n3_m4_abi_grid.parquet')
dd = g[['stay_id', 't_hr', 'onset_hr']].merge(p[['stay_id', 't_hr', 'y', 'p']],
                                              on=['stay_id', 't_hr'], how='inner')
dd = dd.merge(f, on=['stay_id', 't_hr'], how='left', suffixes=('', '_f'))
pos = dd['y'] == 1
r = pd.to_numeric(dd['cr_ratio_base'], errors='coerce')
defn = (r >= 1.5) | (pd.to_numeric(dd['cr_delta_since_adm'], errors='coerce') >= 0.3)
n_obs = int((pos & r.notna()).sum())
n_def = int((pos & r.notna() & defn).sum())
check('MIMIC-IV share of positives definitional', round(n_def / n_obs, 3), 0.053, tol=0.002)

# ratio-alone AUROC on all checkpoints
m = r.notna()
check('cr_ratio_base univariate AUROC (MIMIC)',
      round(roc_auc_score(dd.loc[m, 'y'], r[m]), 3), 0.766, tol=0.002)

def capture_v2(pred, thr=0.15):
    """Definition version 2, re-implemented independently:
    anchor = final positive checkpoint; captured iff an alert fires strictly earlier."""
    out = {}
    for sid, g in pred.groupby('stay_id', sort=False):
        pos_t = g.loc[g['y'] == 1, 't_hr']
        if len(pos_t) == 0:
            out[sid] = False
            continue
        anchor = pos_t.max()
        out[sid] = bool(((g['p'] >= thr) & (g['t_hr'] < anchor)).any())
    return pd.Series(out, name='captured_v2')


# ---------- V4: outcome gradient, recomputed by a different aggregation ----------
print()
print('V4  outcome gradient')
icu = pd.read_csv(DATA / 'mimic-iv-3.1/icu/icustays.csv.gz',
                  usecols=['hadm_id', 'stay_id', 'los'])
adm = pd.read_csv(DATA / 'mimic-iv-3.1/hosp/admissions.csv.gz',
                  usecols=['hadm_id', 'hospital_expire_flag'])
d2 = p.merge(icu, on='stay_id', how='left').merge(adm, on='hadm_id', how='left')
lvl = d2.groupby('stay_id').agg(event=('y', 'max'), max_p=('p', 'max')).reset_index()
meta = d2.drop_duplicates('stay_id')[['stay_id', 'hospital_expire_flag', 'los']]
s = lvl.merge(meta, on='stay_id', how='left')
assert len(s) == lvl.shape[0] and s['hospital_expire_flag'].notna().all(), 'stay-level join broke'
# capture status by the paper's own definition (independent re-implementation)
cv = capture_v2(p).reset_index()
cv.columns = ['stay_id', 'captured_v2']
s = s.merge(cv, on='stay_id', how='left')
s['event'] = s['event'].astype(bool)
check('event-free stay mortality %',
      round(100 * s.loc[~s['event'], 'hospital_expire_flag'].mean(), 1), 9.8, tol=0.2)
check('captured event stay mortality %',
      round(100 * s.loc[s['captured_v2'], 'hospital_expire_flag'].mean(), 1), 41.1, tol=0.2)
check('captured event stays (count)',
      int(s['captured_v2'].sum()), 107, tol=0)
q4 = s[s['max_p'] >= s['max_p'].quantile(0.75)]
check('top-quartile stay mortality %',
      round(100 * q4['hospital_expire_flag'].mean(), 1), 24.3, tol=0.3)

# ---------- V5: eICU outcome gradient ----------
print()
print('V5  eICU outcome gradient')
ep = pd.read_csv(DATA / 'eicu-crd-2.0/patient.csv.gz',
                 usecols=['patientunitstayid', 'unitdischargestatus'], low_memory=False)
ep = ep.rename(columns={'patientunitstayid': 'stay_id'})
e2 = e.merge(ep, on='stay_id', how='left')
e2['died'] = e2['unitdischargestatus'].astype(str).str.lower().str.startswith('expired').astype(float)
elvl = e2.groupby('stay_id').agg(event=('y', 'max'), max_p=('p', 'max')).reset_index()
emeta = e2.drop_duplicates('stay_id')[['stay_id', 'died']]
es = elvl.merge(emeta, on='stay_id', how='left')
assert len(es) == elvl.shape[0] and es['died'].notna().all(), 'eICU stay-level join broke'
ecv = capture_v2(e).reset_index()
ecv.columns = ['stay_id', 'captured_v2']
es = es.merge(ecv, on='stay_id', how='left')
es['event'] = es['event'].astype(bool)
check('eICU event-free unit mortality %',
      round(100 * es.loc[~es['event'], 'died'].mean(), 1), 6.0, tol=0.2)
check('eICU captured event stay mortality %',
      round(100 * es.loc[es['captured_v2'], 'died'].mean(), 1), 35.4, tol=0.3)
check('eICU captured event stays (count)',
      int(es['captured_v2'].sum()), 79, tol=0)

print()
print('=' * 74)
bad = [x for x in ok if not x[1]]
print('checks: %d | passed: %d | failed: %d' % (len(ok), len(ok) - len(bad), len(bad)))
for b in bad:
    print('   FAIL', b[0], 'got', b[2], 'want', b[3])
print()
print('VERDICT:', 'ALL PASS - r13 numbers independently reproduced' if not bad
      else 'FAILURES PRESENT - do not use r13 numbers')
sys.exit(1 if bad else 0)

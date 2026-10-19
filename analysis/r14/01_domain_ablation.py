#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""r14 ANALYSIS 1 - Domain ablation (PI annotation 2026-10-19: "可以补做域消融分析呀").

Replicates the f12 training pipeline exactly (same split, same selection rule,
same hyperparameters, same seed), then retrains with every creatinine-derived
feature removed from the selected set, and scores the same frozen external
cohorts. Canaries assert the full-feature reproduction before any ablation
number is trusted.

Question answered: how much of the external discrimination depends on
creatinine-derived features (which share a measurement stream with the label),
and how much survives on the other 130-odd features alone?
"""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
R = Path('E:/TBI subtype/09_tbi_aki')
J = R / 'results' / 'jinhu'
OUT = R / '_analysis_r14' / 'out'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=SEED, verbose=-1)

print('=' * 74)
print('r14-1  DOMAIN ABLATION: creatinine-family removed, pipeline replicated')
print('=' * 74)

# ---------- 1. replicate f12 pipeline ----------
jf = pd.read_parquet(J / 'jinhu_features.parquet').rename(columns={'stay_id': 'hadm_id'})
jf['hadm_id'] = jf.hadm_id.astype(str)
all_nan = [c for c in jf.columns if c not in ('hadm_id', 't_hr') and jf[c].isna().all()]
CAND = [c for c in jf.columns if c not in ('hadm_id', 't_hr') and c not in all_nan]

sub = pd.read_csv('F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data/'
                  'medical_record_front_page.csv', usecols=['hadm_id', 'subject_id'],
                  dtype=str)
sub['hadm_id'] = sub.hadm_id.astype(str)
jf['subject'] = jf.hadm_id.map(dict(zip(sub.hadm_id, sub.subject_id)))

labs = pd.read_parquet(J / 'jinhu_labs.parquet')
coh = pd.read_parquet(J / 'jinhu_cohort.parquet').set_index('hadm_id')
coh.index = coh.index.astype(str)
cr = labs[labs.fam == 'cr'][['hadm_id', 'tmin', 'val']].copy()
cr['hadm_id'] = cr.hadm_id.astype(str)
cr['hr'] = (cr.tmin - cr.hadm_id.map(coh.icu_start)) / 60.0
cr = cr[cr.val.between(0.1, 30)].sort_values(['hadm_id', 'hr'])
cr['t_hr'] = (cr.hr // 6) * 6
cb = cr.groupby(['hadm_id', 't_hr']).val.max().reset_index()
mg = jf[['hadm_id', 't_hr', 'cr_base']].drop_duplicates().merge(
    cb, on=['hadm_id', 't_hr'], how='left').sort_values(['hadm_id', 't_hr'])
mg['f'] = np.nan
for s in range(9):
    sh = mg.groupby('hadm_id').val.shift(-s)
    mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)
fold = mg.f / mg.cr_base.clip(lower=0.1)
rise = mg.f - mg.cr_base
aki = (fold >= 1.5) | (rise >= 0.3)
st = pd.Series(0, index=mg.index)
st[aki.fillna(False)] = 1
st[fold >= 2.0] = 2
st[(fold >= 3.0) | ((mg.f >= 4.0) & aki)] = 3
st[mg.f.isna()] = 0
for t in [1, 2, 3]:
    mg[f'ge{t}'] = (st >= t).astype(int)
jf = jf.merge(mg[['hadm_id', 't_hr', 'ge1', 'ge2', 'ge3']],
              on=['hadm_id', 't_hr'], how='left')

subjects = np.array(sorted(jf.subject.unique()))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = jf.subject.isin(train_subj).values
te = ~tr
print('train rows %d | test rows %d | candidates %d | subjects %d (f12: 9,837/2,646/280/1,031)'
      % (tr.sum(), te.sum(), len(CAND), len(subjects)))

# ---------- 2. feature selection (same rule) ----------
Xtr_sel = jf[CAND].values[tr]
ytr_sel = jf.ge1.values[tr]
groups_sel = jf.subject.values[tr]
acc = np.zeros(len(CAND))
for a, b in GroupKFold(5).split(Xtr_sel, ytr_sel, groups_sel):
    m = lgb.LGBMClassifier(**{**PARAMS, 'importance_type': 'gain'})
    m.fit(Xtr_sel[a], ytr_sel[a])
    imp = m.feature_importances_.astype(float)
    acc += imp / imp.sum()
imp_mean = acc / 5
order = np.argsort(imp_mean)[::-1]
cum = np.cumsum(imp_mean[order]) / imp_mean.sum()
SEL = [CAND[i] for i in order[:int(np.searchsorted(cum, 0.99) + 1)]]
print('selected %d features (f12 log recorded 166; small drift tolerated if AUROC canary passes)' % len(SEL))
if len(SEL) != 166:
    print('  NOTE: selection count drifted (%d vs 166) - AUROC canary below is the binding gate' % len(SEL))

# ---------- 3. external feature tables + label machinery ----------
def label_machinery(cr_df, feat_df, key):
    mgx = feat_df[[key, 't_hr', 'cr_base']].drop_duplicates().merge(
        cr_df.groupby([key, 't_hr']).cr_val.max().reset_index(),
        on=[key, 't_hr'], how='left').sort_values([key, 't_hr'])
    mgx['f'] = np.nan
    for s in range(9):
        sh = mgx.groupby(key).cr_val.shift(-s)
        mgx['f'] = pd.DataFrame({'c': mgx['f'], 'n': sh}).max(axis=1)
    fo = mgx.f / mgx.cr_base.clip(lower=0.1)
    ri = mgx.f - mgx.cr_base
    ak = (fo >= 1.5) | (ri >= 0.3)
    stx = pd.Series(0, index=mgx.index)
    stx[ak.fillna(False)] = 1
    stx[fo >= 2.0] = 2
    stx[(fo >= 3.0) | ((mgx.f >= 4.0) & ak)] = 3
    stx[mgx.f.isna()] = 0
    for t in [1, 2, 3]:
        mgx[f'ge{t}'] = (stx >= t).astype(int)
    return mgx[[key, 't_hr', 'ge1', 'ge2', 'ge3']]

m4 = pd.read_parquet(R / 'results/n5_abi_rolling_features.parquet')
mlab = pd.read_parquet(R / 'results/n2_m4_abi_cr_serial.parquet').rename(
    columns={'valuenum': 'cr_val'})
mlab = mlab[mlab.cr_val.between(0.1, 30)].copy()
mlab['t_hr'] = (mlab.hr // 6) * 6
m4f = m4[['stay_id', 't_hr'] + SEL].merge(
    label_machinery(mlab[['stay_id', 't_hr', 'cr_val']], m4, 'stay_id'),
    on=['stay_id', 't_hr'], how='left')

eicu = pd.read_parquet(R / 'results/p1_expanded_features.parquet')
elab = pd.read_parquet(R / 'results/n8_2_labs_series.parquet')
ecr = elab[(elab.labname == 'creatinine') & (elab.labtypeid == 1)].rename(
    columns={'sid': 'stay_id', 'labresult': 'cr_val'})
ecr = ecr[ecr.cr_val.between(0.1, 30)].copy()
ecr['t_hr'] = (ecr.hr // 6) * 6
eif = eicu[['stay_id', 't_hr'] + SEL].merge(
    label_machinery(ecr[['stay_id', 't_hr', 'cr_val']], eicu, 'stay_id'),
    on=['stay_id', 't_hr'], how='left')
print('external tables ready: M4 %d rows | eICU %d rows' % (len(m4f), len(eif)))

# ---------- 4. full-model canary ----------
res = {}
for t in [1, 3]:
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(jf[SEL].values[tr], jf[f'ge{t}'].values[tr])
    a_m = roc_auc_score(m4f[f'ge{t}'], m.predict_proba(m4f[SEL].values)[:, 1])
    a_e = roc_auc_score(eif[f'ge{t}'], m.predict_proba(eif[SEL].values)[:, 1])
    res[f'full_ge{t}'] = {'M4': round(a_m, 4), 'eICU': round(a_e, 4)}
    print('CANARY full ge%d: MIMIC %.4f eICU %.4f' % (t, a_m, a_e))
assert abs(res['full_ge1']['M4'] - 0.7694) < 0.001, 'ge1 M4 canary failed'
assert abs(res['full_ge1']['eICU'] - 0.7773) < 0.001, 'ge1 eICU canary failed'

# ---------- 5. ablation: drop creatinine family ----------
CRFAM = [c for c in SEL if c.startswith('cr_')]
SEL_NC = [c for c in SEL if not c.startswith('cr_')]
print()
print('creatinine-family features inside SEL: %d -> %s' % (len(CRFAM), CRFAM))
print('ablated feature set: %d features' % len(SEL_NC))
for t in [1, 3]:
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(jf[SEL_NC].values[tr], jf[f'ge{t}'].values[tr])
    a_m = roc_auc_score(m4f[f'ge{t}'], m.predict_proba(m4f[SEL_NC].values)[:, 1])
    a_e = roc_auc_score(eif[f'ge{t}'], m.predict_proba(eif[SEL_NC].values)[:, 1])
    res[f'nocr_ge{t}'] = {'M4': round(a_m, 4), 'eICU': round(a_e, 4)}
    print('ABLATION no-cr ge%d: MIMIC %.4f eICU %.4f' % (t, a_m, a_e))

res['n_cr_in_sel'] = len(CRFAM)
res['cr_features'] = CRFAM
res['n_sel'] = len(SEL)
res['n_sel_nocr'] = len(SEL_NC)
(OUT / 'domain_ablation.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print()
print('written ->', OUT / 'domain_ablation.json')
print()
print('SUMMARY: any-AKI %.3f/%.3f -> no-cr %.3f/%.3f | severe %.3f/%.3f -> no-cr %.3f/%.3f' % (
    res['full_ge1']['M4'], res['full_ge1']['eICU'],
    res['nocr_ge1']['M4'], res['nocr_ge1']['eICU'],
    res['full_ge3']['M4'], res['full_ge3']['eICU'],
    res['nocr_ge3']['M4'], res['nocr_ge3']['eICU']))

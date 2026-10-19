#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Boundary adjudication with the ORIGINAL sklearn route (same estimator calls
as 02_rich_comparator.py) but READ-ONLY: prints at 6 dp, writes nothing.
Determines whether the frozen 0.6335 (eICU rich AUROC) truly rounds to 0.634,
and the exact MIMIC static-3 bootstrap bounds (frozen 0.1305)."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
R = 'E:/TBI subtype/09_tbi_aki'
J = R + '/results/jinhu'
SEED = 42
BASE = ['age', 'male', 'charlson', 'cr_base']
VITALS = ['hr_rate_locf', 'sbp_locf', 'spo2_locf', 'rr_locf']

jf = pd.read_parquet(J + '/jinhu_features.parquet').rename(columns={'stay_id': 'hadm_id'})
jf['hadm_id'] = jf.hadm_id.astype(str)
sub = pd.read_csv('F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data/'
                  'medical_record_front_page.csv', usecols=['hadm_id', 'subject_id'], dtype=str)
jf['subject'] = jf.hadm_id.map(dict(zip(sub.hadm_id, sub.subject_id)))
labs = pd.read_parquet(J + '/jinhu_labs.parquet')
coh = pd.read_parquet(J + '/jinhu_cohort.parquet').set_index('hadm_id')
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
stg = pd.Series(0, index=mg.index)
stg[aki.fillna(False)] = 1
stg[mg.f.isna()] = 0
mg['ge1'] = (stg >= 1).astype(int)
jf = jf.merge(mg[['hadm_id', 't_hr', 'ge1']], on=['hadm_id', 't_hr'], how='left')
subjects = np.array(sorted(jf.subject.unique()))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = jf.subject.isin(train_subj).values
VARS = BASE + VITALS

lr3 = LogisticRegression(max_iter=1000, class_weight='balanced').fit(
    jf[['age', 'charlson', 'cr_base']].fillna(0).values[tr], jf.ge1.values[tr])
lrR = LogisticRegression(max_iter=1000, class_weight='balanced').fit(
    jf[VARS].fillna(0).values[tr], jf.ge1.values[tr])
print('sklearn lr3 n_iter_', lr3.n_iter_, '| lrR n_iter_', lrR.n_iter_)


def label_ge1(cr_df, feat_df, key):
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
    stx[mgx.f.isna()] = 0
    mgx['ge1'] = (stx >= 1).astype(int)
    return mgx[[key, 't_hr', 'ge1']]


m4 = pd.read_parquet(R + '/results/n5_abi_rolling_features.parquet')
mlab = pd.read_parquet(R + '/results/n2_m4_abi_cr_serial.parquet').rename(columns={'valuenum': 'cr_val'})
mlab = mlab[mlab.cr_val.between(0.1, 30)].copy()
mlab['t_hr'] = (mlab.hr // 6) * 6
m4f = m4.merge(label_ge1(mlab[['stay_id', 't_hr', 'cr_val']], m4, 'stay_id'),
               on=['stay_id', 't_hr'], how='left')
eicu = pd.read_parquet(R + '/results/p1_expanded_features.parquet')
elab = pd.read_parquet(R + '/results/n8_2_labs_series.parquet')
ecr = elab[(elab.labname == 'creatinine') & (elab.labtypeid == 1)].rename(
    columns={'sid': 'stay_id', 'labresult': 'cr_val'})
ecr = ecr[ecr.cr_val.between(0.1, 30)].copy()
ecr['t_hr'] = (ecr.hr // 6) * 6
eif = eicu.merge(label_ge1(ecr[['stay_id', 't_hr', 'cr_val']], eicu, 'stay_id'),
                 on=['stay_id', 't_hr'], how='left')
pm = pd.read_parquet(J + '/f12_preds_M4_ge1.parquet')
pe = pd.read_parquet(J + '/f12_preds_eICU_ge1.parquet')

for name, feat, pred in [('MIMIC-IV', m4f, pm), ('eICU-CRD', eif, pe)]:
    dd = pred.merge(feat, on=['stay_id', 't_hr'], suffixes=('', '_f'))
    X3 = dd[['age', 'charlson', 'cr_base']].fillna(0).values
    XR = dd[VARS].fillna(0).values
    print('%s sklearn static3 AUROC %.6f | rich8 AUROC %.6f' %
          (name, roc_auc_score(dd['y'], lr3.predict_proba(X3)[:, 1]),
           roc_auc_score(dd['y'], lrR.predict_proba(XR)[:, 1])))

# MIMIC static-3 bootstrap with the original estimator's predictions
dd = pm.merge(m4f, on=['stay_id', 't_hr'], suffixes=('', '_f'))
X3 = dd[['age', 'charlson', 'cr_base']].fillna(0).values
p3 = lr3.predict_proba(X3)[:, 1]
rng = np.random.default_rng(SEED)
uniq = dd['stay_id'].unique()
idx = {u: np.where(dd['stay_id'].values == u)[0] for u in uniq}
yv = dd['y'].values
diffs = []
for _ in range(1000):
    sel = rng.choice(uniq, size=len(uniq), replace=True)
    ii = np.concatenate([idx[u] for u in sel])
    if yv[ii].sum() > 0 and (1 - yv[ii]).sum() > 0:
        diffs.append(roc_auc_score(yv[ii], dd['p'].values[ii]) - roc_auc_score(yv[ii], p3[ii]))
dfn = np.array(diffs)
print('MIMIC static3 (sklearn preds) mean %.6f lo %.6f hi %.6f' %
      (dfn.mean(), np.percentile(dfn, 2.5), np.percentile(dfn, 97.5)))

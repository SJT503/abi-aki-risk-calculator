#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""r16-3  Development-side severe-AKI recalibration layer (PI chose option A).

Fits a two-parameter logistic layer on patient-grouped OOF predictions of the
JinhuaNSICU training split (identical protocol to the any-AKI layer), then
applies it unchanged to the internal test set and both external cohorts.

Canaries: (1) the ge1 OOF layer must reproduce the frozen constants
a=0.005067, b=0.268467; (2) the ge1 model must reproduce 0.7694/0.7773.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import expit, logit

sys.stdout.reconfigure(encoding='utf-8')
R = Path('E:/TBI subtype/09_tbi_aki')
J = R / 'results' / 'jinhu'
OUT = R / '_analysis_r16' / 'out'
SEED = 42
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=SEED, verbose=-1)

print('=' * 74)
print('r16-3  SEVERE-AKI recalibration layer (development side)')
print('=' * 74)

# ---- pipeline (identical to r14/r15/r16-2) ----
jf = pd.read_parquet(J / 'jinhu_features.parquet').rename(columns={'stay_id': 'hadm_id'})
jf['hadm_id'] = jf.hadm_id.astype(str)
all_nan = [c for c in jf.columns if c not in ('hadm_id', 't_hr') and jf[c].isna().all()]
CAND = [c for c in jf.columns if c not in ('hadm_id', 't_hr') and c not in all_nan]
sub = pd.read_csv('F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data/'
                  'medical_record_front_page.csv', usecols=['hadm_id', 'subject_id'], dtype=str)
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
assert len(SEL) == 166

# ---- canary 1: ge1 OOF layer reproduces frozen constants ----
oof1 = np.zeros(tr.sum())
for a, b in GroupKFold(5).split(jf[SEL].values[tr], jf.ge1.values[tr], jf.subject.values[tr]):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(jf[SEL].values[tr][a], jf.ge1.values[tr][a])
    oof1[b] = m.predict_proba(jf[SEL].values[tr][b])[:, 1]
lg1 = logit(np.clip(oof1, 1e-12, 1 - 1e-12)).reshape(-1, 1)
cal1 = LogisticRegression(C=1e6, max_iter=2000).fit(lg1, jf.ge1.values[tr])
A1, B1 = float(cal1.intercept_[0]), float(cal1.coef_[0][0])
print('canary ge1 layer: a=%.6f b=%.6f (frozen: 0.005067 0.268467)' % (A1, B1))
assert abs(A1 - 0.005067) < 0.001 and abs(B1 - 0.268467) < 0.001, 'ge1 layer canary failed'

# ---- severe OOF layer ----
oof3 = np.zeros(tr.sum())
for a, b in GroupKFold(5).split(jf[SEL].values[tr], jf.ge3.values[tr], jf.subject.values[tr]):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(jf[SEL].values[tr][a], jf.ge3.values[tr][a])
    oof3[b] = m.predict_proba(jf[SEL].values[tr][b])[:, 1]
print('severe OOF: %d checkpoints, %d positive' % (len(oof3), int(jf.ge3.values[tr].sum())))
lg3 = logit(np.clip(oof3, 1e-12, 1 - 1e-12)).reshape(-1, 1)
cal3 = LogisticRegression(C=1e6, max_iter=2000).fit(lg3, jf.ge3.values[tr])
A3, B3 = float(cal3.intercept_[0]), float(cal3.coef_[0][0])
print('SEVERE layer: a=%.6f b=%.6f  (a=%.4f, b=%.4f for reporting)' % (A3, B3, A3, B3))

# ---- train full severe model and apply layer to external cohorts ----
models = {}
for t in [1, 3]:
    models[t] = lgb.LGBMClassifier(**PARAMS)
    models[t].fit(jf[SEL].values[tr], jf[f'ge{t}'].values[tr])


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
mlab = pd.read_parquet(R / 'results/n2_m4_abi_cr_serial.parquet').rename(columns={'valuenum': 'cr_val'})
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

# ---- canary 2: ge1 model reproduces frozen AUROCs ----
a_m = roc_auc_score(m4f['ge1'], models[1].predict_proba(m4f[SEL].values)[:, 1])
a_e = roc_auc_score(eif['ge1'], models[1].predict_proba(eif[SEL].values)[:, 1])
print('canary ge1 model: MIMIC %.4f eICU %.4f' % (a_m, a_e))
assert abs(a_m - 0.7694) < 0.001 and abs(a_e - 0.7773) < 0.001

# ---- severe raw + recalibrated predictions ----
p3_m = models[3].predict_proba(m4f[SEL].values)[:, 1]
p3_e = models[3].predict_proba(eif[SEL].values)[:, 1]
p3r_m = expit(A3 + B3 * logit(np.clip(p3_m, 1e-12, 1 - 1e-12)))
p3r_e = expit(A3 + B3 * logit(np.clip(p3_e, 1e-12, 1 - 1e-12)))


def slope_ece(y, p):
    x = logit(np.clip(p, 1e-9, 1 - 1e-9)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(x, y)
        sl = float(lr.coef_[0][0])
    except Exception:
        sl = float('nan')
    q = pd.qcut(pd.Series(p), 10, duplicates='drop')
    g = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)}).groupby(q, observed=True).mean()
    ece = float(np.abs(g['p'] - g['y']).mean())
    return sl, ece


print()
print('--- severe calibration: raw -> development-layer recalibrated ---')
res = {'layer': {'a': round(A3, 6), 'b': round(B3, 6)},
       'n_severe_oof_train': int(jf.ge3.values[tr].sum()),
       'layer_basis': '5-fold patient-grouped OOF severe predictions of the training split'}

for name, df, praw, prec in [
        ('MIMIC-IV', m4f, p3_m, p3r_m),
        ('eICU-CRD', eif, p3_e, p3r_e)]:
    y = df['ge3'].values
    auc_r = roc_auc_score(y, praw)
    sl_r, ece_r = slope_ece(y, praw)
    sl_c, ece_c = slope_ece(y, prec)
    # deployment views at the recalibrated scale: what threshold captures what
    event_stays = df[df.ge3 == 1].stay_id.unique()
    for thr in [0.05, 0.10, 0.15, 0.20]:
        alert = prec >= thr
        # stay-level capture
        cap = 0
        for sid in event_stays:
            g = df[df.stay_id == sid]
            g_alert = prec[df.stay_id == sid] >= thr
            g_pos = g.ge3 == 1
            anchor = float(g.loc[g_pos, 't_hr'].max())
            if (g_alert & (g.t_hr < anchor)).any():
                cap += 1
        n_alert = df[alert].stay_id.nunique()
        print('  %s thr%.2f: capture %d/%d (%.1f%%), alert stays %d' % (
            name, thr, cap, len(event_stays), 100 * cap / len(event_stays), n_alert))
    res[name] = {'severe_auc': round(float(auc_r), 4),
                 'slope_raw': round(sl_r, 3), 'slope_recal': round(sl_c, 3),
                 'ece_raw': round(ece_r, 4), 'ece_recal': round(ece_c, 4)}
    print('%s: AUROC %.4f | slope raw %.3f -> recal %.3f | ECE raw %.4f -> recal %.4f' % (
        name, auc_r, sl_r, sl_c, ece_r, ece_c))

(OUT / 'severe_recalibration.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
print()
print('written ->', OUT / 'severe_recalibration.json')

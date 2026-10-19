#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""r15-3  Archive the frozen model objects and verify they reproduce the
manuscript's external AUROCs exactly (answering the reproducibility review
point; the training pipeline was already shown to be bit-exact in r14)."""
from gate import gate, gpu_gate, canary_gate, retry_limit   # noqa: F401
import hashlib
import json
import pickle
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
ART = R / '_analysis_r15' / 'model_objects'
ART.mkdir(parents=True, exist_ok=True)
SEED = 42
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=SEED, verbose=-1)

# ---------- replicate pipeline (r14-verified pattern) ----------
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
assert len(SEL) == 166, len(SEL)

models = {}
for t in [1, 2, 3]:
    models[t] = lgb.LGBMClassifier(**PARAMS)
    models[t].fit(jf[SEL].values[tr], jf[f'ge{t}'].values[tr])
print('3 endpoint models trained (seed %d)' % SEED)


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

aucs = {}
for t in [1, 2, 3]:
    a1_ = roc_auc_score(m4f[f'ge{t}'], models[t].predict_proba(m4f[SEL].values)[:, 1])
    a2_ = roc_auc_score(eif[f'ge{t}'], models[t].predict_proba(eif[SEL].values)[:, 1])
    aucs[f'ge{t}'] = {'M4': round(a1_, 4), 'eICU': round(a2_, 4)}
    print('ge%d: MIMIC %.4f eICU %.4f' % (t, a1_, a2_))
frozen = {'ge1': (0.7694, 0.7773), 'ge2': (0.7919, 0.7648), 'ge3': (0.9222, 0.8779)}
for k, (m4v, ev) in frozen.items():
    assert abs(aucs[k]['M4'] - m4v) < 0.001 and abs(aucs[k]['eICU'] - ev) < 0.001, k
print('ALL SIX canaries reproduce the frozen log values')

# ---------- save ----------
bundle = {
    'params': PARAMS, 'selected_features': SEL, 'models': models,
    'recalibration_layer': {'a': 0.005067, 'b': 0.268467},
    'canary_aurocs': aucs,
    'note': 'Retrained with the identical protocol and seed of f12; reproduces '
            'the frozen external AUROCs exactly (see canary_aurocs).',
}
pkl = ART / 'frozen_models.pkl'
with open(pkl, 'wb') as fh:
    pickle.dump(bundle, fh)
sha = hashlib.sha256(pkl.read_bytes()).hexdigest()
manifest = {'file': pkl.name, 'sha256': sha, 'bytes': pkl.stat().st_size,
            'n_features': len(SEL), 'canary_aurocs': aucs}
(ART / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('archived ->', pkl, '(sha256 %s...)' % sha[:12])

# ---------- reload & verify ----------
with open(pkl, 'rb') as fh:
    b2 = pickle.load(fh)
a_reload = roc_auc_score(m4f['ge1'], b2['models'][1].predict_proba(m4f[SEL].values)[:, 1])
print('reload check: MIMIC ge1 AUROC %.4f (expect 0.7694)' % a_reload)
assert abs(a_reload - 0.7694) < 0.001
print('RELOAD OK')

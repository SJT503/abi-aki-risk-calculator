#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT verification (read-only recomputation) of r16-2 rich comparator.

Own route: Newton-Raphson penalized logistic (ridge lambda=1 on slopes,
intercept exempt, balanced class weights) instead of sklearn; own AUROC
(Mann-Whitney average-rank); identical split/label/bootstrap semantics as
documented for the original (sorted subjects, legacy seed-42 permutation,
first 80%; default_rng(42) stay-level paired bootstrap, 1000 draws).
Prints to stdout only; writes nothing to out/.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import expit

sys.stdout.reconfigure(encoding='utf-8')
R = 'E:/TBI subtype/09_tbi_aki'
J = R + '/results/jinhu'
SEED = 42
BASE = ['age', 'male', 'charlson', 'cr_base']
VITALS = ['hr_rate_locf', 'sbp_locf', 'spo2_locf', 'rr_locf']


def fit_lr(X, y, penalty=1.0, max_iter=400, tol=1e-11):
    """Balanced-weight ridge logistic (intercept unpenalized) via Newton."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    n1 = y.sum()
    w = np.where(y > 0.5, n / (2.0 * n1), n / (2.0 * (n - n1)))
    Xm = np.column_stack([np.ones(n), X])
    m = (w * y).sum() / w.sum()
    beta = np.zeros(p + 1)
    beta[0] = np.log(m / (1.0 - m)) if 0.0 < m < 1.0 else 0.0
    lam = np.zeros(p + 1)
    lam[1:] = penalty

    def nll(b):
        e = Xm @ b
        return -np.sum(w * (y * e - np.logaddexp(0.0, e))) + 0.5 * np.sum(lam * b * b)

    for _ in range(max_iter):
        e = Xm @ beta
        pr = expit(e)
        Wd = w * pr * (1.0 - pr)
        g = Xm.T @ (w * (pr - y)) + lam * beta
        H = (Xm * Wd[:, None]).T @ Xm + np.diag(lam)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        cur = nll(beta)
        t = 1.0
        while t > 1e-10 and nll(beta - t * step) > cur - 1e-14:
            t *= 0.5
        beta = beta - t * step
        if np.max(np.abs(g)) < tol * max(1.0, float(np.abs(beta).max())):
            break
    return beta


def auroc(y, p):
    y = np.asarray(y)
    p = np.asarray(p, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p, method='average')
    return (r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


# ---------------- Jinhua features + subject map ----------------
jf = pd.read_parquet(J + '/jinhu_features.parquet').rename(columns={'stay_id': 'hadm_id'})
jf['hadm_id'] = jf.hadm_id.astype(str)
sub = pd.read_csv('F:/JinhuaNSICU/JinhuaNSICU_db_extracted/JinhuaNSICU_db/data/'
                  'medical_record_front_page.csv', usecols=['hadm_id', 'subject_id'], dtype=str)
jf['subject'] = jf.hadm_id.map(dict(zip(sub.hadm_id, sub.subject_id)))
print('jf rows %d | unmapped subjects: %d' % (len(jf), jf.subject.isna().sum()))

# ---------------- training labels (Jinhua ge1 machinery) ----------------
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
print('train table rows %d | ge1 NaN %d | ge1 rate %.4f' %
      (len(jf), jf.ge1.isna().sum(), jf.ge1.mean()))

subjects = np.array(sorted(jf.subject.unique()))
np.random.seed(SEED)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = jf.subject.isin(train_subj).values
print('subjects %d | train subjects %d | train rows %d' %
      (len(subjects), len(train_subj), tr.sum()))

VARS = BASE + VITALS
print('comparator panel:', VARS)

# ---------------- train static-3 and rich-8 (my own fit) ----------------
X3t = jf[['age', 'charlson', 'cr_base']].fillna(0).values
XRt = jf[VARS].fillna(0).values
b3 = fit_lr(X3t[tr], jf.ge1.values[tr])
bR = fit_lr(XRt[tr], jf.ge1.values[tr])
print('my static-3 coef:', np.round(b3, 4))
print('my rich-8   coef:', np.round(bR, 4))


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


def paired_boot(df, p_model, p_comp, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = df['stay_id'].unique()
    idx = {u: np.where(df['stay_id'].values == u)[0] for u in uniq}
    yv = df['y'].values
    diffs = []
    for _ in range(n):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[u] for u in sel])
        if yv[ii].sum() > 0 and (1 - yv[ii]).sum() > 0:
            diffs.append(auroc(yv[ii], p_model[ii]) - auroc(yv[ii], p_comp[ii]))
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


for name, feat, pred in [('MIMIC-IV', m4f, pm), ('eICU-CRD', eif, pe)]:
    dd = pred.merge(feat, on=['stay_id', 't_hr'], suffixes=('', '_f'))
    print()
    print('%s: pred rows %d -> merged rows %d | stays %d | pos rate %.4f' %
          (name, len(pred), len(dd), dd.stay_id.nunique(), dd['y'].mean()))
    X3 = dd[['age', 'charlson', 'cr_base']].fillna(0).values
    XR = dd[VARS].fillna(0).values
    p3 = expit(b3[0] + X3 @ b3[1:])
    pR = expit(bR[0] + XR @ bR[1:])
    a3 = auroc(dd['y'].values, p3)
    aR = auroc(dd['y'].values, pR)
    am = auroc(dd['y'].values, dd['p'].values)
    dR, loR, hiR = paired_boot(dd, dd['p'].values, pR)
    d3, lo3, hi3 = paired_boot(dd, dd['p'].values, p3)
    print('MY  %s: model %.4f | static-3 %.4f (d %.3f [%.3f, %.3f]) | rich-8 %.4f (d %.3f [%.3f, %.3f])'
          % (name, am, a3, d3, lo3, hi3, aR, dR, loR, hiR))
    print('CLM %s: model 0.7694/0.7773 | s3 0.6116/0.6129 (d 0.158 [0.131,0.185]/0.166) | '
          'rich 0.624/0.634 (d 0.145 [0.122,0.170]/[0.122,0.167])' % name)

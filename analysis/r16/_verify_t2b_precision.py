#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precision pass (read-only), part 2:
(1) hospital-level any-AKI AUROC stats recomputed from frozen eICU predictions
    (median/IQR/range over >=20-positive hospitals; count below pooled values).
(2) cross-check against frozen s1_hospital_auroc.csv.
(3) rounding-boundary numbers from rich comparator (eICU rich AUROC,
    MIMIC static-3 bootstrap bounds) at 5-6 dp.
Prints only; writes nothing."""
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


# ================= part 1: hospital AUROC stats =================
pe = pd.read_parquet(J + '/f12_preds_eICU_ge1.parquet')
hm = pd.read_parquet(R + '/FINAL_20260928/data_current/s1_hospital_map.parquet')
d = pe.merge(hm, on='stay_id', how='inner')
rows = []
for h, gdf in d.groupby('hospitalid'):
    npos = int(gdf['y'].sum())
    if npos >= 20:
        rows.append((h, len(gdf), npos, auroc(gdf['y'].values, gdf['p'].values)))
h = pd.DataFrame(rows, columns=['hospitalid', 'n', 'npos', 'auc']).sort_values('hospitalid')
a = h.auc.values
print('hospitals >=20 pos: %d' % len(h))
print('median %.4f | IQR %.4f-%.4f (width %.4f) | range %.4f-%.4f' %
      (np.median(a), np.quantile(a, .25), np.quantile(a, .75),
       np.quantile(a, .75) - np.quantile(a, .25), a.min(), a.max()))
print('claim: median 0.754, IQR 0.708-0.778 (0.070), range 0.518-0.877')
print('below 0.777 (eICU pooled): %d' % (a < 0.777).sum())
print('below 0.769 (MIMIC value): %d' % (a < 0.769).sum())
print('below pooled-over-50 (my pooled = %.4f): %d' %
      (auroc(d[d.hospitalid.isin(h.hospitalid)]['y'].values,
             d[d.hospitalid.isin(h.hospitalid)]['p'].values),
       (a < auroc(d[d.hospitalid.isin(h.hospitalid)]['y'].values,
                  d[d.hospitalid.isin(h.hospitalid)]['p'].values)).sum()))
print('whole-eICU pooled AUROC = %.4f' % auroc(d['y'].values, d['p'].values))
# tie-check for 34
for thr in (0.769, 0.777):
    print('  hospitals strictly below %.3f: %d | <= : %d' % (thr, (a < thr).sum(), (a <= thr).sum()))

froz = pd.read_csv(R + '/FINAL_20260928/data_current/s1_hospital_auroc.csv').rename(columns={'0': 'auc_f'})
chk = h.merge(froz, on='hospitalid')
print('joined with frozen csv: %d rows | max |my-frozen| = %.6f' %
      (len(chk), (chk.auc - chk.auc_f).abs().max()))

# ================= part 2: precision boundary numbers =================
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
b3 = fit_lr(jf[['age', 'charlson', 'cr_base']].fillna(0).values[tr], jf.ge1.values[tr])
bR = fit_lr(jf[VARS].fillna(0).values[tr], jf.ge1.values[tr])


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
pm = pd.read_parquet(J + '/f12_preds_M4_ge1.parquet')
dd = pm.merge(m4f, on=['stay_id', 't_hr'], suffixes=('', '_f'))
X3 = dd[['age', 'charlson', 'cr_base']].fillna(0).values
p3 = expit(b3[0] + X3 @ b3[1:])
print()
print('MIMIC static-3 AUROC (mine, 6dp): %.6f  (frozen 0.6116)' % auroc(dd['y'].values, p3))
rng = np.random.default_rng(SEED)
uniq = dd['stay_id'].unique()
idx = {u: np.where(dd['stay_id'].values == u)[0] for u in uniq}
yv = dd['y'].values
pmv = dd['p'].values
diffs = []
for _ in range(1000):
    sel = rng.choice(uniq, size=len(uniq), replace=True)
    ii = np.concatenate([idx[u] for u in sel])
    if yv[ii].sum() > 0 and (1 - yv[ii]).sum() > 0:
        diffs.append(auroc(yv[ii], pmv[ii]) - auroc(yv[ii], p3[ii]))
dfn = np.array(diffs)
print('MIMIC static-3 delta mean %.6f lo %.6f hi %.6f  (frozen 0.1578/0.1305/0.1854; ms 0.158 (0.131, 0.185))'
      % (dfn.mean(), np.percentile(dfn, 2.5), np.percentile(dfn, 97.5)))

eicu = pd.read_parquet(R + '/results/p1_expanded_features.parquet')
elab = pd.read_parquet(R + '/results/n8_2_labs_series.parquet')
ecr = elab[(elab.labname == 'creatinine') & (elab.labtypeid == 1)].rename(
    columns={'sid': 'stay_id', 'labresult': 'cr_val'})
ecr = ecr[ecr.cr_val.between(0.1, 30)].copy()
ecr['t_hr'] = (ecr.hr // 6) * 6
eif = eicu.merge(label_ge1(ecr[['stay_id', 't_hr', 'cr_val']], eicu, 'stay_id'),
                 on=['stay_id', 't_hr'], how='left')
pe2 = pd.read_parquet(J + '/f12_preds_eICU_ge1.parquet')
dd2 = pe2.merge(eif, on=['stay_id', 't_hr'], suffixes=('', '_f'))
XR2 = dd2[VARS].fillna(0).values
pR2 = expit(bR[0] + XR2 @ bR[1:])
print('eICU rich-8 AUROC (mine, 6dp): %.6f  (frozen 0.6335; ms 0.634)' % auroc(dd2['y'].values, pR2))
XM3 = dd2[['age', 'charlson', 'cr_base']].fillna(0).values
p3e = expit(b3[0] + XM3 @ b3[1:])
print('eICU static-3 AUROC (mine, 6dp): %.6f  (frozen 0.6129)' % auroc(dd2['y'].values, p3e))
XM4R = dd[VARS].fillna(0).values
pRm = expit(bR[0] + XM4R @ bR[1:])
print('MIMIC rich-8 AUROC (mine, 6dp): %.6f  (frozen 0.624)' % auroc(dd['y'].values, pRm))

# -*- coding: utf-8 -*-
"""s3: the two benchmark-gap statistics.
 1. DeLong paired test: primary (>=Stage 1) model vs admission-variable LR
    baseline, on the same internal-test and external checkpoints.
    (Fast DeLong, Sun-Xu midrank implementation — same as the v1 chain.)
 2. Decision-curve analysis on the recalibrated probabilities, internal test
    + external, thresholds 0.05-0.50 (net benefit vs treat-all / treat-none).
Outputs: results/s3_stats.json + LR prediction parquets + DCA curve csv."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = "results"
FEATS = json.load(open(f"{R}/_frozen291_features.json"))
pi = pd.read_parquet(f"{R}/s1_preds_internal.parquet")
pe = pd.read_parquet(f"{R}/s1_preds_external.parquet")

# ---------- retrain LR baseline on the identical split ----------
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
labs = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet").rename(columns={'valuenum': 'cr_val'})
labs = labs[labs['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
labs['t_hr'] = (labs['hr'] // 6) * 6
cb = labs.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
mg = m4[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
mg['f'] = np.nan
for s in range(9):
    sh = mg.groupby('stay_id')['cr_val'].shift(-s)
    mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)
fold = mg['f'] / mg['cr_base'].clip(lower=0.1)
st = pd.Series(0, index=mg.index)
st[(fold >= 1.5) | ((mg['f'] - mg['cr_base']) >= 0.3)] = 1
st[fold >= 2.0] = 2
st[(fold >= 3.0) | (mg['f'] >= 4.0)] = 3
st[mg['f'].isna()] = 0
mg['ge1'] = (st >= 1).astype(int)
m4f = m4.merge(mg[['stay_id', 't_hr', 'ge1']], on=['stay_id', 't_hr'], how='left')

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(42)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr
CAND = [c for c in ['age', 'charlson', 'cr_base'] if c in FEATS]
lr = LogisticRegression(max_iter=2000, class_weight='balanced')
lr.fit(m4f[CAND].fillna(0).values[tr], m4f['ge1'].values[tr])
eif = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[['stay_id'] + FEATS]
lr_pi = lr.predict_proba(m4f[CAND].fillna(0).values[te])[:, 1]
lr_pe = lr.predict_proba(eif[CAND].fillna(0).values)[:, 1]
pd.DataFrame({'stay_id': m4f.loc[te, 'stay_id'].values, 'p': lr_pi}).to_parquet(f"{R}/s3_lr_preds_int.parquet", index=False)
pd.DataFrame({'stay_id': eif['stay_id'].values, 'p': lr_pe}).to_parquet(f"{R}/s3_lr_preds_ext.parquet", index=False)
print('LR AUCs: int', round(roc_auc_score(pi.y, lr_pi), 4), 'ext', round(roc_auc_score(pe.y, lr_pe), 4))


# ---------- fast DeLong (Sun-Xu midrank) ----------
def midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j + 1)
        i = j
    T2 = np.empty(N, float)
    T2[J] = T
    return T2


def delong_components(y, p):
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    x1, x0 = np.ascontiguousarray(p[pos]), np.ascontiguousarray(p[~pos])
    tx = midrank(np.r_[x1, x0])
    tz = np.r_[midrank(x1), midrank(x0)]
    v10 = (tx[:n1] - tz[:n1]) / n0
    v01 = (tx[n1:] - tz[n1:]) / n1
    s1 = np.var(v10, ddof=1) / n1
    s0 = np.var(v01, ddof=1) / n0
    return v10, v01, s1, s0


def delong_paired(y, p1, p2):
    a1, a2 = roc_auc_score(y, p1), roc_auc_score(y, p2)
    v10a, v01a, s1a, s0a = delong_components(y, p1)
    v10b, v01b, s1b, s0b = delong_components(y, p2)
    cov1 = np.cov(np.stack([v10a, v10b]), ddof=1)[0, 1] / len(v10a)
    cov0 = np.cov(np.stack([v01a, v01b]), ddof=1)[0, 1] / len(v01a)
    var_diff = cov1 + cov0
    diff = a1 - a2
    z = diff / np.sqrt(var_diff)
    from scipy.stats import norm
    pval = 2 * (1 - norm.cdf(abs(z)))
    return {'auc_primary': round(a1, 4), 'auc_baseline': round(a2, 4),
            'diff': round(diff, 4), 'se': round(np.sqrt(var_diff), 4),
            'z': round(z, 2), 'p': f'{pval:.2e}' if pval < 0.001 else round(pval, 4)}


delong = {'internal_test': delong_paired(pi.y.values, pi.p.values, lr_pi),
          'external': delong_paired(pe.y.values, pe.p.values, lr_pe)}
print('DeLong:', json.dumps(delong))

# ---------- DCA ----------
def dca(y, p, thrs):
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thrs:
        al = p >= t
        tp = (al & (y == 1)).sum() / n
        fp = (al & (y == 0)).sum() / n
        nb = tp - fp * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        rows.append({'thr': round(t, 3), 'nb_model': round(nb, 5),
                     'nb_treat_all': round(nb_all, 5), 'nb_none': 0.0})
    return rows


ths = np.arange(0.05, 0.505, 0.025)
dca_int = dca(pi.y.values, pi.p_rec.values, ths)
dca_ext = dca(pe.y.values, pe.p_rec.values, ths)
# range where model NB > treat-all
def wins(rows):
    return all(r['nb_model'] > r['nb_treat_all'] for r in rows)
print('DCA model>treat-all across 0.05-0.50: int', wins(dca_int), 'ext', wins(dca_ext))

pd.DataFrame(dca_int).assign(setting='internal_test').to_csv(f'{R}/s3_dca_curves.csv', index=False)
pd.DataFrame(dca_ext).assign(setting='external').to_csv(f'{R}/s3_dca_curves.csv', mode='a', index=False)

json.dump({'delong': delong,
           'dca': {'internal_test': dca_int, 'external': dca_ext,
                   'model_beats_treat_all_full_range': {'int': wins(dca_int), 'ext': wins(dca_ext)},
                   'nb_ext_at_015': dca_ext[[r['thr'] for r in dca_ext].index(round(0.15, 3))]},
                   }, open(f'{R}/s3_stats.json', 'w'), indent=2)
print('saved s3_stats.json + s3_dca_curves.csv')

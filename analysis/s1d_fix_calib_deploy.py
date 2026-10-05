# -*- coding: utf-8 -*-
"""s1d: fix the recalibration layer for the >=Stage 1 primary model.

Bug in s1: the logistic layer was fitted on IN-SAMPLE train predictions
(the model had seen those rows), producing a miscalibrated layer (slope
dropped 0.51 -> 0.21). Fix: fit the layer on OUT-OF-FOLD train predictions
(5-fold cross_val_predict), apply to internal test + external; recompute
calibration metrics and the deployment views; refresh p_rec in the saved
prediction parquets and update s1_stage_primary.json in place."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = "results"
FEATS = json.load(open(f"{R}/_frozen291_features.json"))
PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
              class_weight='balanced', random_state=42, verbose=-1)
m4 = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")[['stay_id'] + FEATS]
eicu = pd.read_parquet(f"{R}/p1_expanded_features.parquet")[['stay_id'] + FEATS]


def stage_ge1(labs_parquet, feat_df):
    labs = pd.read_parquet(labs_parquet)
    cr = labs.rename(columns={'valuenum': 'cr_val'}) if 'sid' not in labs.columns else \
        labs[labs['labname'] == 'creatinine'].rename(columns={'sid': 'stay_id', 'labresult': 'cr_val'})
    cr = cr[cr['cr_val'].between(0.1, 30)].sort_values(['stay_id', 'hr'])
    cr['t_hr'] = (cr['hr'] // 6) * 6
    cb = cr.groupby(['stay_id', 't_hr'])['cr_val'].max().reset_index().sort_values(['stay_id', 't_hr'])
    mg = feat_df[['stay_id', 't_hr', 'cr_base']].drop_duplicates().merge(cb, on=['stay_id', 't_hr'], how='left').sort_values(['stay_id', 't_hr'])
    mg['f'] = np.nan
    for s in range(9):
        sh = mg.groupby('stay_id')['cr_val'].shift(-s)
        mg['f'] = pd.DataFrame({'c': mg['f'], 'n': sh}).max(axis=1)
    b = mg['cr_base'].clip(lower=0.1); fold = mg['f'] / b
    st = pd.Series(0, index=mg.index)
    st[(fold >= 1.5) | ((mg['f'] - mg['cr_base']) >= 0.3)] = 1
    st[fold >= 2.0] = 2
    st[(fold >= 3.0) | (mg['f'] >= 4.0)] = 3
    st[mg['f'].isna()] = 0
    mg['ge1'] = (st >= 1).astype(int)
    crit = ((fold >= 1.5) | ((mg['f'] - mg['cr_base']) >= 0.3)).fillna(False) & (mg['t_hr'] >= 24)
    mg['crit'] = crit
    onsets = {sid: g['t_hr'].min() for sid, g in mg[mg['crit']].groupby('stay_id')}
    return mg[['stay_id', 't_hr', 'ge1']], onsets


m4_lab, m4_onset = stage_ge1(f"{R}/n2_m4_abi_cr_serial.parquet", m4)
ei_lab, ei_onset = stage_ge1(f"{R}/n8_2_labs_series.parquet", eicu)
m4f = m4.merge(m4_lab, on=['stay_id', 't_hr'], how='left')
eif = eicu.merge(ei_lab, on=['stay_id', 't_hr'], how='left')

n1 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet", columns=['stay_id', 'subject_id'])
s2sub = dict(zip(n1['stay_id'], n1['subject_id']))
subjects = np.array(sorted({s2sub[s] for s in m4f['stay_id'].unique()}))
np.random.seed(42)
perm = np.random.permutation(len(subjects))
train_subj = set(subjects[perm[:int(0.8 * len(subjects))]])
tr = m4f['stay_id'].map(s2sub).isin(train_subj).values
te = ~tr

Xtr, Xte, Xee = m4f[FEATS].values[tr], m4f[FEATS].values[te], eif[FEATS].values
ytr, yte, yee = m4f['ge1'].values[tr], m4f['ge1'].values[te], eif['ge1'].values

mod = lgb.LGBMClassifier(**PARAMS)
mod.fit(Xtr, ytr)
print("re-trained ge1 model (AUROC check):",
      round(roc_auc_score(yte, mod.predict_proba(Xte)[:, 1]), 4))

# ---- OOF recalibration layer ----
print("cross_val_predict (5-fold) for OOF train predictions...", flush=True)
oof = cross_val_predict(lgb.LGBMClassifier(**PARAMS), Xtr, ytr, cv=5,
                        method='predict_proba', n_jobs=None)[:, 1]


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


cal = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1, 1), ytr)
a, b0 = float(cal.intercept_[0]), float(cal.coef_[0][0])
print(f"OOF recal layer: a={a:.4f} b={b0:.4f}")

pte = mod.predict_proba(Xte)[:, 1]
pee = mod.predict_proba(Xee)[:, 1]
ptr = mod.predict_proba(Xtr)[:, 1]


def ece10(y, p):
    q = pd.qcut(pd.Series(p), 10, duplicates='drop')
    g = pd.DataFrame({'y': y, 'p': p, 'q': q}).groupby('q', observed=True)
    return float((g.size() / len(y) * np.abs(g['y'].mean() - g['p'].mean())).sum())


def metrics(y, p):
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return {'slope': round(float(lr.coef_[0][0]), 3), 'ece': round(ece10(y, p), 4),
            'brier': round(float(np.mean((p - y) ** 2)), 4)}


calib = {'recal': {'a': round(a, 4), 'b': round(b0, 4), 'fitted_on': '5-fold out-of-fold train predictions'}}
for nm, yy, pp in [('train_oof', ytr, oof), ('int_test', yte, pte), ('ext', yee, pee)]:
    pr = sigmoid(a + b0 * logit(pp))
    calib[nm] = {'raw': metrics(yy, pp), 'recal': metrics(yy, pr),
                 'brier_null': round(float(np.mean((np.mean(yy) - yy) ** 2)), 4)}
    print(f"  {nm}: raw {calib[nm]['raw']} -> recal {calib[nm]['recal']}")

# ---- deployment with fixed layer ----
pi = pd.read_parquet(f"{R}/s1_preds_internal.parquet")
pe = pd.read_parquet(f"{R}/s1_preds_external.parquet")
pi['p_rec'] = sigmoid(a + b0 * logit(pi['p']))
pe['p_rec'] = sigmoid(a + b0 * logit(pe['p']))


def deploy(df, onsets, thr):
    d = df.copy()
    d['alert'] = d['p_rec'] >= thr
    ev = d[d['y'] == 1].copy()
    nev = d[d['y'] == 0].copy()
    ev['onset'] = ev['stay_id'].map(onsets)
    capt, leads = [], []
    for sid, g in ev.groupby('stay_id'):
        al = g[g['alert'] & (g['t_hr'] < g['onset'])]
        if len(al):
            capt.append(sid)
            leads.append(g['onset'].min() - al['t_hr'].min())
    alert_stays = set(d.loc[d['alert'], 'stay_id'])
    n_capt = len(capt)
    nne = len(alert_stays) / n_capt if n_capt else float('nan')
    fa_stays = alert_stays & set(nev['stay_id'])
    days = nev.groupby('stay_id')['t_hr'].max().clip(lower=6).div(24).sum()
    return {'thr': thr, 'capture': round(n_capt / ev['stay_id'].nunique(), 3),
            'lead_median_h': round(float(np.median(leads)), 1) if leads else None,
            'lead_iqr_h': [round(float(np.percentile(leads, 25)), 1),
                           round(float(np.percentile(leads, 75)), 1)] if leads else None,
            'NNE': round(nne, 2), 'FA_per_100ptd': round(100 * len(fa_stays) / days, 1)}


deploy_int = {f"thr{t}": deploy(pi, m4_onset, t) for t in [0.10, 0.20, 0.30]}
deploy_ext = {f"thr{t}": deploy(pe, ei_onset, t) for t in [0.10, 0.20, 0.30]}
print("ext thr0.20:", json.dumps(deploy_ext['thr0.2']))
print("int thr0.20:", json.dumps(deploy_int['thr0.2']))

pi.to_parquet(f"{R}/s1_preds_internal.parquet", index=False)
pe.to_parquet(f"{R}/s1_preds_external.parquet", index=False)

s1 = json.load(open(f"{R}/s1_stage_primary.json"))
s1['calibration_ge1'] = calib
s1['deployment_ge1'] = {'internal_test': deploy_int, 'external': deploy_ext}
s1['_meta']['calibration_note'] = 'recalibration layer fitted on 5-fold out-of-fold train predictions (s1d fix; in-sample fit was miscalibrated)'
json.dump(s1, open(f"{R}/s1_stage_primary.json", "w"), indent=2)
print("s1_stage_primary.json updated (calibration + deployment fixed)")
